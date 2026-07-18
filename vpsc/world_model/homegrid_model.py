"""Shared action-conditioned HomeGrid world models for the E2-M0 pilot.

The visual/language/action/read encoder and every prediction head are identical
across models.  Only the temporal core changes between a one-layer LSTM, a
one-layer causal Transformer with an explicit 128-transition KV cache, and the
frozen E2 ``hybrid`` condition with ``positive_factor=0.8``.

This module defines model interfaces and the fair factory only.  Losses,
training, dataset loading, and rollout policy deliberately live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

import torch
import torch.nn as nn

from .cores import (
    CausalTransformerCore,
    CoreState,
    E2SignedCore,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
)
from .lm import ModuleParameterStats, module_parameter_stats


Tensor = torch.Tensor
StateT = TypeVar("StateT", bound=CoreState)

VISUAL_PATCHES = 144
VISUAL_VOCAB_SIZE = 64
VISUAL_EMBEDDING_DIM = 8
ACTION_COUNT = 10
READ_CLASS_COUNT = 2
REWARD_CLASS_COUNT = 3
DONE_CLASS_COUNT = 2
TRANSFORMER_CACHE_TOKENS = 128
PARAMETER_TOLERANCE = 0.02
E2_POLICY = "hybrid"
E2_POSITIVE_FACTOR = 0.8


@dataclass(frozen=True)
class HomeGridWorldModelConfig:
    """Frozen shared dimensions for one fair HomeGrid model suite."""

    language_vocab_size: int
    d_model: int = 32
    num_heads: int = 4
    transformer_cache_tokens: int = TRANSFORMER_CACHE_TOKENS
    parameter_tolerance: float = PARAMETER_TOLERANCE

    def __post_init__(self) -> None:
        if self.language_vocab_size <= 1:
            raise ValueError("language_vocab_size must be greater than one")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("num_heads must be positive and divide d_model")
        if self.transformer_cache_tokens <= 0:
            raise ValueError("transformer_cache_tokens must be positive")
        if self.parameter_tolerance <= 0.0:
            raise ValueError("parameter_tolerance must be positive")


@dataclass
class HomeGridWorldModelOutput(Generic[StateT]):
    """Predictions aligned to each input transition plus explicit core state."""

    next_visual_logits: Tensor
    next_language_logits: Tensor
    next_read_logits: Tensor
    reward_logits: Tensor
    done_logits: Tensor
    state: StateT
    hidden_states: Tensor

    @property
    def last_next_visual_logits(self) -> Tensor:
        return self.next_visual_logits[:, -1]

    @property
    def last_next_language_logits(self) -> Tensor:
        return self.next_language_logits[:, -1]


@dataclass(frozen=True)
class HomeGridModelParameterStats:
    """De-duplicated parameter counts for one multimodal model."""

    model: ModuleParameterStats
    core: ModuleParameterStats
    encoder: ModuleParameterStats
    output_norm: ModuleParameterStats
    heads: ModuleParameterStats
    shared_wrapper: ModuleParameterStats
    core_type: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_total": self.model.total,
            "model_trainable": self.model.trainable,
            "core_total": self.core.total,
            "core_trainable": self.core.trainable,
            "encoder_total": self.encoder.total,
            "encoder_trainable": self.encoder.trainable,
            "output_norm_total": self.output_norm.total,
            "output_norm_trainable": self.output_norm.trainable,
            "heads_total": self.heads.total,
            "heads_trainable": self.heads.trainable,
            "shared_wrapper_total": self.shared_wrapper.total,
            "shared_wrapper_trainable": self.shared_wrapper.trainable,
            "core_type": self.core_type,
        }


def _normal_embedding_(embedding: nn.Embedding) -> None:
    nn.init.normal_(
        embedding.weight,
        mean=0.0,
        std=embedding.embedding_dim**-0.5,
    )


def _normal_linear_(linear: nn.Linear) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=linear.in_features**-0.5)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class HomeGridMultimodalEncoder(nn.Module):
    """Encode one ordered 12x12 visual grid plus language/action/read inputs."""

    def __init__(self, language_vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.language_vocab_size = int(language_vocab_size)
        self.d_model = int(d_model)
        self.visual_embedding = nn.Embedding(
            VISUAL_VOCAB_SIZE,
            VISUAL_EMBEDDING_DIM,
        )
        self.patch_position_embedding = nn.Parameter(
            torch.empty(VISUAL_PATCHES, VISUAL_EMBEDDING_DIM)
        )
        self.visual_projection = nn.Linear(
            VISUAL_PATCHES * VISUAL_EMBEDDING_DIM,
            d_model,
        )
        self.language_embedding = nn.Embedding(language_vocab_size, d_model)
        self.action_embedding = nn.Embedding(ACTION_COUNT, d_model)
        self.read_embedding = nn.Embedding(READ_CLASS_COUNT, d_model)
        self.fusion_projection = nn.Linear(4 * d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the same width-aware scale for every shared model instance."""

        _normal_embedding_(self.visual_embedding)
        nn.init.normal_(
            self.patch_position_embedding,
            mean=0.0,
            std=VISUAL_EMBEDDING_DIM**-0.5,
        )
        _normal_linear_(self.visual_projection)
        _normal_embedding_(self.language_embedding)
        _normal_embedding_(self.action_embedding)
        _normal_embedding_(self.read_embedding)
        _normal_linear_(self.fusion_projection)
        self.fusion_norm.reset_parameters()

    def forward(
        self,
        visual_tokens: Tensor,
        language_ids: Tensor,
        actions: Tensor,
        read_flags: Tensor,
    ) -> Tensor:
        batch, time, _ = visual_tokens.shape
        visual = self.visual_embedding(visual_tokens)
        visual = visual + self.patch_position_embedding.view(
            1,
            1,
            VISUAL_PATCHES,
            VISUAL_EMBEDDING_DIM,
        )
        # Patch order is retained: each of the 144 positions owns a distinct
        # contiguous block in the flattened projection input.
        visual = self.visual_projection(
            visual.reshape(batch, time, VISUAL_PATCHES * VISUAL_EMBEDDING_DIM)
        )
        fused = torch.cat(
            (
                visual,
                self.language_embedding(language_ids),
                self.action_embedding(actions),
                self.read_embedding(read_flags.to(dtype=torch.long)),
            ),
            dim=-1,
        )
        return self.fusion_norm(self.fusion_projection(fused))


class HomeGridWorldModel(nn.Module, Generic[StateT]):
    """Action-conditioned multimodal predictor with branchable temporal state."""

    def __init__(
        self,
        language_vocab_size: int,
        d_model: int,
        core: TemporalCore[StateT],
    ) -> None:
        super().__init__()
        if language_vocab_size <= 1:
            raise ValueError("language_vocab_size must be greater than one")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not isinstance(core, TemporalCore):
            raise TypeError("core must implement TemporalCore")
        if core.input_dim != d_model or core.output_dim != d_model:
            raise ValueError(
                "core input/output dimensions must both equal d_model, got "
                f"{core.input_dim}/{core.output_dim} and {d_model}"
            )

        self.language_vocab_size = int(language_vocab_size)
        self.d_model = int(d_model)
        self.core = core
        self.encoder = HomeGridMultimodalEncoder(language_vocab_size, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict(
            {
                "next_visual": nn.Linear(
                    d_model,
                    VISUAL_PATCHES * VISUAL_VOCAB_SIZE,
                ),
                "next_language": nn.Linear(d_model, language_vocab_size),
                "next_read": nn.Linear(d_model, READ_CLASS_COUNT),
                "reward": nn.Linear(d_model, REWARD_CLASS_COUNT),
                "done": nn.Linear(d_model, DONE_CLASS_COUNT),
            }
        )
        self._reset_shared_output_parameters()

    def _reset_shared_output_parameters(self) -> None:
        self.output_norm.reset_parameters()
        for head in self.heads.values():
            if not isinstance(head, nn.Linear):  # pragma: no cover - construction invariant
                raise TypeError("all HomeGrid heads must be linear")
            _normal_linear_(head)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> StateT:
        if device is None:
            device = self.encoder.visual_embedding.weight.device
        if dtype is None:
            dtype = self.encoder.visual_embedding.weight.dtype
        return self.core.initial_state(batch_size, device=device, dtype=dtype)

    @staticmethod
    def _require_tensor(value: Tensor, name: str) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    @staticmethod
    def _validate_integer_ids(
        value: Tensor,
        name: str,
        *,
        minimum: int,
        maximum: int,
        allow_bool: bool = False,
    ) -> None:
        allowed = (torch.int32, torch.int64)
        if value.dtype not in allowed and not (allow_bool and value.dtype == torch.bool):
            expected = "bool/int32/int64" if allow_bool else "int32 or int64"
            raise TypeError(f"{name} must use {expected}")
        if value.numel() == 0:
            raise ValueError(f"{name} cannot be empty")
        observed_min = int(value.min().item())
        observed_max = int(value.max().item())
        if observed_min < minimum or observed_max > maximum:
            raise ValueError(
                f"{name} IDs must be in [{minimum}, {maximum}], got "
                f"[{observed_min}, {observed_max}]"
            )

    def _validate_sequence_inputs(
        self,
        visual_tokens: Tensor,
        language_ids: Tensor,
        actions: Tensor,
        read_flags: Tensor,
    ) -> None:
        for name, value in (
            ("visual_tokens", visual_tokens),
            ("language_ids", language_ids),
            ("actions", actions),
            ("read_flags", read_flags),
        ):
            self._require_tensor(value, name)
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != VISUAL_PATCHES:
            raise ValueError(
                "visual_tokens must be shaped [batch, time, 144], got "
                f"{tuple(visual_tokens.shape)}"
            )
        expected = tuple(visual_tokens.shape[:2])
        if expected[0] == 0 or expected[1] == 0:
            raise ValueError("HomeGrid inputs cannot have an empty batch or time axis")
        for name, value in (
            ("language_ids", language_ids),
            ("actions", actions),
            ("read_flags", read_flags),
        ):
            if value.ndim != 2 or tuple(value.shape) != expected:
                raise ValueError(f"{name} must be shaped {expected}, got {tuple(value.shape)}")
        devices = {
            visual_tokens.device,
            language_ids.device,
            actions.device,
            read_flags.device,
        }
        if len(devices) != 1:
            raise ValueError("all HomeGrid inputs must be on the same device")
        self._validate_integer_ids(
            visual_tokens,
            "visual_tokens",
            minimum=0,
            maximum=VISUAL_VOCAB_SIZE - 1,
        )
        self._validate_integer_ids(
            language_ids,
            "language_ids",
            minimum=0,
            maximum=self.language_vocab_size - 1,
        )
        self._validate_integer_ids(
            actions,
            "actions",
            minimum=0,
            maximum=ACTION_COUNT - 1,
        )
        self._validate_integer_ids(
            read_flags,
            "read_flags",
            minimum=0,
            maximum=1,
            allow_bool=True,
        )

    def _decode(self, hidden_states: Tensor, state: StateT) -> HomeGridWorldModelOutput[StateT]:
        batch, time, _ = hidden_states.shape
        next_visual = self.heads["next_visual"](hidden_states).reshape(
            batch,
            time,
            VISUAL_PATCHES,
            VISUAL_VOCAB_SIZE,
        )
        return HomeGridWorldModelOutput(
            next_visual_logits=next_visual,
            next_language_logits=self.heads["next_language"](hidden_states),
            next_read_logits=self.heads["next_read"](hidden_states),
            reward_logits=self.heads["reward"](hidden_states),
            done_logits=self.heads["done"](hidden_states),
            state=state,
            hidden_states=hidden_states,
        )

    def forward(
        self,
        visual_tokens: Tensor,
        language_ids: Tensor,
        actions: Tensor,
        read_flags: Tensor,
        state: Optional[StateT] = None,
        *,
        detach_state: bool = False,
    ) -> HomeGridWorldModelOutput[StateT]:
        """Process inputs shaped ``[batch,time,...]`` without crossing episodes."""

        self._validate_sequence_inputs(
            visual_tokens,
            language_ids,
            actions,
            read_flags,
        )
        encoded = self.encoder(visual_tokens, language_ids, actions, read_flags)
        core_output = self.core(encoded, state, detach_state=detach_state)
        hidden = self.output_norm(core_output.sequence)
        return self._decode(hidden, core_output.state)

    def step(
        self,
        visual_tokens: Tensor,
        language_ids: Tensor,
        actions: Tensor,
        read_flags: Tensor,
        state: Optional[StateT] = None,
        *,
        detach_state: bool = False,
    ) -> HomeGridWorldModelOutput[StateT]:
        """Consume one transition per batch and preserve a length-one time axis."""

        for name, value, expected_tail in (
            ("visual_tokens", visual_tokens, (VISUAL_PATCHES,)),
            ("language_ids", language_ids, ()),
            ("actions", actions, ()),
            ("read_flags", read_flags, ()),
        ):
            self._require_tensor(value, name)
            if tuple(value.shape[1:]) != expected_tail or value.ndim != len(expected_tail) + 1:
                expected = "[batch, 144]" if expected_tail else "[batch]"
                raise ValueError(f"{name} must be shaped {expected}, got {tuple(value.shape)}")
        self._validate_sequence_inputs(
            visual_tokens.unsqueeze(1),
            language_ids.unsqueeze(1),
            actions.unsqueeze(1),
            read_flags.unsqueeze(1),
        )
        encoded = self.encoder(
            visual_tokens.unsqueeze(1),
            language_ids.unsqueeze(1),
            actions.unsqueeze(1),
            read_flags.unsqueeze(1),
        )
        core_output = self.core.step(
            encoded[:, 0],
            state,
            detach_state=detach_state,
        )
        hidden = self.output_norm(core_output.sequence)
        return self._decode(hidden, core_output.state)

    def parameter_stats(self) -> HomeGridModelParameterStats:
        model = module_parameter_stats(self)
        core = module_parameter_stats(self.core)
        return HomeGridModelParameterStats(
            model=model,
            core=core,
            encoder=module_parameter_stats(self.encoder),
            output_norm=module_parameter_stats(self.output_norm),
            heads=module_parameter_stats(self.heads),
            shared_wrapper=ModuleParameterStats(
                total=model.total - core.total,
                trainable=model.trainable - core.trainable,
            ),
            core_type=type(self.core).__name__,
        )


@dataclass(frozen=True)
class HomeGridParameterBudgetReport:
    totals: Dict[str, int]
    cores: Dict[str, int]
    relative_spread: float
    tolerance: float

    @property
    def within_tolerance(self) -> bool:
        return self.relative_spread <= self.tolerance + 1e-12

    def as_dict(self) -> Dict[str, Any]:
        return {
            "totals": dict(self.totals),
            "cores": dict(self.cores),
            "relative_spread": self.relative_spread,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class HomeGridModelSuite:
    config: HomeGridWorldModelConfig
    lstm: HomeGridWorldModel[Any]
    transformer: HomeGridWorldModel[Any]
    e2: HomeGridWorldModel[Any]
    parameter_report: HomeGridParameterBudgetReport

    @property
    def models(self) -> Dict[str, HomeGridWorldModel[Any]]:
        return {
            "lstm": self.lstm,
            "transformer": self.transformer,
            "e2": self.e2,
        }


class HomeGridParameterBudgetError(AssertionError):
    """Raised when the shared HomeGrid suite exceeds the frozen 2% spread."""


def _parameter_report(
    models: Dict[str, HomeGridWorldModel[Any]],
    tolerance: float,
) -> HomeGridParameterBudgetReport:
    totals = {name: count_parameters(model) for name, model in models.items()}
    cores = {name: count_parameters(model.core) for name, model in models.items()}
    mean = sum(totals.values()) / len(totals)
    relative_spread = (max(totals.values()) - min(totals.values())) / mean
    return HomeGridParameterBudgetReport(
        totals=totals,
        cores=cores,
        relative_spread=float(relative_spread),
        tolerance=float(tolerance),
    )


def assert_homegrid_parameter_budget(
    suite_or_report: HomeGridModelSuite | HomeGridParameterBudgetReport,
    tolerance: Optional[float] = None,
) -> HomeGridParameterBudgetReport:
    report = (
        suite_or_report.parameter_report
        if isinstance(suite_or_report, HomeGridModelSuite)
        else suite_or_report
    )
    allowed = report.tolerance if tolerance is None else float(tolerance)
    if allowed <= 0.0:
        raise ValueError("tolerance must be positive")
    if report.relative_spread > allowed + 1e-12:
        counts = ", ".join(f"{name}={value}" for name, value in report.totals.items())
        raise HomeGridParameterBudgetError(
            f"HomeGrid model parameter spread {report.relative_spread:.4%} exceeds "
            f"{allowed:.4%}: {counts}"
        )
    return report


def build_homegrid_model_suite(
    language_vocab_size: int,
    d_model: int = 32,
    *,
    num_heads: int = 4,
    transformer_cache_tokens: int = TRANSFORMER_CACHE_TOKENS,
    parameter_tolerance: float = PARAMETER_TOLERANCE,
) -> HomeGridModelSuite:
    """Build the frozen M0 suite and enforce total-parameter fairness."""

    config = HomeGridWorldModelConfig(
        language_vocab_size=language_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        transformer_cache_tokens=transformer_cache_tokens,
        parameter_tolerance=parameter_tolerance,
    )
    models: Dict[str, HomeGridWorldModel[Any]] = {
        "lstm": HomeGridWorldModel(
            language_vocab_size,
            d_model,
            StatefulLSTMCore(
                d_model,
                d_model,
                num_layers=1,
                dropout=0.0,
            ),
        ),
        "transformer": HomeGridWorldModel(
            language_vocab_size,
            d_model,
            CausalTransformerCore(
                d_model,
                d_model,
                num_layers=1,
                num_heads=num_heads,
                mlp_ratio=2.0,
                dropout=0.0,
                max_cache_tokens=transformer_cache_tokens,
            ),
        ),
        "e2": HomeGridWorldModel(
            language_vocab_size,
            d_model,
            E2SignedCore(
                d_model,
                d_model,
                policy=E2_POLICY,
                positive_factor=E2_POSITIVE_FACTOR,
            ),
        ),
    }
    report = _parameter_report(models, parameter_tolerance)
    suite = HomeGridModelSuite(
        config=config,
        lstm=models["lstm"],
        transformer=models["transformer"],
        e2=models["e2"],
        parameter_report=report,
    )
    assert_homegrid_parameter_budget(suite, tolerance=parameter_tolerance)
    return suite


__all__ = [
    "ACTION_COUNT",
    "DONE_CLASS_COUNT",
    "E2_POLICY",
    "E2_POSITIVE_FACTOR",
    "HomeGridModelParameterStats",
    "HomeGridModelSuite",
    "HomeGridMultimodalEncoder",
    "HomeGridParameterBudgetError",
    "HomeGridParameterBudgetReport",
    "HomeGridWorldModel",
    "HomeGridWorldModelConfig",
    "HomeGridWorldModelOutput",
    "PARAMETER_TOLERANCE",
    "READ_CLASS_COUNT",
    "REWARD_CLASS_COUNT",
    "TRANSFORMER_CACHE_TOKENS",
    "VISUAL_EMBEDDING_DIM",
    "VISUAL_PATCHES",
    "VISUAL_VOCAB_SIZE",
    "assert_homegrid_parameter_budget",
    "build_homegrid_model_suite",
]
