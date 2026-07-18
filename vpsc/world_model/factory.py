"""Frozen, parameter-audited factory for the three causal-LM baselines.

The factory changes only the temporal core.  Vocabulary size, token embedding
width, wrapper dropout, tied embedding/LM-head weights, padding convention, and
head bias are shared by the LSTM, Transformer, and E2 models.

The default core definitions are intentionally narrow and frozen:

* one-layer LSTM;
* one-layer causal Transformer with a real explicit KV cache and MLP ratio 2;
* signed E/I E2 core with ``state_dim == d_model``.

For unusually small vocabularies the shared embedding/head no longer dominates
the few bias terms by which the cores differ.  If explicitly enabled, automatic
matching searches only a small neighbourhood of E2 state width and Transformer
MLP hidden width.  Its objective is parameter-count spread alone; it never sees
training, validation, or test results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from .cores import (
    CausalTransformerCore,
    E2SignedCore,
    FeedbackPolicy,
    StatefulLSTMCore,
)
from .lm import CausalLanguageModel


DEFAULT_PARAMETER_TOLERANCE = 0.02
_E2_STATE_OFFSETS: Tuple[int, ...] = (0, -1, 1)
_TRANSFORMER_MLP_HIDDEN_OFFSETS: Tuple[int, ...] = (
    0,
    -1,
    1,
    -2,
    2,
    -3,
    3,
    -4,
    4,
)


@dataclass(frozen=True)
class FrozenE2Config:
    """Frozen E2 policy, ablations, and dynamics passed verbatim to the core."""

    state_dim: Optional[int] = None
    policy: FeedbackPolicy = "exact"
    no_positive: bool = False
    state_reset: bool = False
    margin_scale: float = 0.95
    positive_factor: float = 1.0
    negative_factor: float = 1.0
    hybrid_cutoff: float = 1.0
    g_ee: float = 7.75
    g_ei: float = 6.70
    g_ie: float = 10.0
    g_ii: float = 6.30
    theta_e: float = 2.50
    theta_i: float = 5.75
    tau_e: float = 1.0
    tau_i: float = 5.80
    dt: float = 1.0
    micro_steps: int = 1

    def __post_init__(self) -> None:
        if self.state_dim is not None and self.state_dim <= 0:
            raise ValueError("E2 state_dim must be positive or None")
        if self.policy not in {
            "exact",
            "margin",
            "hybrid",
            "exact_full",
            "margin_full",
        }:
            raise ValueError(f"unsupported frozen E2 policy: {self.policy!r}")

    def core_kwargs(self, selected_state_dim: int) -> Dict[str, Any]:
        """Return every frozen field explicitly; no core default is implicit."""

        return {
            "state_dim": selected_state_dim,
            "policy": self.policy,
            "no_positive": self.no_positive,
            "state_reset": self.state_reset,
            "margin_scale": self.margin_scale,
            "positive_factor": self.positive_factor,
            "negative_factor": self.negative_factor,
            "hybrid_cutoff": self.hybrid_cutoff,
            "g_ee": self.g_ee,
            "g_ei": self.g_ei,
            "g_ie": self.g_ie,
            "g_ii": self.g_ii,
            "theta_e": self.theta_e,
            "theta_i": self.theta_i,
            "tau_e": self.tau_e,
            "tau_i": self.tau_i,
            "dt": self.dt,
            "micro_steps": self.micro_steps,
        }


@dataclass(frozen=True)
class FairLMConfig:
    """Common, immutable configuration for a three-model comparison."""

    vocab_size: int
    d_model: int
    num_heads: int = 4
    dropout: float = 0.0
    padding_idx: Optional[int] = 0
    ignore_index: Optional[int] = None
    head_bias: bool = True
    transformer_max_cache_tokens: Optional[int] = None
    e2: FrozenE2Config = field(default_factory=FrozenE2Config)
    parameter_tolerance: float = DEFAULT_PARAMETER_TOLERANCE
    auto_match_parameters: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("num_heads must be positive and divide d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.padding_idx is not None and not 0 <= self.padding_idx < self.vocab_size:
            raise ValueError("padding_idx must be within the vocabulary or None")
        if (
            self.transformer_max_cache_tokens is not None
            and self.transformer_max_cache_tokens <= 0
        ):
            raise ValueError("transformer_max_cache_tokens must be positive or None")
        if self.parameter_tolerance <= 0.0:
            raise ValueError("parameter_tolerance must be positive")


@dataclass(frozen=True)
class ModelParameterCount:
    """Unique total and temporal-core parameter counts for one model."""

    total: int
    core: int


@dataclass(frozen=True)
class ParameterBudgetReport:
    """Three-model counts and the count-only matching decision."""

    lstm: ModelParameterCount
    transformer: ModelParameterCount
    e2: ModelParameterCount
    relative_spread: float
    base_relative_spread: float
    tolerance: float
    selected_e2_state_dim: int
    selected_transformer_mlp_ratio: float
    auto_matched: bool

    @property
    def within_tolerance(self) -> bool:
        return self.relative_spread <= self.tolerance + 1e-12

    @property
    def totals(self) -> Dict[str, int]:
        return {
            "lstm": self.lstm.total,
            "transformer": self.transformer.total,
            "e2": self.e2.total,
        }

    @property
    def cores(self) -> Dict[str, int]:
        return {
            "lstm": self.lstm.core,
            "transformer": self.transformer.core,
            "e2": self.e2.core,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lstm_total": self.lstm.total,
            "lstm_core": self.lstm.core,
            "transformer_total": self.transformer.total,
            "transformer_core": self.transformer.core,
            "e2_total": self.e2.total,
            "e2_core": self.e2.core,
            "relative_spread": self.relative_spread,
            "base_relative_spread": self.base_relative_spread,
            "tolerance": self.tolerance,
            "selected_e2_state_dim": self.selected_e2_state_dim,
            "selected_transformer_mlp_ratio": self.selected_transformer_mlp_ratio,
            "auto_matched": self.auto_matched,
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class FairModelSuite:
    """The three independently trainable models plus their audit report."""

    config: FairLMConfig
    lstm: CausalLanguageModel[Any]
    transformer: CausalLanguageModel[Any]
    e2: CausalLanguageModel[Any]
    parameter_report: ParameterBudgetReport

    @property
    def models(self) -> Dict[str, CausalLanguageModel[Any]]:
        return {
            "lstm": self.lstm,
            "transformer": self.transformer,
            "e2": self.e2,
        }


class ParameterBudgetError(AssertionError):
    """Raised when a three-model suite exceeds its declared count tolerance."""


def _relative_spread(counts: Tuple[int, int, int]) -> float:
    """Return ``(max - min) / mean`` for the three total parameter counts."""

    mean = sum(counts) / len(counts)
    return float((max(counts) - min(counts)) / mean)


def _build_models(
    config: FairLMConfig,
    *,
    e2_state_dim: int,
    transformer_mlp_ratio: float,
) -> Tuple[
    CausalLanguageModel[Any],
    CausalLanguageModel[Any],
    CausalLanguageModel[Any],
]:
    common = {
        "vocab_size": config.vocab_size,
        "dropout": config.dropout,
        "padding_idx": config.padding_idx,
        "ignore_index": config.ignore_index,
        "tie_weights": True,
        "head_bias": config.head_bias,
    }
    # Core-internal dropout is fixed at zero.  The wrapper applies the same
    # configured dropout to every architecture instead of giving one core extra
    # stochastic layers.
    lstm = CausalLanguageModel(
        core=StatefulLSTMCore(
            config.d_model,
            config.d_model,
            num_layers=1,
            dropout=0.0,
        ),
        **common,
    )
    transformer = CausalLanguageModel(
        core=CausalTransformerCore(
            config.d_model,
            config.d_model,
            num_layers=1,
            num_heads=config.num_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=0.0,
            max_cache_tokens=config.transformer_max_cache_tokens,
        ),
        **common,
    )
    e2 = CausalLanguageModel(
        core=E2SignedCore(
            config.d_model,
            config.d_model,
            **config.e2.core_kwargs(e2_state_dim),
        ),
        **common,
    )
    return lstm, transformer, e2


def _counts(
    models: Tuple[
        CausalLanguageModel[Any],
        CausalLanguageModel[Any],
        CausalLanguageModel[Any],
    ],
) -> Tuple[ModelParameterCount, ModelParameterCount, ModelParameterCount]:
    records = []
    for model in models:
        stats = model.parameter_stats()
        records.append(ModelParameterCount(total=stats.model.total, core=stats.core.total))
    return records[0], records[1], records[2]


def _report(
    models: Tuple[
        CausalLanguageModel[Any],
        CausalLanguageModel[Any],
        CausalLanguageModel[Any],
    ],
    *,
    tolerance: float,
    base_relative_spread: float,
    selected_e2_state_dim: int,
    selected_transformer_mlp_ratio: float,
    auto_matched: bool,
) -> ParameterBudgetReport:
    lstm, transformer, e2 = _counts(models)
    spread = _relative_spread((lstm.total, transformer.total, e2.total))
    return ParameterBudgetReport(
        lstm=lstm,
        transformer=transformer,
        e2=e2,
        relative_spread=spread,
        base_relative_spread=base_relative_spread,
        tolerance=tolerance,
        selected_e2_state_dim=selected_e2_state_dim,
        selected_transformer_mlp_ratio=selected_transformer_mlp_ratio,
        auto_matched=auto_matched,
    )


def build_model_suite(config: FairLMConfig) -> FairModelSuite:
    """Build the frozen three-model suite and optionally count-match it.

    Candidate selection is deterministic and uses only total parameter counts.
    No data loader, loss, metric, or model output is accepted by this function,
    making test-set-driven architecture tuning impossible through this API.
    """

    base_state_dim = config.d_model if config.e2.state_dim is None else config.e2.state_dim
    base_mlp_ratio = 2.0
    base_models = _build_models(
        config,
        e2_state_dim=base_state_dim,
        transformer_mlp_ratio=base_mlp_ratio,
    )
    base_counts = _counts(base_models)
    base_spread = _relative_spread(tuple(record.total for record in base_counts))

    selected_state_dim = base_state_dim
    selected_mlp_ratio = base_mlp_ratio
    selected_models = base_models
    selected_spread = base_spread

    if config.auto_match_parameters and base_spread > config.parameter_tolerance:
        base_hidden = 2 * config.d_model
        state_candidates = tuple(
            candidate
            for offset in _E2_STATE_OFFSETS
            if (candidate := base_state_dim + offset) > 0
        )
        hidden_candidates = tuple(
            candidate
            for offset in _TRANSFORMER_MLP_HIDDEN_OFFSETS
            if (candidate := base_hidden + offset) > 0
        )
        best_key = (
            selected_spread,
            0,
            0,
            selected_state_dim,
            base_hidden,
        )
        for state_dim in state_candidates:
            for hidden_dim in hidden_candidates:
                mlp_ratio = hidden_dim / config.d_model
                if state_dim == base_state_dim and hidden_dim == base_hidden:
                    continue
                candidate_models = _build_models(
                    config,
                    e2_state_dim=state_dim,
                    transformer_mlp_ratio=mlp_ratio,
                )
                candidate_counts = _counts(candidate_models)
                candidate_spread = _relative_spread(
                    tuple(record.total for record in candidate_counts)
                )
                distance = abs(state_dim - base_state_dim) + abs(hidden_dim - base_hidden)
                key = (
                    candidate_spread,
                    distance,
                    abs(state_dim - base_state_dim),
                    state_dim,
                    hidden_dim,
                )
                if key < best_key:
                    best_key = key
                    selected_spread = candidate_spread
                    selected_state_dim = state_dim
                    selected_mlp_ratio = mlp_ratio
                    selected_models = candidate_models

    auto_matched = (
        selected_state_dim != base_state_dim
        or abs(selected_mlp_ratio - base_mlp_ratio) > 1e-12
    )
    report = _report(
        selected_models,
        tolerance=config.parameter_tolerance,
        base_relative_spread=base_spread,
        selected_e2_state_dim=selected_state_dim,
        selected_transformer_mlp_ratio=selected_mlp_ratio,
        auto_matched=auto_matched,
    )
    return FairModelSuite(
        config=config,
        lstm=selected_models[0],
        transformer=selected_models[1],
        e2=selected_models[2],
        parameter_report=report,
    )


def assert_parameter_budget(
    suite_or_report: Union[FairModelSuite, ParameterBudgetReport],
    tolerance: Optional[float] = None,
) -> ParameterBudgetReport:
    """Assert total model parameter spread is within the declared tolerance."""

    report = (
        suite_or_report.parameter_report
        if isinstance(suite_or_report, FairModelSuite)
        else suite_or_report
    )
    allowed = report.tolerance if tolerance is None else float(tolerance)
    if allowed <= 0.0:
        raise ValueError("tolerance must be positive")
    if report.relative_spread > allowed + 1e-12:
        formatted = ", ".join(
            f"{name}={count}" for name, count in report.totals.items()
        )
        raise ParameterBudgetError(
            f"model parameter spread {report.relative_spread:.4%} exceeds "
            f"{allowed:.4%}: {formatted}"
        )
    return report


__all__ = [
    "DEFAULT_PARAMETER_TOLERANCE",
    "FairLMConfig",
    "FairModelSuite",
    "FrozenE2Config",
    "ModelParameterCount",
    "ParameterBudgetError",
    "ParameterBudgetReport",
    "assert_parameter_budget",
    "build_model_suite",
]
