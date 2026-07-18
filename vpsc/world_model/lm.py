"""Shared causal language-model wrapper for interchangeable temporal cores.

The embedding, dropout, and language-model head live here rather than inside a
core.  Consequently an LSTM, causal Transformer, and E2 signed recurrent core
can be compared by replacing only :class:`~vpsc.world_model.cores.TemporalCore`.

Two loss conventions are supported explicitly:

``labels=token_ids``
    Standard teacher forcing.  Logits at positions ``[:-1]`` predict labels at
    positions ``[1:]``.

``targets=next_token_ids``
    Already shifted targets aligned one-to-one with every input position.  This
    matches the continuous chunks emitted by ``wikitext.WikiText2Corpus``.

Passing neither returns logits and explicit state without computing a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Literal, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cores import CoreState, TemporalCore, count_parameters


Tensor = torch.Tensor
StateT = TypeVar("StateT", bound=CoreState)
LossMode = Literal["none", "causal_shift", "aligned"]


@dataclass(frozen=True)
class ModuleParameterStats:
    """Unique scalar parameter counts for one module."""

    total: int
    trainable: int


@dataclass(frozen=True)
class CausalLMParameterStats:
    """Parameter breakdown used to construct fair core comparisons.

    ``embedding`` and ``lm_head`` each describe their module in isolation, so a
    tied weight appears in both rows.  ``model`` and ``wrapper_unique`` always
    de-duplicate shared parameters.  ``shared_embedding_head`` makes the overlap
    explicit rather than allowing a table to double-count it accidentally.
    """

    model: ModuleParameterStats
    core: ModuleParameterStats
    embedding: ModuleParameterStats
    output_norm: ModuleParameterStats
    lm_head: ModuleParameterStats
    wrapper_unique: ModuleParameterStats
    shared_embedding_head: int
    tied_weights: bool
    core_type: str

    def as_dict(self) -> Dict[str, Any]:
        """Return a flat, JSON-serialisable logging record."""

        return {
            "model_total": self.model.total,
            "model_trainable": self.model.trainable,
            "core_total": self.core.total,
            "core_trainable": self.core.trainable,
            "embedding_total": self.embedding.total,
            "embedding_trainable": self.embedding.trainable,
            "output_norm_total": self.output_norm.total,
            "output_norm_trainable": self.output_norm.trainable,
            "lm_head_total": self.lm_head.total,
            "lm_head_trainable": self.lm_head.trainable,
            "wrapper_unique_total": self.wrapper_unique.total,
            "wrapper_unique_trainable": self.wrapper_unique.trainable,
            "shared_embedding_head": self.shared_embedding_head,
            "tied_weights": self.tied_weights,
            "core_type": self.core_type,
        }


@dataclass
class CausalLMOutput(Generic[StateT]):
    """Structured output shared by sequence and streaming calls.

    ``logits`` is always ``[batch, time, vocab]``.  ``hidden_states`` is the
    post-dropout, LayerNorm-normalised sequence sent directly to the LM head.
    ``target_count`` remains a scalar tensor to avoid a device synchronisation
    in the training hot path.
    """

    logits: Tensor
    state: StateT
    hidden_states: Tensor
    loss: Optional[Tensor] = None
    target_count: Optional[Tensor] = None
    loss_mode: LossMode = "none"

    @property
    def last_logits(self) -> Tensor:
        """Last-position logits shaped ``[batch, vocab]``."""

        return self.logits[:, -1]


def module_parameter_stats(module: nn.Module) -> ModuleParameterStats:
    """Count total and trainable unique scalar parameters in *module*."""

    return ModuleParameterStats(
        total=count_parameters(module, trainable_only=False),
        trainable=count_parameters(module, trainable_only=True),
    )


class CausalLanguageModel(nn.Module, Generic[StateT]):
    """A core-agnostic autoregressive token model with explicit streaming state.

    Args:
        vocab_size: Number of input/output token IDs.
        core: One temporal core.  Its ``input_dim`` defines embedding width and
            its ``output_dim`` defines LM-head input width.
        dropout: Shared embedding/output dropout probability.
        padding_idx: Embedding padding row.  It is also the default loss ignore
            index; padding is ignored by the loss, not removed from core time.
        ignore_index: Optional explicit loss ignore value.  Defaults to
            ``padding_idx`` (or ``-100`` when no padding index is supplied).
        tie_weights: Share the embedding and LM-head weight matrix.  This
            requires equal core input and output widths.
        head_bias: Include a vocabulary bias in the LM head.
    """

    def __init__(
        self,
        vocab_size: int,
        core: TemporalCore[StateT],
        *,
        dropout: float = 0.0,
        padding_idx: Optional[int] = 0,
        ignore_index: Optional[int] = None,
        tie_weights: bool = False,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if not isinstance(core, TemporalCore):
            raise TypeError("core must implement TemporalCore")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if padding_idx is not None and not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx must be within the vocabulary or None")
        if tie_weights and core.input_dim != core.output_dim:
            raise ValueError(
                "weight tying requires core.input_dim == core.output_dim, got "
                f"{core.input_dim} and {core.output_dim}"
            )

        self.vocab_size = int(vocab_size)
        self.padding_idx = padding_idx
        self.ignore_index = (
            int(ignore_index)
            if ignore_index is not None
            else (-100 if padding_idx is None else int(padding_idx))
        )
        self.tie_weights = bool(tie_weights)
        self.core = core
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=core.input_dim,
            padding_idx=padding_idx,
        )
        # E2' frozen shared-instrument initialisation.  Initialise before tying
        # the output head; no later reset is allowed to overwrite this matrix.
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=core.input_dim**-0.5,
        )
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].zero_()
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(core.output_dim)
        self.lm_head = nn.Linear(core.output_dim, vocab_size, bias=head_bias)
        if tie_weights:
            self.lm_head.weight = self.embedding.weight
        if padding_idx is not None:
            # nn.Embedding already suppresses input-side padding gradients, but
            # a tied LM head would otherwise reintroduce a gradient through its
            # vocabulary row.  Keep the preregistered zero padding row fixed.
            def zero_padding_gradient(gradient: Tensor) -> Tensor:
                gradient = gradient.clone()
                gradient[padding_idx].zero_()
                return gradient

            self.embedding.weight.register_hook(zero_padding_gradient)

    @property
    def embedding_dim(self) -> int:
        return int(self.core.input_dim)

    @property
    def output_dim(self) -> int:
        return int(self.core.output_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> StateT:
        """Create an explicit empty state through the selected core."""

        if device is None:
            device = self.embedding.weight.device
        if dtype is None:
            dtype = self.embedding.weight.dtype
        return self.core.initial_state(batch_size, device=device, dtype=dtype)

    @staticmethod
    def _validate_sequence_ids(token_ids: Tensor, name: str) -> None:
        if token_ids.ndim != 2:
            raise ValueError(
                f"{name} must be shaped [batch, time], got {tuple(token_ids.shape)}"
            )
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError(f"{name} cannot contain an empty batch or time axis")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must use torch.int32 or torch.int64 token IDs")

    def _loss(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = targets.reshape(-1)
        loss_sum = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.ignore_index,
            reduction="sum",
        )
        target_count = flat_targets.ne(self.ignore_index).sum()
        denominator = target_count.clamp_min(1).to(dtype=loss_sum.dtype)
        return loss_sum / denominator, target_count

    def forward(
        self,
        input_ids: Tensor,
        state: Optional[StateT] = None,
        *,
        labels: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        detach_state: bool = False,
    ) -> CausalLMOutput[StateT]:
        """Process a token sequence and optionally compute next-token loss.

        ``labels`` follows standard causal-LM shifting and must match
        ``input_ids``.  ``targets`` is for already shifted, position-aligned
        next-token IDs and must also match ``input_ids``.  Supplying both is an
        error so the loss convention can never change silently.
        """

        self._validate_sequence_ids(input_ids, "input_ids")
        if labels is not None and targets is not None:
            raise ValueError("provide labels or targets, not both")
        if labels is not None:
            self._validate_sequence_ids(labels, "labels")
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if input_ids.shape[1] < 2:
                raise ValueError("causal-shift labels require at least two tokens")
        if targets is not None:
            self._validate_sequence_ids(targets, "targets")
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")

        embedded = self.input_dropout(self.embedding(input_ids))
        core_output = self.core(
            embedded,
            state,
            detach_state=detach_state,
        )
        hidden_states = self.output_norm(self.output_dropout(core_output.sequence))
        logits = self.lm_head(hidden_states)

        loss = None
        target_count = None
        loss_mode: LossMode = "none"
        if labels is not None:
            loss, target_count = self._loss(logits[:, :-1], labels[:, 1:])
            loss_mode = "causal_shift"
        elif targets is not None:
            loss, target_count = self._loss(logits, targets)
            loss_mode = "aligned"

        return CausalLMOutput(
            logits=logits,
            state=core_output.state,
            hidden_states=hidden_states,
            loss=loss,
            target_count=target_count,
            loss_mode=loss_mode,
        )

    def teacher_forced(
        self,
        token_ids: Tensor,
        state: Optional[StateT] = None,
        *,
        detach_state: bool = False,
    ) -> CausalLMOutput[StateT]:
        """Convenience form of ``forward(token_ids, labels=token_ids)``."""

        return self.forward(
            token_ids,
            state,
            labels=token_ids,
            detach_state=detach_state,
        )

    def step(
        self,
        input_id: Tensor,
        state: Optional[StateT] = None,
        *,
        detach_state: bool = False,
    ) -> CausalLMOutput[StateT]:
        """Consume one token per batch item using explicit recurrent/KV state.

        Args:
            input_id: Integer tensor shaped ``[batch]``.
            state: State returned by the preceding sequence or step call.

        Returns:
            A :class:`CausalLMOutput` whose time dimension remains one.
        """

        if input_id.ndim != 1:
            raise ValueError(
                f"input_id must be shaped [batch], got {tuple(input_id.shape)}"
            )
        if input_id.shape[0] == 0:
            raise ValueError("input_id cannot contain an empty batch")
        if input_id.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_id must use torch.int32 or torch.int64 token IDs")

        embedded = self.input_dropout(self.embedding(input_id))
        core_output = self.core.step(
            embedded,
            state,
            detach_state=detach_state,
        )
        hidden_states = self.output_norm(self.output_dropout(core_output.sequence))
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(
            logits=logits,
            state=core_output.state,
            hidden_states=hidden_states,
        )

    def parameter_stats(self) -> CausalLMParameterStats:
        """Return de-duplicated model/core counts plus tied-weight overlap."""

        model_stats = module_parameter_stats(self)
        core_stats = module_parameter_stats(self.core)
        embedding_stats = module_parameter_stats(self.embedding)
        output_norm_stats = module_parameter_stats(self.output_norm)
        head_stats = module_parameter_stats(self.lm_head)
        wrapper_stats = ModuleParameterStats(
            total=model_stats.total - core_stats.total,
            trainable=model_stats.trainable - core_stats.trainable,
        )
        shared = self.embedding.weight.numel() if self.tie_weights else 0
        return CausalLMParameterStats(
            model=model_stats,
            core=core_stats,
            embedding=embedding_stats,
            output_norm=output_norm_stats,
            lm_head=head_stats,
            wrapper_unique=wrapper_stats,
            shared_embedding_head=int(shared),
            tied_weights=self.tie_weights,
            core_type=type(self.core).__name__,
        )


__all__ = [
    "CausalLMOutput",
    "CausalLMParameterStats",
    "CausalLanguageModel",
    "LossMode",
    "ModuleParameterStats",
    "module_parameter_stats",
]
