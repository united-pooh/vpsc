"""Interchangeable temporal cores for language and world-model experiments.

The module deliberately keeps token/observation embedding and prediction heads
outside the temporal core.  Each core consumes a dense, batch-first sequence and
returns the same :class:`CoreOutput` shape, so experiments can replace only the
history mechanism while sharing every encoder and decoder.

The recurrent states are explicit values rather than hidden module attributes.
This is important for imagined rollouts: callers can retain, detach, copy, or
branch an agent state without mutating the model.  ``step`` is just the streaming
form of ``forward`` and always returns a length-one sequence.

The E2 core is a discrete, differentiable descendant of the repository's E/I
ring experiments.  Its four recurrent pathways have fixed signs:

    E -> E  positive       I -> E  negative
    E -> I  positive       I -> I  negative

The channel magnitudes are trainable and row-normalised, but their signs cannot
flip.  ``exact``, ``margin``, and ``hybrid`` retain the frozen E1 meanings;
``no_positive`` removes E -> E, while ``state_reset`` removes temporal memory by
resetting both populations before every token.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generic, Literal, Optional, Tuple, TypeVar, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
FeedbackPolicy = Literal[
    "exact",
    "margin",
    "hybrid",
    "exact_full",
    "margin_full",
]


@dataclass
class LSTMCoreState:
    """Explicit LSTM state, both shaped ``[layers, batch, hidden]``."""

    hidden: Tensor
    cell: Tensor


@dataclass
class LayerKVCache:
    """One attention layer's projected keys and values.

    Both tensors have shape ``[batch, heads, cached_tokens, head_dim]``.
    """

    key: Tensor
    value: Tensor


@dataclass
class TransformerCoreState:
    """Per-layer KV cache plus the absolute position of the next token."""

    layers: Tuple[LayerKVCache, ...]
    position: int = 0


@dataclass
class E2CoreState:
    """Separate excitatory and inhibitory population states."""

    excitatory: Tensor
    inhibitory: Tensor


CoreState = Union[LSTMCoreState, TransformerCoreState, E2CoreState]
StateT = TypeVar("StateT", bound=CoreState)


@dataclass
class CoreOutput(Generic[StateT]):
    """Common output of all temporal cores.

    ``sequence`` is always ``[batch, time, output_dim]``.  The streaming
    :meth:`TemporalCore.step` method keeps the time dimension at length one so
    downstream heads do not need a separate code path.
    """

    sequence: Tensor
    state: StateT

    @property
    def last(self) -> Tensor:
        """Last output as ``[batch, output_dim]``."""

        return self.sequence[:, -1]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count scalar parameters, de-duplicating tied parameters by identity."""

    seen = set()
    total = 0
    for parameter in module.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            total += parameter.numel()
    return int(total)


def state_nbytes(state: Any) -> int:
    """Recursively count tensor storage represented by a core state.

    The helper counts logical tensor bytes (``numel * element_size``), which is
    the appropriate architecture-level comparison for LSTM state, E/I state,
    and Transformer KV cache.  Python object overhead and allocator slack are
    intentionally excluded.
    """

    if isinstance(state, Tensor):
        return int(state.numel() * state.element_size())
    if is_dataclass(state) and not isinstance(state, type):
        return sum(state_nbytes(getattr(state, field.name)) for field in fields(state))
    if isinstance(state, dict):
        return sum(state_nbytes(value) for value in state.values())
    if isinstance(state, (tuple, list)):
        return sum(state_nbytes(value) for value in state)
    return 0


def detach_core_state(state: StateT) -> StateT:
    """Recursively detach every tensor while preserving the state structure."""

    def detach(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.detach()
        if is_dataclass(value) and not isinstance(value, type):
            values = {field.name: detach(getattr(value, field.name)) for field in fields(value)}
            return type(value)(**values)
        if isinstance(value, tuple):
            return tuple(detach(item) for item in value)
        if isinstance(value, list):
            return [detach(item) for item in value]
        if isinstance(value, dict):
            return {key: detach(item) for key, item in value.items()}
        return value

    return cast(StateT, detach(state))


def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def _validate_sequence(x: Tensor, input_dim: int) -> Tuple[int, int]:
    if x.ndim != 3:
        raise ValueError(f"expected x shaped [batch, time, input_dim], got {tuple(x.shape)}")
    if x.shape[-1] != input_dim:
        raise ValueError(f"expected input_dim={input_dim}, got {x.shape[-1]}")
    if x.shape[1] == 0:
        raise ValueError("empty temporal sequences are not supported")
    return int(x.shape[0]), int(x.shape[1])


class TemporalCore(nn.Module, Generic[StateT], ABC):
    """Uniform interface implemented by all temporal histories."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    @abstractmethod
    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> StateT:
        """Create an empty state for a dense batch."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        state: Optional[StateT] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[StateT]:
        """Process ``x`` shaped ``[batch, time, input_dim]``."""

    def step(
        self,
        x_t: Tensor,
        state: Optional[StateT] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[StateT]:
        """Process one streaming token/event and retain a length-one time axis."""

        if x_t.ndim != 2:
            raise ValueError(f"expected x_t shaped [batch, input_dim], got {tuple(x_t.shape)}")
        return self.forward(x_t.unsqueeze(1), state, detach_state=detach_state)


class StatefulLSTMCore(TemporalCore[LSTMCoreState]):
    """Multi-layer LSTM with explicit, branchable recurrent state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LSTMCoreState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        default_device, default_dtype = _module_device_dtype(self)
        device = default_device if device is None else device
        dtype = default_dtype if dtype is None else dtype
        shape = (self.num_layers, batch_size, self.hidden_dim)
        return LSTMCoreState(
            hidden=torch.zeros(shape, device=device, dtype=dtype),
            cell=torch.zeros(shape, device=device, dtype=dtype),
        )

    def _validate_state(self, state: LSTMCoreState, batch_size: int) -> None:
        expected = (self.num_layers, batch_size, self.hidden_dim)
        if tuple(state.hidden.shape) != expected or tuple(state.cell.shape) != expected:
            raise ValueError(
                "invalid LSTM state shapes: "
                f"hidden={tuple(state.hidden.shape)}, cell={tuple(state.cell.shape)}, "
                f"expected={expected}"
            )

    def forward(
        self,
        x: Tensor,
        state: Optional[LSTMCoreState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[LSTMCoreState]:
        batch_size, _ = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        sequence, (hidden, cell) = self.lstm(x, (state.hidden, state.cell))
        next_state = LSTMCoreState(hidden=hidden, cell=cell)
        if detach_state:
            next_state = detach_core_state(next_state)
        return CoreOutput(sequence=sequence, state=next_state)


class _CausalSelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        max_cache_tokens: Optional[int],
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = model_dim // num_heads
        self.dropout = float(dropout)
        self.max_cache_tokens = max_cache_tokens
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)

    def _attention_mask(
        self,
        query_tokens: int,
        past_tokens: int,
        device: torch.device,
    ) -> Tensor:
        total_tokens = past_tokens + query_tokens
        key_index = torch.arange(total_tokens, device=device)
        query_index = past_tokens + torch.arange(query_tokens, device=device)
        allowed = key_index.unsqueeze(0) <= query_index.unsqueeze(1)
        if self.max_cache_tokens is not None:
            lower = query_index.unsqueeze(1) - self.max_cache_tokens + 1
            allowed = allowed & (key_index.unsqueeze(0) >= lower)
        return allowed.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor, cache: LayerKVCache) -> Tuple[Tensor, LayerKVCache]:
        batch_size, query_tokens, _ = x.shape
        qkv = self.qkv(x).view(
            batch_size,
            query_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key_new, value_new = qkv.unbind(dim=0)

        if cache.key.shape[:2] != (batch_size, self.num_heads):
            raise ValueError(
                f"KV cache batch/head shape {tuple(cache.key.shape[:2])} does not match "
                f"{(batch_size, self.num_heads)}"
            )
        if cache.key.shape != cache.value.shape or cache.key.shape[-1] != self.head_dim:
            raise ValueError("invalid or mismatched KV cache shapes")

        past_tokens = int(cache.key.shape[2])
        key = torch.cat((cache.key, key_new), dim=2)
        value = torch.cat((cache.value, value_new), dim=2)

        # Full-sequence training takes the efficient built-in causal path when
        # no history window is requested.  Incremental/chunked and fixed-window
        # paths need an offset-aware boolean mask.
        use_builtin_causal = past_tokens == 0 and self.max_cache_tokens is None
        attention_mask = None
        if not use_builtin_causal:
            attention_mask = self._attention_mask(query_tokens, past_tokens, x.device)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_builtin_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, query_tokens, self.model_dim
        )

        cache_key, cache_value = key, value
        if self.max_cache_tokens is not None:
            cache_key = cache_key[:, :, -self.max_cache_tokens :]
            cache_value = cache_value[:, :, -self.max_cache_tokens :]
        return self.output(attended), LayerKVCache(key=cache_key, value=cache_value)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        max_cache_tokens: Optional[int],
    ) -> None:
        super().__init__()
        hidden = max(1, int(round(model_dim * mlp_ratio)))
        self.norm_attention = nn.LayerNorm(model_dim)
        self.attention = _CausalSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_cache_tokens=max_cache_tokens,
        )
        self.norm_mlp = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, model_dim),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cache: LayerKVCache) -> Tuple[Tensor, LayerKVCache]:
        attended, next_cache = self.attention(self.norm_attention(x), cache)
        x = x + self.residual_dropout(attended)
        x = x + self.mlp(self.norm_mlp(x))
        return x, next_cache


def _sinusoidal_positions(
    start: int,
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Unbounded absolute positions, returned as ``[1, length, dim]``."""

    positions = torch.arange(start, start + length, device=device, dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dim)
    )
    phases = positions.unsqueeze(1) * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(phases)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(phases[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype).unsqueeze(0)


class CausalTransformerCore(TemporalCore[TransformerCoreState]):
    """Pre-norm causal Transformer with explicit incremental KV caches.

    With ``max_cache_tokens=None``, full-sequence training uses standard global
    causal attention and streaming cache grows with history.  A positive
    ``max_cache_tokens`` applies the same causal sliding window in parallel
    training and incremental inference, and crops every layer's cache to that
    exact number of tokens.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_cache_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=model_dim)
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive")
        if mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if max_cache_tokens is not None and max_cache_tokens <= 0:
            raise ValueError("max_cache_tokens must be positive or None")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = model_dim // num_heads
        self.max_cache_tokens = max_cache_tokens
        self.input_projection: nn.Module
        if input_dim == model_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(input_dim, model_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_cache_tokens=max_cache_tokens,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TransformerCoreState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        default_device, default_dtype = _module_device_dtype(self)
        device = default_device if device is None else device
        dtype = default_dtype if dtype is None else dtype
        empty_shape = (batch_size, self.num_heads, 0, self.head_dim)
        layers = tuple(
            LayerKVCache(
                key=torch.empty(empty_shape, device=device, dtype=dtype),
                value=torch.empty(empty_shape, device=device, dtype=dtype),
            )
            for _ in range(self.num_layers)
        )
        return TransformerCoreState(layers=layers, position=0)

    def _validate_state(self, state: TransformerCoreState, batch_size: int) -> None:
        if len(state.layers) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layer caches, got {len(state.layers)}"
            )
        if state.position < 0:
            raise ValueError("cache position cannot be negative")
        for cache in state.layers:
            expected_prefix = (batch_size, self.num_heads)
            if cache.key.shape[:2] != expected_prefix or cache.key.shape[-1] != self.head_dim:
                raise ValueError("cache dimensions do not match this Transformer")
            if cache.key.shape != cache.value.shape:
                raise ValueError("key and value cache shapes must match")
            if self.max_cache_tokens is not None and cache.key.shape[2] > self.max_cache_tokens:
                raise ValueError("provided cache exceeds max_cache_tokens")

    def forward(
        self,
        x: Tensor,
        state: Optional[TransformerCoreState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[TransformerCoreState]:
        batch_size, time = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)

        sequence = self.input_projection(x)
        positions = _sinusoidal_positions(
            state.position,
            time,
            self.model_dim,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        sequence = self.input_dropout(sequence + positions)
        next_layers = []
        for layer, cache in zip(self.layers, state.layers):
            sequence, next_cache = layer(sequence, cache)
            next_layers.append(next_cache)
        sequence = self.output_norm(sequence)
        next_state = TransformerCoreState(
            layers=tuple(next_layers),
            position=state.position + time,
        )
        if detach_state:
            next_state = detach_core_state(next_state)
        return CoreOutput(sequence=sequence, state=next_state)


@dataclass(frozen=True)
class E2FeedbackGains:
    """Effective signed-pathway magnitudes after applying an E1 policy."""

    e_to_e: float
    i_to_e: float
    e_to_i: float
    i_to_i: float


def _normalise_feedback_policy(policy: FeedbackPolicy) -> Literal["exact", "margin", "hybrid"]:
    aliases = {"exact_full": "exact", "margin_full": "margin"}
    normalised = aliases.get(policy, policy)
    if normalised not in {"exact", "margin", "hybrid"}:
        raise ValueError(f"unknown feedback policy: {policy}")
    return cast(Literal["exact", "margin", "hybrid"], normalised)


def _ring_channel_logits(size: int, *, reverse: bool) -> Tensor:
    """Initialise a local-plus-neighbour positive channel before softmax."""

    probabilities = torch.full((size, size), 1e-3)
    for output_index in range(size):
        neighbour = (output_index + (1 if reverse else -1)) % size
        probabilities[output_index, output_index] = 0.85
        probabilities[output_index, neighbour] = 0.15
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities.log()


class E2SignedCore(TemporalCore[E2CoreState]):
    """Explicit signed E/I recurrent core for streaming world models.

    The E1 feedback policy is a construction-time experimental condition, not
    an input-dependent gate:

    * ``exact``: nominal gains and all four pathways;
    * ``margin``: all four gains multiplied by ``margin_scale``;
    * ``hybrid``: margin scaling, plus remove I -> E when
      ``positive_factor < hybrid_cutoff``;
    * ``no_positive``: additionally force E -> E to zero;
    * ``state_reset``: zero E and I before every token, preserving the same
      feed-forward computation while ablating recurrent memory.

    ``state_dim`` defaults to ``hidden_dim``.  With equal input/hidden sizes,
    this gives the same order of parameter count and persistent state bytes as
    an LSTM; experiments should still use :func:`count_parameters` and
    :func:`state_nbytes` rather than assuming exact equality.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        state_dim: Optional[int] = None,
        policy: FeedbackPolicy = "exact",
        no_positive: bool = False,
        state_reset: bool = False,
        margin_scale: float = 0.95,
        positive_factor: float = 1.0,
        negative_factor: float = 1.0,
        hybrid_cutoff: float = 1.0,
        g_ee: float = 7.75,
        g_ei: float = 6.70,
        g_ie: float = 10.0,
        g_ii: float = 6.30,
        theta_e: float = 2.50,
        theta_i: float = 5.75,
        tau_e: float = 1.0,
        tau_i: float = 5.80,
        dt: float = 1.0,
        micro_steps: int = 1,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        state_dim = hidden_dim if state_dim is None else state_dim
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 < margin_scale <= 1.0:
            raise ValueError("margin_scale must be in (0, 1]")
        if positive_factor < 0.0 or negative_factor < 0.0:
            raise ValueError("feedback factors cannot be negative")
        if tau_e <= 0.0 or tau_i <= 0.0 or dt <= 0.0:
            raise ValueError("tau_e, tau_i, and dt must be positive")
        if dt > min(tau_e, tau_i):
            raise ValueError("dt must not exceed either time constant for stable Euler updates")
        if micro_steps <= 0:
            raise ValueError("micro_steps must be positive")
        if min(g_ee, g_ei, g_ie, g_ii) < 0.0:
            raise ValueError("signed-channel gain magnitudes cannot be negative")

        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.policy = _normalise_feedback_policy(policy)
        self.no_positive = bool(no_positive)
        self.state_reset = bool(state_reset)
        self.margin_scale = float(margin_scale)
        self.positive_factor = float(positive_factor)
        self.negative_factor = float(negative_factor)
        self.hybrid_cutoff = float(hybrid_cutoff)
        self.g_ee = float(g_ee)
        self.g_ei = float(g_ei)
        self.g_ie = float(g_ie)
        self.g_ii = float(g_ii)
        self.theta_e = float(theta_e)
        self.theta_i = float(theta_i)
        self.alpha_e = float(dt / tau_e)
        self.alpha_i = float(dt / tau_i)
        self.micro_steps = int(micro_steps)

        self.input_to_e = nn.Linear(input_dim, state_dim)
        self.input_to_i = nn.Linear(input_dim, state_dim)

        # Softmax turns these logits into non-negative row-normalised channel
        # magnitudes.  Signs are introduced only by the four explicit equations.
        self.e_to_e_logits = nn.Parameter(_ring_channel_logits(state_dim, reverse=False))
        self.i_to_e_logits = nn.Parameter(_ring_channel_logits(state_dim, reverse=True))
        self.e_to_i_logits = nn.Parameter(_ring_channel_logits(state_dim, reverse=False))
        self.i_to_i_logits = nn.Parameter(_ring_channel_logits(state_dim, reverse=True))

        self.output_norm = nn.LayerNorm(2 * state_dim)
        self.output_projection = nn.Linear(2 * state_dim, hidden_dim)

    def effective_gains(self) -> E2FeedbackGains:
        if self.policy == "exact":
            scale = 1.0
            remove_negative = False
        elif self.policy == "margin":
            scale = self.margin_scale
            remove_negative = False
        else:
            scale = self.margin_scale
            remove_negative = self.positive_factor < self.hybrid_cutoff

        e_to_e = 0.0 if self.no_positive else self.g_ee * self.positive_factor * scale
        i_to_e = 0.0 if remove_negative else self.g_ei * self.negative_factor * scale
        return E2FeedbackGains(
            e_to_e=e_to_e,
            i_to_e=i_to_e,
            e_to_i=self.g_ie * scale,
            i_to_i=self.g_ii * scale,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> E2CoreState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        default_device, default_dtype = _module_device_dtype(self)
        device = default_device if device is None else device
        dtype = default_dtype if dtype is None else dtype
        shape = (batch_size, self.state_dim)
        return E2CoreState(
            excitatory=torch.zeros(shape, device=device, dtype=dtype),
            inhibitory=torch.zeros(shape, device=device, dtype=dtype),
        )

    def _validate_state(self, state: E2CoreState, batch_size: int) -> None:
        expected = (batch_size, self.state_dim)
        if tuple(state.excitatory.shape) != expected or tuple(state.inhibitory.shape) != expected:
            raise ValueError(
                "invalid E2 state shapes: "
                f"E={tuple(state.excitatory.shape)}, I={tuple(state.inhibitory.shape)}, "
                f"expected={expected}"
            )

    @staticmethod
    def _positive_channel(logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)

    def _advance(
        self,
        x_t: Tensor,
        state: E2CoreState,
        gains: E2FeedbackGains,
    ) -> E2CoreState:
        drive_from_input_e = self.input_to_e(x_t)
        drive_from_input_i = self.input_to_i(x_t)
        excitatory = state.excitatory
        inhibitory = state.inhibitory

        e_to_e = self._positive_channel(self.e_to_e_logits)
        i_to_e = self._positive_channel(self.i_to_e_logits)
        e_to_i = self._positive_channel(self.e_to_i_logits)
        i_to_i = self._positive_channel(self.i_to_i_logits)

        for _ in range(self.micro_steps):
            target_e = torch.sigmoid(
                gains.e_to_e * F.linear(excitatory, e_to_e)
                - gains.i_to_e * F.linear(inhibitory, i_to_e)
                + drive_from_input_e
                - self.theta_e
            )
            target_i = torch.sigmoid(
                gains.e_to_i * F.linear(excitatory, e_to_i)
                - gains.i_to_i * F.linear(inhibitory, i_to_i)
                + drive_from_input_i
                - self.theta_i
            )
            excitatory = excitatory + self.alpha_e * (target_e - excitatory)
            inhibitory = inhibitory + self.alpha_i * (target_i - inhibitory)
        return E2CoreState(excitatory=excitatory, inhibitory=inhibitory)

    def _readout(self, state: E2CoreState) -> Tensor:
        signed_state = torch.cat((state.excitatory, -state.inhibitory), dim=-1)
        return self.output_projection(self.output_norm(signed_state))

    def forward(
        self,
        x: Tensor,
        state: Optional[E2CoreState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E2CoreState]:
        batch_size, time = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)

        gains = self.effective_gains()
        outputs = []
        current = state
        for index in range(time):
            if self.state_reset:
                current = E2CoreState(
                    excitatory=torch.zeros_like(current.excitatory),
                    inhibitory=torch.zeros_like(current.inhibitory),
                )
            current = self._advance(x[:, index], current, gains)
            outputs.append(self._readout(current))
        sequence = torch.stack(outputs, dim=1)
        if detach_state:
            current = detach_core_state(current)
        return CoreOutput(sequence=sequence, state=current)


def _assert_streaming_equivalence(
    core: TemporalCore[Any],
    x: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    core.eval()
    full = core(x)
    state = None
    pieces = []
    for index in range(x.shape[1]):
        streamed = core.step(x[:, index], state)
        pieces.append(streamed.sequence)
        state = streamed.state
    torch.testing.assert_close(full.sequence, torch.cat(pieces, dim=1), atol=atol, rtol=rtol)


def _self_check() -> None:
    """Small shape/cache/equivalence smoke check; it does not train a model."""

    torch.manual_seed(7)
    x = torch.randn(2, 6, 8)
    cores: Tuple[TemporalCore[Any], ...] = (
        StatefulLSTMCore(8, 8),
        CausalTransformerCore(8, 8, num_layers=2, num_heads=2, max_cache_tokens=4),
        E2SignedCore(8, 8, policy="hybrid", positive_factor=0.8),
    )
    with torch.no_grad():
        for core in cores:
            _assert_streaming_equivalence(core, x)
            result = core(x, detach_state=True)
            assert result.sequence.shape == (2, 6, 8)
            assert count_parameters(core) > 0
            assert state_nbytes(result.state) >= 0

        transformer = cast(CausalTransformerCore, cores[1])
        transformer_result = transformer(x)
        assert all(
            cache.key.shape[2] <= cast(int, transformer.max_cache_tokens)
            for cache in transformer_result.state.layers
        )
        e2 = cast(E2SignedCore, cores[2])
        assert e2.effective_gains().i_to_e == 0.0
        assert state_nbytes(e2.initial_state(2)) == 2 * 2 * 8 * 4


if __name__ == "__main__":
    _self_check()
    print("world-model temporal core self-check passed")
