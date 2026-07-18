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
E2ExecutionMode = Literal["reference", "fused"]
E3ExecutionMode = Literal["serial", "scan"]
E3EligibilityForwardMode = Literal["segment", "scan_aligned"]
E3EligibilityBackwardMode = Literal["forward_eligibility", "reverse_adjoint"]
E3FixedPointMode = Literal["serial", "fixed_point"]
E3OscillatorMode = Literal["serial", "scan"]


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


@dataclass
class E3LayerState:
    """Hard-reset residual membrane values for one E/I scan layer."""

    excitatory: Tensor
    inhibitory: Tensor


@dataclass
class E3ScanState:
    """Constant-size streaming state for every cumulative-charge layer."""

    layers: Tuple[E3LayerState, ...]


@dataclass
class E3LayerTrace:
    """Discrete spikes and post-reset membrane sequences for one layer."""

    excitatory_spikes: Tensor
    inhibitory_spikes: Tensor
    excitatory_residuals: Tensor
    inhibitory_residuals: Tensor


@dataclass
class E3GatedTrace:
    """Binary input/write events and slow traces for the AT0 core."""

    excitatory_content: Tensor
    inhibitory_content: Tensor
    excitatory_gate: Tensor
    inhibitory_gate: Tensor
    excitatory_writes: Tensor
    inhibitory_writes: Tensor
    excitatory_spikes: Tensor
    inhibitory_spikes: Tensor
    excitatory_traces: Tensor
    inhibitory_traces: Tensor


@dataclass
class E3OscillatorState:
    """Complex-valued stable oscillator state."""

    value: Tensor


@dataclass
class E3OscillatorTrace:
    """Discrete real/imaginary threshold events plus complex state sequence."""

    excitatory_spikes: Tensor
    inhibitory_spikes: Tensor
    values: Tensor


CoreState = Union[
    LSTMCoreState,
    TransformerCoreState,
    E2CoreState,
    E3ScanState,
    E3OscillatorState,
]
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
        execution_mode: E2ExecutionMode = "fused",
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
        if execution_mode not in ("reference", "fused"):
            raise ValueError("execution_mode must be 'reference' or 'fused'")

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
        self.execution_mode: E2ExecutionMode = execution_mode

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

    def _forward_reference(
        self,
        x: Tensor,
        state: E2CoreState,
        gains: E2FeedbackGains,
    ) -> CoreOutput[E2CoreState]:
        """Original per-token graph retained as the F0 equivalence control."""

        outputs = []
        current = state
        for index in range(x.shape[1]):
            if self.state_reset:
                current = E2CoreState(
                    excitatory=torch.zeros_like(current.excitatory),
                    inhibitory=torch.zeros_like(current.inhibitory),
                )
            current = self._advance(x[:, index], current, gains)
            outputs.append(self._readout(current))
        return CoreOutput(sequence=torch.stack(outputs, dim=1), state=current)

    def _signed_block_weight(self, gains: E2FeedbackGains) -> Tensor:
        """Fuse the four sign-constrained channels into one recurrent matrix."""

        e_to_e = self._positive_channel(self.e_to_e_logits)
        i_to_e = self._positive_channel(self.i_to_e_logits)
        e_to_i = self._positive_channel(self.e_to_i_logits)
        i_to_i = self._positive_channel(self.i_to_i_logits)
        drive_e = torch.cat(
            (gains.e_to_e * e_to_e, -gains.i_to_e * i_to_e), dim=-1
        )
        drive_i = torch.cat(
            (gains.e_to_i * e_to_i, -gains.i_to_i * i_to_i), dim=-1
        )
        return torch.cat((drive_e, drive_i), dim=0)

    def _forward_fused(
        self,
        x: Tensor,
        state: E2CoreState,
        gains: E2FeedbackGains,
    ) -> CoreOutput[E2CoreState]:
        """Algebraically equivalent F0 graph with sequence-level fusion."""

        # The two projections are evaluated on the complete sequence so the
        # backend can parallelise over B*T instead of launching per token.
        input_drive_e = self.input_to_e(x)
        input_drive_i = self.input_to_i(x)
        signed_block = self._signed_block_weight(gains)

        excitatory = state.excitatory
        inhibitory = state.inhibitory
        excitatory_sequence = []
        inhibitory_sequence = []
        for index in range(x.shape[1]):
            if self.state_reset:
                excitatory = torch.zeros_like(excitatory)
                inhibitory = torch.zeros_like(inhibitory)

            for _ in range(self.micro_steps):
                signed_state = torch.cat((excitatory, inhibitory), dim=-1)
                recurrent_drive = F.linear(signed_state, signed_block)
                recurrent_e, recurrent_i = recurrent_drive.chunk(2, dim=-1)
                target_e = torch.sigmoid(
                    recurrent_e + input_drive_e[:, index] - self.theta_e
                )
                target_i = torch.sigmoid(
                    recurrent_i + input_drive_i[:, index] - self.theta_i
                )
                excitatory = excitatory + self.alpha_e * (target_e - excitatory)
                inhibitory = inhibitory + self.alpha_i * (target_i - inhibitory)

            excitatory_sequence.append(excitatory)
            inhibitory_sequence.append(inhibitory)

        excitatory_states = torch.stack(excitatory_sequence, dim=1)
        inhibitory_states = torch.stack(inhibitory_sequence, dim=1)
        signed_sequence = torch.cat(
            (excitatory_states, -inhibitory_states), dim=-1
        )
        sequence = self.output_projection(self.output_norm(signed_sequence))
        return CoreOutput(
            sequence=sequence,
            state=E2CoreState(excitatory=excitatory, inhibitory=inhibitory),
        )

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
        if self.execution_mode == "reference":
            result = self._forward_reference(x, state, gains)
        else:
            result = self._forward_fused(x, state, gains)
        if detach_state:
            result.state = detach_core_state(result.state)
        return result


class _PeriodicSurrogateFloor(torch.autograd.Function):
    """Exact floor forward with a bounded periodic threshold surrogate."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, scale: float) -> Tensor:
        ctx.save_for_backward(value)
        ctx.scale = float(scale)
        return torch.floor(value)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tuple[Tensor, None]:
        (value,) = ctx.saved_tensors
        scale = ctx.scale
        distance = (value - torch.round(value)).abs()
        surrogate = scale / (1.0 + scale * distance).square()
        return gradient * surrogate, None


def _surrogate_floor(value: Tensor, scale: float) -> Tensor:
    return _PeriodicSurrogateFloor.apply(value, scale)


class _StraightThroughRound(torch.autograd.Function):
    """Power-of-two charge quantisation with an identity backward pass."""

    @staticmethod
    def forward(ctx: Any, value: Tensor) -> Tensor:
        del ctx
        return torch.round(value)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tensor:
        del ctx
        return gradient


class E3CumulativeScanCore(TemporalCore[E3ScanState]):
    """Time-parallel signed E/I SNN with exact hard-reset IF dynamics.

    Each layer emits a strictly sub-threshold positive charge.  Cumulative
    charge, integer threshold crossings, binary spike differences, and the
    post-reset residual can therefore be computed with a prefix sum over time.
    ``serial`` retains the same cumulative-charge training graph as a control;
    streaming ``step`` evaluates the equivalent one-token hard reset while
    storing only two residual tensors per layer.

    Same-layer recurrent feedback is intentionally absent in S0: signed E/I
    pathways connect adjacent scan layers, which keeps the time axis fully
    parallel.  Dynamic decay and recurrent reset correction belong to S1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        state_dim: Optional[int] = None,
        num_layers: int = 2,
        max_charge: float = 0.95,
        drive_levels: int = 1024,
        charge_levels: int = 4096,
        surrogate_scale: float = 5.0,
        g_ee: float = 1.0,
        g_ei: float = 1.0,
        g_ie: float = 1.0,
        g_ii: float = 1.0,
        execution_mode: E3ExecutionMode = "scan",
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        state_dim = hidden_dim if state_dim is None else state_dim
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 < max_charge < 1.0:
            raise ValueError("max_charge must be in (0, 1) for binary spikes")
        if drive_levels <= 1 or drive_levels & (drive_levels - 1):
            raise ValueError("drive_levels must be a power of two greater than one")
        if charge_levels <= 1 or charge_levels & (charge_levels - 1):
            raise ValueError("charge_levels must be a power of two greater than one")
        if surrogate_scale <= 0.0:
            raise ValueError("surrogate_scale must be positive")
        if min(g_ee, g_ei, g_ie, g_ii) < 0.0:
            raise ValueError("signed-channel gain magnitudes cannot be negative")
        if execution_mode not in ("serial", "scan"):
            raise ValueError("execution_mode must be 'serial' or 'scan'")

        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.num_layers = int(num_layers)
        self.max_charge = float(max_charge)
        self.drive_levels = int(drive_levels)
        self.charge_levels = int(charge_levels)
        self.surrogate_scale = float(surrogate_scale)
        self.g_ee = float(g_ee)
        self.g_ei = float(g_ei)
        self.g_ie = float(g_ie)
        self.g_ii = float(g_ii)
        self.execution_mode: E3ExecutionMode = execution_mode

        self.input_to_e = nn.Linear(input_dim, state_dim)
        self.input_to_i = nn.Linear(input_dim, state_dim)
        links = num_layers - 1
        self.e_to_e_logits = nn.ParameterList(
            [
                nn.Parameter(_ring_channel_logits(state_dim, reverse=False))
                for _ in range(links)
            ]
        )
        self.i_to_e_logits = nn.ParameterList(
            [
                nn.Parameter(_ring_channel_logits(state_dim, reverse=True))
                for _ in range(links)
            ]
        )
        self.e_to_i_logits = nn.ParameterList(
            [
                nn.Parameter(_ring_channel_logits(state_dim, reverse=False))
                for _ in range(links)
            ]
        )
        self.i_to_i_logits = nn.ParameterList(
            [
                nn.Parameter(_ring_channel_logits(state_dim, reverse=True))
                for _ in range(links)
            ]
        )
        self.layer_bias_e = nn.ParameterList(
            [nn.Parameter(torch.zeros(state_dim)) for _ in range(links)]
        )
        self.layer_bias_i = nn.ParameterList(
            [nn.Parameter(torch.zeros(state_dim)) for _ in range(links)]
        )
        self.output_norm = nn.LayerNorm(4 * state_dim)
        self.output_projection = nn.Linear(4 * state_dim, hidden_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> E3ScanState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        default_device, default_dtype = _module_device_dtype(self)
        device = default_device if device is None else device
        dtype = default_dtype if dtype is None else dtype
        shape = (batch_size, self.state_dim)
        return E3ScanState(
            layers=tuple(
                E3LayerState(
                    excitatory=torch.zeros(shape, device=device, dtype=dtype),
                    inhibitory=torch.zeros(shape, device=device, dtype=dtype),
                )
                for _ in range(self.num_layers)
            )
        )

    def _validate_state(self, state: E3ScanState, batch_size: int) -> None:
        if len(state.layers) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} E3 state layers, got {len(state.layers)}"
            )
        expected = (batch_size, self.state_dim)
        for index, layer in enumerate(state.layers):
            if tuple(layer.excitatory.shape) != expected or tuple(layer.inhibitory.shape) != expected:
                raise ValueError(
                    f"invalid E3 layer {index} state shapes: "
                    f"E={tuple(layer.excitatory.shape)}, "
                    f"I={tuple(layer.inhibitory.shape)}, expected={expected}"
                )

    def _integrate_scan(
        self, charge: Tensor, initial_residual: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cumulative = initial_residual.unsqueeze(1) + torch.cumsum(charge, dim=1)
        cumulative_count = _surrogate_floor(cumulative, self.surrogate_scale)
        previous_count = torch.cat(
            (torch.zeros_like(cumulative_count[:, :1]), cumulative_count[:, :-1]), dim=1
        )
        spikes = cumulative_count - previous_count
        residuals = cumulative - cumulative_count.detach()
        return spikes, residuals, residuals[:, -1]

    def _integrate_serial(
        self, charge: Tensor, initial_residual: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cumulative = initial_residual
        previous_count = torch.zeros_like(initial_residual)
        spikes = []
        residuals = []
        for index in range(charge.shape[1]):
            cumulative = cumulative + charge[:, index]
            cumulative_count = _surrogate_floor(cumulative, self.surrogate_scale)
            spikes.append(cumulative_count - previous_count)
            residuals.append(cumulative - cumulative_count.detach())
            previous_count = cumulative_count
        residual_sequence = torch.stack(residuals, dim=1)
        return torch.stack(spikes, dim=1), residual_sequence, residual_sequence[:, -1]

    def _integrate(
        self, charge: Tensor, initial_residual: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.execution_mode == "serial":
            return self._integrate_serial(charge, initial_residual)
        return self._integrate_scan(charge, initial_residual)

    def _charge(self, drive: Tensor) -> Tensor:
        drive_levels = float(self.drive_levels)
        quantised_drive = _StraightThroughRound.apply(drive * drive_levels) / drive_levels
        continuous = self.max_charge * torch.sigmoid(quantised_drive)
        levels = float(self.charge_levels)
        return _StraightThroughRound.apply(continuous * levels) / levels

    def _signed_layer_drive(
        self,
        link: int,
        excitatory_spikes: Tensor,
        inhibitory_spikes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        e_to_e = torch.softmax(self.e_to_e_logits[link], dim=-1)
        i_to_e = torch.softmax(self.i_to_e_logits[link], dim=-1)
        e_to_i = torch.softmax(self.e_to_i_logits[link], dim=-1)
        i_to_i = torch.softmax(self.i_to_i_logits[link], dim=-1)
        drive_e = (
            self.g_ee * F.linear(excitatory_spikes, e_to_e)
            - self.g_ei * F.linear(inhibitory_spikes, i_to_e)
            + self.layer_bias_e[link]
        )
        drive_i = (
            self.g_ie * F.linear(excitatory_spikes, e_to_i)
            - self.g_ii * F.linear(inhibitory_spikes, i_to_i)
            + self.layer_bias_i[link]
        )
        return drive_e, drive_i

    def forward_dynamics(
        self,
        x: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> Tuple[CoreOutput[E3ScanState], Tuple[E3LayerTrace, ...]]:
        batch_size, _ = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)

        drive_e = self.input_to_e(x)
        drive_i = self.input_to_i(x)
        traces = []
        next_layers = []
        for layer_index in range(self.num_layers):
            if layer_index > 0:
                previous = traces[-1]
                drive_e, drive_i = self._signed_layer_drive(
                    layer_index - 1,
                    previous.excitatory_spikes,
                    previous.inhibitory_spikes,
                )
            charge_e = self._charge(drive_e)
            charge_i = self._charge(drive_i)
            spikes_e, residuals_e, final_e = self._integrate(
                charge_e, state.layers[layer_index].excitatory
            )
            spikes_i, residuals_i, final_i = self._integrate(
                charge_i, state.layers[layer_index].inhibitory
            )
            traces.append(
                E3LayerTrace(
                    excitatory_spikes=spikes_e,
                    inhibitory_spikes=spikes_i,
                    excitatory_residuals=residuals_e,
                    inhibitory_residuals=residuals_i,
                )
            )
            next_layers.append(E3LayerState(excitatory=final_e, inhibitory=final_i))

        final_trace = traces[-1]
        signed_sequence = torch.cat(
            (
                final_trace.excitatory_spikes,
                -final_trace.inhibitory_spikes,
                final_trace.excitatory_residuals,
                -final_trace.inhibitory_residuals,
            ),
            dim=-1,
        )
        sequence = self.output_projection(self.output_norm(signed_sequence))
        next_state = E3ScanState(layers=tuple(next_layers))
        if detach_state:
            next_state = detach_core_state(next_state)
        return CoreOutput(sequence=sequence, state=next_state), tuple(traces)

    def forward(
        self,
        x: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E3ScanState]:
        result, _ = self.forward_dynamics(x, state, detach_state=detach_state)
        return result


class _SurrogateStep(torch.autograd.Function):
    """Binary threshold forward with a bounded fast-sigmoid derivative."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, scale: float) -> Tensor:
        ctx.save_for_backward(value)
        ctx.scale = float(scale)
        return (value >= 0.0).to(dtype=value.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tuple[Tensor, None]:
        (value,) = ctx.saved_tensors
        scale = ctx.scale
        surrogate = scale / (1.0 + scale * value.abs()).square()
        return gradient * surrogate, None


def _surrogate_step(value: Tensor, scale: float) -> Tensor:
    return _SurrogateStep.apply(value, scale)


def _periodic_floor_derivative(value: Tensor, scale: float) -> Tensor:
    """Derivative used by :class:`_PeriodicSurrogateFloor`."""

    distance = (value - torch.round(value)).abs()
    return scale / (1.0 + scale * distance).square()


def _event_step_derivative(value: Tensor, scale: float) -> Tensor:
    """Derivative used by :class:`_SurrogateStep`."""

    return scale / (1.0 + scale * value.abs()).square()


class _TerminalEligibilityScan(torch.autograd.Function):
    """Exact terminal IC0 gradient factored through eligibility traces.

    The forward is the same cumulative hard-reset equation used by
    :class:`E3InputCodedScanCore`.  For a loss that observes only the terminal
    spike/residual, the input projection gradient depends on two aggregated
    eligibility tensors (through ``T`` and ``T-1``) rather than a temporal
    autograd graph.  When the dense input itself requires a gradient, the
    per-time event surrogate derivatives are additionally retained so the
    exact input gradient can be reconstructed in one batched contraction.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        weight_e: Tensor,
        bias_e: Tensor,
        weight_i: Tensor,
        bias_i: Tensor,
        initial_e: Tensor,
        initial_i: Tensor,
        base_charge: float,
        event_charge: float,
        surrogate_scale: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        time_steps = x.shape[1]
        scale = float(surrogate_scale)

        drive_e = F.linear(x, weight_e, bias_e)
        drive_i = F.linear(x, weight_i, bias_i)
        events_e = (drive_e >= 0.0).to(dtype=x.dtype)
        events_i = (drive_i >= 0.0).to(dtype=x.dtype)
        charge_e = float(base_charge) + float(event_charge) * events_e
        charge_i = float(base_charge) + float(event_charge) * events_i

        cumulative_e = initial_e.unsqueeze(1) + torch.cumsum(charge_e, dim=1)
        cumulative_i = initial_i.unsqueeze(1) + torch.cumsum(charge_i, dim=1)
        counts_e = torch.floor(cumulative_e)
        counts_i = torch.floor(cumulative_i)
        # Clone the terminal views before saving them.  Retaining a view here
        # would pin the complete ``[B,T,H]`` cumulative storage and defeat the
        # length-independent core-only eligibility memory bound.
        terminal_q_e = cumulative_e[:, -1].clone()
        terminal_q_i = cumulative_i[:, -1].clone()
        terminal_count_e = counts_e[:, -1]
        terminal_count_i = counts_i[:, -1]
        if time_steps > 1:
            previous_q_e = cumulative_e[:, -2].clone()
            previous_q_i = cumulative_i[:, -2].clone()
            previous_count_e = counts_e[:, -2]
            previous_count_i = counts_i[:, -2]
        else:
            previous_q_e = torch.zeros_like(terminal_q_e)
            previous_q_i = torch.zeros_like(terminal_q_i)
            previous_count_e = torch.zeros_like(terminal_count_e)
            previous_count_i = torch.zeros_like(terminal_count_i)

        spike_e = terminal_count_e - previous_count_e
        spike_i = terminal_count_i - previous_count_i
        final_e = terminal_q_e - terminal_count_e
        final_i = terminal_q_i - terminal_count_i
        raw = torch.cat((spike_e, -spike_i, final_e, -final_i), dim=-1)

        phi_e = _event_step_derivative(drive_e, scale)
        phi_i = _event_step_derivative(drive_i, scale)
        eligibility_e = torch.einsum("bth,btd->bhd", phi_e, x)
        eligibility_i = torch.einsum("bth,btd->bhd", phi_i, x)
        bias_eligibility_e = phi_e.sum(dim=1)
        bias_eligibility_i = phi_i.sum(dim=1)
        if time_steps > 1:
            previous_eligibility_e = torch.einsum(
                "bth,btd->bhd", phi_e[:, :-1], x[:, :-1]
            )
            previous_eligibility_i = torch.einsum(
                "bth,btd->bhd", phi_i[:, :-1], x[:, :-1]
            )
            previous_bias_eligibility_e = phi_e[:, :-1].sum(dim=1)
            previous_bias_eligibility_i = phi_i[:, :-1].sum(dim=1)
        else:
            previous_eligibility_e = torch.zeros_like(eligibility_e)
            previous_eligibility_i = torch.zeros_like(eligibility_i)
            previous_bias_eligibility_e = torch.zeros_like(bias_eligibility_e)
            previous_bias_eligibility_i = torch.zeros_like(bias_eligibility_i)

        saved = [
            terminal_q_e,
            previous_q_e,
            terminal_q_i,
            previous_q_i,
            eligibility_e,
            previous_eligibility_e,
            eligibility_i,
            previous_eligibility_i,
            bias_eligibility_e,
            previous_bias_eligibility_e,
            bias_eligibility_i,
            previous_bias_eligibility_i,
            weight_e,
            weight_i,
        ]
        ctx.input_requires_grad = bool(ctx.needs_input_grad[0])
        if ctx.input_requires_grad:
            saved.extend((phi_e, phi_i))
        ctx.save_for_backward(*saved)
        ctx.time_steps = int(time_steps)
        ctx.event_charge = float(event_charge)
        ctx.surrogate_scale = scale
        return raw, final_e, final_i

    @staticmethod
    def backward(
        ctx: Any,
        grad_raw: Optional[Tensor],
        grad_final_e: Optional[Tensor],
        grad_final_i: Optional[Tensor],
    ) -> Tuple[
        Optional[Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        None,
        None,
        None,
    ]:
        saved = ctx.saved_tensors
        (
            terminal_q_e,
            previous_q_e,
            terminal_q_i,
            previous_q_i,
            eligibility_e,
            previous_eligibility_e,
            eligibility_i,
            previous_eligibility_i,
            bias_eligibility_e,
            previous_bias_eligibility_e,
            bias_eligibility_i,
            previous_bias_eligibility_i,
            weight_e,
            weight_i,
        ) = saved[:14]
        phi_e: Optional[Tensor] = None
        phi_i: Optional[Tensor] = None
        if ctx.input_requires_grad:
            phi_e, phi_i = saved[14:16]

        hidden = terminal_q_e.shape[-1]
        zero = torch.zeros_like(terminal_q_e)
        if grad_raw is None:
            spike_signal_e = zero
            spike_signal_i = zero
            residual_signal_e = zero
            residual_signal_i = zero
        else:
            spike_signal_e = grad_raw[:, :hidden]
            spike_signal_i = -grad_raw[:, hidden : 2 * hidden]
            residual_signal_e = grad_raw[:, 2 * hidden : 3 * hidden]
            residual_signal_i = -grad_raw[:, 3 * hidden :]
        if grad_final_e is not None:
            residual_signal_e = residual_signal_e + grad_final_e
        if grad_final_i is not None:
            residual_signal_i = residual_signal_i + grad_final_i

        scale = ctx.surrogate_scale
        floor_e = _periodic_floor_derivative(terminal_q_e, scale)
        floor_i = _periodic_floor_derivative(terminal_q_i, scale)
        if ctx.time_steps > 1:
            previous_floor_e = _periodic_floor_derivative(previous_q_e, scale)
            previous_floor_i = _periodic_floor_derivative(previous_q_i, scale)
        else:
            previous_floor_e = zero
            previous_floor_i = zero

        terminal_signal_e = residual_signal_e + spike_signal_e * floor_e
        terminal_signal_i = residual_signal_i + spike_signal_i * floor_i
        previous_signal_e = spike_signal_e * previous_floor_e
        previous_signal_i = spike_signal_i * previous_floor_i
        event_charge = ctx.event_charge

        grad_weight_e = event_charge * torch.sum(
            terminal_signal_e.unsqueeze(-1) * eligibility_e
            - previous_signal_e.unsqueeze(-1) * previous_eligibility_e,
            dim=0,
        )
        grad_weight_i = event_charge * torch.sum(
            terminal_signal_i.unsqueeze(-1) * eligibility_i
            - previous_signal_i.unsqueeze(-1) * previous_eligibility_i,
            dim=0,
        )
        grad_bias_e = event_charge * torch.sum(
            terminal_signal_e * bias_eligibility_e
            - previous_signal_e * previous_bias_eligibility_e,
            dim=0,
        )
        grad_bias_i = event_charge * torch.sum(
            terminal_signal_i * bias_eligibility_i
            - previous_signal_i * previous_bias_eligibility_i,
            dim=0,
        )
        grad_initial_e = terminal_signal_e - previous_signal_e
        grad_initial_i = terminal_signal_i - previous_signal_i

        grad_x: Optional[Tensor] = None
        if ctx.input_requires_grad:
            assert phi_e is not None and phi_i is not None
            coefficient_e = terminal_signal_e.unsqueeze(1).expand(
                -1, ctx.time_steps, -1
            )
            coefficient_i = terminal_signal_i.unsqueeze(1).expand(
                -1, ctx.time_steps, -1
            )
            if ctx.time_steps > 1:
                coefficient_e = coefficient_e.clone()
                coefficient_i = coefficient_i.clone()
                coefficient_e[:, :-1] -= previous_signal_e.unsqueeze(1)
                coefficient_i[:, :-1] -= previous_signal_i.unsqueeze(1)
            grad_drive_e = event_charge * phi_e * coefficient_e
            grad_drive_i = event_charge * phi_i * coefficient_i
            grad_x = torch.matmul(grad_drive_e, weight_e) + torch.matmul(
                grad_drive_i, weight_i
            )

        return (
            grad_x,
            grad_weight_e,
            grad_bias_e,
            grad_weight_i,
            grad_bias_i,
            grad_initial_e,
            grad_initial_i,
            None,
            None,
            None,
        )


class _MultiQueryEligibilityScan(torch.autograd.Function):
    """Exact IC0 gradient for a sparse, ordered set of query times.

    The additive IC0 dynamics let every query gradient be contracted with a
    prefix eligibility tensor.  Only ``K`` prefix snapshots are retained for
    ``K`` query outputs, so the core-only backward storage is independent of
    sequence length.  Dense input gradients still require the per-time event
    derivatives and are deliberately accounted for separately.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        query_indices: Tensor,
        weight_e: Tensor,
        bias_e: Tensor,
        weight_i: Tensor,
        bias_i: Tensor,
        initial_e: Tensor,
        initial_i: Tensor,
        base_charge: float,
        event_charge: float,
        surrogate_scale: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        time_steps = x.shape[1]
        scale = float(surrogate_scale)
        query_positions = tuple(int(value) for value in query_indices.tolist())

        drive_e = F.linear(x, weight_e, bias_e)
        drive_i = F.linear(x, weight_i, bias_i)
        events_e = (drive_e >= 0.0).to(dtype=x.dtype)
        events_i = (drive_i >= 0.0).to(dtype=x.dtype)
        charge_e = float(base_charge) + float(event_charge) * events_e
        charge_i = float(base_charge) + float(event_charge) * events_i

        cumulative_e = initial_e.unsqueeze(1) + torch.cumsum(charge_e, dim=1)
        cumulative_i = initial_i.unsqueeze(1) + torch.cumsum(charge_i, dim=1)
        counts_e = torch.floor(cumulative_e)
        counts_i = torch.floor(cumulative_i)

        query_q_e = cumulative_e.index_select(1, query_indices).clone()
        query_q_i = cumulative_i.index_select(1, query_indices).clone()
        query_count_e = counts_e.index_select(1, query_indices)
        query_count_i = counts_i.index_select(1, query_indices)
        previous_q_e = torch.zeros_like(query_q_e)
        previous_q_i = torch.zeros_like(query_q_i)
        previous_count_e = torch.zeros_like(query_count_e)
        previous_count_i = torch.zeros_like(query_count_i)
        positive_query = query_indices > 0
        if bool(positive_query.any()):
            target_columns = torch.nonzero(positive_query, as_tuple=False).squeeze(-1)
            previous_indices = query_indices.index_select(0, target_columns) - 1
            previous_q_e[:, target_columns] = cumulative_e.index_select(
                1, previous_indices
            )
            previous_q_i[:, target_columns] = cumulative_i.index_select(
                1, previous_indices
            )
            previous_count_e[:, target_columns] = counts_e.index_select(
                1, previous_indices
            )
            previous_count_i[:, target_columns] = counts_i.index_select(
                1, previous_indices
            )

        query_spike_e = query_count_e - previous_count_e
        query_spike_i = query_count_i - previous_count_i
        query_residual_e = query_q_e - query_count_e
        query_residual_i = query_q_i - query_count_i
        raw_queries = torch.cat(
            (
                query_spike_e,
                -query_spike_i,
                query_residual_e,
                -query_residual_i,
            ),
            dim=-1,
        )
        final_e = cumulative_e[:, -1] - counts_e[:, -1]
        final_i = cumulative_i[:, -1] - counts_i[:, -1]

        phi_e = _event_step_derivative(drive_e, scale)
        phi_i = _event_step_derivative(drive_i, scale)
        batch_size, _, hidden = phi_e.shape
        input_dim = x.shape[-1]
        running_e = x.new_zeros(batch_size, hidden, input_dim)
        running_i = x.new_zeros(batch_size, hidden, input_dim)
        running_bias_e = x.new_zeros(batch_size, hidden)
        running_bias_i = x.new_zeros(batch_size, hidden)
        eligibility_e = []
        eligibility_i = []
        bias_eligibility_e = []
        bias_eligibility_i = []
        cursor = 0
        for position in query_positions:
            stop = position + 1
            if stop > cursor:
                running_e = running_e + torch.einsum(
                    "bth,btd->bhd", phi_e[:, cursor:stop], x[:, cursor:stop]
                )
                running_i = running_i + torch.einsum(
                    "bth,btd->bhd", phi_i[:, cursor:stop], x[:, cursor:stop]
                )
                running_bias_e = running_bias_e + phi_e[:, cursor:stop].sum(dim=1)
                running_bias_i = running_bias_i + phi_i[:, cursor:stop].sum(dim=1)
            eligibility_e.append(running_e.clone())
            eligibility_i.append(running_i.clone())
            bias_eligibility_e.append(running_bias_e.clone())
            bias_eligibility_i.append(running_bias_i.clone())
            cursor = stop
        if cursor < time_steps:
            running_e = running_e + torch.einsum(
                "bth,btd->bhd", phi_e[:, cursor:], x[:, cursor:]
            )
            running_i = running_i + torch.einsum(
                "bth,btd->bhd", phi_i[:, cursor:], x[:, cursor:]
            )
            running_bias_e = running_bias_e + phi_e[:, cursor:].sum(dim=1)
            running_bias_i = running_bias_i + phi_i[:, cursor:].sum(dim=1)

        query_phi_e = phi_e.index_select(1, query_indices).clone()
        query_phi_i = phi_i.index_select(1, query_indices).clone()
        query_x = x.index_select(1, query_indices).clone()
        saved = [
            query_q_e,
            previous_q_e,
            query_q_i,
            previous_q_i,
            torch.stack(eligibility_e, dim=1),
            torch.stack(eligibility_i, dim=1),
            torch.stack(bias_eligibility_e, dim=1),
            torch.stack(bias_eligibility_i, dim=1),
            query_phi_e,
            query_phi_i,
            query_x,
            running_e.clone(),
            running_i.clone(),
            running_bias_e.clone(),
            running_bias_i.clone(),
            weight_e,
            weight_i,
        ]
        ctx.input_requires_grad = bool(ctx.needs_input_grad[0])
        if ctx.input_requires_grad:
            saved.extend((phi_e, phi_i))
        ctx.save_for_backward(*saved)
        ctx.query_positions = query_positions
        ctx.time_steps = int(time_steps)
        ctx.event_charge = float(event_charge)
        ctx.surrogate_scale = scale
        return raw_queries, final_e, final_i

    @staticmethod
    def backward(
        ctx: Any,
        grad_raw: Optional[Tensor],
        grad_final_e: Optional[Tensor],
        grad_final_i: Optional[Tensor],
    ) -> Tuple[
        Optional[Tensor],
        None,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        None,
        None,
        None,
    ]:
        saved = ctx.saved_tensors
        (
            query_q_e,
            previous_q_e,
            query_q_i,
            previous_q_i,
            eligibility_e,
            eligibility_i,
            bias_eligibility_e,
            bias_eligibility_i,
            query_phi_e,
            query_phi_i,
            query_x,
            final_eligibility_e,
            final_eligibility_i,
            final_bias_eligibility_e,
            final_bias_eligibility_i,
            weight_e,
            weight_i,
        ) = saved[:17]
        phi_e: Optional[Tensor] = None
        phi_i: Optional[Tensor] = None
        if ctx.input_requires_grad:
            phi_e, phi_i = saved[17:19]

        hidden = query_q_e.shape[-1]
        zero_query = torch.zeros_like(query_q_e)
        if grad_raw is None:
            spike_signal_e = zero_query
            spike_signal_i = zero_query
            residual_signal_e = zero_query
            residual_signal_i = zero_query
        else:
            spike_signal_e = grad_raw[:, :, :hidden]
            spike_signal_i = -grad_raw[:, :, hidden : 2 * hidden]
            residual_signal_e = grad_raw[:, :, 2 * hidden : 3 * hidden]
            residual_signal_i = -grad_raw[:, :, 3 * hidden :]

        scale = ctx.surrogate_scale
        floor_e = _periodic_floor_derivative(query_q_e, scale)
        floor_i = _periodic_floor_derivative(query_q_i, scale)
        previous_floor_e = _periodic_floor_derivative(previous_q_e, scale)
        previous_floor_i = _periodic_floor_derivative(previous_q_i, scale)
        for query_index, position in enumerate(ctx.query_positions):
            if position == 0:
                previous_floor_e[:, query_index].zero_()
                previous_floor_i[:, query_index].zero_()

        terminal_signal_e = residual_signal_e + spike_signal_e * floor_e
        terminal_signal_i = residual_signal_i + spike_signal_i * floor_i
        previous_signal_e = spike_signal_e * previous_floor_e
        previous_signal_i = spike_signal_i * previous_floor_i
        final_signal_e = (
            torch.zeros_like(query_q_e[:, 0])
            if grad_final_e is None
            else grad_final_e
        )
        final_signal_i = (
            torch.zeros_like(query_q_i[:, 0])
            if grad_final_i is None
            else grad_final_i
        )

        query_contribution_e = query_phi_e.unsqueeze(-1) * query_x.unsqueeze(-2)
        query_contribution_i = query_phi_i.unsqueeze(-1) * query_x.unsqueeze(-2)
        previous_eligibility_e = eligibility_e - query_contribution_e
        previous_eligibility_i = eligibility_i - query_contribution_i
        previous_bias_eligibility_e = bias_eligibility_e - query_phi_e
        previous_bias_eligibility_i = bias_eligibility_i - query_phi_i
        event_charge = ctx.event_charge

        grad_weight_e = event_charge * (
            torch.sum(
                terminal_signal_e.unsqueeze(-1) * eligibility_e
                - previous_signal_e.unsqueeze(-1) * previous_eligibility_e,
                dim=(0, 1),
            )
            + torch.sum(
                final_signal_e.unsqueeze(-1) * final_eligibility_e,
                dim=0,
            )
        )
        grad_weight_i = event_charge * (
            torch.sum(
                terminal_signal_i.unsqueeze(-1) * eligibility_i
                - previous_signal_i.unsqueeze(-1) * previous_eligibility_i,
                dim=(0, 1),
            )
            + torch.sum(
                final_signal_i.unsqueeze(-1) * final_eligibility_i,
                dim=0,
            )
        )
        grad_bias_e = event_charge * (
            torch.sum(
                terminal_signal_e * bias_eligibility_e
                - previous_signal_e * previous_bias_eligibility_e,
                dim=(0, 1),
            )
            + torch.sum(final_signal_e * final_bias_eligibility_e, dim=0)
        )
        grad_bias_i = event_charge * (
            torch.sum(
                terminal_signal_i * bias_eligibility_i
                - previous_signal_i * previous_bias_eligibility_i,
                dim=(0, 1),
            )
            + torch.sum(final_signal_i * final_bias_eligibility_i, dim=0)
        )
        grad_initial_e = (
            terminal_signal_e - previous_signal_e
        ).sum(dim=1) + final_signal_e
        grad_initial_i = (
            terminal_signal_i - previous_signal_i
        ).sum(dim=1) + final_signal_i

        grad_x: Optional[Tensor] = None
        if ctx.input_requires_grad:
            assert phi_e is not None and phi_i is not None
            coefficient_e = final_signal_e.unsqueeze(1).expand(
                -1, ctx.time_steps, -1
            ).clone()
            coefficient_i = final_signal_i.unsqueeze(1).expand(
                -1, ctx.time_steps, -1
            ).clone()
            for query_index, position in enumerate(ctx.query_positions):
                coefficient_e[:, : position + 1] += terminal_signal_e[
                    :, query_index
                ].unsqueeze(1)
                coefficient_i[:, : position + 1] += terminal_signal_i[
                    :, query_index
                ].unsqueeze(1)
                if position > 0:
                    coefficient_e[:, :position] -= previous_signal_e[
                        :, query_index
                    ].unsqueeze(1)
                    coefficient_i[:, :position] -= previous_signal_i[
                        :, query_index
                    ].unsqueeze(1)
            grad_drive_e = event_charge * phi_e * coefficient_e
            grad_drive_i = event_charge * phi_i * coefficient_i
            grad_x = torch.matmul(grad_drive_e, weight_e) + torch.matmul(
                grad_drive_i, weight_i
            )

        return (
            grad_x,
            None,
            grad_weight_e,
            grad_bias_e,
            grad_weight_i,
            grad_bias_i,
            grad_initial_e,
            grad_initial_i,
            None,
            None,
            None,
        )


class _GatedTraceMultiQueryEligibility(torch.autograd.Function):
    """Exact sparse-query gradient for the AT0 gated affine trace."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        query_indices: Tensor,
        weight: Tensor,
        bias: Tensor,
        decay_logits: Tensor,
        initial_e: Tensor,
        initial_i: Tensor,
        min_decay: float,
        max_decay: float,
        spike_threshold: float,
        surrogate_scale: float,
        scan_aligned: bool,
        reverse_adjoint: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        time_steps = x.shape[1]
        state_dim = initial_e.shape[-1]
        query_positions = tuple(int(value) for value in query_indices.tolist())
        scale = float(surrogate_scale)

        drives = F.linear(x, weight, bias)
        drive_content_e, drive_content_i, drive_gate_e, drive_gate_i = drives.chunk(
            4, dim=-1
        )
        content_e = (drive_content_e >= 0.0).to(dtype=x.dtype)
        content_i = (drive_content_i >= 0.0).to(dtype=x.dtype)
        gate_e = (drive_gate_e >= 0.0).to(dtype=x.dtype)
        gate_i = (drive_gate_i >= 0.0).to(dtype=x.dtype)
        write_e = content_e * gate_e
        write_i = content_i * gate_i

        decay_sigmoid = torch.sigmoid(decay_logits)
        decay_span = float(max_decay) - float(min_decay)
        decays = float(min_decay) + decay_span * decay_sigmoid
        decay_derivative = decay_span * decay_sigmoid * (1.0 - decay_sigmoid)
        decay_e, decay_i = decays[0], decays[1]

        def trace_and_decay_snapshots(
            write: Tensor,
            decay: Tensor,
            decay_logit_derivative: Tensor,
            initial: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            """Evaluate only query/final traces and their decay eligibility."""

            running_trace = initial
            running_decay = torch.zeros_like(initial)
            query_traces = []
            query_decays = []
            cursor = 0

            def advance(stop: int) -> None:
                nonlocal cursor, running_trace, running_decay
                length = stop - cursor
                exponents = torch.arange(
                    length - 1,
                    -1,
                    -1,
                    device=x.device,
                    dtype=x.dtype,
                ).unsqueeze(1)
                powers = decay.unsqueeze(0).pow(exponents)
                exponent_derivative = torch.where(
                    exponents == 0,
                    torch.zeros_like(powers),
                    exponents * decay.unsqueeze(0).pow(exponents - 1.0),
                )
                injection_weights = (1.0 - decay).unsqueeze(0) * powers
                injection_derivative = decay_logit_derivative.unsqueeze(0) * (
                    -powers
                    + (1.0 - decay).unsqueeze(0) * exponent_derivative
                )
                segment = write[:, cursor:stop]
                segment_bias = torch.sum(
                    segment * injection_weights.unsqueeze(0), dim=1
                )
                segment_bias_derivative = torch.sum(
                    segment * injection_derivative.unsqueeze(0), dim=1
                )
                retention = decay.pow(length)
                retention_derivative = (
                    float(length)
                    * decay.pow(length - 1)
                    * decay_logit_derivative
                )
                previous_trace = running_trace
                running_trace = (
                    retention.unsqueeze(0) * previous_trace + segment_bias
                )
                running_decay = (
                    retention.unsqueeze(0) * running_decay
                    + retention_derivative.unsqueeze(0) * previous_trace
                    + segment_bias_derivative
                )
                cursor = stop

            for position in query_positions:
                advance(position + 1)
                query_traces.append(running_trace.clone())
                query_decays.append(running_decay.clone())
            if cursor < time_steps:
                advance(time_steps)
            return (
                torch.stack(query_traces, dim=1),
                torch.stack(query_decays, dim=1),
                running_trace.clone(),
                running_decay.clone(),
            )

        (
            query_trace_pair,
            query_decay_eligibility,
            final_pair,
            final_decay_eligibility,
        ) = trace_and_decay_snapshots(
            torch.cat((write_e, write_i), dim=-1),
            torch.cat((decay_e, decay_i), dim=0),
            decay_derivative.reshape(-1),
            torch.cat((initial_e, initial_i), dim=-1),
        )
        query_trace_e, query_trace_i = query_trace_pair.chunk(2, dim=-1)
        final_e, final_i = final_pair.chunk(2, dim=-1)
        if scan_aligned:
            coefficient_e = decay_e.view(1, 1, -1).expand_as(write_e)
            coefficient_i = decay_i.view(1, 1, -1).expand_as(write_i)
            aligned_e = E3GatedTraceScanCore._affine_prefix_scan(
                coefficient_e,
                (1.0 - coefficient_e) * write_e,
                initial_e,
            )
            aligned_i = E3GatedTraceScanCore._affine_prefix_scan(
                coefficient_i,
                (1.0 - coefficient_i) * write_i,
                initial_i,
            )
            query_trace_e = aligned_e.index_select(1, query_indices).clone()
            query_trace_i = aligned_i.index_select(1, query_indices).clone()
            final_e = aligned_e[:, -1].clone()
            final_i = aligned_i[:, -1].clone()
        query_spike_e = (query_trace_e >= float(spike_threshold)).to(dtype=x.dtype)
        query_spike_i = (query_trace_i >= float(spike_threshold)).to(dtype=x.dtype)
        raw_queries = torch.cat(
            (query_spike_e, -query_spike_i, query_trace_e, -query_trace_i),
            dim=-1,
        )

        phi_content_e = _event_step_derivative(drive_content_e, scale)
        phi_content_i = _event_step_derivative(drive_content_i, scale)
        phi_gate_e = _event_step_derivative(drive_gate_e, scale)
        phi_gate_i = _event_step_derivative(drive_gate_i, scale)
        drive_factors = torch.cat(
            (
                gate_e * phi_content_e,
                gate_i * phi_content_i,
                content_e * phi_gate_e,
                content_i * phi_gate_i,
            ),
            dim=-1,
        )
        ctx.input_requires_grad = bool(ctx.needs_input_grad[0])
        ctx.reverse_adjoint = bool(reverse_adjoint)
        ctx.query_positions = query_positions
        ctx.time_steps = int(time_steps)
        ctx.state_dim = int(state_dim)
        ctx.spike_threshold = float(spike_threshold)
        ctx.surrogate_scale = scale
        if reverse_adjoint:
            if not scan_aligned:
                coefficient_e = decay_e.view(1, 1, -1).expand_as(write_e)
                coefficient_i = decay_i.view(1, 1, -1).expand_as(write_i)
                aligned_e = E3GatedTraceScanCore._affine_prefix_scan(
                    coefficient_e,
                    (1.0 - coefficient_e) * write_e,
                    initial_e,
                )
                aligned_i = E3GatedTraceScanCore._affine_prefix_scan(
                    coefficient_i,
                    (1.0 - coefficient_i) * write_i,
                    initial_i,
                )
            previous_trace_pair = torch.cat(
                (
                    torch.cat((initial_e.unsqueeze(1), aligned_e[:, :-1]), dim=1),
                    torch.cat((initial_i.unsqueeze(1), aligned_i[:, :-1]), dim=1),
                ),
                dim=-1,
            )
            ctx.save_for_backward(
                query_trace_e,
                query_trace_i,
                x,
                previous_trace_pair,
                torch.cat((write_e, write_i), dim=-1),
                drive_factors,
                decay_derivative.reshape(-1),
                decay_e,
                decay_i,
                weight,
            )
            return raw_queries, final_e, final_i

        local_weight = torch.cat(
            (
                (1.0 - decay_e) * gate_e * phi_content_e,
                (1.0 - decay_i) * gate_i * phi_content_i,
                (1.0 - decay_e) * content_e * phi_gate_e,
                (1.0 - decay_i) * content_i * phi_gate_i,
            ),
            dim=-1,
        )
        decay_rows = torch.cat((decay_e, decay_i, decay_e, decay_i), dim=0)

        def matrix_snapshots(
            local: Tensor,
            recurrence: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            batch_size, _, rows = local.shape
            input_dim = x.shape[-1]
            running_weight = x.new_zeros(batch_size, rows, input_dim)
            running_bias = x.new_zeros(batch_size, rows)
            query_weight = []
            query_bias = []
            cursor = 0
            for position in query_positions:
                stop = position + 1
                length = stop - cursor
                powers = recurrence.unsqueeze(0).pow(
                    torch.arange(
                        length - 1,
                        -1,
                        -1,
                        device=x.device,
                        dtype=x.dtype,
                    ).unsqueeze(1)
                )
                weighted = local[:, cursor:stop] * powers.unsqueeze(0)
                retention = recurrence.pow(length)
                running_weight = retention.view(1, -1, 1) * running_weight + torch.einsum(
                    "btr,btd->brd", weighted, x[:, cursor:stop]
                )
                running_bias = retention.unsqueeze(0) * running_bias + weighted.sum(dim=1)
                query_weight.append(running_weight.clone())
                query_bias.append(running_bias.clone())
                cursor = stop
            if cursor < time_steps:
                length = time_steps - cursor
                powers = recurrence.unsqueeze(0).pow(
                    torch.arange(
                        length - 1,
                        -1,
                        -1,
                        device=x.device,
                        dtype=x.dtype,
                    ).unsqueeze(1)
                )
                weighted = local[:, cursor:] * powers.unsqueeze(0)
                retention = recurrence.pow(length)
                running_weight = retention.view(1, -1, 1) * running_weight + torch.einsum(
                    "btr,btd->brd", weighted, x[:, cursor:]
                )
                running_bias = retention.unsqueeze(0) * running_bias + weighted.sum(dim=1)
            return (
                torch.stack(query_weight, dim=1),
                torch.stack(query_bias, dim=1),
                running_weight.clone(),
                running_bias.clone(),
            )

        (
            query_weight_eligibility,
            query_bias_eligibility,
            final_weight_eligibility,
            final_bias_eligibility,
        ) = matrix_snapshots(local_weight, decay_rows)

        saved = [
            query_trace_e,
            query_trace_i,
            query_weight_eligibility,
            query_bias_eligibility,
            query_decay_eligibility,
            final_weight_eligibility,
            final_bias_eligibility,
            final_decay_eligibility,
            decay_e,
            decay_i,
            weight,
        ]
        if ctx.input_requires_grad:
            saved.append(drive_factors)
        ctx.save_for_backward(*saved)
        return raw_queries, final_e, final_i

    @staticmethod
    def backward(
        ctx: Any,
        grad_raw: Optional[Tensor],
        grad_final_e: Optional[Tensor],
        grad_final_i: Optional[Tensor],
    ) -> Tuple[
        Optional[Tensor],
        None,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        saved = ctx.saved_tensors
        if ctx.reverse_adjoint:
            (
                query_trace_e,
                query_trace_i,
                reverse_x,
                previous_trace_pair,
                write_pair,
                drive_factors,
                decay_logit_derivative,
                decay_e,
                decay_i,
                weight,
            ) = saved
        else:
            (
                query_trace_e,
                query_trace_i,
                query_weight_eligibility,
                query_bias_eligibility,
                query_decay_eligibility,
                final_weight_eligibility,
                final_bias_eligibility,
                final_decay_eligibility,
                decay_e,
                decay_i,
                weight,
            ) = saved[:11]
            drive_factors: Optional[Tensor] = None
            if ctx.input_requires_grad:
                drive_factors = saved[11]

        state_dim = ctx.state_dim
        zero_query = torch.zeros_like(query_trace_e)
        if grad_raw is None:
            spike_signal_e = zero_query
            spike_signal_i = zero_query
            trace_signal_e = zero_query
            trace_signal_i = zero_query
        else:
            spike_signal_e = grad_raw[:, :, :state_dim]
            spike_signal_i = -grad_raw[:, :, state_dim : 2 * state_dim]
            trace_signal_e = grad_raw[:, :, 2 * state_dim : 3 * state_dim]
            trace_signal_i = -grad_raw[:, :, 3 * state_dim :]
        output_derivative_e = _event_step_derivative(
            query_trace_e - ctx.spike_threshold, ctx.surrogate_scale
        )
        output_derivative_i = _event_step_derivative(
            query_trace_i - ctx.spike_threshold, ctx.surrogate_scale
        )
        learning_e = trace_signal_e + spike_signal_e * output_derivative_e
        learning_i = trace_signal_i + spike_signal_i * output_derivative_i
        final_signal_e = (
            torch.zeros_like(query_trace_e[:, 0])
            if grad_final_e is None
            else grad_final_e
        )
        final_signal_i = (
            torch.zeros_like(query_trace_i[:, 0])
            if grad_final_i is None
            else grad_final_i
        )

        if ctx.reverse_adjoint:
            direct_e = torch.zeros(
                query_trace_e.shape[0],
                ctx.time_steps,
                state_dim,
                device=query_trace_e.device,
                dtype=query_trace_e.dtype,
            )
            direct_i = torch.zeros_like(direct_e)
            for query_index, position in enumerate(ctx.query_positions):
                direct_e[:, position] += learning_e[:, query_index]
                direct_i[:, position] += learning_i[:, query_index]
            direct_e[:, -1] += final_signal_e
            direct_i[:, -1] += final_signal_i

            def sparse_reverse_adjoint(direct: Tensor, decay: Tensor) -> Tensor:
                result = torch.zeros_like(direct)
                impulse_positions = tuple(
                    sorted(set(ctx.query_positions + (ctx.time_steps - 1,)))
                )
                next_value = torch.zeros_like(direct[:, 0])
                for impulse_index in range(len(impulse_positions) - 1, -1, -1):
                    position = impulse_positions[impulse_index]
                    lower = (
                        impulse_positions[impulse_index - 1] + 1
                        if impulse_index > 0
                        else 0
                    )
                    value_at_impulse = (
                        direct[:, position] + decay.unsqueeze(0) * next_value
                    )
                    exponents = torch.arange(
                        position - lower,
                        -1,
                        -1,
                        device=direct.device,
                        dtype=direct.dtype,
                    ).unsqueeze(1)
                    segment = value_at_impulse.unsqueeze(1) * decay.unsqueeze(0).pow(
                        exponents
                    ).unsqueeze(0)
                    result[:, lower : position + 1] = segment
                    next_value = segment[:, 0]
                return result

            adjoint_e = sparse_reverse_adjoint(direct_e, decay_e)
            adjoint_i = sparse_reverse_adjoint(direct_i, decay_i)
            adjoint_pair = torch.cat((adjoint_e, adjoint_i), dim=-1)
            adjoint_rows = torch.cat(
                (adjoint_e, adjoint_i, adjoint_e, adjoint_i), dim=-1
            )
            decay_rows = torch.cat((decay_e, decay_i, decay_e, decay_i), dim=0)
            grad_drives = (
                (1.0 - decay_rows).view(1, 1, -1)
                * adjoint_rows
                * drive_factors
            )
            grad_weight = torch.einsum("btr,btd->rd", grad_drives, reverse_x)
            grad_bias = grad_drives.sum(dim=(0, 1))
            local_decay = decay_logit_derivative.view(1, 1, -1) * (
                previous_trace_pair - write_pair
            )
            grad_decay_logits = torch.sum(
                adjoint_pair * local_decay, dim=(0, 1)
            ).reshape(2, state_dim)
            initial_gradient = (
                torch.cat((decay_e, decay_i), dim=0).unsqueeze(0)
                * adjoint_pair[:, 0]
            )
            grad_initial_e, grad_initial_i = initial_gradient.chunk(2, dim=-1)
            grad_x = (
                torch.matmul(grad_drives, weight)
                if ctx.input_requires_grad
                else None
            )
            return (
                grad_x,
                None,
                grad_weight,
                grad_bias,
                grad_decay_logits,
                grad_initial_e,
                grad_initial_i,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        learning_rows = torch.cat(
            (learning_e, learning_i, learning_e, learning_i), dim=-1
        )
        final_rows = torch.cat(
            (final_signal_e, final_signal_i, final_signal_e, final_signal_i),
            dim=-1,
        )
        grad_weight = torch.sum(
            learning_rows.unsqueeze(-1) * query_weight_eligibility,
            dim=(0, 1),
        ) + torch.sum(
            final_rows.unsqueeze(-1) * final_weight_eligibility,
            dim=0,
        )
        grad_bias = torch.sum(
            learning_rows * query_bias_eligibility,
            dim=(0, 1),
        ) + torch.sum(final_rows * final_bias_eligibility, dim=0)
        learning_decay = torch.cat((learning_e, learning_i), dim=-1)
        final_decay_signal = torch.cat((final_signal_e, final_signal_i), dim=-1)
        grad_decay_logits = torch.sum(
            learning_decay * query_decay_eligibility,
            dim=(0, 1),
        ) + torch.sum(final_decay_signal * final_decay_eligibility, dim=0)
        grad_decay_logits = grad_decay_logits.reshape(2, state_dim)

        query_powers_e = torch.stack(
            [decay_e.pow(position + 1) for position in ctx.query_positions], dim=0
        )
        query_powers_i = torch.stack(
            [decay_i.pow(position + 1) for position in ctx.query_positions], dim=0
        )
        grad_initial_e = torch.sum(
            learning_e * query_powers_e.unsqueeze(0), dim=1
        ) + final_signal_e * decay_e.pow(ctx.time_steps)
        grad_initial_i = torch.sum(
            learning_i * query_powers_i.unsqueeze(0), dim=1
        ) + final_signal_i * decay_i.pow(ctx.time_steps)

        grad_x: Optional[Tensor] = None
        if ctx.input_requires_grad:
            assert drive_factors is not None
            direct_e = torch.zeros(
                query_trace_e.shape[0],
                ctx.time_steps,
                state_dim,
                device=query_trace_e.device,
                dtype=query_trace_e.dtype,
            )
            direct_i = torch.zeros_like(direct_e)
            for query_index, position in enumerate(ctx.query_positions):
                direct_e[:, position] += learning_e[:, query_index]
                direct_i[:, position] += learning_i[:, query_index]
            direct_e[:, -1] += final_signal_e
            direct_i[:, -1] += final_signal_i
            reverse_e = torch.flip(direct_e, dims=(1,))
            reverse_i = torch.flip(direct_i, dims=(1,))
            coefficient_e = decay_e.view(1, 1, -1).expand_as(reverse_e)
            coefficient_i = decay_i.view(1, 1, -1).expand_as(reverse_i)
            adjoint_e = torch.flip(
                E3GatedTraceScanCore._affine_prefix_scan(
                    coefficient_e,
                    reverse_e,
                    torch.zeros_like(reverse_e[:, 0]),
                ),
                dims=(1,),
            )
            adjoint_i = torch.flip(
                E3GatedTraceScanCore._affine_prefix_scan(
                    coefficient_i,
                    reverse_i,
                    torch.zeros_like(reverse_i[:, 0]),
                ),
                dims=(1,),
            )
            adjoint_rows = torch.cat(
                (adjoint_e, adjoint_i, adjoint_e, adjoint_i), dim=-1
            )
            decay_rows = torch.cat((decay_e, decay_i, decay_e, decay_i), dim=0)
            grad_drives = (
                (1.0 - decay_rows).view(1, 1, -1)
                * adjoint_rows
                * drive_factors
            )
            grad_x = torch.matmul(grad_drives, weight)

        return (
            grad_x,
            None,
            grad_weight,
            grad_bias,
            grad_decay_logits,
            grad_initial_e,
            grad_initial_i,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class E3InputCodedScanCore(E3CumulativeScanCore):
    """One-layer exact-reset scan driven by explicit binary input events."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        state_dim: Optional[int] = None,
        base_charge: float = 0.125,
        event_charge: float = 0.75,
        event_surrogate_scale: float = 5.0,
        execution_mode: E3ExecutionMode = "scan",
    ) -> None:
        if base_charge < 0.0 or event_charge <= 0.0:
            raise ValueError("base_charge must be non-negative and event_charge positive")
        if base_charge + event_charge >= 1.0:
            raise ValueError("base_charge + event_charge must remain below threshold one")
        if event_surrogate_scale <= 0.0:
            raise ValueError("event_surrogate_scale must be positive")
        super().__init__(
            input_dim,
            hidden_dim,
            state_dim=state_dim,
            num_layers=1,
            max_charge=base_charge + event_charge,
            surrogate_scale=event_surrogate_scale,
            execution_mode=execution_mode,
        )
        self.base_charge = float(base_charge)
        self.event_charge = float(event_charge)
        self.event_surrogate_scale = float(event_surrogate_scale)
        nn.init.zeros_(self.input_to_e.bias)
        nn.init.zeros_(self.input_to_i.bias)

    def input_events(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        _validate_sequence(x, self.input_dim)
        return (
            _surrogate_step(self.input_to_e(x), self.event_surrogate_scale),
            _surrogate_step(self.input_to_i(x), self.event_surrogate_scale),
        )

    def _charge(self, drive: Tensor) -> Tensor:
        events = _surrogate_step(drive, self.event_surrogate_scale)
        return self.base_charge + self.event_charge * events

    def _forward_step_tensors_unchecked(
        self,
        x_t: Tensor,
        excitatory: Tensor,
        inhibitory: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pure-tensor inference graph for one exact IC0 event."""

        event_e = (F.linear(x_t, self.input_to_e.weight, self.input_to_e.bias) >= 0.0).to(
            dtype=x_t.dtype
        )
        event_i = (F.linear(x_t, self.input_to_i.weight, self.input_to_i.bias) >= 0.0).to(
            dtype=x_t.dtype
        )
        pre_reset_e = excitatory + self.base_charge + self.event_charge * event_e
        pre_reset_i = inhibitory + self.base_charge + self.event_charge * event_i
        spike_e = torch.floor(pre_reset_e)
        spike_i = torch.floor(pre_reset_i)
        next_e = pre_reset_e - spike_e
        next_i = pre_reset_i - spike_i
        raw = torch.cat((spike_e, -spike_i, next_e, -next_i), dim=-1)
        output = self.output_projection(self.output_norm(raw))
        return output, next_e, next_i, spike_e, spike_i

    def forward_step_tensors(
        self,
        x_t: Tensor,
        excitatory: Tensor,
        inhibitory: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Exact single-event inference without sequence/dataclass overhead.

        The returned tuple is ``(output, next_e, next_i, spike_e, spike_i)``.
        It is deliberately tensor-only so a deployment wrapper can compile the
        complete event update as one graph.  Training should continue to use
        :meth:`forward` or :meth:`forward_terminal_eligibility` because the
        hard comparisons in this inference path do not install surrogates.
        """

        if x_t.ndim != 2 or x_t.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected x_t shaped [batch, {self.input_dim}], got {tuple(x_t.shape)}"
            )
        expected = (x_t.shape[0], self.state_dim)
        if tuple(excitatory.shape) != expected or tuple(inhibitory.shape) != expected:
            raise ValueError(
                "invalid IC0 tensor state shapes: "
                f"E={tuple(excitatory.shape)}, I={tuple(inhibitory.shape)}, "
                f"expected={expected}"
            )
        return self._forward_step_tensors_unchecked(x_t, excitatory, inhibitory)

    def forward_terminal_eligibility(
        self,
        x: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E3ScanState]:
        """Return only the final token using an exact eligibility backward.

        This method is intended for query-sparse objectives whose loss reads
        the final token.  It does not manufacture outputs for earlier tokens.
        With a frozen/detached input tensor, its saved eligibility state is
        independent of sequence length; when ``x.requires_grad`` is true it
        retains per-time event derivatives to reconstruct the exact input
        gradient without a temporal autograd chain.
        """

        batch_size, _ = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        layer = state.layers[0]
        raw, final_e, final_i = _TerminalEligibilityScan.apply(
            x,
            self.input_to_e.weight,
            self.input_to_e.bias,
            self.input_to_i.weight,
            self.input_to_i.bias,
            layer.excitatory,
            layer.inhibitory,
            self.base_charge,
            self.event_charge,
            self.event_surrogate_scale,
        )
        sequence = self.output_projection(self.output_norm(raw)).unsqueeze(1)
        next_state = E3ScanState(
            layers=(E3LayerState(excitatory=final_e, inhibitory=final_i),)
        )
        if detach_state:
            next_state = detach_core_state(next_state)
        return CoreOutput(sequence=sequence, state=next_state)

    def forward_multi_query_eligibility(
        self,
        x: Tensor,
        query_indices: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E3ScanState]:
        """Return sparse query outputs with an exact prefix-eligibility backward.

        ``query_indices`` identifies ordered sequence positions whose outputs
        participate in the loss.  The returned sequence has length ``K`` in
        the same order.  With frozen inputs, backward storage scales with K
        rather than the input sequence length; dense input gradients remain
        exact but retain per-time event derivatives.
        """

        batch_size, time_steps = _validate_sequence(x, self.input_dim)
        if not isinstance(query_indices, Tensor):
            raise TypeError("query_indices must be a torch.Tensor")
        if query_indices.ndim != 1:
            raise ValueError("query_indices must be one-dimensional")
        if query_indices.dtype != torch.long:
            raise ValueError("query_indices must use torch.long dtype")
        if query_indices.numel() == 0:
            raise ValueError("query_indices must be non-empty")
        indices = query_indices.to(device=x.device)
        if int(indices[0].item()) < 0 or int(indices[-1].item()) >= time_steps:
            raise ValueError(
                f"query_indices must lie in [0, {time_steps}), got "
                f"[{int(indices[0].item())}, {int(indices[-1].item())}]"
            )
        if indices.numel() > 1 and not bool(torch.all(indices[1:] > indices[:-1])):
            raise ValueError("query_indices must be strictly increasing and unique")
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        layer = state.layers[0]
        raw, final_e, final_i = _MultiQueryEligibilityScan.apply(
            x,
            indices,
            self.input_to_e.weight,
            self.input_to_e.bias,
            self.input_to_i.weight,
            self.input_to_i.bias,
            layer.excitatory,
            layer.inhibitory,
            self.base_charge,
            self.event_charge,
            self.event_surrogate_scale,
        )
        sequence = self.output_projection(self.output_norm(raw))
        next_state = E3ScanState(
            layers=(E3LayerState(excitatory=final_e, inhibitory=final_i),)
        )
        if detach_state:
            next_state = detach_core_state(next_state)
        return CoreOutput(sequence=sequence, state=next_state)


class E3GatedTraceScanCore(TemporalCore[E3ScanState]):
    """Binary-event SNN with an exact affine scan over slow synaptic traces.

    Four hard-threshold populations encode excitatory/inhibitory content and
    write gates.  Their binary products drive bounded leaky traces; output
    spikes threshold those traces but do not reset them.  This separates the
    slow, exactly scannable memory state from the fast discrete output event.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        state_dim: Optional[int] = None,
        min_decay: float = 0.50,
        max_decay: float = 0.995,
        min_initial_decay: float = 0.55,
        max_initial_decay: float = 0.99,
        spike_threshold: float = 0.50,
        surrogate_scale: float = 5.0,
        execution_mode: E3ExecutionMode = "scan",
        eligibility_forward_mode: E3EligibilityForwardMode = "segment",
        eligibility_backward_mode: E3EligibilityBackwardMode = "forward_eligibility",
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        state_dim = hidden_dim if state_dim is None else state_dim
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 <= min_decay < max_decay < 1.0:
            raise ValueError("decay bounds must satisfy 0 <= min < max < 1")
        if not min_decay < min_initial_decay <= max_initial_decay < max_decay:
            raise ValueError("initial decay range must lie inside decay bounds")
        if not 0.0 < spike_threshold < 1.0:
            raise ValueError("spike_threshold must lie in (0, 1)")
        if surrogate_scale <= 0.0:
            raise ValueError("surrogate_scale must be positive")
        if execution_mode not in ("serial", "scan"):
            raise ValueError("execution_mode must be 'serial' or 'scan'")
        if eligibility_forward_mode not in ("segment", "scan_aligned"):
            raise ValueError(
                "eligibility_forward_mode must be 'segment' or 'scan_aligned'"
            )
        if eligibility_backward_mode not in (
            "forward_eligibility",
            "reverse_adjoint",
        ):
            raise ValueError(
                "eligibility_backward_mode must be 'forward_eligibility' "
                "or 'reverse_adjoint'"
            )

        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.min_decay = float(min_decay)
        self.max_decay = float(max_decay)
        self.spike_threshold = float(spike_threshold)
        self.surrogate_scale = float(surrogate_scale)
        self.execution_mode: E3ExecutionMode = execution_mode
        self.eligibility_forward_mode: E3EligibilityForwardMode = (
            eligibility_forward_mode
        )
        self.eligibility_backward_mode: E3EligibilityBackwardMode = (
            eligibility_backward_mode
        )

        self.input_event_projection = nn.Linear(input_dim, 4 * self.state_dim)
        self.decay_logits = nn.Parameter(torch.empty(2, self.state_dim))
        self.output_norm = nn.LayerNorm(4 * self.state_dim)
        self.output_projection = nn.Linear(4 * self.state_dim, hidden_dim)

        initial_decay = torch.linspace(
            min_initial_decay,
            max_initial_decay,
            steps=self.state_dim,
        )
        normalised = (initial_decay - self.min_decay) / (
            self.max_decay - self.min_decay
        )
        logits = torch.logit(normalised)
        with torch.no_grad():
            self.decay_logits[0].copy_(logits)
            self.decay_logits[1].copy_(logits.flip(0))

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> E3ScanState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        default_device, default_dtype = _module_device_dtype(self)
        device = default_device if device is None else device
        dtype = default_dtype if dtype is None else dtype
        zeros = torch.zeros(
            batch_size, self.state_dim, device=device, dtype=dtype
        )
        return E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=zeros.clone(),
                    inhibitory=zeros.clone(),
                ),
            )
        )

    def _validate_state(self, state: E3ScanState, batch_size: int) -> None:
        if len(state.layers) != 1:
            raise ValueError(f"expected one AT0 state layer, got {len(state.layers)}")
        expected = (batch_size, self.state_dim)
        layer = state.layers[0]
        if tuple(layer.excitatory.shape) != expected or tuple(layer.inhibitory.shape) != expected:
            raise ValueError(
                "invalid AT0 state shapes: "
                f"E={tuple(layer.excitatory.shape)}, I={tuple(layer.inhibitory.shape)}, "
                f"expected={expected}"
            )
        for name, value in (
            ("excitatory", layer.excitatory),
            ("inhibitory", layer.inhibitory),
        ):
            if not bool(torch.all((value >= 0.0) & (value <= 1.0))):
                raise ValueError(f"AT0 {name} trace must lie in [0, 1]")

    def decays(self) -> Tuple[Tensor, Tensor]:
        """Return bounded per-neuron E/I trace decays."""

        span = self.max_decay - self.min_decay
        values = self.min_decay + span * torch.sigmoid(self.decay_logits)
        return values[0], values[1]

    def input_events(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return content, gate, and binary write events for both populations."""

        _validate_sequence(x, self.input_dim)
        logits = self.input_event_projection(x)
        content_e, content_i, gate_e, gate_i = (
            _surrogate_step(part, self.surrogate_scale)
            for part in logits.chunk(4, dim=-1)
        )
        write_e = content_e * gate_e
        write_i = content_i * gate_i
        return content_e, content_i, gate_e, gate_i, write_e, write_i

    def _forward_step_tensors_unchecked(
        self,
        x_t: Tensor,
        excitatory: Tensor,
        inhibitory: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pure-tensor inference graph for one gated trace event."""

        logits = F.linear(
            x_t,
            self.input_event_projection.weight,
            self.input_event_projection.bias,
        )
        content_e, content_i, gate_e, gate_i = (
            (part >= 0.0).to(dtype=x_t.dtype) for part in logits.chunk(4, dim=-1)
        )
        write_e = content_e * gate_e
        write_i = content_i * gate_i
        decay_e, decay_i = self.decays()
        return self._forward_step_tensors_with_decays_unchecked(
            x_t,
            excitatory,
            inhibitory,
            decay_e,
            decay_i,
            write_e=write_e,
            write_i=write_i,
        )

    def _forward_step_tensors_with_decays_unchecked(
        self,
        x_t: Tensor,
        excitatory: Tensor,
        inhibitory: Tensor,
        decay_e: Tensor,
        decay_i: Tensor,
        *,
        write_e: Optional[Tensor] = None,
        write_i: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pure tensor step with deployment-time decay values already cached."""

        if write_e is None or write_i is None:
            logits = F.linear(
                x_t,
                self.input_event_projection.weight,
                self.input_event_projection.bias,
            )
            content_e, content_i, gate_e, gate_i = (
                (part >= 0.0).to(dtype=x_t.dtype)
                for part in logits.chunk(4, dim=-1)
            )
            write_e = content_e * gate_e
            write_i = content_i * gate_i
        next_e = decay_e * excitatory + (1.0 - decay_e) * write_e
        next_i = decay_i * inhibitory + (1.0 - decay_i) * write_i
        spike_e = (next_e >= self.spike_threshold).to(dtype=x_t.dtype)
        spike_i = (next_i >= self.spike_threshold).to(dtype=x_t.dtype)
        raw = torch.cat((spike_e, -spike_i, next_e, -next_i), dim=-1)
        output = self.output_projection(self.output_norm(raw))
        return output, next_e, next_i, spike_e, spike_i, write_e, write_i

    def forward_step_tensors(
        self,
        x_t: Tensor,
        excitatory: Tensor,
        inhibitory: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Exact tensor-only AT0 inference step without sequence overhead."""

        if x_t.ndim != 2 or x_t.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected x_t shaped [batch, {self.input_dim}], got {tuple(x_t.shape)}"
            )
        expected = (x_t.shape[0], self.state_dim)
        if tuple(excitatory.shape) != expected or tuple(inhibitory.shape) != expected:
            raise ValueError(
                "invalid AT0 tensor state shapes: "
                f"E={tuple(excitatory.shape)}, I={tuple(inhibitory.shape)}, "
                f"expected={expected}"
            )
        return self._forward_step_tensors_unchecked(x_t, excitatory, inhibitory)

    def forward_step_tensors_cached_decay(
        self,
        x_t: Tensor,
        excitatory: Tensor,
        inhibitory: Tensor,
        decay_e: Tensor,
        decay_i: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Exact inference step using decays cached after a parameter update.

        A deployment loop should call :meth:`decays` once whenever model
        parameters change, then reuse the two returned ``[state_dim]`` tensors
        for every event.  This removes two sigmoid/bounds evaluations from the
        hot token path without changing the AT0 dynamics.
        """

        if x_t.ndim != 2 or x_t.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected x_t shaped [batch, {self.input_dim}], got {tuple(x_t.shape)}"
            )
        expected = (x_t.shape[0], self.state_dim)
        if tuple(excitatory.shape) != expected or tuple(inhibitory.shape) != expected:
            raise ValueError(
                "invalid AT1 tensor state shapes: "
                f"E={tuple(excitatory.shape)}, I={tuple(inhibitory.shape)}, "
                f"expected={expected}"
            )
        decay_shape = (self.state_dim,)
        if tuple(decay_e.shape) != decay_shape or tuple(decay_i.shape) != decay_shape:
            raise ValueError(
                "invalid AT1 cached decay shapes: "
                f"E={tuple(decay_e.shape)}, I={tuple(decay_i.shape)}, "
                f"expected={decay_shape}"
            )
        return self._forward_step_tensors_with_decays_unchecked(
            x_t,
            excitatory,
            inhibitory,
            decay_e,
            decay_i,
        )

    @staticmethod
    def _affine_prefix_scan(
        coefficient: Tensor, bias: Tensor, initial: Tensor
    ) -> Tensor:
        prefix_a = coefficient
        prefix_b = bias
        offset = 1
        time_steps = coefficient.shape[1]
        while offset < time_steps:
            composed_a = prefix_a[:, offset:] * prefix_a[:, :-offset]
            composed_b = (
                prefix_b[:, offset:]
                + prefix_a[:, offset:] * prefix_b[:, :-offset]
            )
            prefix_a = torch.cat((prefix_a[:, :offset], composed_a), dim=1)
            prefix_b = torch.cat((prefix_b[:, :offset], composed_b), dim=1)
            offset *= 2
        return prefix_a * initial.unsqueeze(1) + prefix_b

    @staticmethod
    def _serial_trace(write: Tensor, decay: Tensor, initial: Tensor) -> Tensor:
        state = initial
        traces = []
        retention = decay.unsqueeze(0)
        injection = 1.0 - retention
        for index in range(write.shape[1]):
            state = retention * state + injection * write[:, index]
            traces.append(state)
        return torch.stack(traces, dim=1)

    def _trace(self, write: Tensor, decay: Tensor, initial: Tensor) -> Tensor:
        if self.execution_mode == "serial":
            return self._serial_trace(write, decay, initial)
        coefficient = decay.view(1, 1, -1).expand_as(write)
        bias = (1.0 - coefficient) * write
        return self._affine_prefix_scan(coefficient, bias, initial)

    def forward_multi_query_eligibility(
        self,
        x: Tensor,
        query_indices: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E3ScanState]:
        """Return exact sparse-query AT0 outputs with compact eligibility.

        The forward dynamics are identical to :meth:`forward`; only the K
        requested positions are materialised at the readout.  With frozen
        inputs, backward saves prefix eligibilities proportional to K instead
        of a time-length autograd graph.  Requiring an input gradient remains
        exact and additionally stores one local event factor per time step.
        """

        batch_size, time_steps = _validate_sequence(x, self.input_dim)
        if not isinstance(query_indices, Tensor):
            raise TypeError("query_indices must be a torch.Tensor")
        if query_indices.ndim != 1:
            raise ValueError("query_indices must be one-dimensional")
        if query_indices.dtype != torch.long:
            raise ValueError("query_indices must use torch.long dtype")
        if query_indices.numel() == 0:
            raise ValueError("query_indices must be non-empty")
        indices = query_indices.to(device=x.device)
        if int(indices[0].item()) < 0 or int(indices[-1].item()) >= time_steps:
            raise ValueError(
                f"query_indices must lie in [0, {time_steps}), got "
                f"[{int(indices[0].item())}, {int(indices[-1].item())}]"
            )
        if indices.numel() > 1 and not bool(torch.all(indices[1:] > indices[:-1])):
            raise ValueError("query_indices must be strictly increasing and unique")
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        layer = state.layers[0]
        raw, final_e, final_i = _GatedTraceMultiQueryEligibility.apply(
            x,
            indices,
            self.input_event_projection.weight,
            self.input_event_projection.bias,
            self.decay_logits,
            layer.excitatory,
            layer.inhibitory,
            self.min_decay,
            self.max_decay,
            self.spike_threshold,
            self.surrogate_scale,
            self.eligibility_forward_mode == "scan_aligned",
            self.eligibility_backward_mode == "reverse_adjoint",
        )
        sequence = self.output_projection(self.output_norm(raw))
        next_state = E3ScanState(
            layers=(E3LayerState(excitatory=final_e, inhibitory=final_i),)
        )
        if detach_state:
            next_state = detach_core_state(next_state)
        return CoreOutput(sequence=sequence, state=next_state)

    def forward_dynamics(
        self,
        x: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> Tuple[CoreOutput[E3ScanState], E3GatedTrace]:
        batch_size, _ = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        layer = state.layers[0]
        content_e, content_i, gate_e, gate_i, write_e, write_i = self.input_events(x)
        decay_e, decay_i = self.decays()
        trace_e = self._trace(write_e, decay_e, layer.excitatory)
        trace_i = self._trace(write_i, decay_i, layer.inhibitory)
        spike_e = _surrogate_step(
            trace_e - self.spike_threshold, self.surrogate_scale
        )
        spike_i = _surrogate_step(
            trace_i - self.spike_threshold, self.surrogate_scale
        )
        raw = torch.cat((spike_e, -spike_i, trace_e, -trace_i), dim=-1)
        sequence = self.output_projection(self.output_norm(raw))
        next_state = E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=trace_e[:, -1],
                    inhibitory=trace_i[:, -1],
                ),
            )
        )
        if detach_state:
            next_state = detach_core_state(next_state)
        diagnostics = E3GatedTrace(
            excitatory_content=content_e,
            inhibitory_content=content_i,
            excitatory_gate=gate_e,
            inhibitory_gate=gate_i,
            excitatory_writes=write_e,
            inhibitory_writes=write_i,
            excitatory_spikes=spike_e,
            inhibitory_spikes=spike_i,
            excitatory_traces=trace_e,
            inhibitory_traces=trace_i,
        )
        return CoreOutput(sequence=sequence, state=next_state), diagnostics

    def forward(
        self,
        x: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E3ScanState]:
        result, _ = self.forward_dynamics(x, state, detach_state=detach_state)
        return result


class E3FixedPointScanCore(TemporalCore[E3ScanState]):
    """Dynamic-decay hard-reset SNN solved by parallel fixed-point scans.

    The serial dynamics are exact.  In ``fixed_point`` mode, hard-reset events
    from the previous iteration turn the membrane recurrence into a diagonal
    affine system, which is composed with a Hillis--Steele prefix scan.  A fixed
    number of global correction rounds trades exact causal propagation for
    logarithmic temporal graph depth; experiments must report convergence to
    the serial reference rather than assuming it.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        state_dim: Optional[int] = None,
        min_decay: float = 0.50,
        max_decay: float = 0.99,
        max_charge: float = 1.50,
        threshold: float = 1.0,
        surrogate_scale: float = 5.0,
        fixed_point_iterations: int = 4,
        execution_mode: E3FixedPointMode = "fixed_point",
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        state_dim = hidden_dim if state_dim is None else state_dim
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 <= min_decay < max_decay < 1.0:
            raise ValueError("decay bounds must satisfy 0 <= min < max < 1")
        if max_charge <= 0.0 or threshold <= 0.0:
            raise ValueError("max_charge and threshold must be positive")
        if surrogate_scale <= 0.0:
            raise ValueError("surrogate_scale must be positive")
        if fixed_point_iterations <= 0:
            raise ValueError("fixed_point_iterations must be positive")
        if execution_mode not in ("serial", "fixed_point"):
            raise ValueError("execution_mode must be 'serial' or 'fixed_point'")

        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.min_decay = float(min_decay)
        self.max_decay = float(max_decay)
        self.max_charge = float(max_charge)
        self.threshold = float(threshold)
        self.surrogate_scale = float(surrogate_scale)
        self.fixed_point_iterations = int(fixed_point_iterations)
        self.execution_mode: E3FixedPointMode = execution_mode

        self.decay_e = nn.Linear(input_dim, state_dim)
        self.decay_i = nn.Linear(input_dim, state_dim)
        self.charge_e = nn.Linear(input_dim, state_dim)
        self.charge_i = nn.Linear(input_dim, state_dim)
        self.output_norm = nn.LayerNorm(4 * state_dim)
        self.output_projection = nn.Linear(4 * state_dim, hidden_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> E3ScanState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        default_device, default_dtype = _module_device_dtype(self)
        device = default_device if device is None else device
        dtype = default_dtype if dtype is None else dtype
        shape = (batch_size, self.state_dim)
        return E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=torch.zeros(shape, device=device, dtype=dtype),
                    inhibitory=torch.zeros(shape, device=device, dtype=dtype),
                ),
            )
        )

    def _validate_state(self, state: E3ScanState, batch_size: int) -> None:
        if len(state.layers) != 1:
            raise ValueError(f"expected one S1 state layer, got {len(state.layers)}")
        expected = (batch_size, self.state_dim)
        layer = state.layers[0]
        if tuple(layer.excitatory.shape) != expected or tuple(layer.inhibitory.shape) != expected:
            raise ValueError(
                "invalid S1 state shapes: "
                f"E={tuple(layer.excitatory.shape)}, I={tuple(layer.inhibitory.shape)}, "
                f"expected={expected}"
            )

    def _decay(self, logits: Tensor) -> Tensor:
        span = self.max_decay - self.min_decay
        return self.min_decay + span * torch.sigmoid(logits)

    def _charge(self, logits: Tensor) -> Tensor:
        return self.max_charge * torch.sigmoid(logits)

    @staticmethod
    def _affine_prefix_scan(
        coefficient: Tensor, bias: Tensor, initial: Tensor
    ) -> Tensor:
        prefix_a = coefficient
        prefix_b = bias
        offset = 1
        time_steps = coefficient.shape[1]
        while offset < time_steps:
            composed_a = prefix_a[:, offset:] * prefix_a[:, :-offset]
            composed_b = prefix_b[:, offset:] + prefix_a[:, offset:] * prefix_b[:, :-offset]
            prefix_a = torch.cat((prefix_a[:, :offset], composed_a), dim=1)
            prefix_b = torch.cat((prefix_b[:, :offset], composed_b), dim=1)
            offset *= 2
        return prefix_a * initial.unsqueeze(1) + prefix_b

    def _serial_population(
        self, decay: Tensor, charge: Tensor, initial: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        membrane = initial
        spikes = []
        residuals = []
        for index in range(decay.shape[1]):
            pre_reset = decay[:, index] * membrane + charge[:, index]
            spike = _surrogate_step(
                pre_reset - self.threshold, self.surrogate_scale
            )
            membrane = pre_reset * (1.0 - spike.detach())
            spikes.append(spike)
            residuals.append(membrane)
        residual_sequence = torch.stack(residuals, dim=1)
        return torch.stack(spikes, dim=1), residual_sequence, residual_sequence[:, -1]

    def _fixed_point_population(
        self, decay: Tensor, charge: Tensor, initial: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        spike_estimate = torch.zeros_like(charge)
        pre_reset = charge
        for _ in range(self.fixed_point_iterations):
            previous_spike = torch.cat(
                (torch.zeros_like(spike_estimate[:, :1]), spike_estimate[:, :-1]),
                dim=1,
            )
            coefficient = decay * (1.0 - previous_spike.detach())
            pre_reset = self._affine_prefix_scan(coefficient, charge, initial)
            spike_estimate = _surrogate_step(
                pre_reset - self.threshold, self.surrogate_scale
            )
        residuals = pre_reset * (1.0 - spike_estimate.detach())
        return spike_estimate, residuals, residuals[:, -1]

    def _population(
        self, decay: Tensor, charge: Tensor, initial: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.execution_mode == "serial":
            return self._serial_population(decay, charge, initial)
        return self._fixed_point_population(decay, charge, initial)

    def forward_dynamics(
        self,
        x: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> Tuple[CoreOutput[E3ScanState], E3LayerTrace]:
        batch_size, _ = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        layer_state = state.layers[0]
        decay_e = self._decay(self.decay_e(x))
        decay_i = self._decay(self.decay_i(x))
        charge_e = self._charge(self.charge_e(x))
        charge_i = self._charge(self.charge_i(x))
        spikes_e, residuals_e, final_e = self._population(
            decay_e, charge_e, layer_state.excitatory
        )
        spikes_i, residuals_i, final_i = self._population(
            decay_i, charge_i, layer_state.inhibitory
        )
        signed_sequence = torch.cat(
            (spikes_e, -spikes_i, residuals_e, -residuals_i), dim=-1
        )
        sequence = self.output_projection(self.output_norm(signed_sequence))
        next_state = E3ScanState(
            layers=(E3LayerState(excitatory=final_e, inhibitory=final_i),)
        )
        if detach_state:
            next_state = detach_core_state(next_state)
        trace = E3LayerTrace(
            excitatory_spikes=spikes_e,
            inhibitory_spikes=spikes_i,
            excitatory_residuals=residuals_e,
            inhibitory_residuals=residuals_i,
        )
        return CoreOutput(sequence=sequence, state=next_state), trace

    def forward(
        self,
        x: Tensor,
        state: Optional[E3ScanState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E3ScanState]:
        result, _ = self.forward_dynamics(x, state, detach_state=detach_state)
        return result


class E3OscillatoryScanCore(TemporalCore[E3OscillatorState]):
    """Stable selective complex recurrence with exact serial/scan equivalence.

    This PRF-style branch emits discrete threshold events but deliberately has
    no hard reset.  It is an oscillatory spiking substrate experiment, not a
    replacement for the strict reset semantics tested by S0/S1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        state_dim: Optional[int] = None,
        min_radius: float = 0.50,
        max_radius: float = 0.995,
        max_phase_modulation: float = math.pi / 4.0,
        drive_scale: float = 0.50,
        spike_threshold: float = 0.50,
        surrogate_scale: float = 5.0,
        execution_mode: E3OscillatorMode = "scan",
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        state_dim = hidden_dim if state_dim is None else state_dim
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not 0.0 <= min_radius < max_radius < 1.0:
            raise ValueError("radius bounds must satisfy 0 <= min < max < 1")
        if max_phase_modulation < 0.0 or drive_scale <= 0.0:
            raise ValueError("phase modulation cannot be negative and drive_scale must be positive")
        if surrogate_scale <= 0.0:
            raise ValueError("surrogate_scale must be positive")
        if execution_mode not in ("serial", "scan"):
            raise ValueError("execution_mode must be 'serial' or 'scan'")

        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.max_phase_modulation = float(max_phase_modulation)
        self.drive_scale = float(drive_scale)
        self.spike_threshold = float(spike_threshold)
        self.surrogate_scale = float(surrogate_scale)
        self.execution_mode: E3OscillatorMode = execution_mode

        self.radius_projection = nn.Linear(input_dim, state_dim)
        self.phase_projection = nn.Linear(input_dim, state_dim)
        self.input_real = nn.Linear(input_dim, state_dim)
        self.input_imag = nn.Linear(input_dim, state_dim)
        base_phase = torch.linspace(0.05, 0.95 * math.pi, state_dim)
        self.base_phase = nn.Parameter(base_phase)
        self.output_norm = nn.LayerNorm(4 * state_dim)
        self.output_projection = nn.Linear(4 * state_dim, hidden_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> E3OscillatorState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        default_device, default_dtype = _module_device_dtype(self)
        device = default_device if device is None else device
        dtype = default_dtype if dtype is None else dtype
        shape = (batch_size, self.state_dim)
        real = torch.zeros(shape, device=device, dtype=dtype)
        return E3OscillatorState(value=torch.complex(real, torch.zeros_like(real)))

    def _validate_state(self, state: E3OscillatorState, batch_size: int) -> None:
        expected = (batch_size, self.state_dim)
        if tuple(state.value.shape) != expected or not state.value.is_complex():
            raise ValueError(
                f"invalid oscillator state {tuple(state.value.shape)} / "
                f"complex={state.value.is_complex()}, expected complex {expected}"
            )

    def _coefficient_and_drive(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        radius = self.min_radius + (self.max_radius - self.min_radius) * torch.sigmoid(
            self.radius_projection(x)
        )
        phase = self.base_phase + self.max_phase_modulation * torch.tanh(
            self.phase_projection(x)
        )
        coefficient = torch.polar(radius, phase)
        drive = self.drive_scale * torch.complex(
            torch.tanh(self.input_real(x)), torch.tanh(self.input_imag(x))
        )
        return coefficient, drive

    @staticmethod
    def _serial_recurrence(
        coefficient: Tensor, drive: Tensor, initial: Tensor
    ) -> Tensor:
        current = initial
        values = []
        for index in range(coefficient.shape[1]):
            current = coefficient[:, index] * current + drive[:, index]
            values.append(current)
        return torch.stack(values, dim=1)

    def forward_dynamics(
        self,
        x: Tensor,
        state: Optional[E3OscillatorState] = None,
        *,
        detach_state: bool = False,
    ) -> Tuple[CoreOutput[E3OscillatorState], E3OscillatorTrace]:
        batch_size, _ = _validate_sequence(x, self.input_dim)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        coefficient, drive = self._coefficient_and_drive(x)
        if self.execution_mode == "serial":
            values = self._serial_recurrence(coefficient, drive, state.value)
        else:
            values = E3FixedPointScanCore._affine_prefix_scan(
                coefficient, drive, state.value
            )
        excitatory_spikes = _surrogate_step(
            values.real - self.spike_threshold, self.surrogate_scale
        )
        inhibitory_spikes = _surrogate_step(
            values.imag - self.spike_threshold, self.surrogate_scale
        )
        features = torch.cat(
            (
                excitatory_spikes,
                -inhibitory_spikes,
                values.real,
                values.imag,
            ),
            dim=-1,
        )
        sequence = self.output_projection(self.output_norm(features))
        next_state = E3OscillatorState(value=values[:, -1])
        if detach_state:
            next_state = detach_core_state(next_state)
        trace = E3OscillatorTrace(
            excitatory_spikes=excitatory_spikes,
            inhibitory_spikes=inhibitory_spikes,
            values=values,
        )
        return CoreOutput(sequence=sequence, state=next_state), trace

    def forward(
        self,
        x: Tensor,
        state: Optional[E3OscillatorState] = None,
        *,
        detach_state: bool = False,
    ) -> CoreOutput[E3OscillatorState]:
        result, _ = self.forward_dynamics(x, state, detach_state=detach_state)
        return result


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
