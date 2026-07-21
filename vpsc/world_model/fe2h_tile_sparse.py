"""FE-2H: neuron-tile sparse gated-trace core.

This module keeps the existing E3 public API untouched and provides a separate
TemporalCore-compatible implementation for FE-2H. The dense-mask path is the
training/oracle reference: all tiles advance every block, but inactive tiles
receive zero writes. The true sparse path is fail-closed inference only: it
projects only active tile slices and advances inactive tiles analytically with
zero-write decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cores import (
    CoreOutput,
    E3GatedTraceScanCore,
    E3LayerState,
    E3ScanState,
    TemporalCore,
    _surrogate_step,
    _validate_sequence,
    count_parameters,
    detach_core_state,
    state_nbytes,
)


TileState = Tuple[E3ScanState, ...]
RouteOverride = Union["FE2HRoute", Tensor]


class FE2HUnsupportedError(RuntimeError):
    """Raised when true tile-sparse execution is requested but unsupported."""


class FE2HFiniteGuardError(RuntimeError):
    """Raised when a finite guard finds the first non-finite tensor."""

    def __init__(
        self,
        *,
        scope: str,
        name: str,
        step: Optional[int],
        index: Optional[Tuple[int, ...]],
        value: Optional[float],
    ) -> None:
        self.scope = scope
        self.name = name
        self.step = step
        self.index = index
        self.value = value
        location = "" if index is None else f" index={index}"
        scalar = "" if value is None else f" value={value}"
        super().__init__(
            f"non-finite detected in {scope}:{name} at step={step}{location}{scalar}"
        )


@dataclass
class FE2HRoute:
    """Blockwise tile-routing result."""

    scores: Tensor
    soft_probs: Tensor
    hard_mask: Tensor
    selected_indices: Tensor
    active_tiles: int
    tile_count: int
    block_size: int
    soft_k_mask: Optional[Tensor] = None
    route_mode: str = "legacy_hard_ST"


@dataclass
class FE2HHomeostasis:
    """Windowed differentiable routing statistics."""

    loss: Tensor
    activation_rate: Tensor
    entropy: Tensor
    gini: Tensor
    p99_tile_load: Tensor
    block_fill: Tensor
    hotspot_share: Tensor
    dead_tile_ratio: Tensor
    target_activation_rate: float


@dataclass
class FE2HMemoryUpperBound:
    """Conservative upper bound for one full materialised forward."""

    parameter_bytes: int
    state_bytes: int
    route_bytes: int
    raw_trace_bytes: int
    output_bytes: int
    active_projection_bytes: int
    forward_only_bytes: int
    forward_only_gib: float
    total_bytes: int
    total_gib: float
    notes: Tuple[str, ...]


@dataclass
class FE2HDiagnostics:
    """Forward-pass diagnostics required by FE-2H gates."""

    route: FE2HRoute
    homeostasis: FE2HHomeostasis
    stable_local_log_energy: Tensor
    route_supervision_loss: Optional[Tensor]
    memory_upper_bound: FE2HMemoryUpperBound
    sparse_supported: bool
    unsupported_reason: Optional[str]
    energy_epsilon: float
    remaining_dense_cost_note: str


class _TileRouter(nn.Module):
    """Block-pooled router over tile scores."""

    def __init__(self, in_features: int, tile_count: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, tile_count),
        )
        # Random router biases dominate the small block-statistics signal at
        # initialization and make deterministic top-k select one fixed route.
        # Zero biases keep the initial ordering input-dependent while leaving
        # both weight matrices trainable.
        for module in self.net:
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, stats: Tensor) -> Tensor:
        return self.net(stats)


def _first_non_finite(tensor: Tensor) -> Tuple[Optional[Tuple[int, ...]], Optional[float]]:
    mask = ~torch.isfinite(tensor)
    if not bool(mask.any().item()):
        return None, None
    flat_index_tensor = mask.reshape(-1).nonzero(as_tuple=False)[0, 0]
    flat_index = int(flat_index_tensor.detach().cpu().item())
    # Report the first non-finite item in flattened scan order so callers get a
    # stable first-hit location across tensor ranks and torch builds.
    index: Tuple[int, ...] = (flat_index,)
    value = float(tensor.reshape(-1)[flat_index].detach().cpu().item())
    return index, value


def _iter_named_tensors(value: Any, prefix: str) -> Iterable[Tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(item, next_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            yield from _iter_named_tensors(item, next_prefix)


def _raise_if_non_finite(
    tensor: Tensor,
    *,
    scope: str,
    name: str,
    step: Optional[int],
) -> None:
    index, value = _first_non_finite(tensor)
    if index is None:
        return
    raise FE2HFiniteGuardError(
        scope=scope,
        name=name,
        step=step,
        index=index,
        value=value,
    )


def _check_loss_terms_finite(
    loss_terms: Optional[Mapping[str, Tensor]],
    *,
    step: Optional[int],
) -> None:
    if loss_terms is None:
        return
    for name, tensor in loss_terms.items():
        _raise_if_non_finite(tensor, scope="loss", name=name, step=step)


def _check_parameters_and_gradients_finite(
    module: nn.Module,
    *,
    step: Optional[int],
) -> None:
    for name, parameter in module.named_parameters():
        _raise_if_non_finite(parameter, scope="parameter", name=name, step=step)
        if parameter.grad is None:
            continue
        _raise_if_non_finite(parameter.grad, scope="gradient", name=name, step=step)


def _check_optimizer_state_finite(
    optimizer: Optional[torch.optim.Optimizer],
    *,
    step: Optional[int],
) -> None:
    if optimizer is None:
        return
    for group_index, group in enumerate(optimizer.param_groups):
        for param_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            prefix = f"group{group_index}.param{param_index}"
            for name, tensor in _iter_named_tensors(state, prefix):
                _raise_if_non_finite(
                    tensor,
                    scope="optimizer_state",
                    name=name,
                    step=step,
                )


def run_fe2h_finite_guard(
    module: nn.Module,
    *,
    loss_terms: Optional[Mapping[str, Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
) -> None:
    """Fail closed on the first non-finite loss/grad/param/optimizer tensor."""

    _check_loss_terms_finite(loss_terms, step=step)
    _check_parameters_and_gradients_finite(module, step=step)
    _check_optimizer_state_finite(optimizer, step=step)


class FE2HNeuronTileCore(TemporalCore):
    """Neuron-tile FE-2H core with dense-mask training and sparse eval."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        state_dim: Optional[int] = None,
        tile_size: int = 32,
        active_tiles: int = 2,
        block_size: int = 32,
        min_decay: float = 0.50,
        max_decay: float = 0.995,
        min_initial_decay: float = 0.55,
        max_initial_decay: float = 0.99,
        spike_threshold: float = 0.50,
        surrogate_scale: float = 5.0,
        router_hidden_dim: int = 64,
        route_temperature: float = 1.0,
        energy_epsilon: float = 1e-8,
        hotspot_cap: float = 0.70,
        dead_tile_floor: float = 1e-3,
        input_event_projection: Optional[nn.Module] = None,
        router_projection: Optional[nn.Module] = None,
        output_norm: Optional[nn.Module] = None,
        output_projection: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        state_dim = hidden_dim if state_dim is None else int(state_dim)
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if state_dim % tile_size != 0:
            raise ValueError("state_dim must be divisible by tile_size")
        tile_count = state_dim // tile_size
        if tile_count < 2:
            raise ValueError("tile_count must be at least two")
        if active_tiles < 2 or active_tiles >= tile_count:
            raise ValueError("active_tiles must be >= 2 and < tile_count")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0.0 <= min_decay < max_decay < 1.0:
            raise ValueError("decay bounds must satisfy 0 <= min < max < 1")
        if not min_decay < min_initial_decay <= max_initial_decay < max_decay:
            raise ValueError("initial decay range must lie inside decay bounds")
        if not 0.0 < spike_threshold < 1.0:
            raise ValueError("spike_threshold must lie in (0, 1)")
        if surrogate_scale <= 0.0:
            raise ValueError("surrogate_scale must be positive")
        if route_temperature <= 0.0:
            raise ValueError("route_temperature must be positive")
        if energy_epsilon <= 0.0:
            raise ValueError("energy_epsilon must be positive")

        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.tile_size = int(tile_size)
        self.tile_count = int(tile_count)
        self.active_tiles = int(active_tiles)
        self.block_size = int(block_size)
        self.min_decay = float(min_decay)
        self.max_decay = float(max_decay)
        self.spike_threshold = float(spike_threshold)
        self.surrogate_scale = float(surrogate_scale)
        self.route_temperature = float(route_temperature)
        self.energy_epsilon = float(energy_epsilon)
        self.hotspot_cap = float(hotspot_cap)
        self.dead_tile_floor = float(dead_tile_floor)

        if input_event_projection is None:
            input_event_projection = nn.Linear(input_dim, 4 * self.state_dim)
        self.input_event_projection = input_event_projection
        self.decay_logits = nn.Parameter(torch.empty(2, self.state_dim))
        if router_projection is None:
            router_projection = _TileRouter(4, self.tile_count, hidden_dim=router_hidden_dim)
        self.router_projection = router_projection
        self.output_norm = (
            nn.LayerNorm(4 * self.state_dim) if output_norm is None else output_norm
        )
        self.output_projection = (
            nn.Linear(4 * self.state_dim, hidden_dim)
            if output_projection is None
            else output_projection
        )

        initial_decay = torch.linspace(
            min_initial_decay,
            max_initial_decay,
            steps=self.state_dim,
        )
        normalised = (initial_decay - self.min_decay) / (self.max_decay - self.min_decay)
        logits = torch.logit(normalised)
        with torch.no_grad():
            self.decay_logits[0].copy_(logits)
            self.decay_logits[1].copy_(logits.flip(0))

        self._last_diagnostics: Optional[FE2HDiagnostics] = None

    @property
    def last_diagnostics(self) -> Optional[FE2HDiagnostics]:
        return self._last_diagnostics

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TileState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device is None or dtype is None:
            parameter = next(self.parameters())
            if device is None:
                device = parameter.device
            if dtype is None:
                dtype = parameter.dtype
        zeros = torch.zeros(batch_size, self.tile_size, device=device, dtype=dtype)
        return tuple(
            E3ScanState(
                layers=(
                    E3LayerState(
                        excitatory=zeros.clone(),
                        inhibitory=zeros.clone(),
                    ),
                )
            )
            for _ in range(self.tile_count)
        )

    def _validate_state(self, state: TileState, batch_size: int) -> None:
        if len(state) != self.tile_count:
            raise ValueError(
                f"expected {self.tile_count} tile states, got {len(state)}"
            )
        expected = (batch_size, self.tile_size)
        for tile_index, tile_state in enumerate(state):
            if len(tile_state.layers) != 1:
                raise ValueError(f"tile {tile_index} must contain exactly one layer")
            layer = tile_state.layers[0]
            if tuple(layer.excitatory.shape) != expected or tuple(layer.inhibitory.shape) != expected:
                raise ValueError(
                    f"invalid tile {tile_index} state shapes: "
                    f"E={tuple(layer.excitatory.shape)} I={tuple(layer.inhibitory.shape)} "
                    f"expected={expected}"
                )
            for name, tensor in (
                ("excitatory", layer.excitatory),
                ("inhibitory", layer.inhibitory),
            ):
                if not bool(torch.all((tensor >= 0.0) & (tensor <= 1.0)).item()):
                    raise ValueError(f"tile {tile_index} {name} trace must lie in [0, 1]")

    def decays(self) -> Tuple[Tensor, Tensor]:
        span = self.max_decay - self.min_decay
        values = self.min_decay + span * torch.sigmoid(self.decay_logits)
        return values[0], values[1]

    def _tile_state(self, state: TileState, tile_index: int) -> Tuple[Tensor, Tensor]:
        layer = state[tile_index].layers[0]
        return layer.excitatory, layer.inhibitory

    def _make_tile_state(self, excitatory: Tensor, inhibitory: Tensor) -> E3ScanState:
        return E3ScanState(layers=(E3LayerState(excitatory=excitatory, inhibitory=inhibitory),))

    def _tile_param_indices(
        self, tile_index: int, *, device: Optional[torch.device] = None
    ) -> Tensor:
        device = self.decay_logits.device if device is None else device
        base = tile_index * self.tile_size
        tile_range = torch.arange(base, base + self.tile_size, device=device)
        return torch.cat(
            (
                tile_range,
                tile_range + self.state_dim,
                tile_range + 2 * self.state_dim,
                tile_range + 3 * self.state_dim,
            ),
            dim=0,
        )

    def _tile_projection_weights(self, tile_index: int) -> Tuple[Tensor, Optional[Tensor]]:
        if not isinstance(self.input_event_projection, nn.Linear):
            raise FE2HUnsupportedError(
                "true tile-sparse inference requires a sliceable nn.Linear input projection"
            )
        indices = self._tile_param_indices(tile_index)
        weight = self.input_event_projection.weight.index_select(0, indices)
        bias: Optional[Tensor] = None
        if getattr(self.input_event_projection, "bias", None) is not None:
            bias = self.input_event_projection.bias.index_select(0, indices)
        return weight, bias

    def _dense_projection_logits(self, block: Tensor) -> Tensor:
        logits = self.input_event_projection(block)
        expected = 4 * self.state_dim
        if logits.ndim != 3 or logits.shape[-1] != expected:
            raise ValueError(
                f"input_event_projection must return [batch, time, {expected}], "
                f"got {tuple(logits.shape)}"
            )
        return logits

    def _tile_logits_from_full(self, logits: Tensor, tile_index: int) -> Tensor:
        indices = self._tile_param_indices(tile_index, device=logits.device)
        return logits.index_select(-1, indices)

    def _iter_blocks(self, time_steps: int) -> Sequence[Tuple[int, int]]:
        return [
            (start, min(time_steps, start + self.block_size))
            for start in range(0, time_steps, self.block_size)
        ]

    def _block_stats(self, x: Tensor, blocks: Sequence[Tuple[int, int]]) -> Tensor:
        features: List[Tensor] = []
        prev_mean = None
        for start, end in blocks:
            block = x[:, start:end]
            block_mean = block.mean(dim=(1, 2))
            block_std = block.std(dim=(1, 2), unbiased=False)
            if prev_mean is None:
                change = torch.zeros_like(block_mean)
            else:
                change = (block_mean - prev_mean).abs()
            norm = block.square().mean(dim=(1, 2)).sqrt()
            prev_mean = block_mean
            features.append(torch.stack((block_mean, block_std, change, norm), dim=-1))
        return torch.stack(features, dim=1)


    def _solve_constrained_soft_k(
        self, scores: Tensor, tau: float
    ) -> Tensor:
        """Solve for soft-k mask: sigmoid((lambda - score_i) / tau) s.t. sum(m_i) = k.

        Uses bisection on lambda in FP32. Returns mask in [0, 1] with sum == k.
        """
        scores_f = scores.float()
        k = float(self.active_tiles)
        tile_n = self.tile_count

        lo = scores_f.min(dim=-1, keepdim=True).values - 10.0
        hi = scores_f.max(dim=-1, keepdim=True).values + 10.0

        for _ in range(50):
            mid = (lo + hi) / 2.0
            mask = torch.sigmoid((mid - scores_f) / tau)
            total = mask.sum(dim=-1, keepdim=True)
            lo = torch.where(total < k, mid, lo)
            hi = torch.where(total >= k, mid, hi)

        mask = torch.sigmoid((lo - scores_f) / tau)
        mask = mask.clamp(0.0, 1.0)
        # Normalize to exactly sum to k per block
        mask = mask * k / mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mask = mask.clamp(0.0, 1.0)
        return mask.to(dtype=scores.dtype)

    def _build_route(
        self,
        x: Tensor,
        blocks: Sequence[Tuple[int, int]],
        route_override: Optional[RouteOverride] = None,
        *,
        route_mode: str = "legacy_hard_ST",
        soft_k_tau: float = 1.0,
    ) -> FE2HRoute:
        batch_size = x.shape[0]
        block_stats = self._block_stats(x, blocks)
        if route_override is None:
            scores = self.router_projection(block_stats.reshape(-1, block_stats.shape[-1]))
            scores = scores.reshape(batch_size, len(blocks), self.tile_count)
            selected = scores.topk(self.active_tiles, dim=-1, largest=False).indices
            hard_mask = torch.zeros_like(scores).scatter_(-1, selected, 1.0)

            if route_mode == "soft_k_annealed":
                soft_k_mask = self._solve_constrained_soft_k(scores, tau=soft_k_tau)
                soft_probs = soft_k_mask / soft_k_mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                return FE2HRoute(
                    scores=scores,
                    soft_probs=soft_probs,
                    hard_mask=hard_mask,
                    selected_indices=selected,
                    active_tiles=self.active_tiles,
                    tile_count=self.tile_count,
                    block_size=self.block_size,
                    soft_k_mask=soft_k_mask,
                    route_mode=route_mode,
                )
            else:
                soft_probs = torch.softmax(-scores / self.route_temperature, dim=-1)
                return FE2HRoute(
                    scores=scores,
                    soft_probs=soft_probs,
                    hard_mask=hard_mask,
                    selected_indices=selected,
                    active_tiles=self.active_tiles,
                    tile_count=self.tile_count,
                    block_size=self.block_size,
                    route_mode="legacy_hard_ST",
                )
        if isinstance(route_override, FE2HRoute):
            route = route_override
            if tuple(route.hard_mask.shape) != (batch_size, len(blocks), self.tile_count):
                raise ValueError("route_override shape does not match the current batch")
            return route
        if route_override.ndim != 3:
            raise ValueError("route_override tensor must be [batch, blocks, tiles]")
        if tuple(route_override.shape) != (batch_size, len(blocks), self.tile_count):
            raise ValueError("route_override shape does not match the current batch")
        hard_mask = route_override.to(dtype=x.dtype, device=x.device)
        if not bool(torch.all((hard_mask == 0.0) | (hard_mask == 1.0)).item()):
            raise ValueError("route_override must be binary")
        if not bool(hard_mask.sum(dim=-1).eq(float(self.active_tiles)).all().item()):
            raise ValueError("route_override must activate exactly active_tiles per block")
        selected = hard_mask.topk(self.active_tiles, dim=-1, largest=True).indices
        soft_probs = hard_mask / hard_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        scores = -hard_mask
        return FE2HRoute(
            scores=scores,
            soft_probs=soft_probs,
            hard_mask=hard_mask,
            selected_indices=selected,
            active_tiles=self.active_tiles,
            tile_count=self.tile_count,
            block_size=self.block_size,
        )

    def route_blocks(
        self,
        x: Tensor,
        route_override: Optional[RouteOverride] = None,
        *,
        route_mode: str = "legacy_hard_ST",
        soft_k_tau: float = 1.0,
    ) -> FE2HRoute:
        _, time_steps = _validate_sequence(x, self.input_dim)
        blocks = self._iter_blocks(time_steps)
        return self._build_route(
            x, blocks, route_override=route_override,
            route_mode=route_mode, soft_k_tau=soft_k_tau,
        )

    def _trace_block(
        self, write: Tensor, decay: Tensor, initial: Tensor, block_length: int
    ) -> Tensor:
        return E3GatedTraceScanCore._blocked_constant_affine_prefix_scan(
            write,
            decay,
            initial,
            block_size=max(1, int(block_length)),
        )

    def _zero_write_trace(self, initial: Tensor, decay: Tensor, block_length: int) -> Tensor:
        powers = decay.unsqueeze(0).pow(
            torch.arange(
                1,
                block_length + 1,
                device=initial.device,
                dtype=initial.dtype,
            ).unsqueeze(1)
        )
        return initial.unsqueeze(1) * powers.unsqueeze(0)

    def _stable_local_log_energy(
        self,
        trace_e: Tensor,
        trace_i: Tensor,
        initial_e: Tensor,
        initial_i: Tensor,
        decay_e: Tensor,
        decay_i: Tensor,
    ) -> Tensor:
        prev_e = torch.cat((initial_e.unsqueeze(1), trace_e[:, :-1]), dim=1)
        prev_i = torch.cat((initial_i.unsqueeze(1), trace_i[:, :-1]), dim=1)
        delta_e = trace_e - decay_e.view(1, 1, -1) * prev_e
        delta_i = trace_i - decay_i.view(1, 1, -1) * prev_i
        energy = delta_e.square().sum(dim=(1, 2)) + delta_i.square().sum(dim=(1, 2))
        return torch.log(energy + self.energy_epsilon)

    def _homeostasis(self, route: FE2HRoute) -> FE2HHomeostasis:
        if route.soft_k_mask is not None:
            soft_usage = route.soft_k_mask
        else:
            soft_usage = (self.active_tiles * route.soft_probs).clamp(0.0, 1.0)
        activation_rate = soft_usage.mean(dim=(0, 1))
        normalized = activation_rate / activation_rate.sum().clamp_min(self.energy_epsilon)
        pairwise = (normalized[:, None] - normalized[None, :]).abs().mean()
        entropy = (
            -(route.soft_probs.clamp_min(self.energy_epsilon) * route.soft_probs.clamp_min(self.energy_epsilon).log())
            .sum(dim=-1)
            .mean()
        )
        tile_loads = soft_usage.reshape(-1, self.tile_count)
        p99_per_tile = torch.quantile(tile_loads, 0.99, dim=0)
        p99_tile_load = p99_per_tile.max()
        block_fill = soft_usage.mean(dim=-1).mean()
        hotspot_share = activation_rate.max()
        dead_tile_ratio = activation_rate.le(self.dead_tile_floor).to(dtype=activation_rate.dtype).mean()
        target_activation = self.active_tiles / self.tile_count
        target = torch.full_like(activation_rate, target_activation)
        rate_loss = F.mse_loss(activation_rate, target)
        fill_loss = (block_fill - target_activation).square()
        hotspot_loss = F.relu(hotspot_share - self.hotspot_cap).square()
        dead_loss = F.relu(self.dead_tile_floor - activation_rate).mean()
        loss = rate_loss + fill_loss + 2.0 * hotspot_loss + dead_loss
        return FE2HHomeostasis(
            loss=loss,
            activation_rate=activation_rate,
            entropy=entropy,
            gini=pairwise,
            p99_tile_load=p99_tile_load,
            block_fill=block_fill,
            hotspot_share=hotspot_share,
            dead_tile_ratio=dead_tile_ratio,
            target_activation_rate=target_activation,
        )

    def estimate_memory_upper_bound(
        self,
        *,
        batch_size: int,
        time_steps: int,
        dtype: Optional[torch.dtype] = None,
    ) -> FE2HMemoryUpperBound:
        if batch_size <= 0 or time_steps <= 0:
            raise ValueError("batch_size and time_steps must be positive")
        if dtype is None:
            dtype = next(self.parameters()).dtype
        element_size = torch.tensor((), dtype=dtype).element_size()
        block_count = len(self._iter_blocks(time_steps))
        parameter_bytes = count_parameters(self, trainable_only=False) * element_size
        state_bytes = state_nbytes(self.initial_state(batch_size, dtype=dtype))
        route_bytes = batch_size * block_count * self.tile_count * element_size * 3
        raw_trace_bytes = batch_size * time_steps * 4 * self.state_dim * element_size
        output_bytes = batch_size * time_steps * self.output_dim * element_size
        active_projection_bytes = (
            batch_size
            * time_steps
            * self.active_tiles
            * 4
            * self.tile_size
            * element_size
        )
        forward_only_bytes = (
            parameter_bytes
            + state_bytes
            + route_bytes
            + raw_trace_bytes
            + output_bytes
            + active_projection_bytes
        )
        gradient_bytes = parameter_bytes
        optimizer_state_bytes = 2 * parameter_bytes
        autograd_reserve_bytes = (
            route_bytes + raw_trace_bytes + output_bytes + active_projection_bytes
        )
        total_bytes = (
            forward_only_bytes
            + gradient_bytes
            + optimizer_state_bytes
            + autograd_reserve_bytes
        )
        notes = (
            "forward_only_gib is the materialised forward footprint for this core only.",
            "total_gib adds gradients, Adam-style optimizer state, and one extra activation reserve for launch gating.",
            "Output projection remains dense and is not claimed as sparse speedup.",
        )
        return FE2HMemoryUpperBound(
            parameter_bytes=parameter_bytes,
            state_bytes=state_bytes,
            route_bytes=route_bytes,
            raw_trace_bytes=raw_trace_bytes,
            output_bytes=output_bytes,
            active_projection_bytes=active_projection_bytes,
            forward_only_bytes=forward_only_bytes,
            forward_only_gib=forward_only_bytes / float(1024 ** 3),
            total_bytes=total_bytes,
            total_gib=total_bytes / float(1024 ** 3),
            notes=notes,
        )

    def _sparse_support_reason(
        self,
        *,
        route_override: Optional[RouteOverride],
    ) -> Optional[str]:
        if self.training:
            return "true tile-sparse inference is eval-only"
        if torch.is_grad_enabled():
            return "true tile-sparse inference requires gradients to be disabled"
        if self.state_dim % self.tile_size != 0:
            return "state_dim must be divisible by tile_size"
        if route_override is None:
            return "freeze the route first with route_override before sparse equivalence/inference"
        if not isinstance(self.input_event_projection, nn.Linear):
            return "true tile-sparse inference requires a sliceable nn.Linear input projection"
        return None

    def forward_dynamics(
        self,
        x: Tensor,
        state: Optional[TileState] = None,
        *,
        detach_state: bool = False,
        sparse_inference: bool = False,
        route_override: Optional[RouteOverride] = None,
        route_mode: str = "legacy_hard_ST",
        soft_k_tau: float = 1.0,
        counterfactual_energy: bool = False,
    ) -> Tuple[CoreOutput[TileState], FE2HDiagnostics]:
        batch_size, time_steps = _validate_sequence(x, self.input_dim)
        blocks = self._iter_blocks(time_steps)
        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            self._validate_state(state, batch_size)
        route = self._build_route(
            x, blocks, route_override=route_override,
            route_mode=route_mode, soft_k_tau=soft_k_tau,
        )

        unsupported_reason = self._sparse_support_reason(route_override=route_override) if sparse_inference else None
        if sparse_inference and unsupported_reason is not None:
            raise FE2HUnsupportedError(unsupported_reason)

        decay_e_all, decay_i_all = self.decays()
        current_state = list(state)
        output = torch.zeros(
            batch_size, time_steps, self.output_dim, device=x.device, dtype=x.dtype
        )
        energy_blocks: List[Tensor] = []
        for block_index, (start, end) in enumerate(blocks):
            block = x[:, start:end]
            block_length = end - start
            dense_logits = None if sparse_inference else self._dense_projection_logits(block)
            spike_e_tiles: List[Tensor] = []
            spike_i_tiles: List[Tensor] = []
            trace_e_tiles: List[Tensor] = []
            trace_i_tiles: List[Tensor] = []
            energy_tiles: List[Tensor] = []
            next_states: List[E3ScanState] = []
            block_mask_hard = route.hard_mask[:, block_index]
            block_mask_soft = route.soft_probs[:, block_index]
            write_mask = block_mask_hard
            if not sparse_inference and route_override is None:
                if route_mode == "soft_k_annealed" and route.soft_k_mask is not None:
                    write_mask = route.soft_k_mask[:, block_index]
                else:
                    write_mask = block_mask_hard.detach() + block_mask_soft - block_mask_soft.detach()
            for tile_index in range(self.tile_count):
                initial_e, initial_i = self._tile_state(tuple(current_state), tile_index)
                tile_start = tile_index * self.tile_size
                tile_end = tile_start + self.tile_size
                decay_e = decay_e_all[tile_start:tile_end]
                decay_i = decay_i_all[tile_start:tile_end]
                if sparse_inference:
                    active_batch = block_mask_hard[:, tile_index].to(dtype=torch.bool)
                    trace_e = self._zero_write_trace(initial_e, decay_e, block_length)
                    trace_i = self._zero_write_trace(initial_i, decay_i, block_length)
                    if bool(active_batch.any().item()):
                        weight, bias = self._tile_projection_weights(tile_index)
                        active_block = block.index_select(0, active_batch.nonzero(as_tuple=False).squeeze(-1))
                        logits = F.linear(active_block, weight, bias)
                        parts = logits.chunk(4, dim=-1)
                        content_e = _surrogate_step(parts[0], self.surrogate_scale)
                        content_i = _surrogate_step(parts[1], self.surrogate_scale)
                        gate_e = _surrogate_step(parts[2], self.surrogate_scale)
                        gate_i = _surrogate_step(parts[3], self.surrogate_scale)
                        write_e = content_e * gate_e
                        write_i = content_i * gate_i
                        active_initial_e = initial_e.index_select(
                            0, active_batch.nonzero(as_tuple=False).squeeze(-1)
                        )
                        active_initial_i = initial_i.index_select(
                            0, active_batch.nonzero(as_tuple=False).squeeze(-1)
                        )
                        trace_e_active = self._trace_block(
                            write_e, decay_e, active_initial_e, block_length
                        )
                        trace_i_active = self._trace_block(
                            write_i, decay_i, active_initial_i, block_length
                        )
                        trace_e[active_batch] = trace_e_active
                        trace_i[active_batch] = trace_i_active
                else:
                    assert dense_logits is not None
                    logits = self._tile_logits_from_full(dense_logits, tile_index)
                    parts = logits.chunk(4, dim=-1)
                    content_e = _surrogate_step(parts[0], self.surrogate_scale)
                    content_i = _surrogate_step(parts[1], self.surrogate_scale)
                    gate_e = _surrogate_step(parts[2], self.surrogate_scale)
                    gate_i = _surrogate_step(parts[3], self.surrogate_scale)
                    write_scale = write_mask[:, tile_index].view(batch_size, 1, 1)
                    write_e = content_e * gate_e * write_scale
                    write_i = content_i * gate_i * write_scale
                    trace_e = self._trace_block(write_e, decay_e, initial_e, block_length)
                    trace_i = self._trace_block(write_i, decay_i, initial_i, block_length)

                spike_e = _surrogate_step(trace_e - self.spike_threshold, self.surrogate_scale)
                spike_i = _surrogate_step(trace_i - self.spike_threshold, self.surrogate_scale)
                spike_e_tiles.append(spike_e)
                spike_i_tiles.append(spike_i)
                trace_e_tiles.append(trace_e)
                trace_i_tiles.append(trace_i)
                energy_tiles.append(
                    self._stable_local_log_energy(
                        trace_e,
                        trace_i,
                        initial_e,
                        initial_i,
                        decay_e,
                        decay_i,
                    )
                )
                next_states.append(
                    self._make_tile_state(
                        trace_e[:, -1],
                        trace_i[:, -1],
                    )
                )

            raw = torch.cat(
                (
                    *spike_e_tiles,
                    *[-value for value in spike_i_tiles],
                    *trace_e_tiles,
                    *[-value for value in trace_i_tiles],
                ),
                dim=-1,
            )
            output[:, start:end] = self.output_projection(self.output_norm(raw))
            energy_blocks.append(torch.stack(energy_tiles, dim=-1))
            current_state = next_states

        stable_local_log_energy = torch.stack(energy_blocks, dim=1)
        route_supervision_loss: Optional[Tensor]
        if route_override is None:
            if counterfactual_energy:
                route_supervision_loss = None
            else:
                route_supervision_loss = F.smooth_l1_loss(
                    route.scores, stable_local_log_energy.detach()
                )
        else:
            route_supervision_loss = None
        homeostasis = self._homeostasis(route)
        diagnostics = FE2HDiagnostics(
            route=route,
            homeostasis=homeostasis,
            stable_local_log_energy=stable_local_log_energy,
            route_supervision_loss=route_supervision_loss,
            memory_upper_bound=self.estimate_memory_upper_bound(
                batch_size=batch_size,
                time_steps=time_steps,
                dtype=x.dtype,
            ),
            sparse_supported=unsupported_reason is None,
            unsupported_reason=unsupported_reason,
            energy_epsilon=self.energy_epsilon,
            remaining_dense_cost_note=(
                "Per-token raw features are still materialised and the output projection remains dense."
            ),
        )
        next_state = tuple(current_state)
        if detach_state:
            next_state = detach_core_state(next_state)
        result = CoreOutput(sequence=output, state=next_state)
        self._last_diagnostics = diagnostics
        return result, diagnostics

    def forward(
        self,
        x: Tensor,
        state: Optional[TileState] = None,
        *,
        detach_state: bool = False,
        sparse_inference: bool = False,
        route_override: Optional[RouteOverride] = None,
        route_mode: str = "legacy_hard_ST",
        soft_k_tau: float = 1.0,
        counterfactual_energy: bool = False,
    ) -> CoreOutput[TileState]:
        result, _ = self.forward_dynamics(
            x,
            state,
            detach_state=detach_state,
            sparse_inference=sparse_inference,
            route_override=route_override,
            route_mode=route_mode,
            soft_k_tau=soft_k_tau,
            counterfactual_energy=counterfactual_energy,
        )
        return result

    def finite_guard(
        self,
        *,
        loss_terms: Optional[Mapping[str, Tensor]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None,
    ) -> None:
        run_fe2h_finite_guard(
            self,
            loss_terms=loss_terms,
            optimizer=optimizer,
            step=step,
        )


__all__ = [
    "FE2HDiagnostics",
    "FE2HFiniteGuardError",
    "FE2HHomeostasis",
    "FE2HMemoryUpperBound",
    "FE2HNeuronTileCore",
    "FE2HRoute",
    "FE2HUnsupportedError",
    "run_fe2h_finite_guard",
]
