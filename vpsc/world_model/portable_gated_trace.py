"""Portable mathematical oracle for parallel gated SNN trace scans.

The SG25F CUDA kernel rewrites the recurrence

``z[t] = a[t] * z[t - 1] + b[t]``

as an associative prefix scan over affine maps.  This module keeps the same
rewrite in ordinary PyTorch so it can serve three purposes without depending
on NVCC: a CPU correctness oracle, a ``torch.compile`` fallback, and the
reference contract for the ROCm/NVIDIA Triton implementation.

The implementation intentionally preserves the hard binary forward and the
bounded fast-sigmoid surrogate used by :class:`E3GatedTraceScanCore`.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


class _SurrogateStep(torch.autograd.Function):
    """Hard threshold forward with the project fast-sigmoid backward."""

    @staticmethod
    def forward(ctx: object, value: Tensor, scale: float) -> Tensor:
        ctx.save_for_backward(value)  # type: ignore[attr-defined]
        ctx.scale = float(scale)  # type: ignore[attr-defined]
        return (value >= 0.0).to(dtype=value.dtype)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> Tuple[Tensor, None]:
        (value,) = ctx.saved_tensors  # type: ignore[attr-defined]
        scale = ctx.scale  # type: ignore[attr-defined]
        derivative = scale / (1.0 + scale * value.abs()).square()
        return gradient * derivative, None


def _surrogate_step(value: Tensor, scale: float) -> Tensor:
    return _SurrogateStep.apply(value, float(scale))


def affine_scan_rounds(time_steps: int) -> int:
    """Return the Hillis--Steele dependency depth for ``time_steps``."""

    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    return int(math.ceil(math.log2(time_steps))) if time_steps > 1 else 0


def inclusive_affine_scan(
    coefficient: Tensor,
    bias: Tensor,
    initial: Tensor,
) -> Tensor:
    """Evaluate a batched affine recurrence with ``O(log T)`` depth.

    Args:
        coefficient: Per-step multipliers shaped ``[batch, time, state]``.
        bias: Per-step injections with the same shape as ``coefficient``.
        initial: State before the first event, shaped ``[batch, state]``.

    Returns:
        Every inclusive recurrence state, shaped ``[batch, time, state]``.

    For a later affine map ``r`` and an earlier map ``l``, composition is
    ``(a_r, b_r) o (a_l, b_l) = (a_r*a_l, a_r*b_l+b_r)``.  That operator is
    associative, which permits a parallel prefix tree for arbitrary lengths;
    no power-of-two padding or token truncation is used here.
    """

    if coefficient.ndim != 3:
        raise ValueError("coefficient must be shaped [batch,time,state]")
    if bias.shape != coefficient.shape:
        raise ValueError("bias must match coefficient")
    if initial.shape != (coefficient.shape[0], coefficient.shape[2]):
        raise ValueError("initial must be shaped [batch,state]")
    if coefficient.shape[1] <= 0:
        raise ValueError("time dimension must be positive")
    if not coefficient.is_floating_point() or not bias.is_floating_point():
        raise TypeError("coefficient and bias must be floating point")
    if coefficient.dtype != bias.dtype or coefficient.device != bias.device:
        raise TypeError("coefficient and bias must share dtype and device")
    if initial.dtype != coefficient.dtype or initial.device != coefficient.device:
        raise TypeError("initial must share coefficient dtype and device")

    prefix_a = coefficient
    prefix_b = bias
    offset = 1
    time_steps = int(coefficient.shape[1])
    while offset < time_steps:
        right_a = prefix_a[:, offset:]
        right_b = prefix_b[:, offset:]
        left_a = prefix_a[:, :-offset]
        left_b = prefix_b[:, :-offset]
        composed_a = right_a * left_a
        composed_b = right_a * left_b + right_b
        prefix_a = torch.cat((prefix_a[:, :offset], composed_a), dim=1)
        prefix_b = torch.cat((prefix_b[:, :offset], composed_b), dim=1)
        offset <<= 1
    return prefix_a * initial.unsqueeze(1) + prefix_b


def _validate_gated_trace_inputs(
    drives: Tensor,
    query_indices: Tensor,
    decays: Tensor,
    initial_e: Tensor,
    initial_i: Tensor,
    *,
    validate_query_values: bool = True,
) -> int:
    if drives.ndim != 3:
        raise ValueError("drives must be shaped [batch,time,4*state]")
    if not drives.is_floating_point():
        raise TypeError("drives must be floating point")
    if drives.shape[1] <= 0 or drives.shape[2] <= 0 or drives.shape[2] % 4:
        raise ValueError("drives require positive time and width divisible by four")
    state_dim = int(drives.shape[2] // 4)
    batch_size = int(drives.shape[0])
    if query_indices.ndim != 2 or query_indices.shape[0] != batch_size:
        raise ValueError("query_indices must be shaped [batch,query]")
    if query_indices.shape[1] <= 0 or query_indices.dtype != torch.long:
        raise TypeError("query_indices must be non-empty int64")
    if query_indices.device != drives.device:
        raise TypeError("query_indices must share the drives device")
    if decays.shape != (2, state_dim):
        raise ValueError("decays must be shaped [2,state]")
    expected_initial = (batch_size, state_dim)
    if initial_e.shape != expected_initial or initial_i.shape != expected_initial:
        raise ValueError("initial E/I must be shaped [batch,state]")
    for name, tensor in (
        ("decays", decays),
        ("initial_e", initial_e),
        ("initial_i", initial_i),
    ):
        if tensor.dtype != drives.dtype or tensor.device != drives.device:
            raise TypeError(f"{name} must share the drives dtype and device")

    if validate_query_values:
        valid = query_indices.ge(0)
        if bool(query_indices.lt(-1).any().item()):
            raise ValueError("query padding must use -1")
        if bool(query_indices[valid].ge(drives.shape[1]).any().item()):
            raise ValueError("query index exceeds the time dimension")
        valid_after_padding = valid & query_indices.eq(-1).cummax(dim=1).values
        if bool(valid_after_padding.any().item()):
            raise ValueError("valid queries must precede -1 padding")
        if query_indices.shape[1] > 1:
            adjacent_valid = valid[:, 1:] & valid[:, :-1]
            not_increasing = adjacent_valid & (
                query_indices[:, 1:] <= query_indices[:, :-1]
            )
            if bool(not_increasing.any().item()):
                raise ValueError("valid queries must be strictly increasing")
    return state_dim


def _event_writes(
    drives: Tensor,
    state_dim: int,
    surrogate_scale: float,
) -> Tensor:
    packed = drives.reshape(drives.shape[0], drives.shape[1], 4, state_dim)
    events = _surrogate_step(packed, surrogate_scale)
    write_e = events[:, :, 0] * events[:, :, 2]
    write_i = events[:, :, 1] * events[:, :, 3]
    return torch.cat((write_e, write_i), dim=-1)


def _query_raw(
    traces: Tensor,
    query_indices: Tensor,
    state_dim: int,
    spike_threshold: float,
    surrogate_scale: float,
) -> Tensor:
    gather_indices = query_indices.clamp_min(0).unsqueeze(-1).expand(
        -1, -1, traces.shape[-1]
    )
    queried = torch.gather(traces, 1, gather_indices)
    trace_e, trace_i = queried.split(state_dim, dim=-1)
    spike_e = _surrogate_step(trace_e - float(spike_threshold), surrogate_scale)
    spike_i = _surrogate_step(trace_i - float(spike_threshold), surrogate_scale)
    raw = torch.cat((spike_e, -spike_i, trace_e, -trace_i), dim=-1)
    return raw * query_indices.ge(0).unsqueeze(-1).to(dtype=raw.dtype)


def _gated_trace(
    drives: Tensor,
    query_indices: Tensor,
    decays: Tensor,
    initial_e: Tensor,
    initial_i: Tensor,
    *,
    spike_threshold: float,
    surrogate_scale: float,
    parallel: bool,
    validate_query_values: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    state_dim = _validate_gated_trace_inputs(
        drives,
        query_indices,
        decays,
        initial_e,
        initial_i,
        validate_query_values=validate_query_values,
    )
    writes = _event_writes(drives, state_dim, surrogate_scale)
    decay_pair = decays.reshape(-1)
    initial_pair = torch.cat((initial_e, initial_i), dim=-1)
    coefficient = decay_pair.view(1, 1, -1).expand_as(writes)
    if parallel:
        traces = inclusive_affine_scan(
            coefficient,
            (1.0 - coefficient) * writes,
            initial_pair,
        )
    else:
        state = initial_pair
        serial_states = []
        injection = 1.0 - decay_pair
        for time_index in range(int(drives.shape[1])):
            state = decay_pair * state + injection * writes[:, time_index]
            serial_states.append(state)
        traces = torch.stack(serial_states, dim=1)
    raw = _query_raw(
        traces,
        query_indices,
        state_dim,
        spike_threshold,
        surrogate_scale,
    )
    final_e, final_i = traces[:, -1].split(state_dim, dim=-1)
    return raw, final_e, final_i


def portable_parallel_gated_trace(
    drives: Tensor,
    query_indices: Tensor,
    decays: Tensor,
    initial_e: Tensor,
    initial_i: Tensor,
    *,
    spike_threshold: float,
    surrogate_scale: float,
    _unchecked: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run the SG25F gated trace through the portable affine prefix tree."""

    return _gated_trace(
        drives,
        query_indices,
        decays,
        initial_e,
        initial_i,
        spike_threshold=spike_threshold,
        surrogate_scale=surrogate_scale,
        parallel=True,
        validate_query_values=not _unchecked,
    )


def serial_gated_trace_reference(
    drives: Tensor,
    query_indices: Tensor,
    decays: Tensor,
    initial_e: Tensor,
    initial_i: Tensor,
    *,
    spike_threshold: float,
    surrogate_scale: float,
    _unchecked: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run the same hard-forward/surrogate-backward dynamics serially."""

    return _gated_trace(
        drives,
        query_indices,
        decays,
        initial_e,
        initial_i,
        spike_threshold=spike_threshold,
        surrogate_scale=surrogate_scale,
        parallel=False,
        validate_query_values=not _unchecked,
    )
