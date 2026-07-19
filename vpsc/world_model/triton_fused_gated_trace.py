"""Fused ROCm/NVIDIA Triton kernels for the SG27A gated SNN trace."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl

from vpsc.world_model.triton_affine_scan import _compose_affine, _num_warps


@triton.jit
def _surrogate_derivative(value, scale: tl.constexpr):
    denominator = 1.0 + scale * tl.abs(value)
    return scale / (denominator * denominator)


@triton.jit
def _gated_trace_forward_kernel(
    drives_ptr,
    decays_ptr,
    initial_e_ptr,
    initial_i_ptr,
    traces_ptr,
    final_e_ptr,
    final_i_ptr,
    TIME: tl.constexpr,
    STATE: tl.constexpr,
    BLOCK_TIME: tl.constexpr,
):
    row = tl.program_id(0)
    pair_width = 2 * STATE
    batch_index = row // pair_width
    pair_index = row - batch_index * pair_width
    polarity = pair_index // STATE
    state_index = pair_index - polarity * STATE
    lanes = tl.arange(0, BLOCK_TIME)
    valid = lanes < TIME

    drive_width = 4 * STATE
    drive_row = batch_index * TIME * drive_width + lanes * drive_width
    content_component = polarity
    gate_component = polarity + 2
    content_offsets = drive_row + content_component * STATE + state_index
    gate_offsets = drive_row + gate_component * STATE + state_index
    content_drive = tl.load(drives_ptr + content_offsets, mask=valid, other=0.0)
    gate_drive = tl.load(drives_ptr + gate_offsets, mask=valid, other=0.0)
    content = (content_drive >= 0.0).to(tl.float32)
    gate = (gate_drive >= 0.0).to(tl.float32)
    write = content * gate

    decay = tl.load(decays_ptr + pair_index)
    coefficient = tl.where(valid, decay, 1.0)
    bias = tl.where(valid, (1.0 - decay) * write, 0.0)
    prefix_a, prefix_b = tl.associative_scan(
        (coefficient, bias),
        axis=0,
        combine_fn=_compose_affine,
    )
    initial_e = tl.load(initial_e_ptr + batch_index * STATE + state_index)
    initial_i = tl.load(initial_i_ptr + batch_index * STATE + state_index)
    initial = tl.where(polarity == 0, initial_e, initial_i)
    trace = prefix_a * initial + prefix_b

    trace_offsets = row * TIME + lanes
    tl.store(traces_ptr + trace_offsets, trace, mask=valid)
    final_offsets = batch_index * STATE + state_index + lanes * 0
    final_mask = valid & (lanes == TIME - 1)
    tl.store(
        final_e_ptr + final_offsets,
        trace,
        mask=final_mask & (polarity == 0),
    )
    tl.store(
        final_i_ptr + final_offsets,
        trace,
        mask=final_mask & (polarity == 1),
    )


@triton.jit
def _gated_query_forward_kernel(
    query_indices_ptr,
    traces_ptr,
    raw_ptr,
    TIME: tl.constexpr,
    STATE: tl.constexpr,
    QUERY: tl.constexpr,
    SPIKE_THRESHOLD: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // QUERY
    state = tl.arange(0, BLOCK_STATE)
    state_valid = state < STATE
    query_index = tl.load(query_indices_ptr + row)
    query_valid = (query_index >= 0) & (query_index < TIME)
    trace_e_offsets = (
        (batch_index * 2 * STATE + state) * TIME + query_index
    )
    trace_i_offsets = trace_e_offsets + STATE * TIME
    load_mask = state_valid & query_valid
    trace_e = tl.load(traces_ptr + trace_e_offsets, mask=load_mask, other=0.0)
    trace_i = tl.load(traces_ptr + trace_i_offsets, mask=load_mask, other=0.0)
    spike_e = (trace_e >= SPIKE_THRESHOLD).to(tl.float32)
    spike_i = (trace_i >= SPIKE_THRESHOLD).to(tl.float32)
    raw_base = row * (4 * STATE)
    tl.store(raw_ptr + raw_base + state, spike_e, mask=state_valid)
    tl.store(raw_ptr + raw_base + STATE + state, -spike_i, mask=state_valid)
    tl.store(raw_ptr + raw_base + 2 * STATE + state, trace_e, mask=state_valid)
    tl.store(raw_ptr + raw_base + 3 * STATE + state, -trace_i, mask=state_valid)


@triton.jit
def _gated_query_backward_kernel(
    grad_raw_ptr,
    query_indices_ptr,
    traces_ptr,
    direct_ptr,
    TIME: tl.constexpr,
    STATE: tl.constexpr,
    QUERY: tl.constexpr,
    SPIKE_THRESHOLD: tl.constexpr,
    SURROGATE_SCALE: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // QUERY
    state = tl.arange(0, BLOCK_STATE)
    state_valid = state < STATE
    query_index = tl.load(query_indices_ptr + row)
    query_valid = (query_index >= 0) & (query_index < TIME)
    load_mask = state_valid & query_valid
    raw_base = row * (4 * STATE)
    grad_spike_e = tl.load(
        grad_raw_ptr + raw_base + state, mask=load_mask, other=0.0
    )
    grad_spike_i = -tl.load(
        grad_raw_ptr + raw_base + STATE + state,
        mask=load_mask,
        other=0.0,
    )
    grad_trace_e = tl.load(
        grad_raw_ptr + raw_base + 2 * STATE + state,
        mask=load_mask,
        other=0.0,
    )
    grad_trace_i = -tl.load(
        grad_raw_ptr + raw_base + 3 * STATE + state,
        mask=load_mask,
        other=0.0,
    )
    trace_e_offsets = (
        (batch_index * 2 * STATE + state) * TIME + query_index
    )
    trace_i_offsets = trace_e_offsets + STATE * TIME
    trace_e = tl.load(traces_ptr + trace_e_offsets, mask=load_mask, other=0.0)
    trace_i = tl.load(traces_ptr + trace_i_offsets, mask=load_mask, other=0.0)
    direct_e = grad_trace_e + grad_spike_e * _surrogate_derivative(
        trace_e - SPIKE_THRESHOLD, SURROGATE_SCALE
    )
    direct_i = grad_trace_i + grad_spike_i * _surrogate_derivative(
        trace_i - SPIKE_THRESHOLD, SURROGATE_SCALE
    )
    direct_e_offsets = (
        (batch_index * 2 * STATE + state) * TIME + query_index
    )
    direct_i_offsets = direct_e_offsets + STATE * TIME
    tl.store(direct_ptr + direct_e_offsets, direct_e, mask=load_mask)
    tl.store(direct_ptr + direct_i_offsets, direct_i, mask=load_mask)


@triton.jit
def _gated_trace_backward_kernel(
    drives_ptr,
    decays_ptr,
    initial_e_ptr,
    initial_i_ptr,
    traces_ptr,
    direct_ptr,
    grad_final_e_ptr,
    grad_final_i_ptr,
    grad_drives_ptr,
    grad_decays_ptr,
    grad_initial_e_ptr,
    grad_initial_i_ptr,
    TIME: tl.constexpr,
    STATE: tl.constexpr,
    SURROGATE_SCALE: tl.constexpr,
    BLOCK_TIME: tl.constexpr,
):
    row = tl.program_id(0)
    pair_width = 2 * STATE
    batch_index = row // pair_width
    pair_index = row - batch_index * pair_width
    polarity = pair_index // STATE
    state_index = pair_index - polarity * STATE
    lanes = tl.arange(0, BLOCK_TIME)
    time_index = TIME - 1 - lanes
    valid = lanes < TIME

    trace_offsets = row * TIME + time_index
    direct = tl.load(direct_ptr + trace_offsets, mask=valid, other=0.0)
    grad_final_e = tl.load(
        grad_final_e_ptr + batch_index * STATE + state_index
    )
    grad_final_i = tl.load(
        grad_final_i_ptr + batch_index * STATE + state_index
    )
    grad_final = tl.where(polarity == 0, grad_final_e, grad_final_i)
    direct += tl.where(valid & (time_index == TIME - 1), grad_final, 0.0)
    decay = tl.load(decays_ptr + pair_index)
    coefficient = tl.where(valid, decay, 1.0)
    _, adjoint = tl.associative_scan(
        (coefficient, direct),
        axis=0,
        combine_fn=_compose_affine,
    )

    previous_from_trace = tl.load(
        traces_ptr + trace_offsets - 1,
        mask=valid & (time_index > 0),
        other=0.0,
    )
    initial_e = tl.load(initial_e_ptr + batch_index * STATE + state_index)
    initial_i = tl.load(initial_i_ptr + batch_index * STATE + state_index)
    initial = tl.where(polarity == 0, initial_e, initial_i)
    previous = tl.where(time_index == 0, initial, previous_from_trace)

    drive_width = 4 * STATE
    drive_row = batch_index * TIME * drive_width + time_index * drive_width
    content_component = polarity
    gate_component = polarity + 2
    content_offsets = drive_row + content_component * STATE + state_index
    gate_offsets = drive_row + gate_component * STATE + state_index
    content_drive = tl.load(drives_ptr + content_offsets, mask=valid, other=0.0)
    gate_drive = tl.load(drives_ptr + gate_offsets, mask=valid, other=0.0)
    content = (content_drive >= 0.0).to(tl.float32)
    gate = (gate_drive >= 0.0).to(tl.float32)
    write = content * gate
    drive_scale = (1.0 - decay) * adjoint
    grad_content = drive_scale * gate * _surrogate_derivative(
        content_drive, SURROGATE_SCALE
    )
    grad_gate = drive_scale * content * _surrogate_derivative(
        gate_drive, SURROGATE_SCALE
    )
    tl.store(grad_drives_ptr + content_offsets, grad_content, mask=valid)
    tl.store(grad_drives_ptr + gate_offsets, grad_gate, mask=valid)

    decay_contribution = tl.where(valid, adjoint * (previous - write), 0.0)
    tl.atomic_add(
        grad_decays_ptr + pair_index,
        tl.sum(decay_contribution, axis=0),
    )
    initial_offsets = batch_index * STATE + state_index + lanes * 0
    initial_mask = valid & (time_index == 0)
    tl.store(
        grad_initial_e_ptr + initial_offsets,
        decay * adjoint,
        mask=initial_mask & (polarity == 0),
    )
    tl.store(
        grad_initial_i_ptr + initial_offsets,
        decay * adjoint,
        mask=initial_mask & (polarity == 1),
    )


def _validate(
    drives: Tensor,
    query_indices: Tensor,
    decays: Tensor,
    initial_e: Tensor,
    initial_i: Tensor,
) -> Tuple[int, int, int, int]:
    if drives.ndim != 3 or drives.shape[1] <= 0 or drives.shape[2] % 4:
        raise ValueError("drives must be [batch,time,4*state]")
    batch_size, time_steps, drive_width = tuple(int(value) for value in drives.shape)
    state_dim = drive_width // 4
    if time_steps > 256:
        raise ValueError("fused Triton trace currently supports time <= 256")
    if not drives.is_cuda or drives.dtype != torch.float32:
        raise TypeError("drives must be float32 on a CUDA/HIP device")
    if not drives.is_contiguous():
        raise ValueError("drives must be contiguous")
    if (
        query_indices.ndim != 2
        or query_indices.shape[0] != batch_size
        or query_indices.shape[1] <= 0
        or query_indices.dtype != torch.long
        or query_indices.device != drives.device
        or not query_indices.is_contiguous()
    ):
        raise TypeError("query_indices must be contiguous device int64 [batch,query]")
    if decays.shape != (2, state_dim):
        raise ValueError("decays must be [2,state]")
    if initial_e.shape != (batch_size, state_dim) or initial_i.shape != (
        batch_size,
        state_dim,
    ):
        raise ValueError("initial E/I must be [batch,state]")
    for name, value in (
        ("decays", decays),
        ("initial_e", initial_e),
        ("initial_i", initial_i),
    ):
        if (
            not value.is_cuda
            or value.dtype != torch.float32
            or value.device != drives.device
            or not value.is_contiguous()
        ):
            raise TypeError(f"{name} must be contiguous on the drives dtype/device")
    return batch_size, time_steps, state_dim, int(query_indices.shape[1])


class _TritonFusedGatedTrace(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        drives: Tensor,
        query_indices: Tensor,
        decays: Tensor,
        initial_e: Tensor,
        initial_i: Tensor,
        spike_threshold: float,
        surrogate_scale: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch, time_steps, state_dim, query_count = _validate(
            drives, query_indices, decays, initial_e, initial_i
        )
        traces = torch.empty(
            batch,
            2,
            state_dim,
            time_steps,
            device=drives.device,
            dtype=drives.dtype,
        )
        final_e = torch.empty_like(initial_e)
        final_i = torch.empty_like(initial_i)
        raw = torch.empty(
            batch,
            query_count,
            4 * state_dim,
            device=drives.device,
            dtype=drives.dtype,
        )
        block_time = triton.next_power_of_2(time_steps)
        _gated_trace_forward_kernel[(batch * 2 * state_dim,)](
            drives,
            decays,
            initial_e,
            initial_i,
            traces,
            final_e,
            final_i,
            TIME=time_steps,
            STATE=state_dim,
            BLOCK_TIME=block_time,
            num_warps=_num_warps(block_time),
        )
        block_state = triton.next_power_of_2(state_dim)
        _gated_query_forward_kernel[(batch * query_count,)](
            query_indices,
            traces,
            raw,
            TIME=time_steps,
            STATE=state_dim,
            QUERY=query_count,
            SPIKE_THRESHOLD=float(spike_threshold),
            BLOCK_STATE=block_state,
            num_warps=1,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            drives,
            query_indices,
            decays,
            initial_e,
            initial_i,
            traces,
        )
        ctx.raw_shape = tuple(raw.shape)
        ctx.spike_threshold = float(spike_threshold)
        ctx.surrogate_scale = float(surrogate_scale)
        return raw, final_e, final_i

    @staticmethod
    def backward(
        ctx: Any,
        grad_raw: Optional[Tensor],
        grad_final_e: Optional[Tensor],
        grad_final_i: Optional[Tensor],
    ) -> Tuple[Tensor, None, Tensor, Tensor, Tensor, None, None]:
        drives, query_indices, decays, initial_e, initial_i, traces = (
            ctx.saved_tensors
        )
        batch, time_steps, state_dim, query_count = _validate(
            drives, query_indices, decays, initial_e, initial_i
        )
        if grad_raw is None:
            grad_raw = torch.zeros(
                ctx.raw_shape,
                device=drives.device,
                dtype=drives.dtype,
            )
        else:
            grad_raw = grad_raw.contiguous()
        if grad_final_e is None:
            grad_final_e = torch.zeros_like(initial_e)
        else:
            grad_final_e = grad_final_e.contiguous()
        if grad_final_i is None:
            grad_final_i = torch.zeros_like(initial_i)
        else:
            grad_final_i = grad_final_i.contiguous()

        direct = torch.zeros_like(traces)
        block_state = triton.next_power_of_2(state_dim)
        _gated_query_backward_kernel[(batch * query_count,)](
            grad_raw,
            query_indices,
            traces,
            direct,
            TIME=time_steps,
            STATE=state_dim,
            QUERY=query_count,
            SPIKE_THRESHOLD=ctx.spike_threshold,
            SURROGATE_SCALE=ctx.surrogate_scale,
            BLOCK_STATE=block_state,
            num_warps=1,
        )
        grad_drives = torch.empty_like(drives)
        grad_decays = torch.zeros_like(decays)
        grad_initial_e = torch.empty_like(initial_e)
        grad_initial_i = torch.empty_like(initial_i)
        block_time = triton.next_power_of_2(time_steps)
        _gated_trace_backward_kernel[(batch * 2 * state_dim,)](
            drives,
            decays,
            initial_e,
            initial_i,
            traces,
            direct,
            grad_final_e,
            grad_final_i,
            grad_drives,
            grad_decays,
            grad_initial_e,
            grad_initial_i,
            TIME=time_steps,
            STATE=state_dim,
            SURROGATE_SCALE=ctx.surrogate_scale,
            BLOCK_TIME=block_time,
            num_warps=_num_warps(block_time),
        )
        return (
            grad_drives,
            None,
            grad_decays,
            grad_initial_e,
            grad_initial_i,
            None,
            None,
        )


def triton_fused_gated_trace(
    drives: Tensor,
    query_indices: Tensor,
    decays: Tensor,
    initial_e: Tensor,
    initial_i: Tensor,
    *,
    spike_threshold: float,
    surrogate_scale: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run the complete hard-forward/surrogate-backward gated trace."""

    return _TritonFusedGatedTrace.apply(
        drives,
        query_indices,
        decays,
        initial_e,
        initial_i,
        spike_threshold,
        surrogate_scale,
    )


def backend_audit() -> Dict[str, Any]:
    target = triton.runtime.driver.active.get_current_target()
    return {
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "triton": triton.__version__,
        "target_backend": target.backend,
        "target_arch": target.arch,
        "max_time": 256,
        "forward_kernel_count": 2,
        "backward_kernel_count_excluding_zero_fill": 2,
    }
