"""Triton implementation of the associative affine recurrence scan.

This is the first ROCm/NVIDIA-neutral execution primitive for SG27A.  It is
kept generic on purpose: the later fused gated-trace kernel can be judged
against this implementation before event thresholding, query gathering, and
the reverse adjoint are fused into a single launch.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _compose_affine(
    left_a,
    left_b,
    right_a,
    right_b,
):
    """Compose ``right o left`` for two scalar affine maps."""

    return right_a * left_a, right_a * left_b + right_b


@triton.jit
def _affine_scan_forward_kernel(
    coefficient_ptr,
    bias_ptr,
    initial_ptr,
    output_ptr,
    TIME: tl.constexpr,
    STATE: tl.constexpr,
    BLOCK_TIME: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // STATE
    state_index = row - batch_index * STATE
    lanes = tl.arange(0, BLOCK_TIME)
    valid = lanes < TIME
    offsets = batch_index * TIME * STATE + lanes * STATE + state_index
    coefficient = tl.load(coefficient_ptr + offsets, mask=valid, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=valid, other=0.0)
    prefix_a, prefix_b = tl.associative_scan(
        (coefficient, bias),
        axis=0,
        combine_fn=_compose_affine,
    )
    initial = tl.load(initial_ptr + batch_index * STATE + state_index)
    output = prefix_a * initial + prefix_b
    tl.store(output_ptr + offsets, output, mask=valid)


@triton.jit
def _affine_scan_backward_kernel(
    grad_output_ptr,
    coefficient_ptr,
    initial_ptr,
    output_ptr,
    grad_coefficient_ptr,
    grad_bias_ptr,
    grad_initial_ptr,
    TIME: tl.constexpr,
    STATE: tl.constexpr,
    BLOCK_TIME: tl.constexpr,
):
    row = tl.program_id(0)
    batch_index = row // STATE
    state_index = row - batch_index * STATE
    lanes = tl.arange(0, BLOCK_TIME)

    # Materialise the original sequence in reverse lane order.  A normal
    # inclusive scan then evaluates lambda[t] = g[t] + a[t+1]*lambda[t+1].
    time_index = TIME - 1 - lanes
    valid = lanes < TIME
    offsets = batch_index * TIME * STATE + time_index * STATE + state_index
    next_offsets = offsets + STATE
    next_valid = valid & (time_index + 1 < TIME)
    next_coefficient = tl.load(
        coefficient_ptr + next_offsets,
        mask=next_valid,
        other=1.0,
    )
    direct_gradient = tl.load(
        grad_output_ptr + offsets,
        mask=valid,
        other=0.0,
    )
    _, adjoint = tl.associative_scan(
        (next_coefficient, direct_gradient),
        axis=0,
        combine_fn=_compose_affine,
    )

    previous_offsets = offsets - STATE
    previous_from_output = tl.load(
        output_ptr + previous_offsets,
        mask=valid & (time_index > 0),
        other=0.0,
    )
    initial = tl.load(initial_ptr + batch_index * STATE + state_index)
    previous = tl.where(time_index == 0, initial, previous_from_output)
    tl.store(
        grad_coefficient_ptr + offsets,
        adjoint * previous,
        mask=valid,
    )
    tl.store(grad_bias_ptr + offsets, adjoint, mask=valid)

    first_coefficient = tl.load(
        coefficient_ptr + batch_index * TIME * STATE + state_index
    )
    grad_initial_offsets = batch_index * STATE + state_index + lanes * 0
    tl.store(
        grad_initial_ptr + grad_initial_offsets,
        first_coefficient * adjoint,
        mask=valid & (time_index == 0),
    )


def _num_warps(block_time: int) -> int:
    if block_time <= 64:
        return 1
    if block_time <= 128:
        return 2
    return 4


def _validate_inputs(
    coefficient: Tensor,
    bias: Tensor,
    initial: Tensor,
) -> Tuple[int, int, int]:
    if coefficient.ndim != 3 or coefficient.shape[1] <= 0:
        raise ValueError("coefficient must be [batch,time,state] with time > 0")
    if bias.shape != coefficient.shape:
        raise ValueError("bias must match coefficient")
    if initial.shape != (coefficient.shape[0], coefficient.shape[2]):
        raise ValueError("initial must be [batch,state]")
    for name, tensor in (
        ("coefficient", coefficient),
        ("bias", bias),
        ("initial", initial),
    ):
        if not tensor.is_cuda or tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be float32 on a CUDA/HIP device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if bias.device != coefficient.device or initial.device != coefficient.device:
        raise TypeError("all inputs must share a device")
    return tuple(int(value) for value in coefficient.shape)


class _TritonAffineScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        coefficient: Tensor,
        bias: Tensor,
        initial: Tensor,
    ) -> Tensor:
        batch, time_steps, state_dim = _validate_inputs(
            coefficient, bias, initial
        )
        output = torch.empty_like(bias)
        block_time = triton.next_power_of_2(time_steps)
        _affine_scan_forward_kernel[(batch * state_dim,)](
            coefficient,
            bias,
            initial,
            output,
            TIME=time_steps,
            STATE=state_dim,
            BLOCK_TIME=block_time,
            num_warps=_num_warps(block_time),
        )
        ctx.save_for_backward(coefficient, initial, output)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        coefficient, initial, output = ctx.saved_tensors
        batch, time_steps, state_dim = tuple(
            int(value) for value in coefficient.shape
        )
        grad_output = grad_output.contiguous()
        grad_coefficient = torch.empty_like(coefficient)
        grad_bias = torch.empty_like(grad_output)
        grad_initial = torch.empty_like(initial)
        block_time = triton.next_power_of_2(time_steps)
        _affine_scan_backward_kernel[(batch * state_dim,)](
            grad_output,
            coefficient,
            initial,
            output,
            grad_coefficient,
            grad_bias,
            grad_initial,
            TIME=time_steps,
            STATE=state_dim,
            BLOCK_TIME=block_time,
            num_warps=_num_warps(block_time),
        )
        return grad_coefficient, grad_bias, grad_initial


def triton_affine_scan(
    coefficient: Tensor,
    bias: Tensor,
    initial: Tensor,
) -> Tensor:
    """Evaluate and differentiate an affine scan through Triton."""

    return _TritonAffineScan.apply(coefficient, bias, initial)


def triton_composed_gated_trace(
    drives: Tensor,
    query_indices: Tensor,
    decays: Tensor,
    initial_e: Tensor,
    initial_i: Tensor,
    *,
    spike_threshold: float,
    surrogate_scale: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compose PyTorch event logic with the Triton affine scan.

    This deliberately unfused route is a semantic bridge and performance
    control for the later all-Triton gated kernel.  It avoids host-value reads
    in the hot path but still materialises the broadcast coefficient tensor,
    so it must not be mistaken for the final memory/launch optimum.
    """

    from vpsc.world_model.portable_gated_trace import (
        _event_writes,
        _query_raw,
    )

    if drives.ndim != 3 or drives.shape[1] <= 0 or drives.shape[2] % 4:
        raise ValueError("drives must be [batch,time,4*state]")
    batch_size, _, drive_width = tuple(int(value) for value in drives.shape)
    state_dim = drive_width // 4
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
    ):
        raise TypeError("query_indices must be device int64 [batch,query]")
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
        ):
            raise TypeError(f"{name} must share the drives dtype/device")

    writes = _event_writes(drives, state_dim, surrogate_scale)
    decay_pair = decays.reshape(-1)
    initial_pair = torch.cat((initial_e, initial_i), dim=-1).contiguous()
    coefficient = decay_pair.view(1, 1, -1).expand_as(writes).contiguous()
    bias = ((1.0 - decay_pair.view(1, 1, -1)) * writes).contiguous()
    traces = triton_affine_scan(coefficient, bias, initial_pair)
    raw = _query_raw(
        traces,
        query_indices,
        state_dim,
        spike_threshold,
        surrogate_scale,
    )
    final_e, final_i = traces[:, -1].split(state_dim, dim=-1)
    return raw, final_e, final_i


def backend_audit() -> Dict[str, Any]:
    """Return the installed compiler/runtime identity after first use."""

    target = triton.runtime.driver.active.get_current_target()
    return {
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "target_backend": target.backend,
        "target_arch": target.arch,
        "target_warp_size": target.warp_size,
    }
