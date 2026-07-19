from __future__ import annotations

import pytest
import torch

from vpsc.world_model.fused_batched_gated_trace_cuda import (
    fused_batched_gated_trace,
)
from vpsc.world_model.fused_parallel_gated_trace_cuda import (
    fused_parallel_gated_trace,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_parallel_scan_matches_serial_batched_trace_and_gradients() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(25_900_001)
    batch, time_steps, state_dim, query_capacity = 3, 96, 5, 6
    queries = torch.tensor(
        ((0, 15, 47, 95, -1, -1), (7, 31, 63, -1, -1, -1), (2, 18, 44, 70, 88, 95)),
        dtype=torch.long,
        device=device,
    )
    serial_drives = torch.randn(
        batch,
        time_steps,
        4 * state_dim,
        device=device,
        requires_grad=True,
    )
    parallel_drives = serial_drives.detach().clone().requires_grad_(True)
    serial_decays = (
        0.5 + 0.45 * torch.rand(2, state_dim, device=device)
    ).requires_grad_(True)
    parallel_decays = serial_decays.detach().clone().requires_grad_(True)
    serial_initial_e = torch.rand(
        batch, state_dim, device=device, requires_grad=True
    )
    serial_initial_i = torch.rand(
        batch, state_dim, device=device, requires_grad=True
    )
    parallel_initial_e = serial_initial_e.detach().clone().requires_grad_(True)
    parallel_initial_i = serial_initial_i.detach().clone().requires_grad_(True)

    serial_raw, serial_final_e, serial_final_i = fused_batched_gated_trace(
        serial_drives,
        queries,
        serial_decays,
        serial_initial_e,
        serial_initial_i,
        spike_threshold=0.5,
        surrogate_scale=4.0,
    )
    parallel_raw, parallel_final_e, parallel_final_i = (
        fused_parallel_gated_trace(
            parallel_drives,
            queries,
            parallel_decays,
            parallel_initial_e,
            parallel_initial_i,
            spike_threshold=0.5,
            surrogate_scale=4.0,
        )
    )
    raw_probe = torch.randn_like(serial_raw)
    final_e_probe = torch.randn_like(serial_final_e)
    final_i_probe = torch.randn_like(serial_final_i)
    (
        (serial_raw * raw_probe).sum()
        + (serial_final_e * final_e_probe).sum()
        + (serial_final_i * final_i_probe).sum()
    ).backward()
    (
        (parallel_raw * raw_probe).sum()
        + (parallel_final_e * final_e_probe).sum()
        + (parallel_final_i * final_i_probe).sum()
    ).backward()

    torch.testing.assert_close(
        parallel_raw, serial_raw, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        parallel_final_e, serial_final_e, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        parallel_final_i, serial_final_i, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        parallel_drives.grad, serial_drives.grad, atol=3e-5, rtol=3e-4
    )
    torch.testing.assert_close(
        parallel_decays.grad, serial_decays.grad, atol=3e-5, rtol=3e-4
    )
    torch.testing.assert_close(
        parallel_initial_e.grad,
        serial_initial_e.grad,
        atol=3e-5,
        rtol=3e-4,
    )
    torch.testing.assert_close(
        parallel_initial_i.grad,
        serial_initial_i.grad,
        atol=3e-5,
        rtol=3e-4,
    )
    assert torch.count_nonzero(parallel_raw[0, 4:]).item() == 0
    assert torch.count_nonzero(parallel_raw[1, 3:]).item() == 0


def test_parallel_trace_rejects_cpu_execution() -> None:
    drives = torch.randn(2, 8, 12)
    queries = torch.tensor(((0, 7), (1, -1)), dtype=torch.long)
    decays = torch.full((2, 3), 0.75)
    initial = torch.zeros(2, 3)

    with pytest.raises(TypeError, match="CUDA float32"):
        fused_parallel_gated_trace(
            drives,
            queries,
            decays,
            initial,
            initial,
            spike_threshold=0.5,
            surrogate_scale=4.0,
        )
