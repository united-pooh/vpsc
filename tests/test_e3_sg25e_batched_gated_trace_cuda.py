from __future__ import annotations

import pytest
import torch

from vpsc.world_model.fused_batched_gated_trace_cuda import (
    fused_batched_gated_trace,
)
from vpsc.world_model.fused_gated_trace_cuda import fused_gated_trace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_batched_queries_match_independent_sg25c_traces_and_gradients() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(25_800_001)
    batch, time_steps, state_dim, query_capacity = 3, 17, 5, 4
    query_indices = torch.tensor(
        ((0, 5, 16, -1), (2, 8, -1, -1), (1, 4, 9, 15)),
        dtype=torch.long,
        device=device,
    )
    valid_counts = (3, 2, 4)
    batched_drives = torch.randn(
        batch,
        time_steps,
        4 * state_dim,
        device=device,
        requires_grad=True,
    )
    reference_drives = batched_drives.detach().clone().requires_grad_(True)
    batched_decays = (
        0.5 + 0.45 * torch.rand(2, state_dim, device=device)
    ).requires_grad_(True)
    reference_decays = batched_decays.detach().clone().requires_grad_(True)
    batched_initial_e = torch.rand(
        batch, state_dim, device=device, requires_grad=True
    )
    batched_initial_i = torch.rand(
        batch, state_dim, device=device, requires_grad=True
    )
    reference_initial_e = batched_initial_e.detach().clone().requires_grad_(True)
    reference_initial_i = batched_initial_i.detach().clone().requires_grad_(True)

    batched_raw, batched_final_e, batched_final_i = fused_batched_gated_trace(
        batched_drives,
        query_indices,
        batched_decays,
        batched_initial_e,
        batched_initial_i,
        spike_threshold=0.5,
        surrogate_scale=4.0,
    )
    reference_rows = []
    reference_final_e_rows = []
    reference_final_i_rows = []
    for row, valid_count in enumerate(valid_counts):
        raw, final_e, final_i = fused_gated_trace(
            reference_drives[row : row + 1],
            query_indices[row, :valid_count],
            reference_decays,
            reference_initial_e[row : row + 1],
            reference_initial_i[row : row + 1],
            spike_threshold=0.5,
            surrogate_scale=4.0,
        )
        padding = torch.zeros(
            1,
            query_capacity - valid_count,
            4 * state_dim,
            device=device,
        )
        reference_rows.append(torch.cat((raw, padding), dim=1))
        reference_final_e_rows.append(final_e)
        reference_final_i_rows.append(final_i)
    reference_raw = torch.cat(reference_rows, dim=0)
    reference_final_e = torch.cat(reference_final_e_rows, dim=0)
    reference_final_i = torch.cat(reference_final_i_rows, dim=0)

    raw_probe = torch.randn_like(batched_raw)
    final_e_probe = torch.randn_like(batched_final_e)
    final_i_probe = torch.randn_like(batched_final_i)
    (
        (batched_raw * raw_probe).sum()
        + (batched_final_e * final_e_probe).sum()
        + (batched_final_i * final_i_probe).sum()
    ).backward()
    (
        (reference_raw * raw_probe).sum()
        + (reference_final_e * final_e_probe).sum()
        + (reference_final_i * final_i_probe).sum()
    ).backward()

    torch.testing.assert_close(
        batched_raw, reference_raw, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        batched_final_e, reference_final_e, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        batched_final_i, reference_final_i, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        batched_drives.grad, reference_drives.grad, atol=3e-5, rtol=3e-4
    )
    torch.testing.assert_close(
        batched_decays.grad, reference_decays.grad, atol=3e-5, rtol=3e-4
    )
    torch.testing.assert_close(
        batched_initial_e.grad,
        reference_initial_e.grad,
        atol=3e-5,
        rtol=3e-4,
    )
    torch.testing.assert_close(
        batched_initial_i.grad,
        reference_initial_i.grad,
        atol=3e-5,
        rtol=3e-4,
    )
    for row, valid_count in enumerate(valid_counts):
        assert torch.count_nonzero(batched_raw[row, valid_count:]).item() == 0


def test_batched_trace_rejects_cpu_execution() -> None:
    drives = torch.randn(2, 8, 12)
    queries = torch.tensor(((0, 7), (1, -1)), dtype=torch.long)
    decays = torch.full((2, 3), 0.75)
    initial = torch.zeros(2, 3)

    with pytest.raises(TypeError, match="CUDA float32"):
        fused_batched_gated_trace(
            drives,
            queries,
            decays,
            initial,
            initial,
            spike_threshold=0.5,
            surrogate_scale=4.0,
        )
