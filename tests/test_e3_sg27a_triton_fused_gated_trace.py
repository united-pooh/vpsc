from __future__ import annotations

import pytest
import torch


pytest.importorskip("triton")

from vpsc.world_model.portable_gated_trace import (  # noqa: E402
    serial_gated_trace_reference,
)
from vpsc.world_model.triton_fused_gated_trace import (  # noqa: E402
    backend_audit,
    triton_fused_gated_trace,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
@pytest.mark.parametrize("time_steps", (1, 3, 31, 96, 128, 160))
def test_fused_gated_trace_matches_serial_surrogate(time_steps: int) -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(27_600_000 + time_steps)
    batch_size, state_dim, query_capacity = 3, 5, 6
    base = torch.randn(batch_size, time_steps, 4 * state_dim, device=device)
    drives = base.sign() * (base.abs() + 0.2)
    decays = 0.55 + 0.35 * torch.rand(2, state_dim, device=device)
    initial_e = 0.05 + 0.2 * torch.rand(batch_size, state_dim, device=device)
    initial_i = 0.05 + 0.2 * torch.rand(batch_size, state_dim, device=device)
    query_rows = []
    for row in range(batch_size):
        positions = sorted(
            {
                min(time_steps - 1, row),
                time_steps // 3,
                2 * time_steps // 3,
                time_steps - 1,
            }
        )
        query_rows.append(
            positions + [-1] * (query_capacity - len(positions))
        )
    queries = torch.tensor(query_rows, dtype=torch.long, device=device)
    reference_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (drives, decays, initial_e, initial_i)
    )
    actual_inputs = tuple(
        value.detach().clone().contiguous().requires_grad_(True)
        for value in (drives, decays, initial_e, initial_i)
    )
    keyword = {"spike_threshold": 0.43, "surrogate_scale": 4.0}
    reference = serial_gated_trace_reference(
        reference_inputs[0],
        queries,
        reference_inputs[1],
        reference_inputs[2],
        reference_inputs[3],
        _unchecked=True,
        **keyword,
    )
    actual = triton_fused_gated_trace(
        actual_inputs[0],
        queries,
        actual_inputs[1],
        actual_inputs[2],
        actual_inputs[3],
        **keyword,
    )
    probes = tuple(torch.randn_like(value) for value in reference)
    sum((value * probe).sum() for value, probe in zip(reference, probes)).backward()
    sum((value * probe).sum() for value, probe in zip(actual, probes)).backward()

    for actual_value, reference_value in zip(actual, reference):
        torch.testing.assert_close(
            actual_value, reference_value, atol=5e-5, rtol=5e-5
        )
    torch.testing.assert_close(
        actual[0][:, :, : 2 * state_dim],
        reference[0][:, :, : 2 * state_dim],
        atol=0,
        rtol=0,
    )
    for actual_input, reference_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad,
            reference_input.grad,
            atol=2e-4,
            rtol=2e-4,
        )
    assert torch.count_nonzero(actual[0][queries.eq(-1)]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
def test_fused_backend_covers_sg26_long_fallback() -> None:
    audit = backend_audit()
    assert audit["target_backend"] == "hip"
    assert audit["target_arch"] == "gfx1101"
    assert audit["max_time"] >= 160


def test_fused_trace_rejects_cpu_execution() -> None:
    drives = torch.randn(2, 8, 12)
    queries = torch.tensor(((0, 7), (1, -1)), dtype=torch.long)
    decays = torch.full((2, 3), 0.75)
    initial = torch.zeros(2, 3)

    with pytest.raises(TypeError, match="CUDA/HIP"):
        triton_fused_gated_trace(
            drives,
            queries,
            decays,
            initial,
            initial,
            spike_threshold=0.5,
            surrogate_scale=4.0,
        )
