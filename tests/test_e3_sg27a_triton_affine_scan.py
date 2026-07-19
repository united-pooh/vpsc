from __future__ import annotations

import pytest
import torch


triton = pytest.importorskip("triton")

from vpsc.world_model.portable_gated_trace import (  # noqa: E402
    inclusive_affine_scan,
    serial_gated_trace_reference,
)
from vpsc.world_model.triton_affine_scan import (  # noqa: E402
    backend_audit,
    triton_affine_scan,
    triton_composed_gated_trace,
)


LENGTHS = (1, 2, 3, 31, 64, 96, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
@pytest.mark.parametrize("time_steps", LENGTHS)
def test_triton_affine_scan_matches_torch_forward_and_backward(
    time_steps: int,
) -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(27_300_000 + time_steps)
    coefficient = 0.55 + 0.4 * torch.rand(
        3, time_steps, 5, device=device
    )
    bias = torch.randn(3, time_steps, 5, device=device)
    initial = torch.randn(3, 5, device=device)
    reference_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (coefficient, bias, initial)
    )
    triton_inputs = tuple(
        value.detach().clone().contiguous().requires_grad_(True)
        for value in (coefficient, bias, initial)
    )
    reference = inclusive_affine_scan(*reference_inputs)
    actual = triton_affine_scan(*triton_inputs)
    probe = torch.randn_like(reference)
    (reference * probe).sum().backward()
    (actual * probe).sum().backward()

    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
    for actual_input, reference_input in zip(triton_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad,
            reference_input.grad,
            atol=8e-5,
            rtol=8e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
def test_triton_backend_is_amd_gfx1101() -> None:
    sample = torch.ones(1, 3, 1, device="cuda", requires_grad=True)
    triton_affine_scan(sample, sample, sample[:, 0]).sum().backward()
    audit = backend_audit()

    assert audit["target_backend"] == "hip"
    assert audit["target_arch"] == "gfx1101"
    assert audit["torch_hip"] is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP is required")
@pytest.mark.parametrize("time_steps", (1, 3, 31, 96, 128))
def test_composed_gated_trace_matches_serial_surrogate(time_steps: int) -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(27_400_000 + time_steps)
    batch_size, state_dim, query_capacity = 3, 5, 5
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
                time_steps // 2,
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
        **keyword,
    )
    actual = triton_composed_gated_trace(
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
            actual_value, reference_value, atol=4e-5, rtol=4e-5
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
            atol=1.5e-4,
            rtol=1.5e-4,
        )
