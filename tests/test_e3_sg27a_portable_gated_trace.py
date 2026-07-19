from __future__ import annotations

import math

import pytest
import torch

from vpsc.world_model.portable_gated_trace import (
    affine_scan_rounds,
    inclusive_affine_scan,
    portable_parallel_gated_trace,
    serial_gated_trace_reference,
)


LENGTHS = (1, 2, 3, 31, 64, 96, 128)


def _serial_affine(
    coefficient: torch.Tensor,
    bias: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    state = initial
    states = []
    for index in range(coefficient.shape[1]):
        state = coefficient[:, index] * state + bias[:, index]
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("time_steps", LENGTHS)
def test_affine_scan_matches_serial_values_and_gradients(time_steps: int) -> None:
    generator = torch.Generator().manual_seed(27_100_000 + time_steps)
    coefficient = 0.55 + 0.4 * torch.rand(
        2, time_steps, 4, generator=generator, dtype=torch.float64
    )
    bias = torch.randn(
        2, time_steps, 4, generator=generator, dtype=torch.float64
    )
    initial = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    serial_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (coefficient, bias, initial)
    )
    parallel_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (coefficient, bias, initial)
    )

    serial = _serial_affine(*serial_inputs)
    parallel = inclusive_affine_scan(*parallel_inputs)
    probe = torch.randn(serial.shape, generator=generator, dtype=torch.float64)
    (serial * probe).sum().backward()
    (parallel * probe).sum().backward()

    torch.testing.assert_close(parallel, serial, atol=2e-12, rtol=2e-12)
    for parallel_input, serial_input in zip(parallel_inputs, serial_inputs):
        torch.testing.assert_close(
            parallel_input.grad,
            serial_input.grad,
            atol=5e-12,
            rtol=5e-12,
        )


def _queries(time_steps: int) -> torch.Tensor:
    first = sorted({0, time_steps // 2, time_steps - 1})
    second = sorted({min(time_steps - 1, 1), max(0, time_steps - 2)})
    capacity = 4
    rows = [row + [-1] * (capacity - len(row)) for row in (first, second)]
    return torch.tensor(rows, dtype=torch.long)


@pytest.mark.parametrize("time_steps", LENGTHS)
def test_portable_gated_trace_matches_serial_surrogate(time_steps: int) -> None:
    generator = torch.Generator().manual_seed(27_200_000 + time_steps)
    batch_size, state_dim = 2, 3
    base_drives = torch.randn(
        batch_size,
        time_steps,
        4 * state_dim,
        generator=generator,
        dtype=torch.float64,
    )
    # Keep drive thresholds away from zero so the hard forward is insensitive
    # to affine reduction roundoff while the surrogate gradient remains active.
    drives = base_drives.sign() * (base_drives.abs() + 0.2)
    decays = 0.55 + 0.35 * torch.rand(
        2, state_dim, generator=generator, dtype=torch.float64
    )
    initial_e = 0.05 + 0.2 * torch.rand(
        batch_size, state_dim, generator=generator, dtype=torch.float64
    )
    initial_i = 0.05 + 0.2 * torch.rand(
        batch_size, state_dim, generator=generator, dtype=torch.float64
    )
    serial_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (drives, decays, initial_e, initial_i)
    )
    parallel_inputs = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (drives, decays, initial_e, initial_i)
    )
    queries = _queries(time_steps)
    keyword = {"spike_threshold": 0.43, "surrogate_scale": 4.0}

    serial = serial_gated_trace_reference(
        serial_inputs[0],
        queries,
        serial_inputs[1],
        serial_inputs[2],
        serial_inputs[3],
        **keyword,
    )
    parallel = portable_parallel_gated_trace(
        parallel_inputs[0],
        queries,
        parallel_inputs[1],
        parallel_inputs[2],
        parallel_inputs[3],
        **keyword,
    )
    probes = tuple(
        torch.randn(value.shape, generator=generator, dtype=torch.float64)
        for value in serial
    )
    sum((value * probe).sum() for value, probe in zip(serial, probes)).backward()
    sum((value * probe).sum() for value, probe in zip(parallel, probes)).backward()

    for parallel_value, serial_value in zip(parallel, serial):
        torch.testing.assert_close(
            parallel_value, serial_value, atol=3e-12, rtol=3e-12
        )
    spike_width = 2 * state_dim
    torch.testing.assert_close(
        parallel[0][:, :, :spike_width],
        serial[0][:, :, :spike_width],
        atol=0,
        rtol=0,
    )
    for parallel_input, serial_input in zip(parallel_inputs, serial_inputs):
        torch.testing.assert_close(
            parallel_input.grad,
            serial_input.grad,
            atol=2e-10,
            rtol=2e-10,
        )
    assert torch.count_nonzero(parallel[0][queries.eq(-1)]).item() == 0


@pytest.mark.parametrize("time_steps", LENGTHS)
def test_affine_scan_round_count_is_logarithmic(time_steps: int) -> None:
    expected = 0 if time_steps == 1 else math.ceil(math.log2(time_steps))
    assert affine_scan_rounds(time_steps) == expected


def test_portable_trace_rejects_queries_after_padding() -> None:
    drives = torch.randn(1, 4, 8)
    decays = torch.full((2, 2), 0.75)
    initial = torch.zeros(1, 2)
    queries = torch.tensor(((0, -1, 3),), dtype=torch.long)

    with pytest.raises(ValueError, match="precede"):
        portable_parallel_gated_trace(
            drives,
            queries,
            decays,
            initial,
            initial,
            spike_threshold=0.5,
            surrogate_scale=4.0,
        )


def test_portable_trace_rejects_non_increasing_queries() -> None:
    drives = torch.randn(1, 4, 8)
    decays = torch.full((2, 2), 0.75)
    initial = torch.zeros(1, 2)
    queries = torch.tensor(((0, 3, 3),), dtype=torch.long)

    with pytest.raises(ValueError, match="strictly increasing"):
        portable_parallel_gated_trace(
            drives,
            queries,
            decays,
            initial,
            initial,
            spike_threshold=0.5,
            surrogate_scale=4.0,
        )
