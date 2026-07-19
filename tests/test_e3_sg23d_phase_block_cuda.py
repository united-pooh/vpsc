from __future__ import annotations

import torch

from experiments import e3_sg19_plan_edge_spikes as sg19
from experiments import e3_sg23_spike_feature_solvers as sg23
from experiments import e3_sg23d_phase_block_cuda as sg23d


def _tiny_states(row_count: int = 14):
    indices = torch.arange(row_count, dtype=torch.long)
    return {
        "keys": torch.stack(
            (indices % 3, indices % 4, indices % 5, indices % 7), dim=1
        ),
        "phases": indices % 3,
        "masks": torch.stack(
            tuple(((indices + bit) % 3 == 0).to(torch.float64) for bit in range(8)),
            dim=1,
        ),
        "plan_current": indices % 5,
        "plan_next": (indices * 2) % 5,
        "return_edges": indices % 2,
        "counts": (indices % 3 + 1).to(torch.float64),
        "target_means": torch.where(
            (indices[:, None] + torch.arange(4)[None, :]) % 2 == 0,
            1.0,
            -1.0,
        ).to(torch.float64),
        "ambiguous_unique_key_count": 0,
    }


def test_phase_groups_are_sorted_and_cover_every_row() -> None:
    states = _tiny_states()
    groups = sg23d.phase_groups(states)
    assert tuple(groups) == (0, 1, 2)
    assert sorted(index for group in groups.values() for index in group) == list(
        range(14)
    )


def test_phase_kernel_is_strictly_block_diagonal() -> None:
    states = _tiny_states()
    kernel = sg19.plan_edge_kernel(states, states)
    groups = sg23d.phase_groups(states)
    assert sg23d.cross_phase_max_abs(kernel, groups) == 0.0


def test_cpu_phase_blocks_preserve_tiny_predictions() -> None:
    states = _tiny_states()
    blocks, structure = sg23d.build_phase_blocks(states)
    dense_coefficients, _metrics, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    result, metrics = sg23d.benchmark_phase_blocks(
        blocks,
        states,
        device=torch.device("cpu"),
        warmups=0,
        repetitions=1,
    )
    assert structure["quadratic_ratio_full_over_blocks"] > 2.0
    assert torch.allclose(
        result["scores"],
        kernel @ dense_coefficients,
        atol=1e-7,
        rtol=0.0,
    )
    backward = sg23d.normalized_backward_error(
        states, sg23d.phase_groups(states), result["coefficients"]
    )
    assert backward["maximum"] <= sg23d.BACKWARD_ERROR_TOLERANCE
    assert metrics["resident"]["sample_count"] == 1
