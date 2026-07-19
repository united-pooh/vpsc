from __future__ import annotations

import torch

from experiments.e3_sg20_strict_return_blocks import (
    _solve_dense_weighted,
    solve_strict_return_blocks,
    strict_return_kernel,
)


def _state() -> dict[str, torch.Tensor]:
    return {
        "keys": torch.tensor(
            [
                [4, 4, 4, 0],
                [4, 4, 4, 1],
                [4, 4, 0, 1],
                [4, 0, 1, 2],
            ],
            dtype=torch.long,
        ),
        "phases": torch.tensor([0, 0, 1, 2], dtype=torch.long),
        "masks": torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        "plan_current": torch.tensor([0, 0, 1, 2], dtype=torch.long),
        "plan_next": torch.tensor([1, 2, 2, 3], dtype=torch.long),
        "return_edges": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }


def test_strict_return_kernel_is_psd_and_cross_blocks_are_zero() -> None:
    state = _state()
    kernel = strict_return_kernel(state, state)
    return_zero = state["return_edges"] == 0
    return_one = state["return_edges"] == 1

    assert torch.allclose(kernel, kernel.T)
    assert torch.count_nonzero(kernel[return_zero][:, return_one]) == 0
    assert torch.count_nonzero(kernel[return_one][:, return_zero]) == 0
    assert float(torch.linalg.eigvalsh(kernel).min().item()) >= -1e-10


def test_exact_block_solve_matches_dense_strict_solve() -> None:
    unique = {
        **_state(),
        "counts": torch.tensor([3.0, 2.0, 4.0, 1.0], dtype=torch.float64),
        "target_means": torch.tensor(
            [
                [1.0, -1.0, 0.5],
                [-1.0, 1.0, 0.25],
                [0.5, 0.5, -1.0],
                [-0.5, 0.25, 1.0],
            ],
            dtype=torch.float64,
        ),
    }

    block_coefficients, report = solve_strict_return_blocks(
        unique, device=torch.device("cpu"), block_workers=1
    )
    dense_coefficients, _kernel = _solve_dense_weighted(
        unique, device=torch.device("cpu")
    )

    assert [block["prototype_count"] for block in report["blocks"]] == [2, 2]
    assert torch.allclose(
        block_coefficients, dense_coefficients, atol=1e-10, rtol=1e-10
    )


def test_two_worker_exact_block_solve_matches_single_worker() -> None:
    unique = {
        **_state(),
        "counts": torch.tensor([3.0, 2.0, 4.0, 1.0], dtype=torch.float64),
        "target_means": torch.tensor(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
                [0.5, -0.5],
                [-0.5, 0.5],
            ],
            dtype=torch.float64,
        ),
    }

    single, _single_report = solve_strict_return_blocks(
        unique, device=torch.device("cpu"), block_workers=1
    )
    parallel, parallel_report = solve_strict_return_blocks(
        unique, device=torch.device("cpu"), block_workers=2
    )

    assert parallel_report["block_workers"] == 2
    assert torch.equal(single, parallel)
