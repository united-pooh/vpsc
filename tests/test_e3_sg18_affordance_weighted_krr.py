from __future__ import annotations

import torch

from experiments.e3_sg18_affordance_weighted_krr import (
    _mask,
    affordance_spike_kernel,
    compress_unique_records,
    weighted_unique_krr_fit,
)


def _duplicate_train() -> dict[str, object]:
    keys = torch.tensor(
        [
            [4, 4, 4, 0],
            [4, 4, 4, 0],
            [4, 4, 4, 0],
            [4, 4, 4, 1],
            [4, 4, 4, 1],
            [4, 4, 4, 1],
        ],
        dtype=torch.long,
    )
    phases = torch.zeros(6, dtype=torch.long)
    masks = torch.tensor(
        [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 3,
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [
            [1.0] * 13,
            [1.0] * 13,
            [-1.0] * 13,
            [-1.0] * 13,
            [1.0] * 13,
            [-1.0] * 13,
        ],
        dtype=torch.float64,
    )
    return {
        "keys": keys,
        "phases": phases,
        "masks": masks,
        "target_code": targets,
        "elapsed_seconds": 0.0,
    }


def test_weighted_unique_solution_matches_expanded_duplicate_krr() -> None:
    train = _duplicate_train()
    unique = compress_unique_records(train)
    _coefficients, audit = weighted_unique_krr_fit(
        train,
        unique,
        ridge_lambda=1e-3,
        device=torch.device("cpu"),
    )
    assert audit["expanded_example_count"] == 6
    assert audit["unique_prototype_count"] == 2
    assert audit["compression_ratio"] == 1 / 3
    assert audit["expanded_train_score_max_abs_difference"] < 1e-9
    assert audit["expanded_prediction_equivalent"]


def test_affordance_product_kernel_is_symmetric_psd() -> None:
    keys = torch.tensor(
        [[4, 4, 4, 0], [4, 4, 4, 1], [4, 4, 0, 1]],
        dtype=torch.long,
    )
    phases = torch.tensor([0, 0, 1], dtype=torch.long)
    masks = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    kernel = affordance_spike_kernel(
        keys, keys, phases, phases, masks, masks
    )
    assert torch.allclose(kernel, kernel.T)
    assert float(torch.linalg.eigvalsh(kernel).min().item()) >= -1e-10


def test_affordance_mask_uses_frozen_action_order() -> None:
    order = ("go east", "inventory", "look")
    assert _mask(("look", "go east"), order) == (1, 0, 1)
