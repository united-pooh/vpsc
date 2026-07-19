from pathlib import Path

import torch

from experiments.e3_sg13_suffix_spike_kernel import suffix_spike_kernel
from experiments.e3_sg14_phase_bound_kernel import (
    ADDITIVE_REFERENCE_SHA256,
    DEFAULT_ADDITIVE_REFERENCE,
    DEFAULT_FRESH_BASELINE,
    FRESH_BASELINE_SHA256,
    KERNEL_SPECS,
    PRIMARY_SPEC,
    _load_reference,
)


def test_sg14_phase_product_binds_suffix_similarity_to_stage() -> None:
    keys = torch.tensor(((0, 1, 2, 3), (0, 1, 2, 3)), dtype=torch.long)
    same_phase = torch.tensor((2, 2), dtype=torch.long)
    different_phase = torch.tensor((2, 3), dtype=torch.long)
    base_plus = suffix_spike_kernel(
        keys[[0]], keys[[1]], different_phase[[0]], different_phase[[1]], PRIMARY_SPEC
    )
    bound = suffix_spike_kernel(
        keys[[0]], keys[[1]], same_phase[[0]], same_phase[[1]], PRIMARY_SPEC
    )
    assert base_plus.item() == 4.0
    assert bound.item() == 8.0

    product_only = next(
        spec for spec in KERNEL_SPECS if spec.name == "phase_product_only"
    )
    assert suffix_spike_kernel(
        keys[[0]],
        keys[[1]],
        different_phase[[0]],
        different_phase[[1]],
        product_only,
    ).item() == 0.0
    assert suffix_spike_kernel(
        keys[[0]], keys[[1]], same_phase[[0]], same_phase[[1]], product_only
    ).item() == 4.0


def test_sg14_all_nonnegative_sum_and_product_kernels_are_psd() -> None:
    keys = torch.tensor(
        (
            (8, 8, 0, 1),
            (8, 2, 0, 1),
            (4, 2, 0, 1),
            (4, 2, 3, 1),
            (4, 2, 3, 5),
        ),
        dtype=torch.long,
    )
    phases = torch.tensor((0, 1, 2, 3, 3), dtype=torch.long)
    for spec in KERNEL_SPECS:
        kernel = suffix_spike_kernel(keys, keys, phases, phases, spec)
        torch.testing.assert_close(kernel, kernel.T)
        assert float(torch.linalg.eigvalsh(kernel).min().item()) >= -1e-10


def test_sg14_frozen_reference_artifacts_match_preregistered_hashes() -> None:
    fresh, fresh_digest = _load_reference(
        Path(DEFAULT_FRESH_BASELINE),
        FRESH_BASELINE_SHA256,
        "E3-SG10 multichannel TextWorld event delta",
    )
    additive, additive_digest = _load_reference(
        Path(DEFAULT_ADDITIVE_REFERENCE),
        ADDITIVE_REFERENCE_SHA256,
        "E3-SG13R fresh-game suffix spike kernel confirmation",
    )
    assert fresh_digest == FRESH_BASELINE_SHA256
    assert additive_digest == ADDITIVE_REFERENCE_SHA256
    assert fresh["decision"]["best_ann_exact_vector_accuracy"] > 0.99
    assert additive["decision"]["primary_test_metrics"][
        "exact_vector_accuracy"
    ] == 0.9333333333333333
