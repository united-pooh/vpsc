from pathlib import Path

import torch

from experiments.e3_sg13_suffix_spike_kernel import suffix_spike_kernel
from experiments.e3_sg15_phase_isolated_kernel import (
    DEFAULT_SG14R_REFERENCE,
    KERNEL_SPECS,
    PRIMARY_SPEC,
    SG14R_REFERENCE_SHA256,
    _load_reference,
)


def test_sg15_strict_kernel_makes_cross_phase_memories_orthogonal() -> None:
    keys = torch.tensor(((0, 1, 2, 3), (0, 1, 2, 3)), dtype=torch.long)
    same = torch.tensor((3, 3), dtype=torch.long)
    different = torch.tensor((2, 3), dtype=torch.long)
    assert suffix_spike_kernel(
        keys[[0]], keys[[1]], same[[0]], same[[1]], PRIMARY_SPEC
    ).item() == 4.0
    assert suffix_spike_kernel(
        keys[[0]], keys[[1]], different[[0]], different[[1]], PRIMARY_SPEC
    ).item() == 0.0


def test_sg15_kernel_family_remains_positive_semidefinite() -> None:
    keys = torch.tensor(
        ((8, 8, 0, 1), (8, 2, 0, 1), (4, 2, 0, 1), (4, 2, 3, 5)),
        dtype=torch.long,
    )
    phases = torch.tensor((0, 1, 2, 3), dtype=torch.long)
    for spec in KERNEL_SPECS:
        kernel = suffix_spike_kernel(keys, keys, phases, phases, spec)
        assert float(torch.linalg.eigvalsh(kernel).min().item()) >= -1e-10


def test_sg15_failed_sg14r_reference_is_hash_locked() -> None:
    reference, digest = _load_reference(
        Path(DEFAULT_SG14R_REFERENCE),
        SG14R_REFERENCE_SHA256,
        "E3-SG14R third-fresh phase-bound spike kernel confirmation",
    )
    assert digest == SG14R_REFERENCE_SHA256
    assert reference["decision"]["primary_test_metrics"][
        "exact_vector_accuracy"
    ] == 0.975
