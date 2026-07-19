from pathlib import Path

import torch

from experiments.e3_sg11_temporal_basis import (
    BASIS_BY_NAME,
    BASIS_SPECS,
    REFERENCE_SHA256,
    _basis_feature,
    _basis_state_after,
    _basis_step,
    _global_variant_selection,
    _reference_artifact,
    augment_hidden_with_basis,
    temporal_basis_values,
)
from experiments.e3_sg8_bilinear_closed_form import _outer_features


REFERENCE = Path("results/e3_scan/e3_sg10_multichannel_delta.json")


def test_sg11_temporal_bases_are_recursive_and_match_their_invariants() -> None:
    device = torch.device("cpu")
    counts = torch.arange(0, 7, dtype=torch.long)

    unit = temporal_basis_values(
        BASIS_BY_NAME["unit_root1"], counts, dtype=torch.float64
    )[:, 0]
    torch.testing.assert_close(unit, counts.to(torch.float64))

    leaky = temporal_basis_values(
        BASIS_BY_NAME["leaky4"], counts, dtype=torch.float64
    )
    taus = torch.tensor((1.0, 2.0, 4.0, 8.0), dtype=torch.float64)
    expected_leaky = 1.0 - torch.exp(-counts[:, None] / taus[None, :])
    torch.testing.assert_close(leaky, expected_leaky, atol=1e-12, rtol=1e-12)

    oscillator = temporal_basis_values(
        BASIS_BY_NAME["oscillator4"], counts, dtype=torch.float64
    )
    torch.testing.assert_close(
        oscillator[:, 0].square() + oscillator[:, 1].square(),
        torch.ones(7, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        oscillator[:, 2].square() + oscillator[:, 3].square(),
        torch.ones(7, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    binary = temporal_basis_values(
        BASIS_BY_NAME["binary3"], counts, dtype=torch.float64
    )
    expected_binary = torch.tensor(
        [
            tuple(1.0 if value & (1 << bit) else -1.0 for bit in range(3))
            for value in range(7)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(binary, expected_binary)

    oracle = temporal_basis_values(
        BASIS_BY_NAME["one_hot6_oracle"],
        torch.arange(1, 7, dtype=torch.long),
        dtype=torch.float64,
    )
    assert torch.unique(oracle, dim=0).shape[0] == 6

    for spec in BASIS_SPECS:
        state = _basis_state_after(
            spec,
            5,
            batch_size=2,
            device=device,
            dtype=torch.float32,
        )
        stepped = _basis_step(spec, state)
        expected = temporal_basis_values(
            spec,
            torch.tensor((6,), dtype=torch.long),
            dtype=torch.float32,
        )[0]
        torch.testing.assert_close(
            _basis_feature(spec, stepped)[0], expected, atol=1e-6, rtol=1e-6
        )


def test_sg11_basis_replaces_coordinates_without_expanding_ridge() -> None:
    torch.manual_seed(11)
    hidden = torch.randn(7, 2, 32)
    counts = torch.tensor(
        [(previous, previous + 1) for previous in range(1, 8)],
        dtype=torch.long,
    )
    baseline_features = _outer_features(
        augment_hidden_with_basis(
            hidden, counts, BASIS_BY_NAME["baseline"]
        )
    )
    assert baseline_features.shape == (7, 1089)
    for spec in BASIS_SPECS:
        augmented = augment_hidden_with_basis(hidden, counts, spec)
        features = _outer_features(augmented)
        assert augmented.shape == hidden.shape
        assert features.shape == (7, 1089)
        if spec.state_dim:
            assert not torch.equal(augmented, hidden)
            torch.testing.assert_close(
                augmented[:, :, : -spec.state_dim],
                hidden[:, :, : -spec.state_dim],
            )


def test_sg11_reference_and_global_selection_do_not_use_test_metrics() -> None:
    reference, digest = _reference_artifact(REFERENCE)
    assert digest == REFERENCE_SHA256
    assert reference["decision"]["best_ann_exact_vector_accuracy"] == 1.0

    def seed(valid_exact_by_name):
        variants = {}
        for spec in BASIS_SPECS:
            exact = valid_exact_by_name.get(spec.name, 0.0)
            variants[spec.name] = {
                "valid": {
                    "exact_vector_accuracy": exact,
                    "macro_channel_accuracy": exact,
                    "mse": 1.0 - exact,
                },
                "test": {"exact_vector_accuracy": 1.0 - exact},
            }
        return {"variants": variants}

    seeds = (
        seed({"unit_root1": 0.8, "leaky4": 0.9}),
        seed({"unit_root1": 0.8, "leaky4": 0.9}),
    )
    selected, audit = _global_variant_selection(seeds)
    assert selected == "leaky4"
    assert {record["name"] for record in audit} == {
        "unit_root1",
        "leaky4",
        "oscillator4",
        "binary3",
    }
