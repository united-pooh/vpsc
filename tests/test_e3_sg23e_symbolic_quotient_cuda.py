from __future__ import annotations

import torch

from experiments import e3_sg19_plan_edge_spikes as sg19
from experiments import e3_sg23_spike_feature_solvers as sg23
from experiments import e3_sg23e_symbolic_quotient_cuda as sg23e


def _tiny_states(row_count: int = 18):
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


def test_gf2_pivots_are_deterministic() -> None:
    packed = torch.tensor(
        [[0b10000000], [0b01000000], [0b11000000], [0b00100000]],
        dtype=torch.uint8,
    ).numpy()
    assert sg23e._gf2_pivots(packed) == (0, 1, 3)


@torch.no_grad()
def test_symbolic_quotient_reconstructs_tiny_kernel() -> None:
    states = _tiny_states()
    features = sg23.build_explicit_features(states)
    quotient = sg23e.build_symbolic_quotient(
        features, coordinate_device=torch.device("cpu")
    )
    kernel = sg19.plan_edge_kernel(states, states)
    assert quotient.metrics["constant_column_max_abs_error"] == 0.0
    assert quotient.metrics["integer_reconstruction_max_abs_error"] == 0.0
    assert torch.equal(
        quotient.basis @ quotient.transform, quotient.support_matrix
    )
    assert torch.equal(
        quotient.basis @ quotient.metric @ quotient.basis.T, kernel
    )


@torch.no_grad()
def test_symbolic_solver_matches_tiny_dense_predictions() -> None:
    states = _tiny_states()
    features = sg23.build_explicit_features(states)
    quotient = sg23e.build_symbolic_quotient(
        features, coordinate_device=torch.device("cpu")
    )
    dense_coefficients, _metrics, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    solved = sg23e._symbolic_operation(
        quotient.basis,
        quotient.metric,
        states["counts"],
        states["target_means"],
    )
    assert torch.allclose(
        solved["scores"],
        kernel @ dense_coefficients,
        atol=1e-7,
        rtol=0.0,
    )
    models = sg23e._recover_models(quotient, solved["beta"])
    assert torch.allclose(
        kernel @ models["dual_prediction_equivalent"],
        solved["scores"],
        atol=1e-9,
        rtol=0.0,
    )
    assert torch.allclose(
        torch.sparse.mm(features.matrix, models["original_feature_model"]),
        solved["scores"],
        atol=1e-9,
        rtol=0.0,
    )
    kernel = sg19.plan_edge_kernel(states, states)
    assert (
        sg23e._original_backward_error(
            states, kernel, quotient.basis, quotient.metric
        )
        <= sg23e.BACKWARD_ERROR_TOLERANCE
    )
