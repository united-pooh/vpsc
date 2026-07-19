from __future__ import annotations

import torch

from experiments import e3_sg19_plan_edge_spikes as sg19
from experiments import e3_sg23_spike_feature_solvers as sg23


def _tiny_states(row_count: int = 8):
    keys = torch.tensor(
        [
            (index % 3, (index + 1) % 4, index % 5, (index * 3) % 7)
            for index in range(row_count)
        ],
        dtype=torch.long,
    )
    masks = torch.tensor(
        [
            tuple(float((index + bit) % 3 == 0) for bit in range(8))
            for index in range(row_count)
        ],
        dtype=torch.float64,
    )
    return {
        "keys": keys,
        "phases": torch.arange(row_count, dtype=torch.long) % 5,
        "masks": masks,
        "plan_current": torch.arange(row_count, dtype=torch.long) % 6,
        "plan_next": (torch.arange(row_count, dtype=torch.long) * 2) % 6,
        "return_edges": torch.arange(row_count, dtype=torch.long) % 2,
        "counts": torch.arange(1, row_count + 1, dtype=torch.float64) % 3 + 1,
        "target_means": torch.where(
            (
                torch.arange(row_count)[:, None]
                + torch.arange(3)[None, :]
            )
            % 2
            == 0,
            1.0,
            -1.0,
        ).to(torch.float64),
        "ambiguous_unique_key_count": 0,
    }


def test_explicit_feature_map_matches_plan_edge_kernel() -> None:
    states = _tiny_states()
    features = sg23.build_explicit_features(states)
    explicit = sg23.explicit_cross_kernel(features.rows, features.rows)
    analytic = sg19.plan_edge_kernel(states, states)
    assert torch.allclose(explicit, analytic, atol=1e-12, rtol=0.0)


def test_sparse_feature_gram_matches_analytic_kernel() -> None:
    states = _tiny_states()
    features = sg23.build_explicit_features(states)
    gram, _seconds = sg23.explicit_dense_gram(features, block_size=3)
    analytic = sg19.plan_edge_kernel(states, states)
    assert torch.allclose(gram, analytic, atol=1e-12, rtol=0.0)


def test_block_pcg_matches_dense_weighted_solution() -> None:
    states = _tiny_states(12)
    features = sg23.build_explicit_features(states)
    dense_coefficients, _metrics, kernel, system = (
        sg23.dense_weighted_cholesky(states)
    )
    pcg_coefficients, metrics = sg23.block_pcg(
        features,
        states,
        preconditioner="return_phase_block",
        dense_system=system,
        relative_tolerance=1e-10,
        max_iterations=128,
    )
    assert metrics["converged"]
    assert torch.allclose(
        kernel @ pcg_coefficients,
        kernel @ dense_coefficients,
        atol=1e-7,
        rtol=0.0,
    )


def test_full_rank_spectral_pcg_matches_dense_solution() -> None:
    states = _tiny_states(12)
    features = sg23.build_explicit_features(states)
    dense_coefficients, _metrics, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    pcg_coefficients, metrics = sg23.block_pcg(
        features,
        states,
        preconditioner="spectral_12",
        relative_tolerance=1e-10,
        max_iterations=64,
    )
    assert metrics["converged"]
    assert torch.allclose(
        kernel @ pcg_coefficients,
        kernel @ dense_coefficients,
        atol=1e-7,
        rtol=0.0,
    )


def test_spectral_iterative_refinement_uses_true_feature_residual() -> None:
    states = _tiny_states(12)
    features = sg23.build_explicit_features(states)
    dense_coefficients, _dense, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    refined_coefficients, metrics = sg23.spectral_iterative_refinement(
        features,
        states,
        rank=12,
        relative_tolerance=1e-10,
        maximum_refinements=4,
    )
    assert metrics["converged"]
    assert torch.allclose(
        kernel @ refined_coefficients,
        kernel @ dense_coefficients,
        atol=1e-7,
        rtol=0.0,
    )


def test_primal_pcg_matches_dense_kernel_predictions() -> None:
    states = _tiny_states(12)
    features = sg23.build_explicit_features(states)
    dense_coefficients, _dense, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    weights, metrics = sg23.primal_pcg(
        features,
        states,
        relative_tolerance=1e-10,
        max_iterations=128,
    )
    assert metrics["converged"]
    assert torch.allclose(
        torch.sparse.mm(features.matrix, weights),
        kernel @ dense_coefficients,
        atol=1e-7,
        rtol=0.0,
    )


def test_online_block_cholesky_matches_batch() -> None:
    states = _tiny_states(12)
    dense_coefficients, _metrics, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    online_coefficients, online = sg23.online_block_cholesky(
        states, block_size=5
    )
    assert online["block_count"] == 3
    assert torch.allclose(
        kernel @ online_coefficients,
        kernel @ dense_coefficients,
        atol=1e-7,
        rtol=0.0,
    )


def test_pivoted_cholesky_reconstructs_at_full_rank() -> None:
    states = _tiny_states(10)
    kernel = sg19.plan_edge_kernel(states, states)
    factor, pivots, _times, metrics = sg23.pivoted_cholesky(
        kernel, maximum_rank=10
    )
    assert len(pivots) == metrics["effective_rank"]
    assert torch.allclose(factor @ factor.T, kernel, atol=1e-9, rtol=0.0)
    pivot_factor = factor.index_select(0, torch.tensor(pivots))
    assert bool((torch.diagonal(pivot_factor) > 0.0).all())


def test_stable_effective_rank_solution_matches_dense_predictions() -> None:
    states = _tiny_states(12)
    dense_coefficients, _dense, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    factor, pivots, _times, metrics = sg23.pivoted_cholesky(
        kernel, maximum_rank=12
    )
    assert metrics["reconstruction_max_abs_error_at_effective_rank"] <= 1e-9
    coefficients, gamma, solve = sg23.low_rank_weighted_solution(
        factor, pivots, states
    )
    pivot_index = torch.tensor(pivots, dtype=torch.long)
    landmark_scores = kernel.index_select(1, pivot_index) @ gamma
    dense_scores = kernel @ dense_coefficients
    assert solve["avoids_dual_lambda_subtraction"]
    assert solve["maximum_small_system_relative_residual"] <= 1e-9
    assert torch.allclose(kernel @ coefficients, dense_scores, atol=1e-7, rtol=0.0)
    assert torch.allclose(landmark_scores, dense_scores, atol=1e-7, rtol=0.0)


def test_matrix_free_pivoted_columns_match_dense_factor() -> None:
    states = _tiny_states(10)
    kernel = sg19.plan_edge_kernel(states, states)
    dense_factor, dense_pivots, _times, _metrics = sg23.pivoted_cholesky(
        kernel, maximum_rank=6
    )
    matrix_free_factor, matrix_free_pivots, _metrics = (
        sg23.pivoted_cholesky_from_states(states, maximum_rank=6)
    )
    assert matrix_free_pivots == dense_pivots
    assert torch.allclose(
        matrix_free_factor, dense_factor, atol=1e-10, rtol=0.0
    )


def test_stress_generator_is_deterministic_and_unique() -> None:
    states = _tiny_states(16)
    first, first_audit = sg23.generate_stress_states(states, 64)
    second, second_audit = sg23.generate_stress_states(states, 64)
    assert first_audit == second_audit
    assert first_audit["unique_state_count"] == 64
    assert torch.equal(first["keys"], second["keys"])


def test_low_rank_rollout_error_fails_quality_closed() -> None:
    assert not sg23._quality_is_perfect(
        {"rollout": {"evaluation_error": {"type": "AssertionError"}}}
    )
