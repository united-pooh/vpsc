from __future__ import annotations

from experiments.e3_ldaa2_diagonal_model_validation import decide, query_indices


def test_query_indices_are_sorted_unique_and_include_late_sequence() -> None:
    indices = query_indices(128, 1 / 32)
    assert indices.tolist() == sorted(set(indices.tolist()))
    assert indices.numel() == 4
    assert int(indices[-1]) == 127


def _operator(speedup=1.5, storage=0.25, exact=True):
    return {
        "exactness": {"passed": exact},
        "segmented_speedup_vs_bptt": speedup,
        "segmented_storage_ratio_to_bptt": storage,
    }


def _training(delta=0.10, parameter=1e-3):
    return [{
        "segmented_minus_bptt_valid_nll": delta,
        "parameter_max_abs_after_training": parameter,
    }]


def test_decision_passes_frozen_boundaries() -> None:
    result = decide(_operator(), _training())
    assert result["overall"] == "PASS"
    assert result["verdict"] == "SECOND_CORE_MODEL_GO_DISPATCHER_REQUIRED"


def test_decision_fails_speed_storage_quality_and_trajectory() -> None:
    assert decide(_operator(speedup=1.49), _training())["overall"] == "FAIL"
    assert decide(_operator(storage=0.251), _training())["overall"] == "FAIL"
    assert decide(_operator(), _training(delta=0.101))["overall"] == "FAIL"
    assert decide(_operator(), _training(parameter=0.00101))["overall"] == "FAIL"
