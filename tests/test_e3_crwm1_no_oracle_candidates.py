from __future__ import annotations

import pytest

from experiments.e3_crwm1_no_oracle_candidates import (
    candidate_actions_from_mask,
    candidate_set_counts,
    decide,
)


ACTION_ORDER = ("a", "b", "c", "d")


def test_candidate_actions_are_derived_only_from_mask_order() -> None:
    assert candidate_actions_from_mask((0, 1, 0, 1), ACTION_ORDER) == (
        "b",
        "d",
    )


def test_candidate_mask_rejects_nonbinary_or_wrong_length() -> None:
    with pytest.raises(ValueError):
        candidate_actions_from_mask((1, 0), ACTION_ORDER)
    with pytest.raises(ValueError):
        candidate_actions_from_mask((1, 0, 2, 0), ACTION_ORDER)


def test_candidate_set_counts_penalize_invalid_and_omitted_actions() -> None:
    assert candidate_set_counts(("a", "b"), ("b", "c")) == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "exact": False,
    }


def _rollout(**updates):
    result = {
        "information_flow_audit": {
            "proposal_materialized_before_evaluator_targets": True,
            "teacher_context_call_count": 0,
            "future_oracle_candidate_proposal_count": 0,
        },
        "candidate_precision": 0.95,
        "candidate_recall": 0.95,
        "candidate_set_exact_accuracy": 0.95,
        "no_oracle_second_exact_accuracy": 0.95,
        "invalid_transition_rate": 0.05,
    }
    result.update(updates)
    return result


def test_frozen_decision_passes_exact_boundary() -> None:
    decision = decide(True, _rollout())
    assert decision["overall"] == "PASS"
    assert decision["verdict"] == "PHASE1_GO_LONG_HORIZON_REQUIRED"


@pytest.mark.parametrize(
    "updates",
    (
        {"candidate_precision": 0.949},
        {"candidate_recall": 0.949},
        {"candidate_set_exact_accuracy": 0.949},
        {"no_oracle_second_exact_accuracy": 0.949},
        {"invalid_transition_rate": 0.051},
    ),
)
def test_frozen_decision_fails_each_quality_gate(updates) -> None:
    decision = decide(True, _rollout(**updates))
    assert decision["overall"] == "FAIL"
    assert decision["verdict"] == "STOP_ORACLE_DEPENDENCE"


def test_information_flow_audit_is_a_hard_gate() -> None:
    rollout = _rollout()
    rollout["information_flow_audit"]["teacher_context_call_count"] = 1
    assert decide(True, rollout)["overall"] == "FAIL"
