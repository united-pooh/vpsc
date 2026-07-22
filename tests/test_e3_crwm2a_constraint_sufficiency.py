from __future__ import annotations

import pytest

from experiments.e3_crwm2a_constraint_sufficiency import (
    decide,
    objective_only_action,
)


def test_objective_only_action_never_receives_candidates() -> None:
    plan = ("go east", "go north", "take coin")
    assert objective_only_action(plan, 0) == "go east"
    assert objective_only_action(plan, 2) == "take coin"
    assert objective_only_action(plan, 3) is None
    with pytest.raises(ValueError):
        objective_only_action(plan, -1)


def _metrics(*, win_rate: float, invalid: float = 0.0, exact: float = 1.0):
    return {
        "win_rate": win_rate,
        "invalid_action_rate": invalid,
        "plan_walkthrough_exact_rate": exact,
    }


def test_constraint_ceiling_stops_textworld_model_matrix() -> None:
    decision = decide(
        True,
        {
            "2": _metrics(win_rate=0.0),
            "8": _metrics(win_rate=1.0),
            "32": _metrics(win_rate=1.0),
        },
    )
    assert decision["residual_value_task_identifiable"] is False
    assert decision["verdict"] == "STOP_TEXTWORLD_TASK_CEILING_PIVOT_ENVIRONMENT"


@pytest.mark.parametrize(
    "h8,h32",
    (
        (_metrics(win_rate=0.875), _metrics(win_rate=1.0)),
        (_metrics(win_rate=1.0, invalid=0.01), _metrics(win_rate=1.0)),
        (_metrics(win_rate=1.0, exact=0.875), _metrics(win_rate=1.0)),
    ),
)
def test_nontrivial_task_proceeds_to_matched_matrix(h8, h32) -> None:
    decision = decide(
        True,
        {"2": _metrics(win_rate=0.0), "8": h8, "32": h32},
    )
    assert decision["residual_value_task_identifiable"] is True
    assert decision["verdict"] == "PROCEED_CRWM2_MATCHED_MATRIX"


def test_data_identity_is_a_hard_failure() -> None:
    decision = decide(
        False,
        {
            "2": _metrics(win_rate=0.0),
            "8": _metrics(win_rate=1.0),
            "32": _metrics(win_rate=1.0),
        },
    )
    assert decision["overall"] == "FAIL"
    assert decision["verdict"] == "STOP_DATA_IDENTITY_FAILURE"
