from __future__ import annotations

from types import SimpleNamespace

from experiments import e3_sg10_multichannel_delta as sg10
from experiments.e3_sg16_closed_loop_planner import PredictionSummary
from experiments.e3_sg17_two_step_rollout import (
    _canonical_fingerprint,
    _rollout_metrics,
    _target_indices,
    _true_context_after,
    imagined_context_after,
    repair_persistent_room_semantics,
)


def _prediction(room: str, *, done: bool = False) -> PredictionSummary:
    return PredictionSummary(
        labels=(
            room,
            sg10.REWARD_LABELS[0],
            sg10.DONE_LABELS[1] if done else sg10.DONE_LABELS[0],
            sg10.EXIT_LABELS[1],
        ),
        semantic_priority=(0, int(done), 0),
        confidence_margin=1.0,
    )


def test_imagined_context_routes_push_pop_hold_and_terminal_stop() -> None:
    context = ("go north", "go east")
    assert imagined_context_after(
        context, "go south", _prediction(sg10.ROOM_LABELS[1])
    ) == context + ("go south",)
    assert imagined_context_after(
        context, "go west", _prediction(sg10.ROOM_LABELS[2])
    ) == ("go north",)
    assert imagined_context_after(
        context, "look", _prediction(sg10.ROOM_LABELS[3])
    ) == context
    assert imagined_context_after(
        context, "inventory", _prediction(sg10.ROOM_LABELS[0])
    ) == context
    assert (
        imagined_context_after(
            context, "take coin", _prediction(sg10.ROOM_LABELS[0], done=True)
        )
        is None
    )


def test_true_previous_room_can_rollback_more_than_one_depth() -> None:
    context = ("go north", "go east", "go south")
    rooms = ("room:a", "room:b", "room:c", "room:d")
    assert _true_context_after(
        context,
        "go west",
        sg10.ROOM_LABELS[2],
        rooms,
        "room:b",
    ) == ("go north",)


def test_rollout_metrics_count_premature_missing_prediction_as_wrong() -> None:
    target = (1, 0, 0, 2)
    metrics = _rollout_metrics((target, None), (target, target))
    assert metrics["example_count"] == 2
    assert metrics["exact_vector_accuracy"] == 0.5
    assert metrics["macro_channel_accuracy"] == 0.5
    assert metrics["missing_prediction_count"] == 1


def test_canonical_tree_fingerprint_is_order_stable_for_mappings() -> None:
    assert _canonical_fingerprint({"b": 2, "a": [1, 3]}) == _canonical_fingerprint(
        {"a": [1, 3], "b": 2}
    )


def test_terminal_target_has_no_actionable_exit_even_if_backend_leaks_commands(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sg10,
        "_room_relation_label",
        lambda *_args, **_kwargs: sg10.ROOM_LABELS[0],
    )
    transition = SimpleNamespace(
        next_observation="terminal",
        reward=1.0,
        done=True,
        info={"admissible_commands": ("go east",)},
    )
    target = _target_indices(None, transition, "room", ())
    assert target[3] == sg10.EXIT_LABELS.index("<exit_count_0>")


def test_repair_persistent_room_semantics_changes_only_legacy_look_target() -> None:
    previous = sg10.ROOM_LABELS.index("<room_previous>")
    same = sg10.ROOM_LABELS.index("<room_same>")
    tree = {
        "canonical_tree_sha256": "legacy",
        "games": ({"seed": 1},),
        "first_records": (
            {
                "actual_relation": sg10.ROOM_LABELS[0],
                "seconds": (
                    {
                        "pair_id": "repair-me",
                        "action": "look",
                        "target_indices": (previous, 0, 0, 0),
                    },
                    {
                        "pair_id": "leave-me",
                        "action": "inventory",
                        "target_indices": (previous, 0, 0, 0),
                    },
                ),
            },
            {
                "actual_relation": "<room_novel>",
                "seconds": (
                    {
                        "pair_id": "also-leave-me",
                        "action": "look",
                        "target_indices": (previous, 0, 0, 0),
                    },
                ),
            },
        ),
    }

    repaired, audit = repair_persistent_room_semantics(tree)

    assert tree["first_records"][0]["seconds"][0]["target_indices"][0] == previous
    assert repaired["first_records"][0]["seconds"][0]["target_indices"][0] == same
    assert repaired["first_records"][0]["seconds"][1]["target_indices"][0] == previous
    assert repaired["first_records"][1]["seconds"][0]["target_indices"][0] == previous
    assert audit["changed_pair_ids"] == ("repair-me",)
    assert audit["changed_pair_count"] == 1
    assert audit["source_tree_sha256"] == "legacy"
    assert audit["repaired_tree_sha256"] != "legacy"
