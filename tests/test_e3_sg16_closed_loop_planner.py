from __future__ import annotations

import torch

from experiments import e3_sg10_multichannel_delta as sg10
from experiments.e3_sg16_closed_loop_planner import (
    CandidateBranch,
    decode_prediction,
    reconcile_topology,
    select_candidate,
)


def _scores(
    *,
    room: str,
    reward: str = sg10.REWARD_LABELS[0],
    done: str = sg10.DONE_LABELS[0],
    exits: str = sg10.EXIT_LABELS[1],
    scale: float = 1.0,
) -> torch.Tensor:
    selected = {
        "room_relation": room,
        "reward": reward,
        "done": done,
        "move_exit_count_after": exits,
    }
    values = -torch.ones(1, sg10.TOTAL_LOGITS) * scale
    for name, labels in sg10.CHANNEL_SPECS:
        start, _stop = sg10.CHANNEL_OFFSETS[name]
        values[0, start + labels.index(selected[name])] = scale
    return values


def test_reward_is_selected_before_novel_room() -> None:
    novel = CandidateBranch(
        "go north", _scores(room=sg10.ROOM_LABELS[1]), 0.1
    )
    terminal = CandidateBranch(
        "take coin",
        _scores(
            room=sg10.ROOM_LABELS[0],
            reward=sg10.REWARD_LABELS[1],
            done=sg10.DONE_LABELS[1],
        ),
        0.1,
    )
    selected, summaries = select_candidate((novel, terminal))
    assert selected.action == "take coin"
    assert summaries["take coin"].semantic_priority > summaries["go north"].semantic_priority


def test_room_priority_margin_and_lexical_tie_break_are_frozen() -> None:
    previous = decode_prediction(_scores(room=sg10.ROOM_LABELS[2]))
    same = decode_prediction(_scores(room=sg10.ROOM_LABELS[3]))
    novel = decode_prediction(_scores(room=sg10.ROOM_LABELS[1]))
    assert previous.semantic_priority < same.semantic_priority < novel.semantic_priority

    branches = (
        CandidateBranch("look", _scores(room=sg10.ROOM_LABELS[3]), 0.1),
        CandidateBranch("inventory", _scores(room=sg10.ROOM_LABELS[3]), 0.1),
    )
    selected, _summaries = select_candidate(branches)
    assert selected.action == "inventory"


def test_confidence_breaks_equal_semantic_predictions_before_action_name() -> None:
    weak = CandidateBranch(
        "aaa", _scores(room=sg10.ROOM_LABELS[1], scale=1.0), 0.1
    )
    strong = CandidateBranch(
        "zzz", _scores(room=sg10.ROOM_LABELS[1], scale=2.0), 0.1
    )
    selected, _summaries = select_candidate((weak, strong))
    assert selected.action == "zzz"


def test_real_observation_topology_push_hold_and_rollback() -> None:
    rooms = ("room:a", "room:b", "room:c")
    relation, unchanged = reconcile_topology(rooms, None)
    assert relation == sg10.ROOM_LABELS[0]
    assert unchanged == rooms

    relation, unchanged = reconcile_topology(rooms, "room:c")
    assert relation == sg10.ROOM_LABELS[3]
    assert unchanged == rooms

    relation, rolled_back = reconcile_topology(rooms, "room:a")
    assert relation == sg10.ROOM_LABELS[2]
    assert rolled_back == ("room:a",)

    relation, pushed = reconcile_topology(rooms, "room:d")
    assert relation == sg10.ROOM_LABELS[1]
    assert pushed == rooms + ("room:d",)
