from __future__ import annotations

from experiments import e3_sg10_multichannel_delta as sg10
from experiments.e3_sg21_episodic_edge_graph import (
    EdgeBinding,
    GraphState,
    plan_path_constraint_mask,
    project_graph_transition,
)


ACTION_ORDER = (
    "examine coin",
    "go east",
    "go north",
    "go south",
    "go west",
    "inventory",
    "look",
    "take coin",
)


def _base_indices() -> tuple[int, int, int, int]:
    return (
        sg10.ROOM_LABELS.index("<room_novel>"),
        sg10.REWARD_LABELS.index("<reward_zero>"),
        sg10.DONE_LABELS.index("<continue>"),
        sg10.EXIT_LABELS.index("<exit_count_1>"),
    )


def test_known_edge_projects_room_mask_and_exit_only() -> None:
    mask_a = (0, 1, 0, 0, 0, 1, 1, 0)
    mask_b = (0, 0, 1, 1, 0, 1, 1, 0)
    graph = GraphState(
        current_room="room:a",
        node_masks={"room:a": mask_a, "room:b": mask_b},
        node_seen_steps={"room:a": 0, "room:b": 1},
        edges={
            ("room:a", "go east"): EdgeBinding(
                "room:b", 1, "observed_forward"
            )
        },
    )

    prediction, mask, next_graph, audit = project_graph_transition(
        _base_indices(),
        (0,) * len(ACTION_ORDER),
        graph,
        "go east",
        ACTION_ORDER,
        branch_tag="known",
    )

    assert prediction[0] == sg10.ROOM_LABELS.index("<room_previous>")
    assert prediction[1:3] == _base_indices()[1:3]
    assert prediction[3] == sg10.EXIT_LABELS.index("<exit_count_2>")
    assert mask == mask_b
    assert next_graph.current_room == "room:b"
    assert audit["kind"] == "known_edge_projection"


def test_unknown_move_writes_imagined_inverse_for_next_step() -> None:
    mask_a = (0, 1, 0, 0, 0, 1, 1, 0)
    predicted_mask = (0, 0, 0, 1, 0, 1, 1, 0)
    graph = GraphState(
        current_room="room:a",
        node_masks={"room:a": mask_a},
        node_seen_steps={"room:a": 0},
        edges={},
    )

    first_prediction, first_mask, imagined_graph, first_audit = (
        project_graph_transition(
            _base_indices(),
            predicted_mask,
            graph,
            "go east",
            ACTION_ORDER,
            branch_tag="imagined",
        )
    )
    second_prediction, second_mask, returned_graph, second_audit = (
        project_graph_transition(
            _base_indices(),
            predicted_mask,
            imagined_graph,
            "go west",
            ACTION_ORDER,
            branch_tag="return",
        )
    )

    assert first_prediction == _base_indices()
    assert first_mask == predicted_mask
    assert first_audit["kind"] == "imagined_edge_residual"
    assert second_prediction[0] == sg10.ROOM_LABELS.index("<room_previous>")
    assert second_mask == mask_a
    assert returned_graph.current_room == "room:a"
    assert second_audit["edge_kind"] == "imagined_inverse"


def test_stationary_action_preserves_current_graph_mask() -> None:
    mask = (0, 1, 0, 1, 0, 1, 1, 0)
    graph = GraphState(
        current_room="room:a",
        node_masks={"room:a": mask},
        node_seen_steps={"room:a": 0},
        edges={},
    )

    prediction, projected_mask, next_graph, audit = project_graph_transition(
        _base_indices(),
        (0,) * len(ACTION_ORDER),
        graph,
        "inventory",
        ACTION_ORDER,
        branch_tag="hold",
    )

    assert prediction[:3] == _base_indices()[:3]
    assert prediction[3] == _base_indices()[3]
    assert projected_mask == mask
    assert next_graph.current_room == "room:a"
    assert audit["kind"] == "stationary_hold"
    assert audit["overrode_mask_exit"] is False


def test_imagined_stationary_mask_never_overrides_learned_exit() -> None:
    three_exit_mask = (0, 1, 1, 1, 0, 1, 1, 0)
    graph = GraphState(
        current_room="imagined:node",
        node_masks={"imagined:node": three_exit_mask},
        node_seen_steps={"imagined:node": -1},
        edges={},
    )

    prediction, projected_mask, _next_graph, audit = project_graph_transition(
        _base_indices(),
        (0,) * len(ACTION_ORDER),
        graph,
        "inventory",
        ACTION_ORDER,
        branch_tag="unsupported-exit",
    )

    assert prediction[3] == _base_indices()[3]
    assert projected_mask == three_exit_mask
    assert audit["overrode_mask_exit"] is False


def test_plan_path_constraint_encodes_inverse_and_next_plan_only() -> None:
    plan = ("go west", "go north", "go east", "go south", "take coin")

    mask = plan_path_constraint_mask("go east", plan, 2, ACTION_ORDER)

    assert mask == (0, 0, 0, 1, 1, 1, 1, 0)


def test_unknown_plan_edge_applies_constraint_before_imagined_binding() -> None:
    graph = GraphState(
        current_room="room:a",
        node_masks={"room:a": (0, 1, 0, 0, 0, 1, 1, 0)},
        node_seen_steps={"room:a": 0},
        edges={},
    )
    plan = ("go east", "go south", "go west", "go north", "take coin")

    prediction, mask, next_graph, audit = project_graph_transition(
        _base_indices(),
        (0,) * len(ACTION_ORDER),
        graph,
        "go east",
        ACTION_ORDER,
        branch_tag="constraint",
        plan=plan,
        phase=0,
        enforce_plan_path_constraint=True,
    )

    assert mask == (0, 0, 0, 1, 1, 1, 1, 0)
    assert prediction[3] == sg10.EXIT_LABELS.index("<exit_count_2>")
    assert next_graph.node_masks[next_graph.current_room] == mask
    assert audit["kind"] == "plan_path_constraint"
