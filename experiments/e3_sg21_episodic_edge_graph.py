"""SG21 sparse episodic edge spikes over the SG19 learned residual."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_sg17_two_step_rollout as sg17  # noqa: E402
from experiments import e3_sg18_affordance_weighted_krr as sg18  # noqa: E402
from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    build_action_alphabet,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, file_sha256  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg21_episodic_edge_graph.json")
DEFAULT_SG19_REFERENCE = Path("results/e3_scan/e3_sg19_plan_edge_spikes.json")
SG19_REFERENCE_SHA256 = (
    "EA2C855CDC9ECA0D41B8345B3CF3918F4BD2BADCCCA0B35F17DFE1F104505C22"
)
SG19_EXPERIMENT = "E3-SG19 objective plan tape and visited-edge spikes"
MOVE_ACTIONS = tuple(sorted(sg19.INVERSE_MOVE))
STATIONARY_ACTIONS = ("examine coin", "inventory", "look")


@dataclass(frozen=True)
class EdgeBinding:
    destination: str
    bound_step: int
    kind: str


@dataclass
class GraphState:
    current_room: Optional[str]
    node_masks: MutableMapping[str, Tuple[int, ...]]
    node_seen_steps: MutableMapping[str, int]
    edges: MutableMapping[Tuple[str, str], EdgeBinding]
    imagined_counter: int = 0

    def clone(self) -> "GraphState":
        return GraphState(
            current_room=self.current_room,
            node_masks=dict(self.node_masks),
            node_seen_steps=dict(self.node_seen_steps),
            edges=dict(self.edges),
            imagined_counter=self.imagined_counter,
        )


def _action_mask(
    actions: Sequence[str], action_order: Sequence[str]
) -> Tuple[int, ...]:
    available = {str(action) for action in actions}
    return tuple(int(action in available) for action in action_order)


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest().upper()


def _snapshot_payload(state: GraphState, root_step: int) -> Dict[str, Any]:
    return {
        "root_step": root_step,
        "current_room": state.current_room,
        "nodes": tuple(
            sorted(
                (
                    room,
                    state.node_seen_steps[room],
                    tuple(state.node_masks[room]),
                )
                for room in state.node_masks
            )
        ),
        "edges": tuple(
            sorted(
                (
                    source,
                    action,
                    binding.destination,
                    binding.bound_step,
                    binding.kind,
                )
                for (source, action), binding in state.edges.items()
            )
        ),
    }


def build_graph_snapshots(
    corpus_root: Path,
    corpus: Any,
    exhaustive: Mapping[str, Any],
    action_order: Sequence[str],
) -> Tuple[Dict[str, Dict[Tuple[int, int], GraphState]], Dict[str, Any]]:
    snapshots: Dict[str, Dict[Tuple[int, int], GraphState]] = {}
    split_audits = {}
    all_fingerprints = []
    for split in SPLITS:
        started = time.perf_counter_ns()
        cache_masks: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        for record in exhaustive[split]["records"]:
            key = (int(record["game_seed"]), int(record["root_step"]))
            mask = tuple(int(value) for value in record["current_mask"])
            existing = cache_masks.setdefault(key, mask)
            if existing != mask:
                raise AssertionError(f"inconsistent exhaustive root mask {key}")
        split_snapshots: Dict[Tuple[int, int], GraphState] = {}
        mask_mismatches = []
        leakage_violations = []
        room_failures = []
        edge_conflicts = []
        game_count = 0
        lines = (corpus_root / split / "episodes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        for line in lines:
            episode = json.loads(line)
            seed = int(episode["seed"])
            game_count += 1
            state = GraphState(None, {}, {}, {})
            for step in episode["steps"]:
                root_step = int(step["step"])
                room = sg10._room_feature(corpus, str(step["observation"]))
                if room is None:
                    room_failures.append((seed, root_step, "current"))
                    continue
                current_mask = _action_mask(
                    step["admissible_actions"], action_order
                )
                state.current_room = room
                state.node_masks[room] = current_mask
                state.node_seen_steps.setdefault(room, root_step)
                snapshot = state.clone()
                split_snapshots[(seed, root_step)] = snapshot
                cached_mask = cache_masks[(seed, root_step)]
                if cached_mask != current_mask:
                    mask_mismatches.append(
                        (seed, root_step, cached_mask, current_mask)
                    )
                for (source, action), binding in snapshot.edges.items():
                    if binding.bound_step >= root_step:
                        leakage_violations.append(
                            (
                                seed,
                                root_step,
                                source,
                                action,
                                binding.bound_step,
                            )
                        )
                payload = _snapshot_payload(snapshot, root_step)
                all_fingerprints.append((split, seed, _canonical_sha(payload)))

                factual_action = str(step["action"])
                next_room = sg10._room_feature(corpus, str(step["next_obs"]))
                if factual_action not in sg19.INVERSE_MOVE:
                    continue
                if next_room is None:
                    room_failures.append((seed, root_step, "move_next"))
                    continue
                bindings = (
                    (
                        (room, factual_action),
                        EdgeBinding(next_room, root_step, "observed_forward"),
                    ),
                    (
                        (next_room, sg19.INVERSE_MOVE[factual_action]),
                        EdgeBinding(room, root_step, "observed_inverse"),
                    ),
                )
                for edge_key, binding in bindings:
                    existing = state.edges.get(edge_key)
                    if existing is not None and existing.destination != binding.destination:
                        edge_conflicts.append(
                            (
                                seed,
                                root_step,
                                edge_key,
                                existing.destination,
                                binding.destination,
                            )
                        )
                    state.edges[edge_key] = binding
        elapsed = (time.perf_counter_ns() - started) / 1e9
        snapshots[split] = split_snapshots
        node_counts = [len(state.node_masks) for state in split_snapshots.values()]
        edge_counts = [len(state.edges) for state in split_snapshots.values()]
        logical_bytes = [
            8
            + len(state.node_masks) * (8 + 1)
            + len(state.edges) * (8 + 1 + 8)
            for state in split_snapshots.values()
        ]
        split_audits[split] = {
            "game_count": game_count,
            "snapshot_count": len(split_snapshots),
            "build_seconds": elapsed,
            "mask_mismatch_count": len(mask_mismatches),
            "mask_mismatches_first_20": tuple(mask_mismatches[:20]),
            "leakage_violation_count": len(leakage_violations),
            "leakage_violations_first_20": tuple(leakage_violations[:20]),
            "room_extraction_failure_count": len(room_failures),
            "room_failures_first_20": tuple(room_failures[:20]),
            "edge_conflict_count": len(edge_conflicts),
            "edge_conflicts_first_20": tuple(edge_conflicts[:20]),
            "maximum_node_count": max(node_counts),
            "maximum_edge_count": max(edge_counts),
            "maximum_logical_graph_bytes": max(logical_bytes),
        }
    return snapshots, {
        "source": "past factual observations and executed moves only",
        "inverse_binding": "same observed physical edge at binding step",
        "all_masks_match_exhaustive_cache": all(
            audit["mask_mismatch_count"] == 0 for audit in split_audits.values()
        ),
        "all_binding_steps_precede_snapshot_root": all(
            audit["leakage_violation_count"] == 0
            for audit in split_audits.values()
        ),
        "all_rooms_unique_and_present": all(
            audit["room_extraction_failure_count"] == 0
            for audit in split_audits.values()
        ),
        "no_edge_conflicts": all(
            audit["edge_conflict_count"] == 0 for audit in split_audits.values()
        ),
        "canonical_snapshot_sha256": _canonical_sha(all_fingerprints),
        "splits": split_audits,
    }


def _indices_from_scores(scores: torch.Tensor) -> Tuple[int, int, int, int]:
    return sg17._prediction_indices(
        sg16.decode_prediction(scores[:, : sg10.TOTAL_LOGITS])
    )


def _summary_from_indices(
    indices: Sequence[int], confidence: float
) -> sg16.PredictionSummary:
    labels = tuple(
        channel_labels[int(index)]
        for (_name, channel_labels), index in zip(sg10.CHANNEL_SPECS, indices)
    )
    return sg16.PredictionSummary(
        labels=labels,  # type: ignore[arg-type]
        semantic_priority=(
            int(labels[1] == sg10.REWARD_LABELS[1]),
            int(labels[2] == sg10.DONE_LABELS[1]),
            sg16.ROOM_PRIORITY[labels[0]],
        ),
        confidence_margin=confidence,
    )


def _exit_index_from_mask(
    mask: Sequence[int], action_order: Sequence[str]
) -> Optional[int]:
    count = sum(
        int(bit)
        for action, bit in zip(action_order, mask)
        if action in sg19.INVERSE_MOVE
    )
    return count if count < len(sg10.EXIT_LABELS) else None


def plan_path_constraint_mask(
    candidate_action: str,
    plan: Sequence[str],
    phase: int,
    action_order: Sequence[str],
) -> Optional[Tuple[int, ...]]:
    plan_current, plan_next = sg19._plan_slots(plan, phase)
    if (
        candidate_action not in sg19.INVERSE_MOVE
        or candidate_action != plan_current
    ):
        return None
    inverse = sg19.INVERSE_MOVE[candidate_action]
    return tuple(
        int(
            action == inverse
            or action == plan_next
            or action in ("inventory", "look")
            or (plan_next == "take coin" and action == "examine coin")
        )
        for action in action_order
    )


def project_graph_transition(
    base_indices: Sequence[int],
    predicted_mask: Sequence[int],
    graph: GraphState,
    candidate_action: str,
    action_order: Sequence[str],
    *,
    branch_tag: str,
    plan: Optional[Sequence[str]] = None,
    phase: Optional[int] = None,
    enforce_plan_path_constraint: bool = False,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, ...], GraphState, Dict[str, Any]]:
    projected = [int(value) for value in base_indices]
    mask = tuple(int(value) for value in predicted_mask)
    next_graph = graph.clone()
    current_room = graph.current_room
    if projected[2] == sg10.DONE_LABELS.index("<done>"):
        next_graph.current_room = None
        return tuple(projected), mask, next_graph, {
            "kind": "terminal",
            "edge_kind": None,
            "overrode_room": False,
            "overrode_mask_exit": False,
        }

    if candidate_action in STATIONARY_ACTIONS:
        if current_room is not None and current_room in graph.node_masks:
            mask = tuple(graph.node_masks[current_room])
        return tuple(projected), mask, next_graph, {
            "kind": "stationary_hold",
            "edge_kind": None,
            "overrode_room": False,
            "overrode_mask_exit": False,
        }

    if candidate_action not in sg19.INVERSE_MOVE or current_room is None:
        return tuple(projected), mask, next_graph, {
            "kind": "learned_residual",
            "edge_kind": None,
            "overrode_room": False,
            "overrode_mask_exit": False,
        }

    binding = graph.edges.get((current_room, candidate_action))
    if binding is not None:
        destination = binding.destination
        projected[0] = sg10.ROOM_LABELS.index("<room_previous>")
        if destination in graph.node_masks:
            mask = tuple(graph.node_masks[destination])
            exit_index = (
                _exit_index_from_mask(mask, action_order)
                if graph.node_seen_steps.get(destination, -1) >= 0
                else None
            )
            if exit_index is not None:
                projected[3] = exit_index
            mask_exit = exit_index is not None
        else:
            mask_exit = False
        next_graph.current_room = destination
        return tuple(projected), mask, next_graph, {
            "kind": "known_edge_projection",
            "edge_kind": binding.kind,
            "overrode_room": True,
            "overrode_mask_exit": mask_exit,
        }

    constraint_applied = False
    if enforce_plan_path_constraint:
        if plan is None or phase is None:
            raise ValueError("plan-path constraint requires plan and phase")
        constrained_mask = plan_path_constraint_mask(
            candidate_action, plan, phase, action_order
        )
        if constrained_mask is not None:
            mask = constrained_mask
            exit_index = _exit_index_from_mask(mask, action_order)
            if exit_index is None:
                raise AssertionError("plan-path mask has unsupported exit count")
            projected[3] = exit_index
            constraint_applied = True

    imagined = f"imagined:{branch_tag}:{graph.imagined_counter}"
    next_graph.imagined_counter += 1
    next_graph.current_room = imagined
    next_graph.node_masks[imagined] = mask
    next_graph.node_seen_steps[imagined] = -1
    next_graph.edges[(current_room, candidate_action)] = EdgeBinding(
        imagined, -1, "imagined_forward"
    )
    next_graph.edges[(imagined, sg19.INVERSE_MOVE[candidate_action])] = EdgeBinding(
        current_room, -1, "imagined_inverse"
    )
    return tuple(projected), mask, next_graph, {
        "kind": (
            "plan_path_constraint"
            if constraint_applied
            else "imagined_edge_residual"
        ),
        "edge_kind": "imagined_forward",
        "overrode_room": False,
        "overrode_mask_exit": constraint_applied,
    }


def evaluate_split_with_graph(
    split_name: str,
    split: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    snapshots: Mapping[Tuple[int, int], GraphState],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    action_order: Sequence[str],
    plans: Optional[Mapping[int, Sequence[str]]] = None,
    enforce_plan_path_constraint: bool = False,
) -> Dict[str, Any]:
    scores = sg19.plan_edge_kernel(split, unique) @ coefficients
    predictions = []
    masks = []
    projection_kinds: Dict[str, int] = {}
    for index, record in enumerate(records):
        seed = int(record["game_seed"])
        root_step = int(record["root_step"])
        snapshot = snapshots[(seed, root_step)]
        base_indices = _indices_from_scores(scores[index : index + 1])
        predicted_mask = tuple(
            int(value)
            for value in (
                scores[index, sg18.NEXT_MASK_OFFSET :]
                > sg18.MASK_DECISION_THRESHOLD
            ).tolist()
        )
        prediction, projected_mask, _next_graph, projection = (
            project_graph_transition(
                base_indices,
                predicted_mask,
                snapshot,
                str(record["candidate_action"]),
                action_order,
                branch_tag=f"{split_name}:{seed}:{root_step}",
                plan=(plans[seed] if plans is not None else None),
                phase=len(record["context_actions"]),
                enforce_plan_path_constraint=enforce_plan_path_constraint,
            )
        )
        predictions.append(prediction)
        masks.append(projected_mask)
        kind = str(projection["kind"])
        projection_kinds[kind] = projection_kinds.get(kind, 0) + 1
    mask_scores = torch.tensor(masks, dtype=torch.float64) * 2.0 - 1.0
    return {
        "delta": sg17._rollout_metrics(
            predictions,
            tuple(tuple(int(value) for value in row) for row in split["targets"]),
        ),
        "next_affordance": sg18._mask_metrics(
            mask_scores, split["next_masks"], action_order
        ),
        "projection_kind_counts": projection_kinds,
    }


def evaluate_two_step_with_graph(
    tree: Mapping[str, Any],
    exhaustive_test: Sequence[Mapping[str, Any]],
    snapshots: Mapping[Tuple[int, int], GraphState],
    plans: Mapping[int, Sequence[str]],
    *,
    alphabet_index: Mapping[str, int],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    action_order: Sequence[str],
    device: torch.device,
    score_fn=None,
    enforce_plan_path_constraint: bool = False,
) -> Dict[str, Any]:
    lookup = {
        (
            int(record["game_seed"]),
            int(record["root_step"]),
            str(record["candidate_action"]),
        ): record
        for record in exhaustive_test
    }
    if score_fn is None:
        runtime_unique = {
            name: tensor.to(
                device=device,
                dtype=(torch.float32 if name == "masks" else tensor.dtype),
            )
            for name, tensor in unique.items()
            if isinstance(tensor, torch.Tensor)
            and name
            in (
                "keys",
                "phases",
                "masks",
                "plan_current",
                "plan_next",
                "return_edges",
            )
        }
        runtime_coefficients = coefficients.to(
            device=device, dtype=torch.float32
        )

        def score_fn(
            context_actions: Sequence[str],
            current_mask: Sequence[int],
            candidate_action: str,
            plan: Sequence[str],
            last_move: Optional[str],
        ) -> Tuple[torch.Tensor, Tuple[int, ...], float]:
            return sg19._runtime_score(
                context_actions,
                current_mask,
                candidate_action,
                plan,
                last_move,
                alphabet_index=alphabet_index,
                unique=runtime_unique,
                coefficients=runtime_coefficients,
                device=device,
            )
    first_predictions = []
    first_targets = []
    first_mask_predictions = []
    first_mask_targets = []
    teacher_predictions = []
    self_predictions = []
    second_targets = []
    teacher_timings = []
    self_timings = []
    projection_kinds: Dict[str, int] = {}
    first_projection_kinds: Dict[str, int] = {}
    routing_correct = 0
    routed_count = 0
    premature = 0
    errors = []
    for first in tree["first_records"]:
        seed = int(first["game_seed"])
        root_step = int(first["root_step"])
        action = str(first["action"])
        root_record = lookup[(seed, root_step, action)]
        context = tuple(first["context_actions"])
        root_last_move = sg19._last_move(context)
        first_scores, predicted_mask, model_elapsed = score_fn(
            context,
            root_record["current_mask"],
            action,
            plans[seed],
            root_last_move,
        )
        base_summary = sg16.decode_prediction(
            first_scores[:, : sg10.TOTAL_LOGITS]
        )
        projection_started = time.perf_counter_ns()
        first_prediction, first_mask, branch_graph, first_projection = (
            project_graph_transition(
                sg17._prediction_indices(base_summary),
                predicted_mask,
                snapshots[(seed, root_step)],
                action,
                action_order,
                branch_tag=f"{seed}:{root_step}:{action}",
                plan=plans[seed],
                phase=len(context),
                enforce_plan_path_constraint=enforce_plan_path_constraint,
            )
        )
        first_elapsed = model_elapsed + (
            time.perf_counter_ns() - projection_started
        ) / 1e6
        first_predictions.append(first_prediction)
        first_targets.append(tuple(first["target_indices"]))
        first_mask_predictions.append(first_mask)
        first_mask_targets.append(tuple(root_record["next_mask"]))
        first_kind = str(first_projection["kind"])
        first_projection_kinds[first_kind] = (
            first_projection_kinds.get(first_kind, 0) + 1
        )
        if not first["seconds"]:
            continue
        projected_summary = _summary_from_indices(
            first_prediction, base_summary.confidence_margin
        )
        true_context = tuple(first["true_context_actions"])
        self_context = sg17.imagined_context_after(
            context, action, projected_summary
        )
        next_last_move = (
            action if action in sg19.INVERSE_MOVE else root_last_move
        )
        routed_count += 1
        routing_correct += self_context == true_context
        if self_context is None:
            premature += 1
        for second in first["seconds"]:
            second_action = str(second["action"])
            teacher_scores, teacher_mask, teacher_model_elapsed = (
                score_fn(
                    true_context,
                    root_record["next_mask"],
                    second_action,
                    plans[seed],
                    next_last_move,
                )
            )
            projection_started = time.perf_counter_ns()
            teacher_prediction, _teacher_projected_mask, _teacher_graph, (
                teacher_projection
            ) = project_graph_transition(
                _indices_from_scores(teacher_scores),
                teacher_mask,
                branch_graph,
                second_action,
                action_order,
                branch_tag=f"teacher:{seed}:{root_step}:{action}:{second_action}",
                plan=plans[seed],
                phase=len(true_context),
                enforce_plan_path_constraint=enforce_plan_path_constraint,
            )
            teacher_elapsed = teacher_model_elapsed + (
                time.perf_counter_ns() - projection_started
            ) / 1e6
            target = tuple(second["target_indices"])
            teacher_predictions.append(teacher_prediction)
            second_targets.append(target)
            teacher_timings.append(first_elapsed + teacher_elapsed)
            kind = str(teacher_projection["kind"])
            projection_kinds[kind] = projection_kinds.get(kind, 0) + 1

            self_prediction = None
            self_elapsed = first_elapsed
            if self_context is not None:
                self_scores, self_mask, self_model_elapsed = score_fn(
                    self_context,
                    first_mask,
                    second_action,
                    plans[seed],
                    next_last_move,
                )
                projection_started = time.perf_counter_ns()
                self_prediction, _self_projected_mask, _self_graph, (
                    self_projection
                ) = project_graph_transition(
                    _indices_from_scores(self_scores),
                    self_mask,
                    branch_graph,
                    second_action,
                    action_order,
                    branch_tag=(
                        f"self:{seed}:{root_step}:{action}:{second_action}"
                    ),
                    plan=plans[seed],
                    phase=len(self_context),
                    enforce_plan_path_constraint=enforce_plan_path_constraint,
                )
                self_elapsed += self_model_elapsed + (
                    time.perf_counter_ns() - projection_started
                ) / 1e6
                if self_projection["kind"] != teacher_projection["kind"]:
                    raise AssertionError("teacher/self graph projection kind diverged")
            self_predictions.append(self_prediction)
            self_timings.append(self_elapsed)
            if self_prediction != target and len(errors) < 100:
                errors.append(
                    {
                        "pair_id": second["pair_id"],
                        "first_action": action,
                        "second_action": second_action,
                        "projection": teacher_projection,
                        "target_indices": target,
                        "teacher_prediction": teacher_prediction,
                        "self_prediction": self_prediction,
                    }
                )
    teacher_metrics = sg17._rollout_metrics(teacher_predictions, second_targets)
    self_metrics = sg17._rollout_metrics(self_predictions, second_targets)
    first_mask_tensor = torch.tensor(first_mask_targets, dtype=torch.bool)
    first_mask_scores = (
        torch.tensor(first_mask_predictions, dtype=torch.float64) * 2.0 - 1.0
    )
    return {
        "first_all_admissible": sg17._rollout_metrics(
            first_predictions, first_targets
        ),
        "first_next_affordance": sg18._mask_metrics(
            first_mask_scores, first_mask_tensor, action_order
        ),
        "teacher_forced_second": teacher_metrics,
        "self_rollout_second": self_metrics,
        "self_minus_teacher_exact": (
            self_metrics["exact_vector_accuracy"]
            - teacher_metrics["exact_vector_accuracy"]
        ),
        "first_routing_accuracy": routing_correct / routed_count,
        "premature_stop_first_branch_count": premature,
        "first_projection_kind_counts": first_projection_kinds,
        "second_projection_kind_counts": projection_kinds,
        "teacher_pair_timing": sg16._timing_summary(teacher_timings),
        "self_pair_timing": sg16._timing_summary(self_timings),
        "self_error_records_first_100": tuple(errors),
    }


def _decision(
    graph_audit: Mapping[str, Any],
    base_fit: Mapping[str, Any],
    graph_test: Mapping[str, Any],
    rollout: Mapping[str, Any],
    sg17_reference: Mapping[str, Any],
    logical_total_bytes: int,
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "no_leak_graph_gate": "SMOKE",
            "mechanism_gate": "SMOKE",
            "world_state_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "response_speed_gate": "SMOKE",
            "storage_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg21_episodic_edge_graph",
        }
    no_leak = (
        graph_audit["all_masks_match_exhaustive_cache"]
        and graph_audit["all_binding_steps_precede_snapshot_root"]
        and graph_audit["all_rooms_unique_and_present"]
        and graph_audit["no_edge_conflicts"]
    )
    mechanism = (
        rollout["teacher_forced_second"]["exact_vector_accuracy"] == 1.0
        and rollout["self_rollout_second"]["exact_vector_accuracy"] == 1.0
        and all(
            value == 1.0
            for value in rollout["self_rollout_second"][
                "channel_accuracy"
            ].values()
        )
        and rollout["self_minus_teacher_exact"] == 0.0
        and rollout["first_routing_accuracy"] == 1.0
        and rollout["premature_stop_first_branch_count"] == 0
    )
    world_state = (
        graph_test["delta"]["exact_vector_accuracy"] == 1.0
        and all(
            value == 1.0
            for value in graph_test["delta"]["channel_accuracy"].values()
        )
        and graph_test["next_affordance"]["bit_accuracy"] >= 0.98
        and graph_test["next_affordance"]["exact_mask_accuracy"] >= 0.95
    )
    graph_train_wall = graph_audit["splits"]["train"]["build_seconds"]
    total_train_wall = (
        base_fit["deployment_training_wall_seconds"] + graph_train_wall
    )
    ann_training = [
        replication["training"][name]["elapsed_seconds"]
        for replication in sg17_reference["replications"]
        for name in sg16.ANN_MODEL_NAMES
    ]
    training = total_train_wall < min(ann_training)
    response_records = []
    for replication in sg17_reference["replications"]:
        for name in sg16.ANN_MODEL_NAMES:
            ann = replication["rollout"][name]["teacher_pair_timing"]
            record = {
                "seed": replication["seed"],
                "ann_model": name,
                "snn_p50_ms": rollout["teacher_pair_timing"]["p50_ms"],
                "ann_p50_ms": ann["p50_ms"],
                "snn_p95_ms": rollout["teacher_pair_timing"]["p95_ms"],
                "ann_p95_ms": ann["p95_ms"],
            }
            record["passed"] = (
                record["snn_p50_ms"] <= record["ann_p50_ms"]
                and record["snn_p95_ms"] <= record["ann_p95_ms"]
            )
            response_records.append(record)
    response = all(record["passed"] for record in response_records)
    min_ann_bytes = min(
        replication["parameter_counts"][name] * 4
        for replication in sg17_reference["replications"]
        for name in sg16.ANN_MODEL_NAMES
    )
    storage = logical_total_bytes <= min_ann_bytes
    gates = {
        "no_leak_graph_gate": no_leak,
        "mechanism_gate": mechanism,
        "world_state_gate": world_state,
        "training_speed_gate": training,
        "response_speed_gate": response,
        "storage_gate": storage,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = "sg21r_sixth_fresh_matched_ann_confirmation"
    elif mechanism and not world_state:
        next_route = "sg22_sparse_observation_affordance_features"
    elif not no_leak:
        next_route = "sg21_environment_graph_identity_audit"
    else:
        next_route = "sg21_graph_projection_error_diagnostic"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "graph_plus_base_training_wall_seconds": total_train_wall,
        "response_comparisons": response_records,
        "next_route": next_route,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    sg19_reference, sg19_digest = sg16._load_reference(
        args.sg19_reference.expanduser().resolve(),
        SG19_REFERENCE_SHA256,
        SG19_EXPERIMENT,
    )
    sg18_reference, sg18_digest = sg16._load_reference(
        args.sg18_reference.expanduser().resolve(),
        sg19.SG18_REFERENCE_SHA256,
        sg19.SG18_EXPERIMENT,
    )
    sg17_reference, sg17_digest = sg16._load_reference(
        args.sg17_reference.expanduser().resolve(),
        sg19.SG17_REFERENCE_SHA256,
        sg19.SG17_EXPERIMENT,
    )
    cache_path = args.exhaustive_cache.expanduser().resolve()
    cache_digest = file_sha256(cache_path).upper()
    if cache_digest != sg19.EXHAUSTIVE_CACHE_SHA256:
        raise ValueError("SG21 exhaustive cache SHA mismatch")
    exhaustive = json.loads(cache_path.read_text(encoding="utf-8"))["exhaustive"]

    corpus = load_event_corpus(corpus_root)
    base_examples, vocabulary = sg10.build_multichannel_examples(
        corpus_root, corpus
    )
    alphabet = build_action_alphabet(base_examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_order = tuple(sg18_reference["configuration"]["action_order"])
    plans, plan_audit = sg19.load_objective_plans(corpus_root)
    tensors = {
        split: sg19.tensorize_extended(
            exhaustive[split]["records"],
            plans[split],
            alphabet_index=alphabet_index,
            device=device,
        )
        for split in SPLITS
    }
    unique = sg19.compress_extended(tensors["train"])
    coefficients, base_fit = sg19.fit_weighted_extended(
        tensors["train"], unique, device=device
    )
    base_split_metrics = {
        split: sg19.evaluate_split(
            tensors[split], unique, coefficients, action_order
        )
        for split in SPLITS
    }
    snapshots, graph_audit = build_graph_snapshots(
        corpus_root, corpus, exhaustive, action_order
    )
    graph_split_metrics = {
        split: evaluate_split_with_graph(
            split,
            tensors[split],
            exhaustive[split]["records"],
            snapshots[split],
            unique,
            coefficients,
            action_order,
        )
        for split in SPLITS
    }
    repaired_tree, tree_repair_audit = sg17.repair_persistent_room_semantics(
        sg17_reference["branch_tree"]
    )
    rollout = evaluate_two_step_with_graph(
        repaired_tree,
        exhaustive["test"]["records"],
        snapshots["test"],
        plans["test"],
        alphabet_index=alphabet_index,
        unique=unique,
        coefficients=coefficients,
        action_order=action_order,
        device=device,
    )
    base_model_bytes = int(
        unique["keys"].shape[0]
        * (
            unique["keys"].shape[1]
            + 1
            + len(action_order)
            + 3
            + coefficients.shape[1] * 4
        )
    )
    max_graph_bytes = max(
        graph_audit["splits"][split]["maximum_logical_graph_bytes"]
        for split in SPLITS
    )
    logical_total_bytes = base_model_bytes + max_graph_bytes
    decision = _decision(
        graph_audit,
        base_fit,
        graph_split_metrics["test"],
        rollout,
        sg17_reference,
        logical_total_bytes,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG21 episodic edge spikes and causal output projection",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "mechanism repair on observed fifth games",
            "statement": (
                "One-shot sparse bindings should replay experienced topology "
                "while the SG19 kernel predicts only unknown residual dynamics."
            ),
            "what_if": (
                "What if remembering experienced facts is a local spike write, "
                "not a gradient-learning problem?"
            ),
        },
        "references": {
            "sg19_plan_edge_negative": {
                "path": str(args.sg19_reference.expanduser().resolve()),
                "sha256": sg19_digest,
            },
            "sg18_affordance_negative": {
                "path": str(args.sg18_reference.expanduser().resolve()),
                "sha256": sg18_digest,
            },
            "sg17_rollout_tree": {
                "path": str(args.sg17_reference.expanduser().resolve()),
                "sha256": sg17_digest,
            },
            "exhaustive_cache": {
                "path": str(cache_path),
                "sha256": cache_digest,
            },
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "threads": args.threads,
            "base_kernel": base_fit["kernel"],
            "graph_update": "one-shot local source-action-destination binding",
            "known_edge_projection": (
                "room relation plus stored affordance and derived exit count"
            ),
            "reward_done_source": "unchanged SG19 learned residual",
            "objective_plan_source": "public objective string only",
            "walkthrough_role": "audit equality only; never model input",
            "action_order": action_order,
        },
        "dataset": {
            "base_vocabulary_fingerprint": vocabulary.fingerprint,
            "base_action_alphabet": alphabet,
            "objective_plan_audit": plan_audit,
            "persistent_room_tree_repair": tree_repair_audit,
            "exhaustive_counts": {
                split: exhaustive[split]["record_count"] for split in SPLITS
            },
            "graph_audit": graph_audit,
        },
        "base_weighted_fit": base_fit,
        "base_split_metrics": base_split_metrics,
        "graph_split_metrics": graph_split_metrics,
        "two_step_rollout": rollout,
        "storage": {
            "base_model_logical_bytes": base_model_bytes,
            "maximum_episode_graph_logical_bytes": max_graph_bytes,
            "combined_logical_bytes": logical_total_bytes,
        },
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=sg19.DEFAULT_CORPUS)
    parser.add_argument(
        "--sg19-reference", type=Path, default=DEFAULT_SG19_REFERENCE
    )
    parser.add_argument(
        "--sg18-reference", type=Path, default=sg19.DEFAULT_SG18_REFERENCE
    )
    parser.add_argument(
        "--sg17-reference", type=Path, default=sg19.DEFAULT_SG17_REFERENCE
    )
    parser.add_argument(
        "--exhaustive-cache", type=Path, default=sg19.DEFAULT_EXHAUSTIVE_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
