"""SG17 two-step composition on official TextWorld counterfactual trees.

The official interpreter proposes legal second-step actions and supplies only
evaluation targets.  SNN, LSTM, and Transformer recursively route their own
first prediction into a second latent transition.  This isolates dynamics
composition from the separate future-affordance generation problem.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _percentile,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg9_atomic_event_stream import (  # noqa: E402
    _generic_candidate_hidden,
    _prefill_previous_event,
    action_event_token,
)
from experiments.e3_sg12_spike_delay_rls import build_action_alphabet  # noqa: E402
from experiments.e3_sg13_suffix_spike_kernel import (  # noqa: E402
    extract_kernel_records,
)
from vpsc.world_model.cores import count_parameters  # noqa: E402
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.textworld import TextWorldAdapter, open_textworld  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary, file_sha256  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg16r_l5")
DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg17_two_step_rollout.json")
DEFAULT_SG16R_REFERENCE = Path(
    "results/e3_scan/e3_sg16r_fifth_fresh_closed_loop_confirmation.json"
)
DEFAULT_SG17_REFERENCE = Path("results/e3_scan/e3_sg17_two_step_rollout.json")
SG16R_REFERENCE_SHA256 = (
    "CFD3E2FF3F3F384EE1D6EAB445D468432DF6AB5A0BC534B86EC63114B250598E"
)
SG16R_EXPERIMENT = (
    "E3-SG16R fifth-fresh closed-loop candidate planner confirmation"
)
SG17_EXPERIMENT = "E3-SG17 two-step official branch rollout composition"
ANN_MODEL_NAMES = sg16.ANN_MODEL_NAMES
EXPECTED_COUNTS = sg16.EXPECTED_COUNTS
EXPECTED_GROUPS = sg16.EXPECTED_GROUPS
MECHANISM_SEEDS = sg16.CONFIRMATION_SEEDS
CONFIRMATION_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20270101, 20270109)),
    "test": tuple(range(20270109, 20270117)),
}


def _target_indices(
    corpus: Any,
    transition: Any,
    current_observation: str,
    prior_observations: Sequence[str],
) -> Tuple[int, int, int, int]:
    after_actions = (
        ()
        if bool(transition.done)
        else tuple(transition.info.get("admissible_commands", ()) or ())
    )
    labels = (
        sg10._room_relation_label(
            corpus,
            str(transition.next_observation),
            current_observation,
            prior_observations,
        ),
        sg10.REWARD_LABELS[1]
        if float(transition.reward) > 0.0
        else sg10.REWARD_LABELS[0],
        sg10.DONE_LABELS[1] if bool(transition.done) else sg10.DONE_LABELS[0],
        sg10._move_exit_label(after_actions),
    )
    return tuple(
        channel_labels.index(label)
        for (_name, channel_labels), label in zip(sg10.CHANNEL_SPECS, labels)
    )  # type: ignore[return-value]


def _true_context_after(
    context_actions: Sequence[str],
    first_action: str,
    relation: str,
    root_rooms: Sequence[str],
    next_room: Optional[str],
) -> Tuple[str, ...]:
    context = tuple(context_actions)
    if relation == sg10.ROOM_LABELS[1]:
        return context + (first_action,)
    if relation == sg10.ROOM_LABELS[2]:
        if next_room is None or next_room not in root_rooms:
            raise ValueError("previous-room relation lacks a known target room")
        target = max(
            index for index, room in enumerate(root_rooms) if room == next_room
        )
        return context[:target]
    return context


def imagined_context_after(
    context_actions: Sequence[str],
    first_action: str,
    prediction: sg16.PredictionSummary,
) -> Optional[Tuple[str, ...]]:
    if prediction.labels[2] == sg10.DONE_LABELS[1]:
        return None
    context = tuple(context_actions)
    relation = prediction.labels[0]
    if relation == sg10.ROOM_LABELS[1]:
        return context + (first_action,)
    if relation == sg10.ROOM_LABELS[2]:
        return context[:-1] if context else context
    return context


def _canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def collect_two_step_tree(
    corpus_root: Path,
    corpus: Any,
    games: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    episode_by_seed = {
        int(episode["seed"]): episode
        for episode in (
            json.loads(line)
            for line in (corpus_root / "test" / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    first_records = []
    game_audit = []
    clone_timings = []
    for game in games:
        seed = int(game["seed"])
        episode = episode_by_seed[seed]
        game_file = Path(str(game["game_file"]))
        actual_sha = file_sha256(game_file).upper()
        if actual_sha != str(game["game_sha256"]).upper():
            raise ValueError(f"SG17 game SHA mismatch for seed {seed}")
        adapter = open_textworld(game_file, extras=())
        live_mutation_free = True
        try:
            current_observation = adapter.reset()
            factual_actions = tuple(str(step["action"]) for step in episode["steps"])
            prior_observations: list[str] = []
            root_rooms: list[str] = []
            initial_room = sg10._room_feature(corpus, current_observation)
            if initial_room is None:
                raise ValueError("SG17 root has no room feature")
            root_rooms.append(initial_room)
            for root_step, stored_step in enumerate(episode["steps"]):
                if sg0.normalize_textworld_observation(current_observation) != sg0.normalize_textworld_observation(
                    str(stored_step["observation"])
                ):
                    raise ValueError("SG17 live root observation diverged from episode")
                live_actions = tuple(sorted(set(adapter.admissible_actions)))
                if live_actions != tuple(sorted(stored_step["admissible_actions"])):
                    raise ValueError("SG17 live root actions diverged from episode")
                live_transition_count = len(adapter.transitions)
                for first_action in live_actions:
                    _sync(torch.device("cpu"))
                    clone_started = time.perf_counter_ns()
                    branch_env = adapter.env.copy()
                    branch = TextWorldAdapter(
                        branch_env, source=f"{game_file}#root={root_step}"
                    )
                    branch.state = adapter.state
                    branch.initial_observation = current_observation
                    branch.objective = adapter.objective
                    try:
                        first_transition = branch.step(first_action)
                        clone_timings.append(
                            (time.perf_counter_ns() - clone_started) / 1e6
                        )
                        first_target = _target_indices(
                            corpus,
                            first_transition,
                            current_observation,
                            prior_observations,
                        )
                        first_relation = sg10.CHANNEL_SPECS[0][1][first_target[0]]
                        next_room = sg10._room_feature(
                            corpus, first_transition.next_observation
                        )
                        true_context = _true_context_after(
                            factual_actions[:root_step],
                            first_action,
                            first_relation,
                            root_rooms,
                            next_room,
                        )
                        second_records = []
                        if not bool(first_transition.done):
                            second_actions = tuple(
                                sorted(set(branch.admissible_actions))
                            )
                            for second_action in second_actions:
                                second_started = time.perf_counter_ns()
                                second_transition = branch.counterfactual(second_action)
                                clone_timings.append(
                                    (time.perf_counter_ns() - second_started) / 1e6
                                )
                                first_next_has_room = (
                                    sg10._room_feature(
                                        corpus, first_transition.next_observation
                                    )
                                    is not None
                                )
                                second_target = _target_indices(
                                    corpus,
                                    second_transition,
                                    (
                                        str(first_transition.next_observation)
                                        if first_next_has_room
                                        else current_observation
                                    ),
                                    (
                                        (*prior_observations, current_observation)
                                        if first_next_has_room
                                        else tuple(prior_observations)
                                    ),
                                )
                                second_records.append(
                                    {
                                        "pair_id": (
                                            f"{seed}:{root_step}:{first_action}:"
                                            f"{second_action}"
                                        ),
                                        "action": second_action,
                                        "target_indices": second_target,
                                    }
                                )
                        first_records.append(
                            {
                                "first_id": f"{seed}:{root_step}:{first_action}",
                                "game_seed": seed,
                                "root_step": root_step,
                                "context_actions": factual_actions[:root_step],
                                "action": first_action,
                                "target_indices": first_target,
                                "actual_relation": first_relation,
                                "actual_done": bool(first_transition.done),
                                "true_context_actions": true_context,
                                "seconds": tuple(second_records),
                            }
                        )
                    finally:
                        branch.close()
                live_mutation_free = live_mutation_free and (
                    len(adapter.transitions) == live_transition_count
                    and tuple(sorted(set(adapter.admissible_actions))) == live_actions
                )
                factual = adapter.step(factual_actions[root_step])
                if (
                    float(factual.reward) != float(stored_step["reward"])
                    or bool(factual.done) != bool(stored_step["done"])
                    or sg0.normalize_textworld_observation(factual.next_observation)
                    != sg0.normalize_textworld_observation(str(stored_step["next_obs"]))
                ):
                    raise ValueError("SG17 live factual advance diverged from episode")
                prior_observations.append(current_observation)
                current_observation = str(factual.next_observation)
                room = sg10._room_feature(corpus, current_observation)
                if room is not None and room not in root_rooms:
                    root_rooms.append(room)
            won = bool(adapter.transitions[-1].info.get("won", False))
            game_audit.append(
                {
                    "seed": seed,
                    "game_sha256": actual_sha,
                    "root_count": len(episode["steps"]),
                    "live_factual_won": won,
                    "counterfactuals_did_not_mutate_live": live_mutation_free,
                }
            )
        finally:
            adapter.close()
    second_count = sum(len(record["seconds"]) for record in first_records)
    fingerprint_payload = {
        "games": game_audit,
        "first_records": first_records,
    }
    return {
        "source": "official TextWorld 1.7 core Environment.copy() twice",
        "future_candidate_set_role": "evaluator oracle proposal only",
        "game_count": len(game_audit),
        "root_count": sum(game["root_count"] for game in game_audit),
        "first_branch_count": len(first_records),
        "terminal_first_branch_count": sum(
            bool(record["actual_done"]) for record in first_records
        ),
        "second_pair_count": second_count,
        "all_live_factual_won": all(
            bool(game["live_factual_won"]) for game in game_audit
        ),
        "all_counterfactuals_non_mutating": all(
            bool(game["counterfactuals_did_not_mutate_live"])
            for game in game_audit
        ),
        "canonical_tree_sha256": _canonical_fingerprint(fingerprint_payload),
        "clone_timing": sg16._timing_summary(clone_timings),
        "games": tuple(game_audit),
        "first_records": tuple(first_records),
    }


def repair_persistent_room_semantics(
    tree: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Repair legacy two-step targets that forgot room across no-text actions."""

    repaired = json.loads(json.dumps(tree))
    changed = []
    previous_index = sg10.ROOM_LABELS.index("<room_previous>")
    same_index = sg10.ROOM_LABELS.index("<room_same>")
    for first in repaired["first_records"]:
        if first["actual_relation"] != sg10.ROOM_LABELS[0]:
            continue
        for second in first["seconds"]:
            if (
                second["action"] == "look"
                and int(second["target_indices"][0]) == previous_index
            ):
                second["target_indices"][0] = same_index
                changed.append(second["pair_id"])
    fingerprint_payload = {
        "games": repaired["games"],
        "first_records": repaired["first_records"],
    }
    old_sha = repaired["canonical_tree_sha256"]
    new_sha = _canonical_fingerprint(fingerprint_payload)
    repaired["canonical_tree_sha256"] = new_sha
    return repaired, {
        "rule": (
            "if first delta has no room text, preserve physical current room; "
            "a following look is room_same, not room_previous"
        ),
        "changed_pair_count": len(changed),
        "changed_pair_ids": tuple(changed),
        "source_tree_sha256": old_sha,
        "repaired_tree_sha256": new_sha,
    }


def _prediction_indices(
    prediction: sg16.PredictionSummary,
) -> Tuple[int, int, int, int]:
    return tuple(
        labels.index(label)
        for (_name, labels), label in zip(
            sg10.CHANNEL_SPECS, prediction.labels
        )
    )  # type: ignore[return-value]


def _rollout_metrics(
    predictions: Sequence[Optional[Tuple[int, int, int, int]]],
    targets: Sequence[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    if len(predictions) != len(targets) or not targets:
        raise ValueError("rollout predictions and targets must be non-empty and aligned")
    exact = sum(
        prediction is not None and prediction == target
        for prediction, target in zip(predictions, targets)
    )
    channel_accuracy = {}
    class_recall = {}
    for channel_index, (name, labels) in enumerate(sg10.CHANNEL_SPECS):
        correct = Counter()
        total = Counter()
        channel_correct = 0
        for prediction, target in zip(predictions, targets):
            target_index = target[channel_index]
            total[target_index] += 1
            if prediction is not None and prediction[channel_index] == target_index:
                correct[target_index] += 1
                channel_correct += 1
        channel_accuracy[name] = channel_correct / len(targets)
        class_recall[name] = {
            label: correct[index] / total[index] if total[index] else None
            for index, label in enumerate(labels)
        }
    return {
        "example_count": len(targets),
        "exact_vector_accuracy": exact / len(targets),
        "macro_channel_accuracy": sg0._mean(channel_accuracy.values()),
        "channel_accuracy": channel_accuracy,
        "class_recall": class_recall,
        "missing_prediction_count": sum(value is None for value in predictions),
    }


def _ann_prefix_stack(
    model: sg10.MultiChannelBilinearModel,
    vocabulary: Vocabulary,
    context_actions: Sequence[str],
    *,
    device: torch.device,
) -> Tuple[sg16.AnnPrefix, ...]:
    start_id = vocabulary.token_id(sg10.START_EVENT)
    hidden, state = _prefill_previous_event(
        model.language_model, start_id, device=device
    )
    stack = [sg16.AnnPrefix(hidden, state)]
    for action in context_actions:
        token_id = vocabulary.token_id(action_event_token(action))
        hidden, state = _generic_candidate_hidden(
            model.language_model, token_id, stack[-1].state, device=device
        )
        stack.append(sg16.AnnPrefix(hidden, state))
    return tuple(stack)


def _ann_score(
    model: sg10.MultiChannelBilinearModel,
    vocabulary: Vocabulary,
    prefix: sg16.AnnPrefix,
    action: str,
    *,
    device: torch.device,
) -> sg16.CandidateBranch:
    token_id = vocabulary.token_id(action_event_token(action))
    if token_id == vocabulary.unk_id:
        raise KeyError(f"SG17 ANN action is OOV: {action}")
    _sync(device)
    started = time.perf_counter_ns()
    hidden, state = _generic_candidate_hidden(
        model.language_model, token_id, prefix.state, device=device
    )
    scores = model.relation_head(prefix.previous_hidden, hidden)
    scores.sum().item()
    _sync(device)
    return sg16.CandidateBranch(
        action=action,
        scores=scores,
        elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
        next_hidden=hidden,
        next_state=state,
    )


def _prefix_for_context(
    expert_context: Sequence[str],
    routed_context: Sequence[str],
    expert_stack: Sequence[sg16.AnnPrefix],
    first_branch: sg16.CandidateBranch,
) -> sg16.AnnPrefix:
    expert = tuple(expert_context)
    routed = tuple(routed_context)
    if routed == expert + (first_branch.action,):
        if first_branch.next_hidden is None or first_branch.next_state is None:
            raise AssertionError("novel ANN branch lacks cached state")
        return sg16.AnnPrefix(first_branch.next_hidden, first_branch.next_state)
    if routed == expert:
        return expert_stack[-1]
    if len(routed) <= len(expert) and routed == expert[: len(routed)]:
        return expert_stack[len(routed)]
    raise ValueError("SG17 routed context is not reachable from expert prefix")


def evaluate_rollout_model(
    name: str,
    tree: Mapping[str, Any],
    *,
    device: torch.device,
    kernel_backend: Optional[sg16.KernelPlannerBackend] = None,
    ann_model: Optional[sg10.MultiChannelBilinearModel] = None,
    vocabulary: Optional[Vocabulary] = None,
) -> Dict[str, Any]:
    is_kernel = kernel_backend is not None
    if is_kernel == (ann_model is not None):
        raise ValueError("provide exactly one SG17 model backend")
    if ann_model is not None and vocabulary is None:
        raise ValueError("ANN rollout requires vocabulary")
    if ann_model is not None:
        ann_model.eval()
    first_predictions = []
    first_targets = []
    teacher_predictions = []
    self_predictions = []
    second_targets = []
    teacher_pair_timings = []
    self_pair_timings = []
    routing_correct = 0
    routed_first_count = 0
    premature_stops = 0
    error_records = []
    prefix_cache: Dict[Tuple[str, ...], Tuple[sg16.AnnPrefix, ...]] = {}
    with torch.inference_mode():
        for first in tree["first_records"]:
            context = tuple(first["context_actions"])
            if is_kernel:
                first_branch = kernel_backend.score(context, (first["action"],))[0]  # type: ignore[union-attr]
                expert_stack = None
            else:
                if context not in prefix_cache:
                    prefix_cache[context] = _ann_prefix_stack(
                        ann_model, vocabulary, context, device=device  # type: ignore[arg-type]
                    )
                expert_stack = prefix_cache[context]
                first_branch = _ann_score(
                    ann_model,  # type: ignore[arg-type]
                    vocabulary,  # type: ignore[arg-type]
                    expert_stack[-1],
                    first["action"],
                    device=device,
                )
            first_summary = sg16.decode_prediction(first_branch.scores)
            first_prediction = _prediction_indices(first_summary)
            first_target = tuple(first["target_indices"])
            first_predictions.append(first_prediction)
            first_targets.append(first_target)
            if not first["seconds"]:
                continue
            true_context = tuple(first["true_context_actions"])
            self_context = imagined_context_after(
                context, first["action"], first_summary
            )
            routed_first_count += 1
            routing_correct += self_context == true_context
            if is_kernel:
                teacher_prefix = None
                self_prefix = None
            else:
                teacher_prefix = _prefix_for_context(
                    context, true_context, expert_stack, first_branch  # type: ignore[arg-type]
                )
                self_prefix = (
                    None
                    if self_context is None
                    else _prefix_for_context(
                        context, self_context, expert_stack, first_branch  # type: ignore[arg-type]
                    )
                )
            if self_context is None:
                premature_stops += 1
            for second in first["seconds"]:
                if is_kernel:
                    teacher_branch = kernel_backend.score(  # type: ignore[union-attr]
                        true_context, (second["action"],)
                    )[0]
                else:
                    teacher_branch = _ann_score(
                        ann_model,  # type: ignore[arg-type]
                        vocabulary,  # type: ignore[arg-type]
                        teacher_prefix,  # type: ignore[arg-type]
                        second["action"],
                        device=device,
                    )
                teacher_prediction = _prediction_indices(
                    sg16.decode_prediction(teacher_branch.scores)
                )
                teacher_predictions.append(teacher_prediction)
                target = tuple(second["target_indices"])
                second_targets.append(target)
                teacher_pair_timings.append(
                    first_branch.elapsed_ms + teacher_branch.elapsed_ms
                )
                self_prediction = None
                self_elapsed = first_branch.elapsed_ms
                if self_context is not None:
                    if is_kernel:
                        self_branch = kernel_backend.score(  # type: ignore[union-attr]
                            self_context, (second["action"],)
                        )[0]
                    else:
                        self_branch = _ann_score(
                            ann_model,  # type: ignore[arg-type]
                            vocabulary,  # type: ignore[arg-type]
                            self_prefix,  # type: ignore[arg-type]
                            second["action"],
                            device=device,
                        )
                    self_prediction = _prediction_indices(
                        sg16.decode_prediction(self_branch.scores)
                    )
                    self_elapsed += self_branch.elapsed_ms
                self_predictions.append(self_prediction)
                self_pair_timings.append(self_elapsed)
                if self_prediction != target and len(error_records) < 100:
                    error_records.append(
                        {
                            "pair_id": second["pair_id"],
                            "first_action": first["action"],
                            "second_action": second["action"],
                            "true_context": true_context,
                            "self_context": self_context,
                            "target_indices": target,
                            "teacher_prediction": teacher_prediction,
                            "self_prediction": self_prediction,
                        }
                    )
    first_metrics = _rollout_metrics(first_predictions, first_targets)
    teacher_metrics = _rollout_metrics(teacher_predictions, second_targets)
    self_metrics = _rollout_metrics(self_predictions, second_targets)
    return {
        "model": name,
        "first_all_admissible": first_metrics,
        "teacher_forced_second": teacher_metrics,
        "self_rollout_second": self_metrics,
        "self_minus_teacher_exact": (
            self_metrics["exact_vector_accuracy"]
            - teacher_metrics["exact_vector_accuracy"]
        ),
        "first_routing_accuracy": routing_correct / routed_first_count,
        "premature_stop_first_branch_count": premature_stops,
        "teacher_pair_timing": sg16._timing_summary(teacher_pair_timings),
        "self_pair_timing": sg16._timing_summary(self_pair_timings),
        "self_error_records_first_100": tuple(error_records),
    }


def _frozen_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "sg16_protocol": sg16._frozen_protocol(args),
        "rollout_depth": 2,
        "root_state_source": "stored factual path replayed in official core",
        "first_actions": "all current official admissible actions",
        "second_actions": "all official branch admissible actions; evaluator only",
        "teacher_routing": "actual first room relation push/pop/hold",
        "self_routing": "predicted first room/done push/pop/hold or stop",
        "previous_room_self_rule": "pop exactly one expert depth",
        "training_data_added": False,
    }


def _decision(
    data_audit: Mapping[str, Any],
    action_coverage: Mapping[str, Any],
    tree: Mapping[str, Any],
    spike: Mapping[str, Any],
    replications: Sequence[Mapping[str, Any]],
    *,
    fresh_confirmation: bool,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "tree_gate": "SMOKE",
            "offline_model_gate": "SMOKE",
            "rollout_task_gate": "SMOKE",
            "snn_composition_quality_gate": "SMOKE",
            "quality_relative_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "rollout_response_gate": "SMOKE",
            "storage_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg17_two_step_rollout",
        }
    tree_gate = (
        tree["game_count"] == 8
        and tree["root_count"] == 40
        and tree["first_branch_count"] > 0
        and tree["second_pair_count"] > 0
        and tree["all_live_factual_won"]
        and tree["all_counterfactuals_non_mutating"]
        and action_coverage["passed"]
    )
    offline_model = (
        spike["offline"]["test"]["exact_vector_accuracy"] >= 0.98
        and all(
            replication["offline"]["transformer"]["exact_vector_accuracy"]
            >= 0.90
            for replication in replications
        )
    )
    rollout_task = all(
        max(
            replication["rollout"][name]["teacher_forced_second"][
                "exact_vector_accuracy"
            ]
            for name in ANN_MODEL_NAMES
        )
        >= 0.85
        for replication in replications
    )
    snn_quality = all(
        replication["rollout"]["strict_phase_snn"]["first_all_admissible"][
            "exact_vector_accuracy"
        ]
        >= 0.98
        and replication["rollout"]["strict_phase_snn"][
            "teacher_forced_second"
        ]["exact_vector_accuracy"]
        >= 0.98
        and replication["rollout"]["strict_phase_snn"]["self_rollout_second"][
            "exact_vector_accuracy"
        ]
        >= 0.95
        and replication["rollout"]["strict_phase_snn"][
            "self_minus_teacher_exact"
        ]
        >= -0.03
        and all(
            value >= 0.95
            for value in replication["rollout"]["strict_phase_snn"][
                "self_rollout_second"
            ]["channel_accuracy"].values()
        )
        and replication["rollout"]["strict_phase_snn"][
            "premature_stop_first_branch_count"
        ]
        == 0
        for replication in replications
    )
    relative = all(
        replication["rollout"]["strict_phase_snn"]["self_rollout_second"][
            "exact_vector_accuracy"
        ]
        >= max(
            replication["rollout"][name]["self_rollout_second"][
                "exact_vector_accuracy"
            ]
            for name in ANN_MODEL_NAMES
        )
        - 0.02
        and replication["rollout"]["strict_phase_snn"][
            "self_rollout_second"
        ]["macro_channel_accuracy"]
        >= max(
            replication["rollout"][name]["self_rollout_second"][
                "macro_channel_accuracy"
            ]
            for name in ANN_MODEL_NAMES
        )
        - 0.02
        for replication in replications
    )
    spike_training = spike["training"]["deployment_training_wall_seconds"]
    training_speed = all(
        spike_training < replication["training"][name]["elapsed_seconds"]
        for replication in replications
        for name in ANN_MODEL_NAMES
    )
    response_comparisons = []
    for replication in replications:
        snn = replication["rollout"]["strict_phase_snn"][
            "teacher_pair_timing"
        ]
        for name in ANN_MODEL_NAMES:
            ann = replication["rollout"][name]["teacher_pair_timing"]
            record = {
                "seed": replication["seed"],
                "ann_model": name,
                "snn_p50_ms": snn["p50_ms"],
                "ann_p50_ms": ann["p50_ms"],
                "snn_p95_ms": snn["p95_ms"],
                "ann_p95_ms": ann["p95_ms"],
            }
            record["passed"] = (
                record["snn_p50_ms"] <= record["ann_p50_ms"]
                and record["snn_p95_ms"] <= record["ann_p95_ms"]
            )
            response_comparisons.append(record)
    response = all(record["passed"] for record in response_comparisons)
    spike_bytes = spike[
        "logical_model_storage_bytes_uint8_keys_phase_float32_alpha"
    ]
    storage = all(
        spike_bytes <= replication["parameter_counts"][name] * 4
        for replication in replications
        for name in ANN_MODEL_NAMES
    )
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "tree_gate": tree_gate,
        "offline_model_gate": offline_model,
        "rollout_task_gate": rollout_task,
        "snn_composition_quality_gate": snn_quality,
        "quality_relative_gate": relative,
        "training_speed_gate": training_speed,
        "rollout_response_gate": response,
        "storage_gate": storage,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = (
            "sg18_sparse_affordance_head_model_only_mpc"
            if fresh_confirmation
            else "sg17r_sixth_fresh_two_step_confirmation"
        )
    elif all(
        replication["rollout"]["strict_phase_snn"]["teacher_forced_second"][
            "exact_vector_accuracy"
        ]
        >= 0.98
        for replication in replications
    ):
        next_route = "sg17_uncertainty_routed_latent_transition"
    elif not response:
        next_route = "sg17_unique_prototype_vectorized_rollout"
    else:
        next_route = "sg17_observation_reservoir_times_strict_phase_kernel"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "fresh_confirmation": fresh_confirmation,
        "independent_confirmation_required": not fresh_confirmation,
        "response_comparisons": response_comparisons,
        "next_route": next_route,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    expected_seeds = CONFIRMATION_SEEDS if args.fresh_confirmation else MECHANISM_SEEDS
    sg16_reference, sg16_digest = sg16._load_reference(
        args.sg16r_reference.expanduser().resolve(),
        SG16R_REFERENCE_SHA256,
        SG16R_EXPERIMENT,
    )
    if sg16_reference["decision"]["overall"] != "PASS":
        raise ValueError("SG17 requires a passing independent SG16R reference")
    sg17_reference = None
    sg17_digest = None
    if args.fresh_confirmation:
        if not args.sg17_reference_sha:
            raise ValueError("fresh confirmation requires --sg17-reference-sha")
        sg17_reference, sg17_digest = sg16._load_reference(
            args.sg17_reference.expanduser().resolve(),
            args.sg17_reference_sha,
            SG17_EXPERIMENT,
        )
        if sg17_reference["decision"]["overall"] != "PASS":
            raise ValueError("SG17 mechanism reference must pass before confirmation")
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=expected_seeds
    )
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    data_audit = sg10.audit_multichannel_examples(
        examples,
        vocabulary,
        expected_counts=EXPECTED_COUNTS,
        expected_groups=EXPECTED_GROUPS,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG17 multichannel data audit failed")
    alphabet = build_action_alphabet(examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_coverage = sg16._action_coverage(
        corpus_root, vocabulary, alphabet_index
    )
    if not action_coverage["passed"]:
        raise AssertionError("SG17 action coverage failed")
    train_hashes = sg16._artifact_hashes(corpus_root, "train")
    frozen_protocol = _frozen_protocol(args)
    confirmation_reproduction = None
    if sg17_reference is not None:
        confirmation_reproduction = {
            "protocol_equal": json.loads(json.dumps(frozen_protocol))
            == sg17_reference["configuration"]["frozen_protocol"],
            "train_artifacts_equal": train_hashes
            == sg17_reference["dataset"]["train_artifact_sha256"],
            "vocabulary_fingerprint_equal": vocabulary.fingerprint
            == sg17_reference["dataset"]["vocabulary_fingerprint"],
            "action_alphabet_equal": list(alphabet)
            == sg17_reference["dataset"]["action_alphabet"],
        }
        if not all(confirmation_reproduction.values()):
            raise AssertionError("SG17R frozen protocol reproduction failed")
    games = sg16._game_records(corpus_root, expected_seeds["test"])
    if args.game_limit:
        games = games[: args.game_limit]
    tree = collect_two_step_tree(corpus_root, corpus, games)
    records = {
        split: extract_kernel_records(
            examples[split], alphabet_index=alphabet_index, device=device
        )
        for split in SPLITS
    }
    spike_result, spike_runtime = sg16._fit_spike_kernel(
        records, examples, batch_groups=args.batch_groups, device=device
    )
    class_weights = sg10.build_class_weights(examples["train"], device=device)
    replications = []
    for seed in args.seeds:
        all_models = sg10.build_multichannel_models(
            10_200_000 + 100 * seed,
            vocabulary,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_heads=args.num_heads,
            device=device,
        )
        models = {name: all_models[name] for name in ANN_MODEL_NAMES}
        schedule = sg10.build_length_stratified_schedule(
            examples["train"],
            epochs=args.epochs,
            batch_groups=args.batch_groups,
            seed=10_201_000 + seed,
        )
        training = {
            name: sg10.train_multichannel(
                name,
                model,
                examples["train"],
                schedule,
                class_weights,
                epochs=args.epochs,
                batches_per_epoch=10,
                device=device,
            )
            for name, model in models.items()
        }
        offline = {
            name: sg10.evaluate_multichannel(
                model,
                examples["test"],
                class_weights,
                device=device,
                include_records=False,
            )
            for name, model in models.items()
        }
        kernel_backend = sg16.KernelPlannerBackend(
            alphabet_index=alphabet_index,
            prototype_keys=spike_runtime["prototype_keys"],
            prototype_phases=spike_runtime["prototype_phases"],
            alpha=spike_runtime["alpha"],
            device=device,
        )
        rollout = {
            "strict_phase_snn": evaluate_rollout_model(
                "strict_phase_snn",
                tree,
                device=device,
                kernel_backend=kernel_backend,
            )
        }
        for name, model in models.items():
            rollout[name] = evaluate_rollout_model(
                name,
                tree,
                device=device,
                ann_model=model,
                vocabulary=vocabulary,
            )
        replications.append(
            {
                "seed": seed,
                "parameter_counts": {
                    name: count_parameters(model) for name, model in models.items()
                },
                "training": training,
                "offline": offline,
                "rollout": rollout,
            }
        )
        del models
        del all_models
    decision = _decision(
        data_audit,
        action_coverage,
        tree,
        spike_result,
        replications,
        fresh_confirmation=args.fresh_confirmation,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": (
            "E3-SG17R sixth-fresh two-step rollout confirmation"
            if args.fresh_confirmation
            else SG17_EXPERIMENT
        ),
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": (
                "independent sixth procedural game confirmation"
                if args.fresh_confirmation
                else "rollout mechanism on already observed fifth games"
            ),
            "statement": (
                "Strict phase spike memory composes its own first delta into a "
                "second transition without BPTT or attention history."
            ),
            "what_if": (
                "What if local spike-state composition is the minimal real-time "
                "unit of world-model thought?"
            ),
        },
        "references": {
            "sg16r_closed_loop": {
                "path": str(args.sg16r_reference.expanduser().resolve()),
                "sha256": sg16_digest,
            },
            "sg17_frozen_rollout": (
                {
                    "path": str(args.sg17_reference.expanduser().resolve()),
                    "sha256": sg17_digest,
                }
                if sg17_reference is not None
                else None
            ),
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "fresh_confirmation": args.fresh_confirmation,
            "threads": args.threads if device.type == "cpu" else None,
            "frozen_protocol": frozen_protocol,
            "actual_game_count": len(games),
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "train_artifact_sha256": train_hashes,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "action_alphabet": alphabet,
            "audit": data_audit,
            "action_coverage": action_coverage,
        },
        "branch_tree": tree,
        "spike_kernel": spike_result,
        "confirmation_reproduction": confirmation_reproduction,
        "replications": replications,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--sg16r-reference", type=Path, default=DEFAULT_SG16R_REFERENCE
    )
    parser.add_argument("--sg17-reference", type=Path, default=DEFAULT_SG17_REFERENCE)
    parser.add_argument("--sg17-reference-sha", type=str, default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--game-limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh-confirmation", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.epochs,
        args.threads,
        args.d_model,
        args.state_dim,
        args.num_heads,
        args.batch_groups,
        args.max_actions,
    ) <= 0:
        parser.error("numeric model controls must be positive")
    if min(args.seeds) < 0 or args.game_limit < 0:
        parser.error("seeds and game-limit must be nonnegative")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
    if args.game_limit and not args.quick:
        parser.error("game-limit is smoke-only")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.game_limit = args.game_limit or 1
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
