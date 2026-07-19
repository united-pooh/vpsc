"""SG19 objective plan tape and visited-edge spikes for two-step rollout."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment, _sync  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_sg17_two_step_rollout as sg17  # noqa: E402
from experiments import e3_sg18_affordance_weighted_krr as sg18  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    action_index,
    build_action_alphabet,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, file_sha256  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg16r_l5")
DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg19_plan_edge_spikes.json")
DEFAULT_SG18_REFERENCE = Path(
    "results/e3_scan/e3_sg18_affordance_weighted_krr.json"
)
DEFAULT_SG17_REFERENCE = Path("results/e3_scan/e3_sg17_two_step_rollout.json")
DEFAULT_EXHAUSTIVE_CACHE = sg18.DEFAULT_EXHAUSTIVE_CACHE
SG18_REFERENCE_SHA256 = (
    "86D069D6AEAC497ABBCAD64CD6E327DB8A4CB7A85C081EB3A68DC92A1CF08FE3"
)
SG18_EXPERIMENT = (
    "E3-SG18 exhaustive affordance spikes with weighted unique KRR"
)
SG17_REFERENCE_SHA256 = sg18.SG17_REFERENCE_SHA256
SG17_EXPERIMENT = sg18.SG17_EXPERIMENT
EXHAUSTIVE_CACHE_SHA256 = (
    "2E6F7462C2620163CC1F89C3F52D7EE6851C2EAE4C52BADBA5DB08287561A9AB"
)
FROZEN_LAMBDA = 1e-6
PLAN_PAD = "<plan_pad>"
_DIRECTION_PATTERN = re.compile(r"\b(north|south|east|west)\b", re.IGNORECASE)
INVERSE_MOVE = {
    "go east": "go west",
    "go west": "go east",
    "go north": "go south",
    "go south": "go north",
}


def parse_objective_plan(objective: str) -> Tuple[str, ...]:
    directions = tuple(
        f"go {match.lower()}" for match in _DIRECTION_PATTERN.findall(objective)
    )
    return directions + ("take coin",)


def _last_move(context_actions: Sequence[str]) -> Optional[str]:
    for action in reversed(tuple(context_actions)):
        if action in INVERSE_MOVE:
            return action
    return None


def return_edge_spike(last_move: Optional[str], candidate_action: str) -> int:
    return int(
        last_move is not None
        and INVERSE_MOVE.get(last_move) == candidate_action
    )


def _plan_slots(plan: Sequence[str], phase: int) -> Tuple[str, str]:
    current = plan[phase] if phase < len(plan) else PLAN_PAD
    following = plan[phase + 1] if phase + 1 < len(plan) else PLAN_PAD
    return current, following


def load_objective_plans(
    corpus_root: Path,
) -> Tuple[Dict[str, Dict[int, Tuple[str, ...]]], Dict[str, Any]]:
    plans: Dict[str, Dict[int, Tuple[str, ...]]] = {}
    records = []
    for split in SPLITS:
        split_plans = {}
        for line in (corpus_root / split / "episodes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            episode = json.loads(line)
            seed = int(episode["seed"])
            parsed = parse_objective_plan(str(episode["objective"]))
            walkthrough = tuple(str(action) for action in episode["walkthrough"])
            split_plans[seed] = parsed
            records.append(
                {
                    "split": split,
                    "seed": seed,
                    "parsed_plan": parsed,
                    "walkthrough": walkthrough,
                    "equal": parsed == walkthrough,
                }
            )
        plans[split] = split_plans
    return plans, {
        "source": "public objective text only",
        "compiler": "word-boundary compass directions in text order plus take coin",
        "game_count": len(records),
        "all_plans_equal_walkthrough_for_audit": all(
            bool(record["equal"]) for record in records
        ),
        "records": tuple(records),
    }


def tensorize_extended(
    records: Sequence[Mapping[str, Any]],
    plans: Mapping[int, Sequence[str]],
    *,
    alphabet_index: Mapping[str, int],
    device: torch.device,
) -> Dict[str, Any]:
    base = sg18.tensorize_records(
        records, alphabet_index=alphabet_index, device=device
    )
    plan_pad_index = len(alphabet_index)
    plan_current = []
    plan_next = []
    return_edges = []
    started = time.perf_counter_ns()
    for record in records:
        seed = int(record["game_seed"])
        phase = len(record["context_actions"])
        current, following = _plan_slots(plans[seed], phase)
        plan_current.append(
            plan_pad_index
            if current == PLAN_PAD
            else action_index(current, alphabet_index)
        )
        plan_next.append(
            plan_pad_index
            if following == PLAN_PAD
            else action_index(following, alphabet_index)
        )
        return_edges.append(
            return_edge_spike(
                _last_move(record["context_actions"]),
                str(record["candidate_action"]),
            )
        )
    base.update(
        {
            "plan_current": torch.tensor(
                plan_current, dtype=torch.long, device=device
            ),
            "plan_next": torch.tensor(
                plan_next, dtype=torch.long, device=device
            ),
            "return_edges": torch.tensor(
                return_edges, dtype=torch.long, device=device
            ),
            "extended_state_encoding_seconds": (
                time.perf_counter_ns() - started
            )
            / 1e9,
        }
    )
    return base


def plan_edge_kernel(
    query: Mapping[str, torch.Tensor],
    prototypes: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    base = sg18.affordance_spike_kernel(
        query["keys"],
        prototypes["keys"],
        query["phases"],
        prototypes["phases"],
        query["masks"],
        prototypes["masks"],
        dtype=dtype,
    )
    return_equal = (
        query["return_edges"][:, None]
        == prototypes["return_edges"][None, :]
    ).to(dtype)
    current_equal = (
        query["plan_current"][:, None]
        == prototypes["plan_current"][None, :]
    ).to(dtype)
    next_equal = (
        query["plan_next"][:, None]
        == prototypes["plan_next"][None, :]
    ).to(dtype)
    return base * (1.0 + return_equal) * (
        1.0 + current_equal + next_equal
    )


def compress_extended(train: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter_ns()
    combined = torch.cat(
        (
            train["keys"],
            train["phases"][:, None],
            train["masks"].to(torch.long),
            train["plan_current"][:, None],
            train["plan_next"][:, None],
            train["return_edges"][:, None],
        ),
        dim=1,
    )
    unique, inverse, counts = torch.unique(
        combined,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    sums = torch.zeros(
        unique.shape[0],
        train["target_code"].shape[1],
        dtype=torch.float64,
        device=combined.device,
    )
    sums.index_add_(0, inverse, train["target_code"])
    target_means = sums / counts[:, None].to(torch.float64)
    target_sets: Dict[int, set[Tuple[float, ...]]] = defaultdict(set)
    for index, group in enumerate(inverse.detach().cpu().tolist()):
        target_sets[int(group)].add(
            tuple(float(value) for value in train["target_code"][index].cpu())
        )
    key_stop = train["keys"].shape[1]
    phase_index = key_stop
    mask_start = phase_index + 1
    mask_stop = mask_start + train["masks"].shape[1]
    return {
        "keys": unique[:, :key_stop],
        "phases": unique[:, phase_index],
        "masks": unique[:, mask_start:mask_stop].to(torch.float64),
        "plan_current": unique[:, mask_stop],
        "plan_next": unique[:, mask_stop + 1],
        "return_edges": unique[:, mask_stop + 2],
        "counts": counts.to(torch.float64),
        "target_means": target_means,
        "ambiguous_unique_key_count": sum(
            len(values) > 1 for values in target_sets.values()
        ),
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def fit_weighted_extended(
    train: Mapping[str, Any],
    unique: Mapping[str, Any],
    *,
    device: torch.device,
    kernel_fn=plan_edge_kernel,
    kernel_name: str = (
        "affordance_phase_suffix_times_return_edge_and_objective_plan_tape"
    ),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    kernel_started = time.perf_counter_ns()
    kernel = kernel_fn(unique, unique)
    _sync(device)
    kernel_seconds = (time.perf_counter_ns() - kernel_started) / 1e9
    sqrt_counts = unique["counts"].sqrt()
    system = (
        sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
        + FROZEN_LAMBDA
        * torch.eye(kernel.shape[0], dtype=torch.float64, device=device)
    )
    rhs = sqrt_counts[:, None] * unique["target_means"]
    solve_started = time.perf_counter_ns()
    factor = torch.linalg.cholesky(system)
    coefficients = sqrt_counts[:, None] * torch.cholesky_solve(rhs, factor)
    _sync(device)
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9

    audit_started = time.perf_counter_ns()
    expanded_kernel = kernel_fn(train, train)
    expanded_factor = torch.linalg.cholesky(
        expanded_kernel
        + FROZEN_LAMBDA
        * torch.eye(
            expanded_kernel.shape[0], dtype=torch.float64, device=device
        )
    )
    expanded_alpha = torch.cholesky_solve(
        train["target_code"], expanded_factor
    )
    unique_train_kernel = kernel_fn(train, unique)
    unique_scores = unique_train_kernel @ coefficients
    expanded_scores = expanded_kernel @ expanded_alpha
    difference = float((unique_scores - expanded_scores).abs().max().item())
    equivalent = bool(
        torch.equal(
            sg10._prediction_matrix(unique_scores[:, : sg10.TOTAL_LOGITS]),
            sg10._prediction_matrix(expanded_scores[:, : sg10.TOTAL_LOGITS]),
        )
        and torch.equal(
            unique_scores[:, sg18.NEXT_MASK_OFFSET :]
            > sg18.MASK_DECISION_THRESHOLD,
            expanded_scores[:, sg18.NEXT_MASK_OFFSET :]
            > sg18.MASK_DECISION_THRESHOLD,
        )
    )
    audit_seconds = (time.perf_counter_ns() - audit_started) / 1e9
    return coefficients, {
        "kernel": kernel_name,
        "ridge_lambda": FROZEN_LAMBDA,
        "mask_decision_threshold": sg18.MASK_DECISION_THRESHOLD,
        "expanded_example_count": train["keys"].shape[0],
        "unique_prototype_count": unique["keys"].shape[0],
        "compression_ratio": unique["keys"].shape[0] / train["keys"].shape[0],
        "ambiguous_unique_key_count": unique["ambiguous_unique_key_count"],
        "single_pass_aggregation_seconds": unique["elapsed_seconds"],
        "unique_kernel_seconds": kernel_seconds,
        "weighted_cholesky_solve_seconds": solve_seconds,
        "deployment_training_wall_seconds": (
            train["elapsed_seconds"]
            + train["extended_state_encoding_seconds"]
            + unique["elapsed_seconds"]
            + kernel_seconds
            + solve_seconds
        ),
        "expanded_equivalence_audit_seconds_excluded": audit_seconds,
        "expanded_train_score_max_abs_difference": difference,
        "expanded_prediction_equivalent": equivalent,
    }


def evaluate_split(
    split: Mapping[str, Any],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    action_order: Sequence[str],
    *,
    kernel_fn=plan_edge_kernel,
) -> Dict[str, Any]:
    scores = kernel_fn(split, unique) @ coefficients
    return {
        "delta": sg10._ridge_multichannel_metrics(
            scores[:, : sg10.TOTAL_LOGITS],
            split["targets"],
            split["group_ids"],
        ),
        "next_affordance": sg18._mask_metrics(
            scores[:, sg18.NEXT_MASK_OFFSET :],
            split["next_masks"],
            action_order,
        ),
    }


def _runtime_score(
    context_actions: Sequence[str],
    current_mask: Sequence[int],
    candidate_action: str,
    plan: Sequence[str],
    last_move: Optional[str],
    *,
    alphabet_index: Mapping[str, int],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    device: torch.device,
    kernel_fn=plan_edge_kernel,
) -> Tuple[torch.Tensor, Tuple[int, ...], float]:
    phase_value = len(context_actions)
    plan_current, plan_next = _plan_slots(plan, phase_value)
    pad_index = len(alphabet_index)
    query = {
        "keys": torch.tensor(
            [
                sg18.sg13._padded_history_key(
                    context_actions,
                    candidate_action,
                    alphabet_index=alphabet_index,
                    pad_index=pad_index,
                )
            ],
            dtype=torch.long,
            device=device,
        ),
        "phases": torch.tensor((phase_value,), dtype=torch.long, device=device),
        "masks": torch.tensor(
            [current_mask], dtype=torch.float32, device=device
        ),
        "plan_current": torch.tensor(
            (
                pad_index
                if plan_current == PLAN_PAD
                else action_index(plan_current, alphabet_index),
            ),
            dtype=torch.long,
            device=device,
        ),
        "plan_next": torch.tensor(
            (
                pad_index
                if plan_next == PLAN_PAD
                else action_index(plan_next, alphabet_index),
            ),
            dtype=torch.long,
            device=device,
        ),
        "return_edges": torch.tensor(
            (return_edge_spike(last_move, candidate_action),),
            dtype=torch.long,
            device=device,
        ),
    }
    _sync(device)
    started = time.perf_counter_ns()
    scores = kernel_fn(query, unique, dtype=torch.float32) @ coefficients
    scores.sum().item()
    _sync(device)
    predicted_mask = tuple(
        int(value)
        for value in (
            scores[0, sg18.NEXT_MASK_OFFSET :]
            > sg18.MASK_DECISION_THRESHOLD
        )
        .cpu()
        .tolist()
    )
    return scores, predicted_mask, (time.perf_counter_ns() - started) / 1e6


def evaluate_two_step(
    tree: Mapping[str, Any],
    exhaustive_test: Sequence[Mapping[str, Any]],
    plans: Mapping[int, Sequence[str]],
    *,
    alphabet_index: Mapping[str, int],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    action_order: Sequence[str],
    device: torch.device,
    kernel_fn=plan_edge_kernel,
) -> Dict[str, Any]:
    lookup = {
        (
            int(record["game_seed"]),
            int(record["root_step"]),
            str(record["candidate_action"]),
        ): record
        for record in exhaustive_test
    }
    runtime_unique = {
        name: tensor.to(
            device=device,
            dtype=(
                torch.float32
                if name in ("masks",)
                else tensor.dtype
            ),
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
    runtime_coefficients = coefficients.to(device=device, dtype=torch.float32)
    first_predictions = []
    first_targets = []
    first_mask_predictions = []
    first_mask_targets = []
    teacher_predictions = []
    self_predictions = []
    second_targets = []
    teacher_timings = []
    self_timings = []
    routing_correct = 0
    routed_count = 0
    premature = 0
    previous_errors = 0
    previous_return_errors = 0
    previous_nonreturn_errors = 0
    errors = []
    for first in tree["first_records"]:
        seed = int(first["game_seed"])
        root_record = lookup[(seed, int(first["root_step"]), str(first["action"]))]
        context = tuple(first["context_actions"])
        root_last_move = _last_move(context)
        first_scores, predicted_mask, first_elapsed = _runtime_score(
            context,
            root_record["current_mask"],
            str(first["action"]),
            plans[seed],
            root_last_move,
            alphabet_index=alphabet_index,
            unique=runtime_unique,
            coefficients=runtime_coefficients,
            device=device,
            kernel_fn=kernel_fn,
        )
        first_summary = sg16.decode_prediction(
            first_scores[:, : sg10.TOTAL_LOGITS]
        )
        first_predictions.append(sg17._prediction_indices(first_summary))
        first_targets.append(tuple(first["target_indices"]))
        first_mask_predictions.append(predicted_mask)
        first_mask_targets.append(tuple(root_record["next_mask"]))
        if not first["seconds"]:
            continue
        true_context = tuple(first["true_context_actions"])
        self_context = sg17.imagined_context_after(
            context, str(first["action"]), first_summary
        )
        next_last_move = (
            str(first["action"])
            if str(first["action"]) in INVERSE_MOVE
            else root_last_move
        )
        routed_count += 1
        routing_correct += self_context == true_context
        if self_context is None:
            premature += 1
        for second in first["seconds"]:
            teacher_scores, _teacher_mask, teacher_elapsed = _runtime_score(
                true_context,
                root_record["next_mask"],
                str(second["action"]),
                plans[seed],
                next_last_move,
                alphabet_index=alphabet_index,
                unique=runtime_unique,
                coefficients=runtime_coefficients,
                device=device,
                kernel_fn=kernel_fn,
            )
            teacher_prediction = sg17._prediction_indices(
                sg16.decode_prediction(
                    teacher_scores[:, : sg10.TOTAL_LOGITS]
                )
            )
            target = tuple(second["target_indices"])
            teacher_predictions.append(teacher_prediction)
            second_targets.append(target)
            teacher_timings.append(first_elapsed + teacher_elapsed)
            self_prediction = None
            self_elapsed = first_elapsed
            if self_context is not None:
                self_scores, _self_mask, second_elapsed = _runtime_score(
                    self_context,
                    predicted_mask,
                    str(second["action"]),
                    plans[seed],
                    next_last_move,
                    alphabet_index=alphabet_index,
                    unique=runtime_unique,
                    coefficients=runtime_coefficients,
                    device=device,
                    kernel_fn=kernel_fn,
                )
                self_prediction = sg17._prediction_indices(
                    sg16.decode_prediction(
                        self_scores[:, : sg10.TOTAL_LOGITS]
                    )
                )
                self_elapsed += second_elapsed
            self_predictions.append(self_prediction)
            self_timings.append(self_elapsed)
            if self_prediction != target:
                if target[0] == sg10.ROOM_LABELS.index("<room_previous>"):
                    previous_errors += 1
                    if return_edge_spike(next_last_move, str(second["action"])):
                        previous_return_errors += 1
                    else:
                        previous_nonreturn_errors += 1
                if len(errors) < 100:
                    errors.append(
                        {
                            "pair_id": second["pair_id"],
                            "first_action": first["action"],
                            "second_action": second["action"],
                            "last_move": next_last_move,
                            "return_edge": return_edge_spike(
                                next_last_move, str(second["action"])
                            ),
                            "true_context": true_context,
                            "self_context": self_context,
                            "target_indices": target,
                            "teacher_prediction": teacher_prediction,
                            "self_prediction": self_prediction,
                        }
                    )
    first_mask_tensor = torch.tensor(first_mask_targets, dtype=torch.bool)
    first_mask_scores = (
        torch.tensor(first_mask_predictions, dtype=torch.float64) * 2.0 - 1.0
    )
    teacher_metrics = sg17._rollout_metrics(teacher_predictions, second_targets)
    self_metrics = sg17._rollout_metrics(self_predictions, second_targets)
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
        "previous_room_self_error_count": previous_errors,
        "previous_room_return_edge_error_count": previous_return_errors,
        "previous_room_nonreturn_error_count": previous_nonreturn_errors,
        "teacher_pair_timing": sg16._timing_summary(teacher_timings),
        "self_pair_timing": sg16._timing_summary(self_timings),
        "self_error_records_first_100": tuple(errors),
    }


def _decision(
    plan_audit: Mapping[str, Any],
    fit: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    rollout: Mapping[str, Any],
    sg17_reference: Mapping[str, Any],
    logical_model_bytes: int,
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "language_state_gate": "SMOKE",
            "weighted_math_gate": "SMOKE",
            "one_step_state_gate": "SMOKE",
            "two_step_composition_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "response_speed_gate": "SMOKE",
            "storage_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg19_plan_edge_spikes",
        }
    language = (
        plan_audit["game_count"] == 48
        and plan_audit["all_plans_equal_walkthrough_for_audit"]
    )
    math_gate = (
        fit["compression_ratio"] < 1.0
        and fit["expanded_train_score_max_abs_difference"] <= 1e-6
        and fit["expanded_prediction_equivalent"]
    )
    one_step = (
        test_metrics["delta"]["exact_vector_accuracy"] == 1.0
        and all(
            value == 1.0
            for value in test_metrics["delta"]["channel_accuracy"].values()
        )
        and test_metrics["next_affordance"]["bit_accuracy"] >= 0.98
        and test_metrics["next_affordance"]["exact_mask_accuracy"] >= 0.95
    )
    two_step = (
        rollout["teacher_forced_second"]["exact_vector_accuracy"] >= 0.98
        and rollout["self_rollout_second"]["exact_vector_accuracy"] >= 0.98
        and rollout["self_minus_teacher_exact"] >= -0.01
        and all(
            value >= 0.98
            for value in rollout["self_rollout_second"]["channel_accuracy"].values()
        )
        and rollout["first_routing_accuracy"] == 1.0
        and rollout["premature_stop_first_branch_count"] == 0
        and rollout["previous_room_self_error_count"] <= 5
    )
    ann_training = [
        replication["training"][name]["elapsed_seconds"]
        for replication in sg17_reference["replications"]
        for name in sg16.ANN_MODEL_NAMES
    ]
    training = fit["deployment_training_wall_seconds"] < min(ann_training)
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
    storage = logical_model_bytes <= min_ann_bytes
    gates = {
        "language_state_gate": language,
        "weighted_math_gate": math_gate,
        "one_step_state_gate": one_step,
        "two_step_composition_gate": two_step,
        "training_speed_gate": training,
        "response_speed_gate": response,
        "storage_gate": storage,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = "sg19r_matched_ann_and_fresh_game_confirmation"
    elif rollout["previous_room_self_error_count"] > 5:
        next_route = "sg20_strict_return_edge_isolation"
    elif test_metrics["next_affordance"]["exact_mask_accuracy"] < 0.95:
        next_route = "sg19_raw_objective_token_reservoir"
    else:
        next_route = "sg19_unique_kernel_or_state_diagnostic"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "response_comparisons": response_records,
        "next_route": next_route,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    sg18_reference, sg18_digest = sg16._load_reference(
        args.sg18_reference.expanduser().resolve(),
        SG18_REFERENCE_SHA256,
        SG18_EXPERIMENT,
    )
    sg17_reference, sg17_digest = sg16._load_reference(
        args.sg17_reference.expanduser().resolve(),
        SG17_REFERENCE_SHA256,
        SG17_EXPERIMENT,
    )
    cache_path = args.exhaustive_cache.expanduser().resolve()
    cache_digest = file_sha256(cache_path).upper()
    if cache_digest != EXHAUSTIVE_CACHE_SHA256:
        raise ValueError("SG19 exhaustive cache SHA mismatch")
    exhaustive = json.loads(cache_path.read_text(encoding="utf-8"))["exhaustive"]
    corpus = load_event_corpus(corpus_root)
    base_examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    alphabet = build_action_alphabet(base_examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_order = tuple(sg18_reference["configuration"]["action_order"])
    plans, plan_audit = load_objective_plans(corpus_root)
    if not plan_audit["all_plans_equal_walkthrough_for_audit"]:
        raise AssertionError("SG19 objective plan compiler audit failed")
    tensors = {
        split: tensorize_extended(
            exhaustive[split]["records"],
            plans[split],
            alphabet_index=alphabet_index,
            device=device,
        )
        for split in SPLITS
    }
    unique = compress_extended(tensors["train"])
    coefficients, fit = fit_weighted_extended(
        tensors["train"], unique, device=device
    )
    split_metrics = {
        split: evaluate_split(
            tensors[split], unique, coefficients, action_order
        )
        for split in SPLITS
    }
    repaired_tree, tree_repair_audit = sg17.repair_persistent_room_semantics(
        sg17_reference["branch_tree"]
    )
    if repaired_tree["canonical_tree_sha256"] == sg17_reference["branch_tree"][
        "canonical_tree_sha256"
    ]:
        raise AssertionError("SG19 expected a legacy persistent-room tree repair")
    rollout = evaluate_two_step(
        repaired_tree,
        exhaustive["test"]["records"],
        plans["test"],
        alphabet_index=alphabet_index,
        unique=unique,
        coefficients=coefficients,
        action_order=action_order,
        device=device,
    )
    logical_model_bytes = int(
        unique["keys"].shape[0]
        * (
            unique["keys"].shape[1]
            + 1
            + len(action_order)
            + 3
            + coefficients.shape[1] * 4
        )
    )
    decision = _decision(
        plan_audit,
        fit,
        split_metrics["test"],
        rollout,
        sg17_reference,
        logical_model_bytes,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG19 objective plan tape and visited-edge spikes",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "mechanism repair on observed fifth games",
            "statement": (
                "Public objective-plan spikes and one visited-edge relation "
                "complete the discrete state needed for two-step composition."
            ),
            "what_if": (
                "What if language goals are sparse future events and episodic "
                "topology needs only the last physical edge?"
            ),
        },
        "references": {
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
            "objective_plan_source": "public objective string only",
            "walkthrough_role": "audit equality only; never model input",
            "inverse_move_map": INVERSE_MOVE,
            "action_order": action_order,
            "kernel": fit["kernel"],
            "ridge_lambda": FROZEN_LAMBDA,
        },
        "dataset": {
            "base_vocabulary_fingerprint": vocabulary.fingerprint,
            "base_action_alphabet": alphabet,
            "objective_plan_audit": plan_audit,
            "persistent_room_tree_repair": tree_repair_audit,
            "exhaustive_counts": {
                split: exhaustive[split]["record_count"] for split in SPLITS
            },
        },
        "weighted_unique_fit": fit,
        "split_metrics": split_metrics,
        "two_step_rollout": rollout,
        "logical_model_storage_bytes": logical_model_bytes,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sg18-reference", type=Path, default=DEFAULT_SG18_REFERENCE)
    parser.add_argument("--sg17-reference", type=Path, default=DEFAULT_SG17_REFERENCE)
    parser.add_argument(
        "--exhaustive-cache", type=Path, default=DEFAULT_EXHAUSTIVE_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("threads must be positive")
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
