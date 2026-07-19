"""SG18 exhaustive affordance-spike state with weighted unique KRR.

Every legal action at each expert root is collected from the official
TextWorld interpreter.  Current and next admissible-action masks become sparse
world-state channels.  Duplicate spike keys are reduced to weighted sufficient
statistics and solved in closed form without changing the expanded KRR
function.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg13_suffix_spike_kernel as sg13  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_sg17_two_step_rollout as sg17  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    PRIMARY_ORDER,
    build_action_alphabet,
)
from experiments.e3_sg15_phase_isolated_kernel import PRIMARY_SPEC  # noqa: E402
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.textworld import open_textworld  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, file_sha256  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg16r_l5")
DEFAULT_OUTPUT = Path(
    "results/e3_scan/e3_sg18_affordance_weighted_krr.json"
)
DEFAULT_EXHAUSTIVE_CACHE = Path(
    "results/e3_scan/e3_sg18_exhaustive_affordance_cache.json"
)
DEFAULT_SG17_REFERENCE = Path("results/e3_scan/e3_sg17_two_step_rollout.json")
SG17_REFERENCE_SHA256 = (
    "E46E5F24C3D57A40A3405D8BCFBF737A5223C151371FC235BC527DC2096CC7EF"
)
SG17_EXPERIMENT = "E3-SG17 two-step official branch rollout composition"
EXPECTED_DATA_SEEDS = sg16.CONFIRMATION_SEEDS
EXPECTED_EXHAUSTIVE_COUNTS = {"train": 640, "valid": 160, "test": 160}
FROZEN_LAMBDA = 1e-6
MASK_DECISION_THRESHOLD = 1e-6
NEXT_MASK_OFFSET = sg10.TOTAL_LOGITS


def _action_order(corpus_root: Path) -> Tuple[str, ...]:
    actions = set()
    for line in (corpus_root / "train" / "episodes.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        episode = json.loads(line)
        for step in episode["steps"]:
            actions.update(str(action) for action in step["admissible_actions"])
    if not actions:
        raise ValueError("SG18 train action order is empty")
    return tuple(sorted(actions))


def _mask(actions: Sequence[str], action_order: Sequence[str]) -> Tuple[int, ...]:
    values = set(actions)
    unknown = values - set(action_order)
    if unknown:
        raise KeyError(f"SG18 action mask contains OOV actions: {sorted(unknown)}")
    return tuple(int(action in values) for action in action_order)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest().upper()


def _base_artifact_hashes(corpus_root: Path) -> Dict[str, Dict[str, str]]:
    return {
        split: {
            name: file_sha256(corpus_root / split / name).upper()
            for name in ("manifest.json", "episodes.jsonl")
        }
        for split in SPLITS
    }


def collect_exhaustive_split(
    corpus_root: Path,
    corpus: Any,
    split: str,
    games: Sequence[Mapping[str, Any]],
    action_order: Sequence[str],
) -> Dict[str, Any]:
    episodes = {
        int(episode["seed"]): episode
        for episode in (
            json.loads(line)
            for line in (corpus_root / split / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    records = []
    game_audit = []
    clone_timings = []
    for game in games:
        seed = int(game["seed"])
        episode = episodes[seed]
        game_file = Path(str(game["game_file"]))
        actual_sha = file_sha256(game_file).upper()
        if actual_sha != str(game["game_sha256"]).upper():
            raise ValueError(f"SG18 game SHA mismatch for {split} seed {seed}")
        adapter = open_textworld(game_file, extras=())
        non_mutating = True
        try:
            current_observation = adapter.reset()
            prior_observations: list[str] = []
            factual_actions = tuple(str(step["action"]) for step in episode["steps"])
            for root_step, stored_step in enumerate(episode["steps"]):
                live_actions = tuple(sorted(set(adapter.admissible_actions)))
                if live_actions != tuple(sorted(stored_step["admissible_actions"])):
                    raise ValueError("SG18 live admissible actions diverged from episode")
                current_mask = _mask(live_actions, action_order)
                transition_count = len(adapter.transitions)
                for candidate_action in live_actions:
                    started = time.perf_counter_ns()
                    transition = adapter.counterfactual(candidate_action)
                    clone_timings.append(
                        (time.perf_counter_ns() - started) / 1e6
                    )
                    after_actions = (
                        ()
                        if bool(transition.done)
                        else tuple(
                            transition.info.get("admissible_commands", ()) or ()
                        )
                    )
                    target = sg17._target_indices(
                        corpus,
                        transition,
                        current_observation,
                        prior_observations,
                    )
                    records.append(
                        {
                            "record_id": (
                                f"{split}:{seed}:{root_step}:{candidate_action}"
                            ),
                            "group_id": f"{split}:{seed}:{root_step}",
                            "game_seed": seed,
                            "root_step": root_step,
                            "context_actions": factual_actions[:root_step],
                            "current_mask": current_mask,
                            "candidate_action": candidate_action,
                            "target_indices": target,
                            "next_mask": _mask(after_actions, action_order),
                        }
                    )
                non_mutating = non_mutating and (
                    len(adapter.transitions) == transition_count
                    and tuple(sorted(set(adapter.admissible_actions))) == live_actions
                )
                factual = adapter.step(factual_actions[root_step])
                if (
                    float(factual.reward) != float(stored_step["reward"])
                    or bool(factual.done) != bool(stored_step["done"])
                    or sg0.normalize_textworld_observation(factual.next_observation)
                    != sg0.normalize_textworld_observation(str(stored_step["next_obs"]))
                ):
                    raise ValueError("SG18 factual replay diverged from episode")
                prior_observations.append(current_observation)
                current_observation = str(factual.next_observation)
            won = bool(adapter.transitions[-1].info.get("won", False))
            game_audit.append(
                {
                    "seed": seed,
                    "game_sha256": actual_sha,
                    "root_count": len(episode["steps"]),
                    "record_count": sum(
                        len(step["admissible_actions"])
                        for step in episode["steps"]
                    ),
                    "live_factual_won": won,
                    "counterfactuals_did_not_mutate_live": non_mutating,
                }
            )
        finally:
            adapter.close()
    fingerprint_value = {"split": split, "games": game_audit, "records": records}
    return {
        "split": split,
        "game_count": len(game_audit),
        "root_count": sum(game["root_count"] for game in game_audit),
        "record_count": len(records),
        "all_live_factual_won": all(
            bool(game["live_factual_won"]) for game in game_audit
        ),
        "all_counterfactuals_non_mutating": all(
            bool(game["counterfactuals_did_not_mutate_live"])
            for game in game_audit
        ),
        "canonical_sha256": _fingerprint(fingerprint_value),
        "clone_timing": sg16._timing_summary(clone_timings),
        "games": tuple(game_audit),
        "records": tuple(records),
    }


def tensorize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    alphabet_index: Mapping[str, int],
    device: torch.device,
) -> Dict[str, Any]:
    pad_index = len(alphabet_index)
    started = time.perf_counter_ns()
    keys = torch.tensor(
        [
            sg13._padded_history_key(
                record["context_actions"],
                str(record["candidate_action"]),
                alphabet_index=alphabet_index,
                pad_index=pad_index,
            )
            for record in records
        ],
        dtype=torch.long,
        device=device,
    )
    phases = torch.tensor(
        [len(record["context_actions"]) for record in records],
        dtype=torch.long,
        device=device,
    )
    masks = torch.tensor(
        [record["current_mask"] for record in records],
        dtype=torch.float64,
        device=device,
    )
    targets = torch.tensor(
        [record["target_indices"] for record in records],
        dtype=torch.long,
        device=device,
    )
    next_masks = torch.tensor(
        [record["next_mask"] for record in records],
        dtype=torch.bool,
        device=device,
    )
    target_code = torch.cat(
        (
            sg10._ridge_target_code(targets),
            next_masks.to(torch.float64) * 2.0 - 1.0,
        ),
        dim=1,
    )
    _sync(device)
    return {
        "keys": keys,
        "phases": phases,
        "masks": masks,
        "targets": targets,
        "next_masks": next_masks,
        "target_code": target_code,
        "group_ids": tuple(str(record["group_id"]) for record in records),
        "record_ids": tuple(str(record["record_id"]) for record in records),
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def affordance_spike_kernel(
    query_keys: torch.Tensor,
    prototype_keys: torch.Tensor,
    query_phases: torch.Tensor,
    prototype_phases: torch.Tensor,
    query_masks: torch.Tensor,
    prototype_masks: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if query_masks.ndim != 2 or prototype_masks.ndim != 2:
        raise ValueError("SG18 affordance masks must be matrices")
    if query_masks.shape[1] != prototype_masks.shape[1]:
        raise ValueError("SG18 affordance mask widths differ")
    suffix = sg13.suffix_spike_kernel(
        query_keys,
        prototype_keys,
        query_phases,
        prototype_phases,
        PRIMARY_SPEC,
        dtype=dtype,
    )
    mask_dimension = query_masks.shape[1]
    overlap = query_masks.to(dtype) @ prototype_masks.to(dtype).T
    return suffix * (1.0 + overlap / float(mask_dimension))


def compress_unique_records(train: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter_ns()
    combined = torch.cat(
        (
            train["keys"],
            train["phases"][:, None],
            train["masks"].to(torch.long),
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
    means = sums / counts[:, None].to(torch.float64)
    target_sets: Dict[int, set[Tuple[float, ...]]] = defaultdict(set)
    for index, group in enumerate(inverse.detach().cpu().tolist()):
        target_sets[int(group)].add(
            tuple(float(value) for value in train["target_code"][index].cpu())
        )
    key_width = train["keys"].shape[1]
    mask_start = key_width + 1
    return {
        "keys": unique[:, :key_width],
        "phases": unique[:, key_width],
        "masks": unique[:, mask_start:].to(torch.float64),
        "counts": counts.to(torch.float64),
        "target_means": means,
        "inverse": inverse,
        "ambiguous_unique_key_count": sum(
            len(values) > 1 for values in target_sets.values()
        ),
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def weighted_unique_krr_fit(
    train: Mapping[str, Any],
    unique: Mapping[str, Any],
    *,
    ridge_lambda: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    started_kernel = time.perf_counter_ns()
    kernel = affordance_spike_kernel(
        unique["keys"],
        unique["keys"],
        unique["phases"],
        unique["phases"],
        unique["masks"],
        unique["masks"],
    )
    _sync(device)
    kernel_seconds = (time.perf_counter_ns() - started_kernel) / 1e9
    sqrt_counts = unique["counts"].sqrt()
    system = (
        sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
        + float(ridge_lambda)
        * torch.eye(kernel.shape[0], dtype=torch.float64, device=device)
    )
    rhs = sqrt_counts[:, None] * unique["target_means"]
    started_solve = time.perf_counter_ns()
    factor = torch.linalg.cholesky(system)
    transformed = torch.cholesky_solve(rhs, factor)
    coefficients = sqrt_counts[:, None] * transformed
    _sync(device)
    solve_seconds = (time.perf_counter_ns() - started_solve) / 1e9

    audit_started = time.perf_counter_ns()
    expanded_kernel = affordance_spike_kernel(
        train["keys"],
        train["keys"],
        train["phases"],
        train["phases"],
        train["masks"],
        train["masks"],
    )
    expanded_system = expanded_kernel + float(ridge_lambda) * torch.eye(
        expanded_kernel.shape[0], dtype=torch.float64, device=device
    )
    expanded_factor = torch.linalg.cholesky(expanded_system)
    expanded_alpha = torch.cholesky_solve(train["target_code"], expanded_factor)
    unique_train_kernel = affordance_spike_kernel(
        train["keys"],
        unique["keys"],
        train["phases"],
        unique["phases"],
        train["masks"],
        unique["masks"],
    )
    unique_scores = unique_train_kernel @ coefficients
    expanded_scores = expanded_kernel @ expanded_alpha
    score_difference = float((unique_scores - expanded_scores).abs().max().item())
    prediction_equivalent = bool(
        torch.equal(
            sg10._prediction_matrix(unique_scores[:, : sg10.TOTAL_LOGITS]),
            sg10._prediction_matrix(expanded_scores[:, : sg10.TOTAL_LOGITS]),
        )
        and torch.equal(
            unique_scores[:, NEXT_MASK_OFFSET:] > MASK_DECISION_THRESHOLD,
            expanded_scores[:, NEXT_MASK_OFFSET:] > MASK_DECISION_THRESHOLD,
        )
    )
    audit_seconds = (time.perf_counter_ns() - audit_started) / 1e9
    return coefficients, {
        "kernel": "strict_phase_suffix_times_one_plus_affordance_bit_inner_product",
        "ridge_lambda": ridge_lambda,
        "expanded_example_count": train["keys"].shape[0],
        "unique_prototype_count": unique["keys"].shape[0],
        "compression_ratio": unique["keys"].shape[0] / train["keys"].shape[0],
        "ambiguous_unique_key_count": unique["ambiguous_unique_key_count"],
        "single_pass_aggregation_seconds": unique["elapsed_seconds"],
        "unique_kernel_seconds": kernel_seconds,
        "weighted_cholesky_solve_seconds": solve_seconds,
        "deployment_training_wall_seconds": (
            train["elapsed_seconds"]
            + unique["elapsed_seconds"]
            + kernel_seconds
            + solve_seconds
        ),
        "expanded_equivalence_audit_seconds_excluded": audit_seconds,
        "expanded_train_score_max_abs_difference": score_difference,
        "expanded_prediction_equivalent": prediction_equivalent,
        "mask_decision_threshold": MASK_DECISION_THRESHOLD,
    }


def _mask_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    action_order: Sequence[str],
) -> Dict[str, Any]:
    predictions = scores > MASK_DECISION_THRESHOLD
    bit_accuracy = float((predictions == targets).to(torch.float64).mean().item())
    exact = float((predictions == targets).all(dim=1).to(torch.float64).mean().item())
    by_action = {}
    for index, action in enumerate(action_order):
        target = targets[:, index]
        prediction = predictions[:, index]
        positives = int(target.sum().item())
        negatives = int((~target).sum().item())
        by_action[action] = {
            "accuracy": float((prediction == target).to(torch.float64).mean().item()),
            "positive_recall": (
                float((prediction & target).sum().item()) / positives
                if positives
                else None
            ),
            "negative_recall": (
                float(((~prediction) & (~target)).sum().item()) / negatives
                if negatives
                else None
            ),
        }
    return {
        "example_count": targets.shape[0],
        "bit_accuracy": bit_accuracy,
        "exact_mask_accuracy": exact,
        "by_action": by_action,
    }


def evaluate_split(
    split: Mapping[str, Any],
    train_unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    action_order: Sequence[str],
) -> Tuple[Dict[str, Any], torch.Tensor]:
    kernel = affordance_spike_kernel(
        split["keys"],
        train_unique["keys"],
        split["phases"],
        train_unique["phases"],
        split["masks"],
        train_unique["masks"],
    )
    scores = kernel @ coefficients
    return {
        "delta": sg10._ridge_multichannel_metrics(
            scores[:, : sg10.TOTAL_LOGITS],
            split["targets"],
            split["group_ids"],
        ),
        "next_affordance": _mask_metrics(
            scores[:, NEXT_MASK_OFFSET:], split["next_masks"], action_order
        ),
    }, scores


def _runtime_score(
    context_actions: Sequence[str],
    current_mask: Sequence[int],
    candidate_action: str,
    *,
    alphabet_index: Mapping[str, int],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, ...], float]:
    _sync(device)
    started = time.perf_counter_ns()
    key = torch.tensor(
        [
            sg13._padded_history_key(
                context_actions,
                candidate_action,
                alphabet_index=alphabet_index,
                pad_index=len(alphabet_index),
            )
        ],
        dtype=torch.long,
        device=device,
    )
    phase = torch.tensor(
        (len(context_actions),), dtype=torch.long, device=device
    )
    mask = torch.tensor(
        [current_mask], dtype=torch.float32, device=device
    )
    kernel = affordance_spike_kernel(
        key,
        unique["keys"],
        phase,
        unique["phases"],
        mask,
        unique["masks"],
        dtype=torch.float32,
    )
    scores = kernel @ coefficients
    scores.sum().item()
    _sync(device)
    next_mask = tuple(
        int(value)
        for value in (
            scores[0, NEXT_MASK_OFFSET:] > MASK_DECISION_THRESHOLD
        ).cpu().tolist()
    )
    return scores, next_mask, (time.perf_counter_ns() - started) / 1e6


def evaluate_two_step(
    tree: Mapping[str, Any],
    exhaustive_test: Sequence[Mapping[str, Any]],
    *,
    alphabet_index: Mapping[str, int],
    unique: Mapping[str, Any],
    coefficients: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    lookup = {
        (
            int(record["game_seed"]),
            int(record["root_step"]),
            str(record["candidate_action"]),
        ): record
        for record in exhaustive_test
    }
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
    errors = []
    evaluated_first_branch_count = 0
    runtime_unique = {
        "keys": unique["keys"].to(device=device),
        "phases": unique["phases"].to(device=device),
        "masks": unique["masks"].to(device=device, dtype=torch.float32),
    }
    runtime_coefficients = coefficients.to(device=device, dtype=torch.float32)
    for first in tree["first_records"]:
        lookup_key = (
            int(first["game_seed"]),
            int(first["root_step"]),
            str(first["action"]),
        )
        if lookup_key not in lookup:
            continue
        root_record = lookup[lookup_key]
        evaluated_first_branch_count += 1
        context = tuple(first["context_actions"])
        first_scores, predicted_mask, first_elapsed = _runtime_score(
            context,
            root_record["current_mask"],
            str(first["action"]),
            alphabet_index=alphabet_index,
            unique=runtime_unique,
            coefficients=runtime_coefficients,
            device=device,
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
        routed_count += 1
        routing_correct += self_context == true_context
        if self_context is None:
            premature += 1
        for second in first["seconds"]:
            teacher_scores, _teacher_mask, teacher_elapsed = _runtime_score(
                true_context,
                root_record["next_mask"],
                str(second["action"]),
                alphabet_index=alphabet_index,
                unique=runtime_unique,
                coefficients=runtime_coefficients,
                device=device,
            )
            teacher_prediction = sg17._prediction_indices(
                sg16.decode_prediction(
                    teacher_scores[:, : sg10.TOTAL_LOGITS]
                )
            )
            teacher_predictions.append(teacher_prediction)
            target = tuple(second["target_indices"])
            second_targets.append(target)
            teacher_timings.append(first_elapsed + teacher_elapsed)
            self_prediction = None
            self_elapsed = first_elapsed
            if self_context is not None:
                self_scores, _self_mask, second_elapsed = _runtime_score(
                    self_context,
                    predicted_mask,
                    str(second["action"]),
                    alphabet_index=alphabet_index,
                    unique=runtime_unique,
                    coefficients=runtime_coefficients,
                    device=device,
                )
                self_prediction = sg17._prediction_indices(
                    sg16.decode_prediction(
                        self_scores[:, : sg10.TOTAL_LOGITS]
                    )
                )
                self_elapsed += second_elapsed
            self_predictions.append(self_prediction)
            self_timings.append(self_elapsed)
            if self_prediction != target and len(errors) < 100:
                errors.append(
                    {
                        "pair_id": second["pair_id"],
                        "true_context": true_context,
                        "self_context": self_context,
                        "actual_first_next_mask": root_record["next_mask"],
                        "predicted_first_next_mask": predicted_mask,
                        "target_indices": target,
                        "teacher_prediction": teacher_prediction,
                        "self_prediction": self_prediction,
                    }
                )
    first_mask_tensor = torch.tensor(first_mask_targets, dtype=torch.bool)
    first_mask_score_proxy = torch.tensor(
        first_mask_predictions, dtype=torch.float64
    ) * 2.0 - 1.0
    teacher_metrics = sg17._rollout_metrics(teacher_predictions, second_targets)
    self_metrics = sg17._rollout_metrics(self_predictions, second_targets)
    return {
        "evaluated_first_branch_count": evaluated_first_branch_count,
        "evaluated_second_pair_count": len(second_targets),
        "first_all_admissible": sg17._rollout_metrics(
            first_predictions, first_targets
        ),
        "first_next_affordance": _mask_metrics(
            first_mask_score_proxy,
            first_mask_tensor,
            tuple(str(index) for index in range(first_mask_tensor.shape[1])),
        ),
        "teacher_forced_second": teacher_metrics,
        "self_rollout_second": self_metrics,
        "self_minus_teacher_exact": (
            self_metrics["exact_vector_accuracy"]
            - teacher_metrics["exact_vector_accuracy"]
        ),
        "first_routing_accuracy": routing_correct / routed_count,
        "premature_stop_first_branch_count": premature,
        "teacher_pair_timing": sg16._timing_summary(teacher_timings),
        "self_pair_timing": sg16._timing_summary(self_timings),
        "self_error_records_first_100": tuple(errors),
    }


def _decision(
    exhaustive: Mapping[str, Mapping[str, Any]],
    fit: Mapping[str, Any],
    split_metrics: Mapping[str, Mapping[str, Any]],
    rollout: Mapping[str, Any],
    sg17_reference: Mapping[str, Any],
    logical_model_bytes: int,
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "SMOKE",
            "weighted_math_gate": "SMOKE",
            "one_step_state_gate": "SMOKE",
            "two_step_quality_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "rollout_response_gate": "SMOKE",
            "storage_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg18_affordance_weighted_krr",
        }
    data = all(
        exhaustive[split]["record_count"] == EXPECTED_EXHAUSTIVE_COUNTS[split]
        and exhaustive[split]["all_live_factual_won"]
        and exhaustive[split]["all_counterfactuals_non_mutating"]
        for split in SPLITS
    )
    math_gate = (
        fit["compression_ratio"] < 1.0
        and fit["expanded_train_score_max_abs_difference"] <= 1e-6
        and fit["expanded_prediction_equivalent"]
    )
    test = split_metrics["test"]
    one_step = (
        test["delta"]["exact_vector_accuracy"] >= 0.95
        and all(value >= 0.95 for value in test["delta"]["channel_accuracy"].values())
        and test["next_affordance"]["bit_accuracy"] >= 0.95
        and test["next_affordance"]["exact_mask_accuracy"] >= 0.90
        and (
            test["delta"]["exact_vector_accuracy"] >= 0.95
            or test["delta"]["exact_vector_accuracy"] >= 0.90
        )
    )
    two_step = (
        rollout["teacher_forced_second"]["exact_vector_accuracy"] >= 0.95
        and rollout["self_rollout_second"]["exact_vector_accuracy"] >= 0.90
        and rollout["self_minus_teacher_exact"] >= -0.05
        and all(
            value >= 0.90
            for value in rollout["self_rollout_second"]["channel_accuracy"].values()
        )
        and rollout["premature_stop_first_branch_count"] == 0
        and rollout["self_rollout_second"]["exact_vector_accuracy"]
        >= sg17_reference["replications"][0]["rollout"]["strict_phase_snn"][
            "self_rollout_second"
        ]["exact_vector_accuracy"]
        + 0.15
    )
    ann_training = [
        replication["training"][name]["elapsed_seconds"]
        for replication in sg17_reference["replications"]
        for name in sg16.ANN_MODEL_NAMES
    ]
    training_speed = fit["deployment_training_wall_seconds"] < min(ann_training)
    response_comparisons = []
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
            response_comparisons.append(record)
    response = all(record["passed"] for record in response_comparisons)
    min_ann_bytes = min(
        replication["parameter_counts"][name] * 4
        for replication in sg17_reference["replications"]
        for name in sg16.ANN_MODEL_NAMES
    )
    storage = logical_model_bytes <= min_ann_bytes
    gates = {
        "data_gate": data,
        "weighted_math_gate": math_gate,
        "one_step_state_gate": one_step,
        "two_step_quality_gate": two_step,
        "training_speed_gate": training_speed,
        "rollout_response_gate": response,
        "storage_gate": storage,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = "sg18r_affordance_matched_ann_and_fresh_games"
    elif (
        rollout["teacher_forced_second"]["exact_vector_accuracy"] >= 0.95
        and rollout["self_rollout_second"]["exact_vector_accuracy"] < 0.90
    ):
        next_route = "sg18_objective_language_plan_spike_tape"
    elif not math_gate or not response:
        next_route = "sg18_incremental_unique_cholesky_or_pruning"
    else:
        next_route = "sg18_raw_observation_reservoir_content"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
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
    sg17_reference, sg17_digest = sg16._load_reference(
        args.sg17_reference.expanduser().resolve(),
        SG17_REFERENCE_SHA256,
        SG17_EXPERIMENT,
    )
    if sg17_reference["decision"]["overall"] != "FAIL":
        raise ValueError("SG18 expects the corrected SG17 negative reference")
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=EXPECTED_DATA_SEEDS
    )
    corpus = load_event_corpus(corpus_root)
    sg10_examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    data_audit = sg10.audit_multichannel_examples(
        sg10_examples,
        vocabulary,
        expected_counts=sg16.EXPECTED_COUNTS,
        expected_groups=sg16.EXPECTED_GROUPS,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG18 base corpus audit failed")
    alphabet = build_action_alphabet(sg10_examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_order = _action_order(corpus_root)
    if len(action_order) != 8:
        raise AssertionError(f"SG18 expected eight actions, got {action_order}")
    base_artifact_hashes = _base_artifact_hashes(corpus_root)
    cache_path = args.exhaustive_cache.expanduser().resolve()
    cache_reused = False
    cache_sha256 = None
    if not args.quick and cache_path.is_file() and not args.refresh_cache:
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        expected_cache_identity = {
            "schema_version": 1,
            "corpus_dir": str(corpus_root),
            "base_artifact_sha256": base_artifact_hashes,
            "action_order": list(action_order),
            "expected_data_seeds": {
                split: list(EXPECTED_DATA_SEEDS[split]) for split in SPLITS
            },
        }
        actual_cache_identity = {
            key: cache_payload[key] for key in expected_cache_identity
        }
        if actual_cache_identity != expected_cache_identity:
            raise ValueError("SG18 exhaustive cache identity mismatch")
        exhaustive = cache_payload["exhaustive"]
        cache_reused = True
        cache_sha256 = file_sha256(cache_path).upper()
    else:
        exhaustive = {}
        for split in SPLITS:
            games = sg16._game_records(
                corpus_root, EXPECTED_DATA_SEEDS[split], split=split
            )
            if args.game_limit:
                games = games[: args.game_limit]
            exhaustive[split] = collect_exhaustive_split(
                corpus_root, corpus, split, games, action_order
            )
        if not args.quick:
            cache_payload = {
                "schema_version": 1,
                "corpus_dir": str(corpus_root),
                "base_artifact_sha256": base_artifact_hashes,
                "action_order": action_order,
                "expected_data_seeds": EXPECTED_DATA_SEEDS,
                "exhaustive": exhaustive,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cache_sha256 = file_sha256(cache_path).upper()
    tensors = {
        split: tensorize_records(
            exhaustive[split]["records"],
            alphabet_index=alphabet_index,
            device=device,
        )
        for split in SPLITS
    }
    unique = compress_unique_records(tensors["train"])
    coefficients, fit = weighted_unique_krr_fit(
        tensors["train"],
        unique,
        ridge_lambda=FROZEN_LAMBDA,
        device=device,
    )
    split_metrics = {}
    for split in SPLITS:
        split_metrics[split], _scores = evaluate_split(
            tensors[split], unique, coefficients, action_order
        )
    rollout = evaluate_two_step(
        sg17_reference["branch_tree"],
        exhaustive["test"]["records"],
        alphabet_index=alphabet_index,
        unique=unique,
        coefficients=coefficients,
        device=device,
    )
    logical_model_bytes = int(
        unique["keys"].shape[0]
        * (
            unique["keys"].shape[1]
            + 1
            + len(action_order)
            + coefficients.shape[1] * 4
        )
    )
    decision = _decision(
        exhaustive,
        fit,
        split_metrics,
        rollout,
        sg17_reference,
        logical_model_bytes,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG18 exhaustive affordance spikes with weighted unique KRR",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "mechanism repair on observed fifth games",
            "statement": (
                "Sparse current/next affordance state plus exhaustive candidates "
                "repairs two-step branch composition without backpropagation."
            ),
            "what_if": (
                "What if actionable world state is an explicit spike mask and "
                "duplicate experience can be compressed into exact weighted "
                "sufficient statistics?"
            ),
        },
        "references": {
            "sg17_corrected_negative": {
                "path": str(args.sg17_reference.expanduser().resolve()),
                "sha256": sg17_digest,
            }
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "threads": args.threads if device.type == "cpu" else None,
            "ridge_lambda": FROZEN_LAMBDA,
            "mask_decision_threshold": MASK_DECISION_THRESHOLD,
            "delay_order": PRIMARY_ORDER,
            "action_order": action_order,
            "kernel": fit["kernel"],
            "future_candidate_set_role": "SG17 evaluator oracle proposal only",
            "exhaustive_cache": {
                "path": str(cache_path) if not args.quick else None,
                "sha256": cache_sha256,
                "reused": cache_reused,
            },
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "base_audit": data_audit,
            "base_vocabulary_fingerprint": vocabulary.fingerprint,
            "base_action_alphabet": alphabet,
            "exhaustive": exhaustive,
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
    parser.add_argument("--sg17-reference", type=Path, default=DEFAULT_SG17_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exhaustive-cache", type=Path, default=DEFAULT_EXHAUSTIVE_CACHE
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--game-limit", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.game_limit < 0:
        parser.error("threads must be positive and game-limit nonnegative")
    if args.game_limit and not args.quick:
        parser.error("game-limit is smoke-only")
    if args.quick:
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
