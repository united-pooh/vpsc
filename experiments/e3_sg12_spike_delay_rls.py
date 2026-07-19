"""SG12 sparse spike delay-line with primal ridge and block-RLS training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _percentile,
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments.e3_sg8_bilinear_closed_form import RIDGE_LAMBDAS  # noqa: E402
from experiments.e3_sg9_atomic_event_stream import action_event_token  # noqa: E402
from experiments.e3_sg11_temporal_basis import (  # noqa: E402
    DEFAULT_REFERENCE,
    _reference_artifact,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg12_spike_delay_rls.json")
DELAY_ORDERS = (1, 2, 3, 4)
PRIMARY_ORDER = 3


def build_action_alphabet(
    examples: Mapping[str, Sequence[sg10.MultiChannelExample]],
) -> Tuple[str, ...]:
    alphabet = tuple(
        sorted(
            {
                action_event_token(action)
                for example in examples["train"]
                for action in (*example.context_actions, example.candidate_action)
            }
        )
    )
    if not alphabet:
        raise ValueError("SG12 train action alphabet is empty")
    return alphabet


def action_index(
    action: str, alphabet_index: Mapping[str, int]
) -> int:
    token = action_event_token(action)
    if token not in alphabet_index:
        raise KeyError(f"SG12 action event is outside train alphabet: {token}")
    return alphabet_index[token]


def delay_initial(
    batch_size: int,
    order: int,
    alphabet_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if min(batch_size, order, alphabet_size) <= 0:
        raise ValueError("delay dimensions must be positive")
    return torch.zeros(
        batch_size, order, alphabet_size, dtype=torch.bool, device=device
    )


def delay_step(state: torch.Tensor, action_spike: torch.Tensor) -> torch.Tensor:
    if state.ndim != 3 or state.dtype != torch.bool:
        raise ValueError("delay state must be bool [batch, order, alphabet]")
    if (
        action_spike.ndim != 2
        or action_spike.dtype != torch.bool
        or action_spike.shape[0] != state.shape[0]
        or action_spike.shape[1] != state.shape[2]
    ):
        raise ValueError("action spike must be bool [batch, alphabet]")
    if not bool(torch.all(action_spike.sum(dim=1) == 1)):
        raise ValueError("each action event must contain exactly one spike")
    return torch.cat((action_spike[:, None, :], state[:, :-1, :]), dim=1)


def delay_state_after(
    actions: Sequence[str],
    *,
    order: int,
    alphabet_index: Mapping[str, int],
    device: torch.device,
) -> torch.Tensor:
    alphabet_size = len(alphabet_index)
    eye = torch.eye(alphabet_size, dtype=torch.bool, device=device)
    state = delay_initial(1, order, alphabet_size, device=device)
    for action in actions:
        index = action_index(action, alphabet_index)
        state = delay_step(state, eye[index : index + 1])
    return state


def delay_feature_tensor(
    context_state: torch.Tensor,
    candidate_spike: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if context_state.ndim != 3 or candidate_spike.ndim != 2:
        raise ValueError("delay features require state [B,K,A] and spike [B,A]")
    context = context_state.flatten(1).to(dtype=dtype)
    candidate = candidate_spike.to(dtype=dtype)
    outer = torch.einsum("bi,bj->bij", context, candidate).flatten(1)
    ones = torch.ones(
        context.shape[0], 1, dtype=dtype, device=context.device
    )
    return torch.cat((ones, context, candidate, outer), dim=1)


def expected_feature_dimension(order: int, alphabet_size: int) -> int:
    context = order * alphabet_size
    return 1 + context + alphabet_size + context * alphabet_size


def extract_delay_features(
    examples: Sequence[sg10.MultiChannelExample],
    *,
    order: int,
    alphabet_index: Mapping[str, int],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    alphabet_size = len(alphabet_index)
    eye = torch.eye(alphabet_size, dtype=torch.bool, device=device)
    features = []
    targets = []
    group_ids = []
    active_counts = []
    started = time.perf_counter_ns()
    for example in examples:
        state = delay_state_after(
            example.context_actions,
            order=order,
            alphabet_index=alphabet_index,
            device=device,
        )
        index = action_index(example.candidate_action, alphabet_index)
        candidate = eye[index : index + 1]
        feature = delay_feature_tensor(state, candidate, dtype=dtype)
        features.append(feature)
        targets.append(example.target_indices)
        active_counts.append(int((feature != 0.0).sum().item()))
        group_ids.append(example.step_group_id)
    _sync(device)
    feature_tensor = torch.cat(features)
    expected = expected_feature_dimension(order, alphabet_size)
    if feature_tensor.shape[1] != expected:
        raise AssertionError(
            f"SG12 feature dimension mismatch: {feature_tensor.shape[1]} != {expected}"
        )
    return {
        "features": feature_tensor,
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
        "group_ids": tuple(group_ids),
        "active_feature_count": {
            "min": min(active_counts),
            "max": max(active_counts),
            "mean": sg0._mean(active_counts),
        },
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def conditional_history_audit(
    examples: Mapping[str, Sequence[sg10.MultiChannelExample]],
    *,
    orders: Sequence[int],
) -> Dict[str, Any]:
    results = {}
    for order in orders:
        train_outputs: Dict[
            Tuple[Tuple[str, ...], str], Counter[Tuple[int, ...]]
        ] = defaultdict(Counter)
        for example in examples["train"]:
            key = (
                tuple(example.context_actions[-order:]),
                example.candidate_action,
            )
            train_outputs[key][example.target_indices] += 1
        split_records = {}
        for split in SPLITS:
            covered = 0
            correct = 0
            for example in examples[split]:
                key = (
                    tuple(example.context_actions[-order:]),
                    example.candidate_action,
                )
                if key in train_outputs:
                    covered += 1
                    prediction = train_outputs[key].most_common(1)[0][0]
                    correct += int(prediction == example.target_indices)
            split_records[split] = {
                "covered_examples": covered,
                "total_examples": len(examples[split]),
                "covered_majority_accuracy": correct / covered if covered else None,
            }
        results[str(order)] = {
            "train_key_count": len(train_outputs),
            "train_ambiguous_key_count": sum(
                len(outputs) > 1 for outputs in train_outputs.values()
            ),
            "train_examples_in_ambiguous_keys": sum(
                sum(outputs.values())
                for outputs in train_outputs.values()
                if len(outputs) > 1
            ),
            "splits": split_records,
        }
    return results


def audit_delay_data(
    examples: Mapping[str, Sequence[sg10.MultiChannelExample]],
    alphabet: Sequence[str],
) -> Dict[str, Any]:
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    unknown = {}
    for split in SPLITS:
        values = []
        for example in examples[split]:
            for action in (*example.context_actions, example.candidate_action):
                token = action_event_token(action)
                if token not in alphabet_index:
                    values.append(token)
        unknown[split] = tuple(sorted(set(values)))
    history = conditional_history_audit(examples, orders=DELAY_ORDERS)
    passed = (
        len(alphabet) == 8
        and all(not values for values in unknown.values())
        and history[str(PRIMARY_ORDER)]["train_ambiguous_key_count"] == 0
        and history["2"]["train_ambiguous_key_count"] > 0
    )
    return {
        "action_alphabet": tuple(alphabet),
        "action_alphabet_size": len(alphabet),
        "heldout_unknown_action_events": unknown,
        "conditional_history": history,
        "primary_order": PRIMARY_ORDER,
        "test_labels_already_observed_during_mechanism_design": True,
        "independent_confirmation_required": True,
        "passed": passed,
    }


def _standardize_train_features(
    train_x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = train_x[:, 1:].mean(dim=0)
    scale = train_x[:, 1:].std(dim=0, unbiased=False).clamp_min(1e-8)
    return mean, scale


def transform_features(
    values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        (values[:, :1], (values[:, 1:] - mean) / scale), dim=1
    )


def compile_raw_readout(
    transformed_weights: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    raw = torch.empty_like(transformed_weights)
    raw[1:] = transformed_weights[1:] / scale[:, None]
    raw[0] = transformed_weights[0] - (
        mean / scale
    ) @ transformed_weights[1:]
    return raw


def fit_primal_ridge(
    extracted: Mapping[str, Mapping[str, Any]],
    *,
    lambdas: Sequence[float],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    train_x = extracted["train"]["features"]
    train_y = sg10._ridge_target_code(extracted["train"]["targets"])
    mean, scale = _standardize_train_features(train_x)
    transformed = {
        split: transform_features(extracted[split]["features"], mean, scale)
        for split in SPLITS
    }
    gram = transformed["train"].T @ transformed["train"]
    rhs = transformed["train"].T @ train_y
    identity = torch.eye(gram.shape[0], dtype=torch.float64, device=device)
    candidates = []
    weights_by_lambda = {}
    started = time.perf_counter_ns()
    for ridge_lambda in lambdas:
        weights = torch.linalg.solve(
            gram + float(ridge_lambda) * identity, rhs
        )
        weights_by_lambda[float(ridge_lambda)] = weights
        candidates.append(
            {
                "lambda": float(ridge_lambda),
                "valid": sg10._ridge_multichannel_metrics(
                    transformed["valid"] @ weights,
                    extracted["valid"]["targets"],
                    extracted["valid"]["group_ids"],
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda record: (
            -record["valid"]["exact_vector_accuracy"],
            -record["valid"]["macro_channel_accuracy"],
            record["valid"]["mse"],
            record["lambda"],
        ),
    )
    weights = weights_by_lambda[selected["lambda"]]
    raw_weights = compile_raw_readout(weights, mean, scale)
    _sync(device)
    fit_seconds = (time.perf_counter_ns() - started) / 1e9
    metrics = {
        split: sg10._ridge_multichannel_metrics(
            transformed[split] @ weights,
            extracted[split]["targets"],
            extracted[split]["group_ids"],
        )
        for split in SPLITS
    }
    compile_error = max(
        float(
            (
                transformed[split] @ weights
                - extracted[split]["features"] @ raw_weights
            )
            .abs()
            .max()
            .item()
        )
        for split in SPLITS
    )
    feature_seconds = sum(
        extracted[split]["elapsed_seconds"] for split in ("train", "valid")
    )
    result = {
        "feature_dimension": train_x.shape[1],
        "output_dimension": sg10.TOTAL_LOGITS,
        "readout_parameter_count": weights.numel(),
        "selected_lambda": selected["lambda"],
        "lambda_candidates": tuple(float(value) for value in lambdas),
        "selection_rule": "max valid exact, max valid macro, min MSE, min lambda",
        "validation_candidates": candidates,
        "train": metrics["train"],
        "valid": selected["valid"],
        "test": metrics["test"],
        "active_feature_count": {
            split: extracted[split]["active_feature_count"] for split in SPLITS
        },
        "max_compiled_raw_score_difference": compile_error,
        "timing": {
            "feature_extraction_seconds": {
                split: extracted[split]["elapsed_seconds"] for split in SPLITS
            },
            "lambda_grid_primal_solve_seconds": fit_seconds,
            "selection_training_wall_seconds": feature_seconds + fit_seconds,
        },
    }
    runtime = {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "transformed_weights": weights.detach(),
        "raw_weights": raw_weights.detach(),
        "x_train": transformed["train"].detach(),
        "y_train": train_y.detach(),
    }
    return result, runtime


def block_rls(
    x: torch.Tensor,
    y: torch.Tensor,
    schedule: Sequence[Sequence[int]],
    *,
    ridge_lambda: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if ridge_lambda <= 0.0:
        raise ValueError("RLS lambda must be positive")
    dimension = x.shape[1]
    outputs = y.shape[1]
    covariance = torch.eye(
        dimension, dtype=x.dtype, device=device
    ) / float(ridge_lambda)
    weights = torch.zeros(dimension, outputs, dtype=x.dtype, device=device)
    timings = []
    residual_mse = []
    started_all = time.perf_counter_ns()
    for indices in schedule:
        index = torch.tensor(indices, dtype=torch.long, device=device)
        block_x = x.index_select(0, index)
        block_y = y.index_select(0, index)
        _sync(device)
        started = time.perf_counter_ns()
        x_covariance = block_x @ covariance
        innovation_covariance = torch.eye(
            block_x.shape[0], dtype=x.dtype, device=device
        ) + x_covariance @ block_x.T
        gain = torch.linalg.solve(
            innovation_covariance, x_covariance
        ).T
        innovation = block_y - block_x @ weights
        weights = weights + gain @ innovation
        covariance = covariance - gain @ x_covariance
        covariance = 0.5 * (covariance + covariance.T)
        _sync(device)
        timings.append((time.perf_counter_ns() - started) / 1e6)
        residual_mse.append(float(innovation.square().mean().item()))
    elapsed = (time.perf_counter_ns() - started_all) / 1e9
    return weights, covariance, {
        "block_updates": len(schedule),
        "examples_seen": sum(len(indices) for indices in schedule),
        "block_timing": {
            **_sample_summary(timings, 1),
            "p95_ms": _percentile(timings, 0.95),
        },
        "elapsed_seconds": elapsed,
        "residual_mse_first": residual_mse[0],
        "residual_mse_last": residual_mse[-1],
    }


def evaluate_rls_replication(
    extracted: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, torch.Tensor],
    batch_result: Mapping[str, Any],
    schedule: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    weights, covariance, timing = block_rls(
        runtime["x_train"],
        runtime["y_train"],
        schedule,
        ridge_lambda=float(batch_result["selected_lambda"]),
        device=device,
    )
    batch_weights = runtime["transformed_weights"]
    metrics = {}
    max_score_difference = 0.0
    prediction_equivalent = True
    for split in SPLITS:
        transformed = transform_features(
            extracted[split]["features"], runtime["mean"], runtime["scale"]
        )
        rls_scores = transformed @ weights
        batch_scores = transformed @ batch_weights
        max_score_difference = max(
            max_score_difference,
            float((rls_scores - batch_scores).abs().max().item()),
        )
        prediction_equivalent = prediction_equivalent and bool(
            torch.equal(
                sg10._prediction_matrix(rls_scores),
                sg10._prediction_matrix(batch_scores),
            )
        )
        metrics[split] = sg10._ridge_multichannel_metrics(
            rls_scores,
            extracted[split]["targets"],
            extracted[split]["group_ids"],
        )
    return {
        "algorithm": "block recursive least squares via Woodbury identity",
        "selected_lambda": batch_result["selected_lambda"],
        "max_weight_abs_difference_from_batch": float(
            (weights - batch_weights).abs().max().item()
        ),
        "max_score_abs_difference_from_batch": max_score_difference,
        "prediction_equivalent_to_batch": prediction_equivalent,
        "covariance_symmetry_max_abs_error": float(
            (covariance - covariance.T).abs().max().item()
        ),
        "train": metrics["train"],
        "valid": metrics["valid"],
        "test": metrics["test"],
        "timing": timing,
    }


def evaluate_cached_delay_stream(
    examples: Sequence[sg10.MultiChannelExample],
    *,
    order: int,
    alphabet_index: Mapping[str, int],
    raw_weights: torch.Tensor,
    device: torch.device,
    timing_repeats: int,
    timing_warmup_repeats: int,
) -> Dict[str, Any]:
    alphabet_size = len(alphabet_index)
    eye = torch.eye(alphabet_size, dtype=torch.bool, device=device)
    deploy_weights = raw_weights.to(device=device, dtype=torch.float32)
    groups: Dict[str, list[sg10.MultiChannelExample]] = defaultdict(list)
    for example in examples:
        groups[example.step_group_id].append(example)
    accumulator = sg10._metric_accumulator()
    timings = []
    state_sizes = []
    active_counts = []
    max_full_difference = 0.0
    with torch.inference_mode():
        for group_id in sorted(groups):
            group = sorted(groups[group_id], key=lambda value: value.candidate_index)
            context_actions = group[0].context_actions
            if any(example.context_actions != context_actions for example in group):
                raise ValueError("SG12 group context mismatch")
            prefix_state = delay_state_after(
                context_actions,
                order=order,
                alphabet_index=alphabet_index,
                device=device,
            )
            state_sizes.append(prefix_state.numel() * prefix_state.element_size())
            for example in group:
                candidate_index = action_index(
                    example.candidate_action, alphabet_index
                )

                def forward_candidate() -> Tuple[torch.Tensor, torch.Tensor, int]:
                    candidate = eye[candidate_index : candidate_index + 1]
                    next_state = delay_step(prefix_state, candidate)
                    features = delay_feature_tensor(
                        prefix_state, candidate, dtype=torch.float32
                    )
                    scores = features @ deploy_weights
                    active = int((features != 0.0).sum().item())
                    return scores, next_state, active

                scores, _next_state, active = forward_candidate()
                active_counts.append(active)
                prediction = sg10._prediction_matrix(scores)
                target = torch.tensor(
                    [example.target_indices], dtype=torch.long, device=device
                )
                sg10._accumulate_predictions(
                    accumulator, prediction, target, (example.step_group_id,)
                )
                for repeat in range(timing_warmup_repeats + timing_repeats):
                    _sync(device)
                    started = time.perf_counter_ns()
                    timed_scores, timed_next, _active = forward_candidate()
                    (timed_scores.sum() + timed_next.sum()).item()
                    _sync(device)
                    if repeat >= timing_warmup_repeats:
                        timings.append((time.perf_counter_ns() - started) / 1e6)
                rebuilt_state = delay_state_after(
                    example.context_actions,
                    order=order,
                    alphabet_index=alphabet_index,
                    device=device,
                )
                candidate = eye[candidate_index : candidate_index + 1]
                full_features = delay_feature_tensor(
                    rebuilt_state, candidate, dtype=torch.float32
                )
                full_scores = full_features @ deploy_weights
                max_full_difference = max(
                    max_full_difference,
                    float((scores - full_scores).abs().max().item()),
                )
    result = sg10._finalize_metrics(accumulator)
    result.update(
        {
            "mode": "persistent_sparse_spike_delay_line",
            "delay_order": order,
            "state_dtype": "bool",
            "state_bytes_max": max(state_sizes),
            "feature_dimension": raw_weights.shape[0],
            "dense_readout_multiply_adds": raw_weights.numel(),
            "active_feature_count": {
                "min": min(active_counts),
                "max": max(active_counts),
                "mean": sg0._mean(active_counts),
            },
            "max_full_score_abs_difference": max_full_difference,
            "candidate_timing": {
                **_sample_summary(timings, 1),
                "p95_ms": _percentile(timings, 0.95),
                "p99_ms": _percentile(timings, 0.99),
            },
            "candidate_timing_sample_count": len(timings),
        }
    )
    return result


def _quality_gate(metrics: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    return (
        metrics["exact_vector_accuracy"] >= 0.98
        and metrics["macro_channel_accuracy"] >= 0.98
        and all(value >= 0.95 for value in metrics["channel_accuracy"].values())
        and metrics["class_recall"]["reward"][sg10.REWARD_LABELS[1]] >= 0.90
        and metrics["class_recall"]["done"][sg10.DONE_LABELS[1]] >= 0.90
        and metrics["exact_vector_accuracy"]
        >= reference["decision"]["best_ann_exact_vector_accuracy"] - 0.02
        and metrics["macro_channel_accuracy"]
        >= reference["decision"]["best_ann_macro_channel_accuracy"] - 0.02
    )


def _decision(
    data_audit: Mapping[str, Any],
    order_results: Mapping[str, Mapping[str, Any]],
    rls_results: Sequence[Mapping[str, Any]],
    stream_results: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "primary_quality_gate": "SMOKE",
            "mechanism_gate": "SMOKE",
            "rls_equivalence_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "cached_stream_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg12_spike_delay_rls",
        }
    primary = order_results[str(PRIMARY_ORDER)]["test"]
    quality = _quality_gate(primary, reference)
    primary_exact = primary["exact_vector_accuracy"]
    mechanism_comparisons = {}
    for order in (1, 2):
        control = order_results[str(order)]["test"]["exact_vector_accuracy"]
        passed = primary_exact >= control + 0.01 or control < 0.98
        mechanism_comparisons[str(order)] = {
            "control_exact": control,
            "primary_minus_control": primary_exact - control,
            "passed": passed,
        }
    mechanism = all(
        record["passed"] for record in mechanism_comparisons.values()
    )
    rls_equivalent = all(
        result["prediction_equivalent_to_batch"]
        and result["max_score_abs_difference_from_batch"] <= 1e-6
        for result in rls_results
    )
    primary_selection_wall = order_results[str(PRIMARY_ORDER)]["timing"][
        "selection_training_wall_seconds"
    ]
    end_to_end_rls_walls = [
        primary_selection_wall + result["timing"]["elapsed_seconds"]
        for result in rls_results
    ]
    transformer_walls = [
        seed["training"]["transformer"]["elapsed_seconds"]
        for seed in reference["seeds"]
    ]
    training_speed = rls_equivalent and all(
        observed <= expected
        for observed, expected in zip(end_to_end_rls_walls, transformer_walls)
    )
    stream_comparison = []
    for schedule_seed, (stream, reference_seed) in enumerate(
        zip(stream_results, reference["seeds"])
    ):
        ann = reference_seed["cached_stream"]["transformer"]["generic"][
            "candidate_timing"
        ]
        quality_equivalent = (
            abs(stream["exact_vector_accuracy"] - primary_exact) <= 1e-12
        )
        passed = (
            quality_equivalent
            and stream["max_full_score_abs_difference"] <= 1e-6
            and stream["candidate_timing"]["p50_ms"] <= ann["p50_ms"]
            and stream["candidate_timing"]["p95_ms"] <= ann["p95_ms"]
        )
        stream_comparison.append(
            {
                "timing_replication": schedule_seed,
                "snn_p50_ms": stream["candidate_timing"]["p50_ms"],
                "snn_p95_ms": stream["candidate_timing"]["p95_ms"],
                "transformer_p50_ms": ann["p50_ms"],
                "transformer_p95_ms": ann["p95_ms"],
                "quality_equivalent": quality_equivalent,
                "passed": passed,
            }
        )
    stream_pass = quality and all(
        record["passed"] for record in stream_comparison
    )
    overall = (
        data_audit["passed"]
        and quality
        and mechanism
        and rls_equivalent
        and training_speed
        and stream_pass
    )
    return {
        "data_gate": "PASS" if data_audit["passed"] else "FAIL",
        "primary_quality_gate": "PASS" if quality else "FAIL",
        "mechanism_gate": "PASS" if mechanism else "FAIL",
        "rls_equivalence_gate": "PASS" if rls_equivalent else "FAIL",
        "training_speed_gate": "PASS" if training_speed else "FAIL",
        "cached_stream_gate": "PASS" if stream_pass else "FAIL",
        "overall": "PASS" if overall else "FAIL",
        "primary_order": PRIMARY_ORDER,
        "primary_test_metrics": primary,
        "mechanism_comparisons": mechanism_comparisons,
        "rls_end_to_end_training_wall_seconds": end_to_end_rls_walls,
        "transformer_training_wall_seconds": transformer_walls,
        "per_replication_stream_comparison": stream_comparison,
        "independent_confirmation_required": True,
        "next_route": (
            "sg12r_fresh_game_seed_confirmation_and_closed_loop"
            if overall
            else (
                "sg13_order4_sparse_associative_state"
                if _quality_gate(order_results["4"]["test"], reference)
                else "sg13_hierarchical_suffix_spike_kernel"
            )
        ),
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
    reference_path = args.reference.expanduser().resolve()
    reference, reference_digest = _reference_artifact(reference_path)
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    base_audit = sg10.audit_multichannel_examples(
        examples,
        vocabulary,
        expected_counts=dict(zip(SPLITS, args.expected_counts)),
        expected_groups=dict(zip(SPLITS, args.expected_groups)),
    )
    alphabet = build_action_alphabet(examples)
    delay_audit = audit_delay_data(examples, alphabet)
    data_audit = {
        "sg10_multichannel": base_audit,
        "spike_delay_line": delay_audit,
        "passed": base_audit["passed"] and delay_audit["passed"],
    }
    if not data_audit["passed"]:
        raise AssertionError("SG12 data audit failed")
    alphabet_index = {token: index for index, token in enumerate(alphabet)}

    order_results = {}
    order_runtimes = {}
    extracted_by_order = {}
    for order in DELAY_ORDERS:
        extracted = {
            split: extract_delay_features(
                examples[split],
                order=order,
                alphabet_index=alphabet_index,
                device=device,
            )
            for split in SPLITS
        }
        result, runtime = fit_primal_ridge(
            extracted, lambdas=args.ridge_lambdas, device=device
        )
        result["delay_order"] = order
        result["delay_state_bits"] = order * len(alphabet)
        order_results[str(order)] = result
        order_runtimes[order] = runtime
        extracted_by_order[order] = extracted

    primary_result = order_results[str(PRIMARY_ORDER)]
    primary_runtime = order_runtimes[PRIMARY_ORDER]
    primary_extracted = extracted_by_order[PRIMARY_ORDER]
    rls_results = []
    stream_results = []
    for seed in args.seeds:
        schedule = sg10.build_length_stratified_schedule(
            examples["train"],
            epochs=1,
            batch_groups=args.batch_groups,
            seed=12_001_000 + seed,
        )
        rls = evaluate_rls_replication(
            primary_extracted,
            primary_runtime,
            primary_result,
            schedule,
            device=device,
        )
        rls["schedule_seed"] = seed
        rls_results.append(rls)
        stream = evaluate_cached_delay_stream(
            examples["test"],
            order=PRIMARY_ORDER,
            alphabet_index=alphabet_index,
            raw_weights=primary_runtime["raw_weights"],
            device=device,
            timing_repeats=args.timing_repeats,
            timing_warmup_repeats=args.timing_warmup_repeats,
        )
        stream["timing_replication"] = seed
        stream_results.append(stream)
    decision = _decision(
        data_audit,
        order_results,
        rls_results,
        stream_results,
        reference,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG12 sparse spike delay-line with block RLS",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "mechanism experiment; original test already observed",
            "statement": (
                "A fixed three-event binary spike delay-line is the minimal "
                "sufficient action state for SG10 and supports exact one-pass RLS."
            ),
            "what_if": (
                "What if sparse causal events should be preserved exactly for a "
                "short horizon instead of approximated by decaying traces?"
            ),
        },
        "reference": {
            "path": str(reference_path),
            "sha256": reference_digest,
            "experiment": reference["experiment"],
            "qualified_ann": "transformer",
            "best_ann_exact_vector_accuracy": reference["decision"][
                "best_ann_exact_vector_accuracy"
            ],
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "orders": DELAY_ORDERS,
            "primary_order": PRIMARY_ORDER,
            "schedule_seeds": tuple(args.seeds),
            "threads": args.threads if device.type == "cpu" else None,
            "batch_groups": args.batch_groups,
            "block_examples": args.batch_groups * 3,
            "blocks_per_rls_pass": 10,
            "timing_repeats_per_candidate": args.timing_repeats,
            "timing_warmup_repeats_per_candidate": args.timing_warmup_repeats,
            "ridge_lambdas": tuple(args.ridge_lambdas),
            "deployment_readout_dtype": "float32",
            "training_dtype": "float64",
        },
        "dataset": {
            "synthetic": False,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "audit": data_audit,
        },
        "order_results": order_results,
        "rls_replications": rls_results,
        "cached_stream_replications": stream_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=sg10.DEFAULT_CORPUS_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=128)
    parser.add_argument("--timing-warmup-repeats", type=int, default=16)
    parser.add_argument(
        "--ridge-lambdas", nargs="+", type=float, default=RIDGE_LAMBDAS
    )
    parser.add_argument(
        "--expected-counts",
        nargs=3,
        type=int,
        default=tuple(sg10.EXPECTED_COUNTS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-groups",
        nargs=3,
        type=int,
        default=tuple(sg10.EXPECTED_GROUPS[split] for split in SPLITS),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.threads,
        args.batch_groups,
        args.timing_repeats,
        *args.ridge_lambdas,
        *args.expected_counts,
        *args.expected_groups,
    ) <= 0:
        parser.error("all numeric experiment controls must be positive")
    if args.timing_warmup_repeats < 0:
        parser.error("timing-warmup-repeats must be nonnegative")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.timing_repeats = 1
        args.timing_warmup_repeats = 0
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


if __name__ == "__main__":
    main()
