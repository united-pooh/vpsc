"""SG13 hierarchical suffix spike-kernel associative world-state memory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments.e3_sg8_bilinear_closed_form import RIDGE_LAMBDAS  # noqa: E402
from experiments.e3_sg11_temporal_basis import (  # noqa: E402
    DEFAULT_REFERENCE,
    _reference_artifact,
)
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    PRIMARY_ORDER,
    action_index,
    audit_delay_data,
    build_action_alphabet,
    delay_state_after,
    delay_step,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg13_suffix_spike_kernel.json")
DEFAULT_SG12_REFERENCE = Path("results/e3_scan/e3_sg12_spike_delay_rls.json")
DEFAULT_SG13_REFERENCE = Path("results/e3_scan/e3_sg13_suffix_spike_kernel.json")
SG12_REFERENCE_SHA256 = (
    "EB776BC2884CA6E23EEA319A81312098EF8AC93111FC62919BE8E7EEBE5EA76C"
)
SG13_REFERENCE_SHA256 = (
    "1DF3593277FED31B2624DAC27AD486E368203B3D9C76079A36BA91F5FFEC8C6E"
)
CONFIRMATION_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20260901, 20260909)),
    "test": tuple(range(20260909, 20260917)),
}


@dataclass(frozen=True)
class KernelSpec:
    name: str
    suffix_weights: Tuple[float, float, float, float]
    phase_weight: float
    primary: bool = False
    phase_suffix_weights: Tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )


KERNEL_SPECS = (
    KernelSpec("candidate_only", (1.0, 0.0, 0.0, 0.0), 0.0),
    KernelSpec("suffix1", (1.0, 1.0, 0.0, 0.0), 0.0),
    KernelSpec("suffix2", (1.0, 1.0, 1.0, 0.0), 0.0),
    KernelSpec("suffix3_no_phase", (1.0, 1.0, 1.0, 1.0), 0.0),
    KernelSpec(
        "suffix3_phase", (1.0, 1.0, 1.0, 1.0), 1.0, primary=True
    ),
    KernelSpec("depth_weighted_phase", (1.0, 2.0, 4.0, 8.0), 1.0),
)
SPEC_BY_NAME = {spec.name: spec for spec in KERNEL_SPECS}
PRIMARY_SPEC = next(spec for spec in KERNEL_SPECS if spec.primary)


def _load_sg12_reference(path: Path) -> Tuple[Dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != SG12_REFERENCE_SHA256:
        raise ValueError(
            f"SG12 reference SHA mismatch: expected {SG12_REFERENCE_SHA256}, got {digest}"
        )
    value = json.loads(payload)
    if value["experiment"] != "E3-SG12 sparse spike delay-line with block RLS":
        raise ValueError("unexpected SG12 reference experiment")
    return value, digest


def _load_sg13_reference(path: Path) -> Tuple[Dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != SG13_REFERENCE_SHA256:
        raise ValueError(
            f"SG13 reference SHA mismatch: expected {SG13_REFERENCE_SHA256}, got {digest}"
        )
    value = json.loads(payload)
    if value["experiment"] != (
        "E3-SG13 hierarchical suffix spike kernel associative memory"
    ):
        raise ValueError("unexpected SG13 reference experiment")
    return value, digest


def _padded_history_key(
    context_actions: Sequence[str],
    candidate_action: str,
    *,
    alphabet_index: Mapping[str, int],
    pad_index: int,
) -> Tuple[int, int, int, int]:
    history = tuple(
        action_index(action, alphabet_index) for action in context_actions[-3:]
    )
    padded = (pad_index,) * (3 - len(history)) + history
    return padded + (action_index(candidate_action, alphabet_index),)


def extract_kernel_records(
    examples: Sequence[sg10.MultiChannelExample],
    *,
    alphabet_index: Mapping[str, int],
    device: torch.device,
) -> Dict[str, Any]:
    pad_index = len(alphabet_index)
    started = time.perf_counter_ns()
    keys = torch.tensor(
        [
            _padded_history_key(
                example.context_actions,
                example.candidate_action,
                alphabet_index=alphabet_index,
                pad_index=pad_index,
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )
    phases = torch.tensor(
        [len(example.context_actions) for example in examples],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [example.target_indices for example in examples],
        dtype=torch.long,
        device=device,
    )
    _sync(device)
    return {
        "keys": keys,
        "phases": phases,
        "targets": targets,
        "target_code": sg10._ridge_target_code(targets),
        "group_ids": tuple(example.step_group_id for example in examples),
        "game_seeds": tuple(example.game_seed for example in examples),
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def suffix_spike_kernel(
    query_keys: torch.Tensor,
    prototype_keys: torch.Tensor,
    query_phases: torch.Tensor,
    prototype_phases: torch.Tensor,
    spec: KernelSpec,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if query_keys.ndim != 2 or query_keys.shape[1] != 4:
        raise ValueError("query keys must have shape [query, 4]")
    if prototype_keys.ndim != 2 or prototype_keys.shape[1] != 4:
        raise ValueError("prototype keys must have shape [prototype, 4]")
    if query_phases.shape != (query_keys.shape[0],):
        raise ValueError("query phases must align with keys")
    if prototype_phases.shape != (prototype_keys.shape[0],):
        raise ValueError("prototype phases must align with keys")
    cumulative = query_keys[:, 3, None] == prototype_keys[None, :, 3]
    phase_equal = query_phases[:, None] == prototype_phases[None, :]
    result = float(spec.suffix_weights[0]) * cumulative.to(dtype=dtype)
    product_weight = float(spec.phase_suffix_weights[0])
    if product_weight:
        result = result + product_weight * (cumulative & phase_equal).to(
            dtype=dtype
        )
    for depth in range(1, 4):
        slot = 3 - depth
        cumulative = cumulative & (
            query_keys[:, slot, None] == prototype_keys[None, :, slot]
        )
        weight = float(spec.suffix_weights[depth])
        if weight:
            result = result + weight * cumulative.to(dtype=dtype)
        product_weight = float(spec.phase_suffix_weights[depth])
        if product_weight:
            result = result + product_weight * (
                cumulative & phase_equal
            ).to(dtype=dtype)
    if spec.phase_weight:
        result = result + float(spec.phase_weight) * phase_equal.to(dtype=dtype)
    return result


def build_game_folds(
    game_seeds: Sequence[int], *, fold_count: int
) -> Tuple[Tuple[int, ...], ...]:
    unique = tuple(sorted(set(game_seeds)))
    if len(unique) % fold_count:
        raise ValueError("train games must divide fold count")
    seed_to_fold = {seed: index % fold_count for index, seed in enumerate(unique)}
    folds = tuple(
        tuple(
            index
            for index, seed in enumerate(game_seeds)
            if seed_to_fold[seed] == fold
        )
        for fold in range(fold_count)
    )
    expected = len(game_seeds) // fold_count
    if any(len(indices) != expected for indices in folds):
        raise AssertionError("SG13 game folds are not example-balanced")
    return folds


def _fit_kernel_spec(
    spec: KernelSpec,
    records: Mapping[str, Mapping[str, Any]],
    *,
    lambdas: Sequence[float],
    fold_count: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    train = records["train"]
    started_kernel = time.perf_counter_ns()
    train_kernel = suffix_spike_kernel(
        train["keys"],
        train["keys"],
        train["phases"],
        train["phases"],
        spec,
    )
    _sync(device)
    train_kernel_seconds = (time.perf_counter_ns() - started_kernel) / 1e9
    folds = build_game_folds(train["game_seeds"], fold_count=fold_count)
    all_indices = torch.arange(train_kernel.shape[0], device=device)
    cv_scores = {
        float(value): torch.zeros_like(train["target_code"]) for value in lambdas
    }
    cv_started = time.perf_counter_ns()
    fold_audit = []
    for fold_index, validation_tuple in enumerate(folds):
        validation = torch.tensor(validation_tuple, dtype=torch.long, device=device)
        mask = torch.ones(train_kernel.shape[0], dtype=torch.bool, device=device)
        mask[validation] = False
        training = all_indices[mask]
        kernel_train = train_kernel.index_select(0, training).index_select(1, training)
        kernel_valid = train_kernel.index_select(0, validation).index_select(1, training)
        targets_train = train["target_code"].index_select(0, training)
        identity = torch.eye(
            training.numel(), dtype=torch.float64, device=device
        )
        for ridge_lambda in lambdas:
            alpha = torch.linalg.solve(
                kernel_train + float(ridge_lambda) * identity,
                targets_train,
            )
            cv_scores[float(ridge_lambda)][validation] = kernel_valid @ alpha
        fold_audit.append(
            {
                "fold": fold_index,
                "train_examples": training.numel(),
                "valid_examples": validation.numel(),
                "train_game_count": len(
                    {train["game_seeds"][int(index)] for index in training.cpu()}
                ),
                "valid_game_count": len(
                    {train["game_seeds"][int(index)] for index in validation.cpu()}
                ),
            }
        )
    _sync(device)
    cv_seconds = (time.perf_counter_ns() - cv_started) / 1e9
    candidates = []
    for ridge_lambda in lambdas:
        candidates.append(
            {
                "lambda": float(ridge_lambda),
                "cross_validated": sg10._ridge_multichannel_metrics(
                    cv_scores[float(ridge_lambda)],
                    train["targets"],
                    train["group_ids"],
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda record: (
            -record["cross_validated"]["exact_vector_accuracy"],
            -record["cross_validated"]["macro_channel_accuracy"],
            record["cross_validated"]["mse"],
            record["lambda"],
        ),
    )
    fit_started = time.perf_counter_ns()
    identity = torch.eye(
        train_kernel.shape[0], dtype=torch.float64, device=device
    )
    alpha = torch.linalg.solve(
        train_kernel + float(selected["lambda"]) * identity,
        train["target_code"],
    )
    _sync(device)
    full_fit_seconds = (time.perf_counter_ns() - fit_started) / 1e9
    split_scores = {"train": train_kernel @ alpha}
    evaluation_kernel_seconds = {}
    for split in ("valid", "test"):
        started = time.perf_counter_ns()
        kernel = suffix_spike_kernel(
            records[split]["keys"],
            train["keys"],
            records[split]["phases"],
            train["phases"],
            spec,
        )
        split_scores[split] = kernel @ alpha
        _sync(device)
        evaluation_kernel_seconds[split] = (
            time.perf_counter_ns() - started
        ) / 1e9
    metrics = {
        split: sg10._ridge_multichannel_metrics(
            split_scores[split],
            records[split]["targets"],
            records[split]["group_ids"],
        )
        for split in SPLITS
    }
    result = {
        "kernel": {
            "name": spec.name,
            "suffix_weights": spec.suffix_weights,
            "phase_weight": spec.phase_weight,
            "phase_suffix_weights": spec.phase_suffix_weights,
            "primary": spec.primary,
            "positive_semidefinite_by_construction": True,
        },
        "prototype_count": train_kernel.shape[0],
        "alpha_parameter_count": alpha.numel(),
        "selected_lambda": selected["lambda"],
        "lambda_candidates": tuple(float(value) for value in lambdas),
        "selection_split": "4-fold train-game cross-validation",
        "official_valid_or_test_used_for_selection": False,
        "cross_validation_folds": fold_audit,
        "cross_validation_candidates": candidates,
        "cross_validated": selected["cross_validated"],
        "train": metrics["train"],
        "valid": metrics["valid"],
        "test": metrics["test"],
        "timing": {
            "train_kernel_seconds": train_kernel_seconds,
            "cross_validation_solve_seconds": cv_seconds,
            "full_fit_seconds": full_fit_seconds,
            "selection_plus_full_fit_wall_seconds": (
                records["train"]["elapsed_seconds"]
                + train_kernel_seconds
                + cv_seconds
                + full_fit_seconds
            ),
            "evaluation_kernel_seconds": evaluation_kernel_seconds,
        },
    }
    runtime = {
        "alpha": alpha.detach(),
        "train_kernel": train_kernel.detach(),
        "prototype_keys": train["keys"].detach(),
        "prototype_phases": train["phases"].detach(),
    }
    return result, runtime


def block_schur_kernel_fit(
    records: Mapping[str, Mapping[str, Any]],
    spec: KernelSpec,
    schedule: Sequence[Sequence[int]],
    *,
    ridge_lambda: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    train = records["train"]
    ordered_indices: list[int] = []
    cholesky: Optional[torch.Tensor] = None
    timings = []
    started_all = time.perf_counter_ns()
    for block_tuple in schedule:
        new_indices = tuple(int(index) for index in block_tuple)
        new = torch.tensor(new_indices, dtype=torch.long, device=device)
        new_keys = train["keys"].index_select(0, new)
        new_phases = train["phases"].index_select(0, new)
        _sync(device)
        started = time.perf_counter_ns()
        diagonal = suffix_spike_kernel(
            new_keys, new_keys, new_phases, new_phases, spec
        ) + float(ridge_lambda) * torch.eye(
            new.numel(), dtype=torch.float64, device=device
        )
        if cholesky is None:
            cholesky = torch.linalg.cholesky(diagonal)
        else:
            old = torch.tensor(ordered_indices, dtype=torch.long, device=device)
            old_keys = train["keys"].index_select(0, old)
            old_phases = train["phases"].index_select(0, old)
            cross = suffix_spike_kernel(
                old_keys, new_keys, old_phases, new_phases, spec
            )
            projected = torch.linalg.solve_triangular(
                cholesky, cross, upper=False
            )
            schur = diagonal - projected.T @ projected
            schur = 0.5 * (schur + schur.T)
            schur_cholesky = torch.linalg.cholesky(schur)
            old_dimension = cholesky.shape[0]
            new_cholesky = torch.zeros(
                old_dimension + new.numel(),
                old_dimension + new.numel(),
                dtype=torch.float64,
                device=device,
            )
            new_cholesky[:old_dimension, :old_dimension] = cholesky
            new_cholesky[old_dimension:, :old_dimension] = projected.T
            new_cholesky[old_dimension:, old_dimension:] = schur_cholesky
            cholesky = new_cholesky
        _sync(device)
        timings.append((time.perf_counter_ns() - started) / 1e6)
        ordered_indices.extend(new_indices)
    if cholesky is None or len(ordered_indices) != train["keys"].shape[0]:
        raise AssertionError("SG13 online kernel schedule did not cover train")
    ordered = torch.tensor(ordered_indices, dtype=torch.long, device=device)
    ordered_targets = train["target_code"].index_select(0, ordered)
    alpha_ordered = torch.cholesky_solve(ordered_targets, cholesky)
    alpha = torch.zeros_like(alpha_ordered)
    alpha[ordered] = alpha_ordered
    elapsed = (time.perf_counter_ns() - started_all) / 1e9
    ordered_keys = train["keys"].index_select(0, ordered)
    ordered_phases = train["phases"].index_select(0, ordered)
    ordered_system = suffix_spike_kernel(
        ordered_keys,
        ordered_keys,
        ordered_phases,
        ordered_phases,
        spec,
    ) + float(ridge_lambda) * torch.eye(
        ordered.numel(), dtype=torch.float64, device=device
    )
    return alpha, {
        "factorization": "block Cholesky update of each Schur complement",
        "block_updates": len(schedule),
        "examples_seen": len(ordered_indices),
        "block_timing": {
            **_sample_summary(timings, 1),
            "p95_ms": _percentile(timings, 0.95),
        },
        "elapsed_seconds": elapsed,
        "final_factor_dimension": cholesky.shape[0],
        "final_factor_reconstruction_max_abs_error": float(
            (cholesky @ cholesky.T - ordered_system).abs().max().item()
        ),
    }


def evaluate_online_kernel_replication(
    records: Mapping[str, Mapping[str, Any]],
    spec: KernelSpec,
    runtime: Mapping[str, torch.Tensor],
    batch_result: Mapping[str, Any],
    schedule: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    alpha, timing = block_schur_kernel_fit(
        records,
        spec,
        schedule,
        ridge_lambda=float(batch_result["selected_lambda"]),
        device=device,
    )
    batch_alpha = runtime["alpha"]
    metrics = {}
    max_score_difference = 0.0
    prediction_equivalent = True
    for split in SPLITS:
        kernel = suffix_spike_kernel(
            records[split]["keys"],
            runtime["prototype_keys"],
            records[split]["phases"],
            runtime["prototype_phases"],
            spec,
        )
        online_scores = kernel @ alpha
        batch_scores = kernel @ batch_alpha
        max_score_difference = max(
            max_score_difference,
            float((online_scores - batch_scores).abs().max().item()),
        )
        prediction_equivalent = prediction_equivalent and bool(
            torch.equal(
                sg10._prediction_matrix(online_scores),
                sg10._prediction_matrix(batch_scores),
            )
        )
        metrics[split] = sg10._ridge_multichannel_metrics(
            online_scores,
            records[split]["targets"],
            records[split]["group_ids"],
        )
    return {
        "algorithm": (
            "block Cholesky-Schur kernel recursive least squares; "
            "no explicit ill-conditioned inverse"
        ),
        "max_alpha_abs_difference_from_batch": float(
            (alpha - batch_alpha).abs().max().item()
        ),
        "max_score_abs_difference_from_batch": max_score_difference,
        "prediction_equivalent_to_batch": prediction_equivalent,
        "train": metrics["train"],
        "valid": metrics["valid"],
        "test": metrics["test"],
        "timing": timing,
    }


def _key_from_delay_state(
    state: torch.Tensor, candidate_index: int, *, pad_index: int
) -> torch.Tensor:
    if state.shape[0] != 1 or state.shape[1] != 3:
        raise ValueError("SG13 stream state must be [1,3,alphabet]")
    ordered = torch.flip(state[0], dims=(0,))
    active = ordered.any(dim=1)
    indices = ordered.to(torch.long).argmax(dim=1)
    indices = torch.where(
        active,
        indices,
        torch.full_like(indices, pad_index),
    )
    candidate = torch.tensor(
        (candidate_index,), dtype=torch.long, device=state.device
    )
    return torch.cat((indices, candidate), dim=0)[None]


def evaluate_cached_kernel_stream(
    examples: Sequence[sg10.MultiChannelExample],
    *,
    alphabet_index: Mapping[str, int],
    spec: KernelSpec,
    runtime: Mapping[str, torch.Tensor],
    device: torch.device,
    timing_repeats: int,
    timing_warmup_repeats: int,
) -> Dict[str, Any]:
    alphabet_size = len(alphabet_index)
    pad_index = alphabet_size
    eye = torch.eye(alphabet_size, dtype=torch.bool, device=device)
    prototype_keys = runtime["prototype_keys"].to(device=device)
    prototype_phases = runtime["prototype_phases"].to(device=device)
    alpha = runtime["alpha"].to(device=device, dtype=torch.float32)
    groups: Dict[str, list[sg10.MultiChannelExample]] = {}
    for example in examples:
        groups.setdefault(example.step_group_id, []).append(example)
    accumulator = sg10._metric_accumulator()
    timings = []
    max_full_difference = 0.0
    state_bytes = []
    with torch.inference_mode():
        for group_id in sorted(groups):
            group = sorted(groups[group_id], key=lambda value: value.candidate_index)
            context_actions = group[0].context_actions
            prefix_state = delay_state_after(
                context_actions,
                order=PRIMARY_ORDER,
                alphabet_index=alphabet_index,
                device=device,
            )
            prefix_phase = torch.tensor(
                (len(context_actions),), dtype=torch.long, device=device
            )
            state_bytes.append(
                prefix_state.numel() * prefix_state.element_size()
                + prefix_phase.numel() * prefix_phase.element_size()
            )
            for example in group:
                candidate_index = action_index(
                    example.candidate_action, alphabet_index
                )

                def forward_candidate() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    candidate_spike = eye[candidate_index : candidate_index + 1]
                    next_state = delay_step(prefix_state, candidate_spike)
                    next_phase = prefix_phase + 1
                    query_key = _key_from_delay_state(
                        prefix_state, candidate_index, pad_index=pad_index
                    )
                    kernel = suffix_spike_kernel(
                        query_key,
                        prototype_keys,
                        prefix_phase,
                        prototype_phases,
                        spec,
                        dtype=torch.float32,
                    )
                    scores = kernel @ alpha
                    return scores, next_state, next_phase

                scores, _next_state, _next_phase = forward_candidate()
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
                    timed_scores, timed_state, timed_phase = forward_candidate()
                    (
                        timed_scores.sum()
                        + timed_state.sum()
                        + timed_phase.sum()
                    ).item()
                    _sync(device)
                    if repeat >= timing_warmup_repeats:
                        timings.append((time.perf_counter_ns() - started) / 1e6)
                rebuilt_state = delay_state_after(
                    example.context_actions,
                    order=PRIMARY_ORDER,
                    alphabet_index=alphabet_index,
                    device=device,
                )
                rebuilt_key = _key_from_delay_state(
                    rebuilt_state, candidate_index, pad_index=pad_index
                )
                full_kernel = suffix_spike_kernel(
                    rebuilt_key,
                    prototype_keys,
                    prefix_phase,
                    prototype_phases,
                    spec,
                    dtype=torch.float32,
                )
                full_scores = full_kernel @ alpha
                max_full_difference = max(
                    max_full_difference,
                    float((scores - full_scores).abs().max().item()),
                )
    result = sg10._finalize_metrics(accumulator)
    logical_model_bytes = (
        prototype_keys.numel()
        + prototype_phases.numel()
        + alpha.numel() * alpha.element_size()
    )
    result.update(
        {
            "mode": "persistent_spike_delay_plus_suffix_kernel_memory",
            "kernel": spec.name,
            "prototype_count": prototype_keys.shape[0],
            "alpha_parameter_count": alpha.numel(),
            "persistent_state_bytes_max": max(state_bytes),
            "logical_model_storage_bytes_uint8_keys_float32_alpha": logical_model_bytes,
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
    kernel_results: Mapping[str, Mapping[str, Any]],
    online_results: Sequence[Mapping[str, Any]],
    stream_results: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    sg12_reference: Mapping[str, Any],
    *,
    quick: bool,
    fresh_confirmation: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "primary_quality_gate": "SMOKE",
            "mechanism_gate": "SMOKE",
            "online_equivalence_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "cached_stream_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": (
                "formal_sg13r_fresh_game_confirmation"
                if fresh_confirmation
                else "formal_sg13_suffix_spike_kernel"
            ),
        }
    primary = kernel_results[PRIMARY_SPEC.name]
    primary_test = primary["test"]
    quality = _quality_gate(primary_test, reference)
    suffix2_cv = kernel_results["suffix2"]["cross_validated"][
        "exact_vector_accuracy"
    ]
    sg12_exact = sg12_reference["decision"]["primary_test_metrics"][
        "exact_vector_accuracy"
    ]
    mechanism = (
        primary["cross_validated"]["exact_vector_accuracy"] >= suffix2_cv
        and (
            primary_test["exact_vector_accuracy"] >= sg12_exact + 0.04
            or quality
        )
    )
    online_equivalent = all(
        result["prediction_equivalent_to_batch"]
        and result["max_score_abs_difference_from_batch"] <= 1e-6
        for result in online_results
    )
    transformer_walls = [
        seed["training"]["transformer"]["elapsed_seconds"]
        for seed in reference["seeds"]
    ]
    selection_wall = primary["timing"]["selection_plus_full_fit_wall_seconds"]
    training_speed = (
        online_equivalent
        and selection_wall <= sg0._mean(transformer_walls)
        and all(
            result["timing"]["elapsed_seconds"] <= transformer_wall
            for result, transformer_wall in zip(online_results, transformer_walls)
        )
    )
    transformer_parameter_bytes = max(
        seed["parameter_counts"]["transformer"]["total"] * 4
        for seed in reference["seeds"]
    )
    stream_comparison = []
    for replication, (stream, reference_seed) in enumerate(
        zip(stream_results, reference["seeds"])
    ):
        ann = reference_seed["cached_stream"]["transformer"]["generic"][
            "candidate_timing"
        ]
        passed = (
            abs(
                stream["exact_vector_accuracy"]
                - primary_test["exact_vector_accuracy"]
            )
            <= 1e-12
            and stream["max_full_score_abs_difference"] <= 1e-6
            and stream["candidate_timing"]["p50_ms"] <= ann["p50_ms"]
            and stream["candidate_timing"]["p95_ms"] <= ann["p95_ms"]
            and stream[
                "logical_model_storage_bytes_uint8_keys_float32_alpha"
            ]
            <= transformer_parameter_bytes
        )
        stream_comparison.append(
            {
                "replication": replication,
                "snn_p50_ms": stream["candidate_timing"]["p50_ms"],
                "snn_p95_ms": stream["candidate_timing"]["p95_ms"],
                "transformer_p50_ms": ann["p50_ms"],
                "transformer_p95_ms": ann["p95_ms"],
                "snn_model_bytes": stream[
                    "logical_model_storage_bytes_uint8_keys_float32_alpha"
                ],
                "transformer_parameter_bytes": transformer_parameter_bytes,
                "passed": passed,
            }
        )
    stream_pass = quality and all(record["passed"] for record in stream_comparison)
    overall = (
        data_audit["passed"]
        and quality
        and mechanism
        and online_equivalent
        and training_speed
        and stream_pass
    )
    return {
        "data_gate": "PASS" if data_audit["passed"] else "FAIL",
        "primary_quality_gate": "PASS" if quality else "FAIL",
        "mechanism_gate": "PASS" if mechanism else "FAIL",
        "online_equivalence_gate": "PASS" if online_equivalent else "FAIL",
        "training_speed_gate": "PASS" if training_speed else "FAIL",
        "cached_stream_gate": "PASS" if stream_pass else "FAIL",
        "overall": "PASS" if overall else "FAIL",
        "primary_kernel": PRIMARY_SPEC.name,
        "primary_test_metrics": primary_test,
        "primary_cross_validated_metrics": primary["cross_validated"],
        "suffix2_cross_validated_exact": suffix2_cv,
        "sg12_primary_exact": sg12_exact,
        "selection_plus_fit_wall_seconds": selection_wall,
        "transformer_mean_training_wall_seconds": sg0._mean(transformer_walls),
        "per_replication_stream_comparison": stream_comparison,
        "independent_fresh_game_confirmation_required": not fresh_confirmation,
        "fresh_game_confirmation": fresh_confirmation,
        "next_route": (
            (
                "sg14_closed_loop_candidate_planner"
                if fresh_confirmation
                else "sg13r_fresh_games_then_closed_loop_planner"
            )
            if overall
            else "sg14_frozen_reservoir_plus_exact_delay_phase_kernel"
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
    reference, reference_digest = _reference_artifact(
        args.reference.expanduser().resolve()
    )
    sg12_reference, sg12_digest = _load_sg12_reference(
        args.sg12_reference.expanduser().resolve()
    )
    sg13_reference = None
    sg13_digest = None
    if args.fresh_confirmation:
        sg13_reference, sg13_digest = _load_sg13_reference(
            args.sg13_reference.expanduser().resolve()
        )
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
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    records = {
        split: extract_kernel_records(
            examples[split], alphabet_index=alphabet_index, device=device
        )
        for split in SPLITS
    }
    folds = build_game_folds(
        records["train"]["game_seeds"], fold_count=args.cv_folds
    )
    data_audit = {
        "sg10_multichannel": base_audit,
        "spike_delay_line": delay_audit,
        "cv_fold_count": len(folds),
        "cv_fold_example_counts": tuple(len(fold) for fold in folds),
        "official_valid_or_test_used_for_selection": False,
        "passed": (
            base_audit["passed"]
            and delay_audit["passed"]
            and len(folds) == args.cv_folds
            and len(set(len(fold) for fold in folds)) == 1
        ),
    }
    if args.fresh_confirmation:
        observed_seeds = {
            split: tuple(sorted({example.game_seed for example in examples[split]}))
            for split in SPLITS
        }
        seed_match = observed_seeds == CONFIRMATION_SEEDS
        pairwise_disjoint = all(
            not (set(observed_seeds[left]) & set(observed_seeds[right]))
            for left_index, left in enumerate(SPLITS)
            for right in SPLITS[left_index + 1 :]
        )
        data_audit["fresh_confirmation"] = {
            "observed_seeds": observed_seeds,
            "expected_seeds": CONFIRMATION_SEEDS,
            "exact_seed_match": seed_match,
            "pairwise_seed_disjoint": pairwise_disjoint,
            "kernel_and_lambda_frozen_before_generation": True,
        }
        data_audit["passed"] = (
            data_audit["passed"] and seed_match and pairwise_disjoint
        )
    if not data_audit["passed"]:
        raise AssertionError("SG13 data audit failed")
    kernel_results = {}
    runtimes = {}
    for spec in KERNEL_SPECS:
        kernel_results[spec.name], runtimes[spec.name] = _fit_kernel_spec(
            spec,
            records,
            lambdas=args.ridge_lambdas,
            fold_count=args.cv_folds,
            device=device,
        )
    primary_result = kernel_results[PRIMARY_SPEC.name]
    primary_runtime = runtimes[PRIMARY_SPEC.name]
    confirmation_reproduction = None
    if args.fresh_confirmation:
        if sg13_reference is None:
            raise AssertionError("SG13 confirmation reference was not loaded")
        frozen = sg13_reference["kernel_results"][PRIMARY_SPEC.name]
        confirmation_reproduction = {
            "original_primary_kernel": frozen["kernel"]["name"],
            "current_primary_kernel": primary_result["kernel"]["name"],
            "original_selected_lambda": frozen["selected_lambda"],
            "current_selected_lambda": primary_result["selected_lambda"],
            "cross_validated_exact_difference": abs(
                frozen["cross_validated"]["exact_vector_accuracy"]
                - primary_result["cross_validated"]["exact_vector_accuracy"]
            ),
            "train_exact_difference": abs(
                frozen["train"]["exact_vector_accuracy"]
                - primary_result["train"]["exact_vector_accuracy"]
            ),
        }
        confirmation_reproduction["passed"] = (
            confirmation_reproduction["original_primary_kernel"]
            == confirmation_reproduction["current_primary_kernel"]
            == PRIMARY_SPEC.name
            and confirmation_reproduction["original_selected_lambda"]
            == confirmation_reproduction["current_selected_lambda"]
            == 1e-6
            and confirmation_reproduction["cross_validated_exact_difference"]
            <= 1e-12
            and confirmation_reproduction["train_exact_difference"] <= 1e-12
        )
        if not confirmation_reproduction["passed"]:
            raise AssertionError("SG13R frozen train/kernel reproduction failed")
    online_results = []
    stream_results = []
    for seed in args.seeds:
        schedule = sg10.build_length_stratified_schedule(
            examples["train"],
            epochs=1,
            batch_groups=args.batch_groups,
            seed=13_001_000 + seed,
        )
        online = evaluate_online_kernel_replication(
            records,
            PRIMARY_SPEC,
            primary_runtime,
            primary_result,
            schedule,
            device=device,
        )
        online["schedule_seed"] = seed
        online_results.append(online)
        stream = evaluate_cached_kernel_stream(
            examples["test"],
            alphabet_index=alphabet_index,
            spec=PRIMARY_SPEC,
            runtime=primary_runtime,
            device=device,
            timing_repeats=args.timing_repeats,
            timing_warmup_repeats=args.timing_warmup_repeats,
        )
        stream["timing_replication"] = seed
        stream_results.append(stream)
    decision = _decision(
        data_audit,
        kernel_results,
        online_results,
        stream_results,
        reference,
        sg12_reference,
        quick=args.quick,
        fresh_confirmation=args.fresh_confirmation,
    )
    return {
        "schema_version": 1,
        "experiment": (
            "E3-SG13R fresh-game suffix spike kernel confirmation"
            if args.fresh_confirmation
            else "E3-SG13 hierarchical suffix spike kernel associative memory"
        ),
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": (
                "independent fresh procedural game confirmation"
                if args.fresh_confirmation
                else "mechanism experiment; original test already observed"
            ),
            "statement": (
                "A PSD hierarchy of exact and backed-off spike suffix matches can "
                "generalize the exact delay state without iterative backpropagation."
            ),
            "what_if": (
                "What if sparse event memory should interpolate through a suffix "
                "kernel rather than expand every multi-event conjunction?"
            ),
        },
        "references": {
            "sg10": {
                "path": str(args.reference.expanduser().resolve()),
                "sha256": reference_digest,
            },
            "sg12": {
                "path": str(args.sg12_reference.expanduser().resolve()),
                "sha256": sg12_digest,
            },
            "sg13_frozen_architecture": (
                {
                    "path": str(args.sg13_reference.expanduser().resolve()),
                    "sha256": sg13_digest,
                }
                if args.fresh_confirmation
                else None
            ),
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "primary_kernel": PRIMARY_SPEC.name,
            "kernel_specs": tuple(spec.name for spec in KERNEL_SPECS),
            "cv_folds": args.cv_folds,
            "cv_split_key": "train game seed",
            "ridge_lambdas": tuple(args.ridge_lambdas),
            "schedule_seeds": tuple(args.seeds),
            "batch_groups": args.batch_groups,
            "block_examples": args.batch_groups * 3,
            "threads": args.threads if device.type == "cpu" else None,
            "timing_repeats_per_candidate": args.timing_repeats,
            "timing_warmup_repeats_per_candidate": args.timing_warmup_repeats,
            "training_dtype": "float64",
            "deployment_alpha_dtype": "float32",
            "fresh_confirmation": args.fresh_confirmation,
        },
        "dataset": {
            "synthetic": False,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "action_alphabet": alphabet,
            "audit": data_audit,
        },
        "kernel_results": kernel_results,
        "confirmation_reproduction": confirmation_reproduction,
        "online_replications": online_results,
        "cached_stream_replications": stream_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=sg10.DEFAULT_CORPUS_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sg12-reference", type=Path, default=DEFAULT_SG12_REFERENCE)
    parser.add_argument("--sg13-reference", type=Path, default=DEFAULT_SG13_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=4)
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
    parser.add_argument("--fresh-confirmation", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.threads,
        args.cv_folds,
        args.batch_groups,
        args.timing_repeats,
        *args.ridge_lambdas,
        *args.expected_counts,
        *args.expected_groups,
    ) <= 0:
        parser.error("all numeric controls must be positive")
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
