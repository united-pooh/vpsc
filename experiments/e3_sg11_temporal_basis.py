"""SG11 recursive temporal-basis diagnostics for persistent SNN world state."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
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
from experiments.e3_sg8_bilinear_closed_form import (  # noqa: E402
    RIDGE_LAMBDAS,
    _outer_features,
    _query_hidden,
)
from experiments.e3_sg9_atomic_event_stream import (  # noqa: E402
    _snn_cached_decay_candidate_hidden,
)
from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore,
    E3ScanState,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS  # noqa: E402


REFERENCE_SHA256 = (
    "56BD001A17AD7093F4B3A37329B9B2083AD127F848072F5881148C941C02A77F"
)
DEFAULT_REFERENCE = Path("results/e3_scan/e3_sg10_multichannel_delta.json")
DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg11_temporal_basis.json")


@dataclass(frozen=True)
class BasisSpec:
    name: str
    state_dim: int
    deployable: bool
    recurrence: str


BASIS_SPECS = (
    BasisSpec("baseline", 0, False, "original SG10 gated trace only"),
    BasisSpec("unit_root1", 1, True, "c <- c + 1"),
    BasisSpec(
        "leaky4",
        4,
        True,
        "z <- exp(-1/tau) z + (1-exp(-1/tau)), tau=(1,2,4,8)",
    ),
    BasisSpec(
        "oscillator4",
        4,
        True,
        "two fixed 2D rotations with periods 8 and 16 events",
    ),
    BasisSpec(
        "binary3",
        3,
        True,
        "three nonlinear spike bits increment modulo 8",
    ),
    BasisSpec(
        "one_hot6_oracle",
        6,
        False,
        "six-state ring shift diagnostic upper bound",
    ),
)
BASIS_BY_NAME = {spec.name: spec for spec in BASIS_SPECS}
DEPLOYABLE_NAMES = tuple(spec.name for spec in BASIS_SPECS if spec.deployable)


def _basis_initial(
    spec: BasisSpec,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("basis batch_size must be positive")
    if spec.name == "baseline":
        return torch.empty(batch_size, 0, device=device, dtype=dtype)
    if spec.name == "oscillator4":
        return torch.tensor(
            (0.0, 1.0, 0.0, 1.0), device=device, dtype=dtype
        ).expand(batch_size, -1).clone()
    if spec.name == "one_hot6_oracle":
        state = torch.zeros(batch_size, 6, device=device, dtype=dtype)
        state[:, 0] = 1.0
        return state
    return torch.zeros(batch_size, spec.state_dim, device=device, dtype=dtype)


def _basis_step(spec: BasisSpec, state: torch.Tensor) -> torch.Tensor:
    if tuple(state.shape[-1:]) != (spec.state_dim,):
        raise ValueError(
            f"invalid {spec.name} state width {state.shape[-1]}, "
            f"expected {spec.state_dim}"
        )
    if spec.name == "baseline":
        return state
    if spec.name == "unit_root1":
        return state + 1.0
    if spec.name == "leaky4":
        taus = torch.tensor(
            (1.0, 2.0, 4.0, 8.0), device=state.device, dtype=state.dtype
        )
        decay = torch.exp(-1.0 / taus)
        return decay * state + (1.0 - decay)
    if spec.name == "oscillator4":
        parts = []
        for start, period in ((0, 8.0), (2, 16.0)):
            sine = state[:, start]
            cosine = state[:, start + 1]
            theta = 2.0 * math.pi / period
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            parts.extend(
                (
                    sine * cos_theta + cosine * sin_theta,
                    cosine * cos_theta - sine * sin_theta,
                )
            )
        return torch.stack(parts, dim=1)
    if spec.name == "binary3":
        bit0, bit1, bit2 = state.unbind(dim=1)
        carry1 = bit0 * bit1
        next0 = 1.0 - bit0
        next1 = bit1 + bit0 - 2.0 * bit1 * bit0
        next2 = bit2 + carry1 - 2.0 * bit2 * carry1
        return torch.stack((next0, next1, next2), dim=1)
    if spec.name == "one_hot6_oracle":
        return torch.roll(state, shifts=1, dims=1)
    raise KeyError(f"unknown temporal basis {spec.name}")


def _basis_feature(spec: BasisSpec, state: torch.Tensor) -> torch.Tensor:
    if spec.name == "binary3":
        return 2.0 * state - 1.0
    return state


def _basis_state_after(
    spec: BasisSpec,
    event_count: int,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if event_count < 0:
        raise ValueError("event_count must be nonnegative")
    state = _basis_initial(
        spec, batch_size, device=device, dtype=dtype
    )
    for _ in range(event_count):
        state = _basis_step(spec, state)
    return state


def temporal_basis_values(
    spec: BasisSpec,
    event_counts: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Evaluate a basis by recurrence only, never from labels or game identity."""

    if event_counts.dtype != torch.long or bool(torch.any(event_counts < 0)):
        raise ValueError("event_counts must be a nonnegative long tensor")
    if spec.state_dim == 0:
        return torch.empty(
            *event_counts.shape,
            0,
            device=event_counts.device,
            dtype=dtype,
        )
    table = []
    state = _basis_initial(
        spec, 1, device=event_counts.device, dtype=dtype
    )
    table.append(_basis_feature(spec, state)[0])
    maximum = int(event_counts.max().item()) if event_counts.numel() else 0
    for _ in range(maximum):
        state = _basis_step(spec, state)
        table.append(_basis_feature(spec, state)[0])
    values = torch.stack(table, dim=0)
    return values[event_counts]


def augment_hidden_with_basis(
    hidden: torch.Tensor,
    event_counts: torch.Tensor,
    spec: BasisSpec,
) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[1] != 2:
        raise ValueError("SG11 hidden must have shape [batch, 2, dimension]")
    if tuple(event_counts.shape) != tuple(hidden.shape[:2]):
        raise ValueError("event_counts must align with the two hidden queries")
    if spec.state_dim == 0:
        return hidden
    if spec.state_dim >= hidden.shape[-1]:
        raise ValueError("temporal basis must leave event-content coordinates")
    basis = temporal_basis_values(spec, event_counts, dtype=hidden.dtype)
    augmented = hidden.clone()
    augmented[:, :, -spec.state_dim :] = basis
    return augmented


def _augment_single_hidden(
    hidden: torch.Tensor,
    state: torch.Tensor,
    spec: BasisSpec,
) -> torch.Tensor:
    if hidden.ndim != 2:
        raise ValueError("single hidden must be [batch, dimension]")
    if spec.state_dim == 0:
        return hidden
    result = hidden.clone()
    result[:, -spec.state_dim :] = _basis_feature(spec, state).to(hidden.dtype)
    return result


def _reference_artifact(path: Path) -> Tuple[Dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != REFERENCE_SHA256:
        raise ValueError(
            f"SG10 reference SHA mismatch: expected {REFERENCE_SHA256}, got {digest}"
        )
    result = json.loads(payload)
    if result["experiment"] != "E3-SG10 multichannel TextWorld event delta":
        raise ValueError("unexpected SG10 reference experiment")
    return result, digest


def _extract_hidden_splits(
    language_model: Any,
    examples: Mapping[str, Sequence[sg10.MultiChannelExample]],
    *,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    language_model.eval()
    extracted = {}
    with torch.inference_mode():
        for split in SPLITS:
            hidden_parts = []
            target_parts = []
            count_parts = []
            group_ids = []
            by_length: Dict[int, list[int]] = defaultdict(list)
            for index, example in enumerate(examples[split]):
                by_length[len(example.prompt_ids)].append(index)
            started = time.perf_counter_ns()
            for length in sorted(by_length):
                values = by_length[length]
                for start in range(0, len(values), 96):
                    indices = tuple(values[start : start + 96])
                    input_ids, query_indices, targets = sg10._batch_tensors(
                        examples[split], indices, device=device
                    )
                    hidden, _state = _query_hidden(
                        language_model,
                        input_ids,
                        query_indices,
                        use_eligibility=False,
                        detach_state=True,
                    )
                    hidden_parts.append(hidden)
                    target_parts.append(targets)
                    count_parts.append(
                        torch.tensor(
                            ((length - 1, length),),
                            dtype=torch.long,
                            device=device,
                        ).expand(len(indices), -1)
                    )
                    group_ids.extend(
                        examples[split][index].step_group_id for index in indices
                    )
            _sync(device)
            extracted[split] = {
                "hidden": torch.cat(hidden_parts),
                "targets": torch.cat(target_parts),
                "event_counts": torch.cat(count_parts),
                "group_ids": tuple(group_ids),
                "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
            }
    return extracted


def _variant_features(
    raw: Mapping[str, Any], spec: BasisSpec
) -> Tuple[torch.Tensor, float]:
    started = time.perf_counter_ns()
    augmented = augment_hidden_with_basis(
        raw["hidden"], raw["event_counts"], spec
    )
    features = _outer_features(augmented).to(torch.float64)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    return features, elapsed


def _fit_variant(
    spec: BasisSpec,
    raw: Mapping[str, Mapping[str, Any]],
    *,
    lambdas: Sequence[float],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    feature_sets = {}
    feature_times = {}
    for split in SPLITS:
        feature_sets[split], feature_times[split] = _variant_features(
            raw[split], spec
        )
    train_x = feature_sets["train"]
    train_y = sg10._ridge_target_code(raw["train"]["targets"])
    mean = train_x[:, 1:].mean(dim=0)
    scale = train_x[:, 1:].std(dim=0, unbiased=False).clamp_min(1e-8)

    def transform(values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (values[:, :1], (values[:, 1:] - mean) / scale), dim=1
        )

    transformed = {split: transform(feature_sets[split]) for split in SPLITS}
    identity = torch.eye(train_x.shape[0], dtype=torch.float64, device=device)
    gram = transformed["train"] @ transformed["train"].T
    candidates = []
    weights_by_lambda = {}
    started_fit = time.perf_counter_ns()
    for ridge_lambda in lambdas:
        alpha = torch.linalg.solve(
            gram + float(ridge_lambda) * identity, train_y
        )
        weights = transformed["train"].T @ alpha
        weights_by_lambda[float(ridge_lambda)] = weights
        candidates.append(
            {
                "lambda": float(ridge_lambda),
                "valid": sg10._ridge_multichannel_metrics(
                    transformed["valid"] @ weights,
                    raw["valid"]["targets"],
                    raw["valid"]["group_ids"],
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
    _sync(device)
    fit_seconds = (time.perf_counter_ns() - started_fit) / 1e9
    metrics = {
        split: sg10._ridge_multichannel_metrics(
            transformed[split] @ weights,
            raw[split]["targets"],
            raw[split]["group_ids"],
        )
        for split in SPLITS
    }
    raw_training_seconds = sum(
        raw[split]["elapsed_seconds"] for split in ("train", "valid")
    )
    variant_training_seconds = sum(
        feature_times[split] for split in ("train", "valid")
    )
    result = {
        "basis": {
            "name": spec.name,
            "state_dimension": spec.state_dim,
            "deployable": spec.deployable,
            "recurrence": spec.recurrence,
            "label_or_game_conditioned": False,
        },
        "feature_dimension": transformed["train"].shape[1],
        "output_dimension": sg10.TOTAL_LOGITS,
        "readout_parameter_count": weights.numel(),
        "selected_lambda": selected["lambda"],
        "lambda_candidates": tuple(float(value) for value in lambdas),
        "selection_rule": "max valid exact, max valid macro, min MSE, min lambda",
        "validation_candidates": candidates,
        "train": metrics["train"],
        "valid": selected["valid"],
        "test": metrics["test"],
        "timing": {
            "raw_reservoir_extraction_seconds": {
                split: raw[split]["elapsed_seconds"] for split in SPLITS
            },
            "basis_feature_seconds": feature_times,
            "fit_seconds": fit_seconds,
            "independent_training_wall_seconds": (
                raw_training_seconds + variant_training_seconds + fit_seconds
            ),
        },
    }
    runtime = {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "weights": weights.detach(),
    }
    return result, runtime


def _global_variant_selection(
    seed_results: Sequence[Mapping[str, Any]],
) -> Tuple[str, Tuple[Dict[str, Any], ...]]:
    audit = []
    for name in DEPLOYABLE_NAMES:
        records = [seed["variants"][name]["valid"] for seed in seed_results]
        audit.append(
            {
                "name": name,
                "state_dimension": BASIS_BY_NAME[name].state_dim,
                "mean_valid_exact_vector_accuracy": sg0._mean(
                    record["exact_vector_accuracy"] for record in records
                ),
                "mean_valid_macro_channel_accuracy": sg0._mean(
                    record["macro_channel_accuracy"] for record in records
                ),
                "mean_valid_mse": sg0._mean(record["mse"] for record in records),
            }
        )
    order = {name: index for index, name in enumerate(DEPLOYABLE_NAMES)}
    selected = min(
        audit,
        key=lambda record: (
            -record["mean_valid_exact_vector_accuracy"],
            -record["mean_valid_macro_channel_accuracy"],
            record["mean_valid_mse"],
            record["state_dimension"],
            order[record["name"]],
        ),
    )
    return selected["name"], tuple(audit)


def _deployment_transform(
    features: torch.Tensor, runtime: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    mean = runtime["mean"].to(device=features.device, dtype=features.dtype)
    scale = runtime["scale"].to(device=features.device, dtype=features.dtype)
    return torch.cat(
        (features[:, :1], (features[:, 1:] - mean) / scale), dim=1
    )


def evaluate_cached_basis_stream(
    language_model: Any,
    examples: Sequence[sg10.MultiChannelExample],
    spec: BasisSpec,
    runtime: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    timing_repeats: int,
    timing_warmup_repeats: int,
) -> Dict[str, Any]:
    if not spec.deployable:
        raise ValueError("cached stream requires a deployable temporal basis")
    core = language_model.core
    if not isinstance(core, E3GatedTraceScanCore):
        raise TypeError("SG11 cached stream requires the gated trace SNN")
    language_model.eval()
    groups: Dict[str, list[sg10.MultiChannelExample]] = defaultdict(list)
    example_indices = {id(example): index for index, example in enumerate(examples)}
    for example in examples:
        groups[example.step_group_id].append(example)
    accumulator = sg10._metric_accumulator()
    timings = []
    prefix_timings = []
    state_sizes = []
    max_full_difference = 0.0
    decays = core.decays()
    deploy_runtime = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in runtime.items()
    }

    with torch.inference_mode():
        for group_id in sorted(groups):
            group = sorted(groups[group_id], key=lambda value: value.candidate_index)
            context = group[0].prompt_ids[:-1]
            prefix_input = torch.tensor([context], dtype=torch.long, device=device)
            prefix_query = torch.tensor(
                (len(context) - 1,), dtype=torch.long, device=device
            )
            _sync(device)
            prefix_started = time.perf_counter_ns()
            hidden, prefix_state = _query_hidden(
                language_model,
                prefix_input,
                prefix_query,
                use_eligibility=False,
                detach_state=True,
            )
            if not isinstance(prefix_state, E3ScanState):
                raise TypeError("SG11 prefix returned invalid SNN state")
            prefix_basis_state = _basis_state_after(
                spec,
                len(context),
                batch_size=1,
                device=device,
                dtype=hidden.dtype,
            )
            previous_hidden = _augment_single_hidden(
                hidden[:, 0], prefix_basis_state, spec
            )
            previous_hidden.sum().item()
            _sync(device)
            prefix_timings.append(
                (time.perf_counter_ns() - prefix_started) / 1e6
            )
            state_sizes.append(
                state_nbytes(prefix_state)
                + prefix_basis_state.numel() * prefix_basis_state.element_size()
            )

            for example in group:
                candidate_id = example.prompt_ids[-1]

                def forward_candidate() -> torch.Tensor:
                    candidate_hidden, _next_state = (
                        _snn_cached_decay_candidate_hidden(
                            language_model,
                            candidate_id,
                            prefix_state,
                            decays,
                            device=device,
                        )
                    )
                    candidate_basis_state = _basis_step(spec, prefix_basis_state)
                    augmented_candidate = _augment_single_hidden(
                        candidate_hidden, candidate_basis_state, spec
                    )
                    pair = torch.stack(
                        (previous_hidden, augmented_candidate), dim=1
                    )
                    features = _outer_features(pair)
                    transformed = _deployment_transform(features, deploy_runtime)
                    return transformed @ deploy_runtime["weights"]

                scores = forward_candidate()
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
                    timed_scores = forward_candidate()
                    timed_scores.sum().item()
                    _sync(device)
                    if repeat >= timing_warmup_repeats:
                        timings.append((time.perf_counter_ns() - started) / 1e6)

                index = example_indices[id(example)]
                full_input, full_query, _targets = sg10._batch_tensors(
                    examples, (index,), device=device
                )
                full_hidden, _full_state = _query_hidden(
                    language_model,
                    full_input,
                    full_query,
                    use_eligibility=False,
                    detach_state=True,
                )
                counts = torch.tensor(
                    ((len(example.prompt_ids) - 1, len(example.prompt_ids)),),
                    dtype=torch.long,
                    device=device,
                )
                full_augmented = augment_hidden_with_basis(
                    full_hidden, counts, spec
                )
                full_features = _outer_features(full_augmented)
                full_scores = _deployment_transform(
                    full_features, deploy_runtime
                ) @ deploy_runtime["weights"]
                max_full_difference = max(
                    max_full_difference,
                    float((scores - full_scores).abs().max().item()),
                )

    result = sg10._finalize_metrics(accumulator)
    result.update(
        {
            "basis": spec.name,
            "deployment_dtype": "float32",
            "mode": "snn_cached_decay_plus_recursive_basis_ridge",
            "max_full_score_abs_difference": max_full_difference,
            "prefix_timing": {
                **_sample_summary(prefix_timings, 1),
                "p99_ms": _percentile(prefix_timings, 0.99),
            },
            "candidate_timing": {
                **_sample_summary(timings, 1),
                "p99_ms": _percentile(timings, 0.99),
            },
            "candidate_timing_sample_count": len(timings),
            "prefix_state_bytes_max": max(state_sizes),
        }
    )
    return result


def _mean_test_metrics(
    seed_results: Sequence[Mapping[str, Any]], selected_name: str
) -> Dict[str, Any]:
    tests = [seed["variants"][selected_name]["test"] for seed in seed_results]
    channel_accuracy = {
        name: sg0._mean(test["channel_accuracy"][name] for test in tests)
        for name, _labels in sg10.CHANNEL_SPECS
    }
    return {
        "exact_vector_accuracy": sg0._mean(
            test["exact_vector_accuracy"] for test in tests
        ),
        "macro_channel_accuracy": sg0._mean(
            test["macro_channel_accuracy"] for test in tests
        ),
        "channel_accuracy": channel_accuracy,
        "reward_positive_recall": sg0._mean(
            test["class_recall"]["reward"][sg10.REWARD_LABELS[1]]
            for test in tests
        ),
        "done_positive_recall": sg0._mean(
            test["class_recall"]["done"][sg10.DONE_LABELS[1]]
            for test in tests
        ),
    }


def _decision(
    seed_results: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    selected_name: str,
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "diagnostic_gate": "SMOKE",
            "deployable_quality_gate": "SMOKE",
            "closed_form_speed_gate": "SMOKE",
            "cached_stream_gate": "SMOKE",
            "overall": "SMOKE",
            "selected_deployable_basis": selected_name,
            "next_route": "formal_sg11_temporal_basis",
        }
    baseline_replication = []
    for seed, reference_seed in zip(seed_results, reference["seeds"]):
        observed = seed["variants"]["baseline"]["test"]
        expected = reference_seed["closed_form_ridge"]["test"]
        baseline_replication.append(
            {
                "seed": seed["seed"],
                "observed_exact": observed["exact_vector_accuracy"],
                "expected_exact": expected["exact_vector_accuracy"],
                "absolute_exact_difference": abs(
                    observed["exact_vector_accuracy"]
                    - expected["exact_vector_accuracy"]
                ),
                "passed": abs(
                    observed["exact_vector_accuracy"]
                    - expected["exact_vector_accuracy"]
                )
                <= 1.0 / 60.0 + 1e-12,
            }
        )
    oracle_exact = sg0._mean(
        seed["variants"]["one_hot6_oracle"]["test"][
            "exact_vector_accuracy"
        ]
        for seed in seed_results
    )
    diagnostic_pass = all(record["passed"] for record in baseline_replication) and (
        oracle_exact >= 0.98
    )
    mean_metrics = _mean_test_metrics(seed_results, selected_name)
    quality_pass = (
        mean_metrics["exact_vector_accuracy"] >= 0.98
        and mean_metrics["macro_channel_accuracy"] >= 0.98
        and all(value >= 0.95 for value in mean_metrics["channel_accuracy"].values())
        and mean_metrics["reward_positive_recall"] >= 0.90
        and mean_metrics["done_positive_recall"] >= 0.90
        and mean_metrics["exact_vector_accuracy"]
        >= reference["decision"]["best_ann_exact_vector_accuracy"] - 0.02
        and mean_metrics["macro_channel_accuracy"]
        >= reference["decision"]["best_ann_macro_channel_accuracy"] - 0.02
    )
    selected_walls = [
        seed["variants"][selected_name]["timing"][
            "independent_training_wall_seconds"
        ]
        for seed in seed_results
    ]
    transformer_walls = [
        seed["training"]["transformer"]["elapsed_seconds"]
        for seed in reference["seeds"]
    ]
    closed_form_speed = sg0._mean(selected_walls) <= sg0._mean(transformer_walls)
    stream_comparison = []
    for seed, reference_seed in zip(seed_results, reference["seeds"]):
        stream = seed["cached_stream"]
        ann = reference_seed["cached_stream"]["transformer"]["generic"][
            "candidate_timing"
        ]
        equivalent = stream["max_full_score_abs_difference"] <= 1e-5
        quality_equivalent = (
            abs(
                stream["exact_vector_accuracy"]
                - seed["variants"][selected_name]["test"][
                    "exact_vector_accuracy"
                ]
            )
            <= 1e-12
        )
        passed = (
            equivalent
            and quality_equivalent
            and stream["candidate_timing"]["p50_ms"] <= ann["p50_ms"]
            and stream["candidate_timing"]["p95_ms"] <= ann["p95_ms"]
        )
        stream_comparison.append(
            {
                "seed": seed["seed"],
                "snn_p50_ms": stream["candidate_timing"]["p50_ms"],
                "snn_p95_ms": stream["candidate_timing"]["p95_ms"],
                "transformer_p50_ms": ann["p50_ms"],
                "transformer_p95_ms": ann["p95_ms"],
                "full_cached_equivalent": equivalent,
                "float32_quality_equivalent": quality_equivalent,
                "passed": passed,
            }
        )
    cached_speed = quality_pass and all(
        record["passed"] for record in stream_comparison
    )
    overall = (
        diagnostic_pass and quality_pass and closed_form_speed and cached_speed
    )
    return {
        "diagnostic_gate": "PASS" if diagnostic_pass else "FAIL",
        "deployable_quality_gate": "PASS" if quality_pass else "FAIL",
        "closed_form_speed_gate": "PASS" if closed_form_speed else "FAIL",
        "cached_stream_gate": "PASS" if cached_speed else "FAIL",
        "overall": "PASS" if overall else "FAIL",
        "selected_deployable_basis": selected_name,
        "baseline_replication": baseline_replication,
        "one_hot_oracle_mean_test_exact": oracle_exact,
        "selected_mean_test_metrics": mean_metrics,
        "mean_selected_training_wall_seconds": sg0._mean(selected_walls),
        "mean_transformer_training_wall_seconds": sg0._mean(transformer_walls),
        "per_seed_stream_comparison": stream_comparison,
        "next_route": (
            "sg12_online_rls_closed_loop_rollout"
            if overall
            else (
                "sg12_learned_critical_or_unit_circle_basis"
                if diagnostic_pass
                else "sg12_action_history_associative_state"
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
    expected_counts = dict(zip(SPLITS, args.expected_counts))
    expected_groups = dict(zip(SPLITS, args.expected_groups))
    data_audit = sg10.audit_multichannel_examples(
        examples,
        vocabulary,
        expected_counts=expected_counts,
        expected_groups=expected_groups,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG11 reused SG10 data audit failed")

    seed_results = []
    runtimes = []
    reservoirs = []
    for seed in args.seeds:
        models = sg10.build_multichannel_models(
            10_200_000 + 100 * seed,
            vocabulary,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_heads=args.num_heads,
            device=device,
        )
        reservoir = copy.deepcopy(models["snn_ra0"].language_model)
        raw = _extract_hidden_splits(reservoir, examples, device=device)
        variants = {}
        variant_runtimes = {}
        for spec in BASIS_SPECS:
            variants[spec.name], variant_runtimes[spec.name] = _fit_variant(
                spec,
                raw,
                lambdas=args.ridge_lambdas,
                device=device,
            )
            if variants[spec.name]["feature_dimension"] != 1089:
                raise AssertionError("SG11 feature dimension must remain fixed")
        seed_results.append(
            {
                "seed": seed,
                "frozen_reservoir_parameter_count": count_parameters(reservoir),
                "variants": variants,
            }
        )
        runtimes.append(variant_runtimes)
        reservoirs.append(reservoir)

    selected_name, selection_audit = _global_variant_selection(seed_results)
    selected_spec = BASIS_BY_NAME[selected_name]
    for index, seed_result in enumerate(seed_results):
        seed_result["cached_stream"] = evaluate_cached_basis_stream(
            reservoirs[index],
            examples["test"],
            selected_spec,
            runtimes[index][selected_name],
            device=device,
            timing_repeats=args.timing_repeats,
            timing_warmup_repeats=args.timing_warmup_repeats,
        )
    decision = _decision(
        seed_results,
        reference,
        selected_name,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG11 recursive temporal basis for persistent SNN state",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "mechanism diagnostic plus deployable recurrence",
            "statement": (
                "A tiny autonomous temporal basis can restore event phase to a "
                "persistent gated-trace SNN without expanding its ridge feature space."
            ),
            "what_if": (
                "What if SG10 lacks an autonomous phase carrier rather than action "
                "content, and a recurrent spike clock closes the Transformer gap?"
            ),
        },
        "reference": {
            "path": str(reference_path),
            "sha256": reference_digest,
            "experiment": reference["experiment"],
            "best_ann": "transformer",
            "best_ann_exact_vector_accuracy": reference["decision"][
                "best_ann_exact_vector_accuracy"
            ],
            "best_ann_macro_channel_accuracy": reference["decision"][
                "best_ann_macro_channel_accuracy"
            ],
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "seeds": tuple(args.seeds),
            "threads": args.threads if device.type == "cpu" else None,
            "d_model": args.d_model,
            "state_dim": args.state_dim,
            "num_heads": args.num_heads,
            "timing_repeats_per_candidate": args.timing_repeats,
            "timing_warmup_repeats_per_candidate": args.timing_warmup_repeats,
            "ridge_lambdas": tuple(args.ridge_lambdas),
            "variant_names": tuple(spec.name for spec in BASIS_SPECS),
            "deployable_names": DEPLOYABLE_NAMES,
            "global_selection_rule": (
                "max mean valid exact, max mean valid macro, min mean valid MSE, "
                "min state dimension, frozen candidate order"
            ),
            "fixed_feature_dimension": 1089,
            "fixed_readout_parameter_count": 1089 * sg10.TOTAL_LOGITS,
        },
        "dataset": {
            "synthetic": False,
            "audit": data_audit,
            "vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
            },
        },
        "global_variant_selection": {
            "selected": selected_name,
            "test_used_for_selection": False,
            "candidates": selection_audit,
        },
        "seeds": seed_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=sg10.DEFAULT_CORPUS_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=31)
    parser.add_argument("--num-heads", type=int, default=4)
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
        args.d_model,
        args.state_dim,
        args.num_heads,
        args.timing_repeats,
        *args.ridge_lambdas,
        *args.expected_counts,
        *args.expected_groups,
    ) <= 0:
        parser.error("all numeric experiment controls must be positive")
    if args.timing_warmup_repeats < 0:
        parser.error("timing-warmup-repeats must be nonnegative")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
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
