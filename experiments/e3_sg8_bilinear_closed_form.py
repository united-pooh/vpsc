"""SG8 bilinear relation binding and closed-form SNN reservoir readout."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
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
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg1_history_generation import build_history_models  # noqa: E402
from experiments.e3_sg4_move_pair_ranking import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    EXPECTED_DATA_SEEDS,
)
from experiments.e3_sg6_move_delta import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_STEP_GROUPS,
    LABELS,
    MODEL_NAMES,
    MoveDeltaExample,
    audit_move_delta_examples,
    build_move_delta_examples,
)
from experiments.e3_sg7_paired_binary_batch import (  # noqa: E402
    build_paired_batch_schedule,
)
from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


RIDGE_LAMBDAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)


class BilinearRelationModel(nn.Module):
    def __init__(self, language_model: Any, d_model: int) -> None:
        super().__init__()
        self.language_model = language_model
        self.relation_head = nn.Bilinear(d_model, d_model, 2, bias=True)


def build_bilinear_models(
    seed: int,
    vocabulary: Vocabulary,
    *,
    d_model: int,
    state_dim: int,
    num_heads: int,
    device: torch.device,
) -> Dict[str, BilinearRelationModel]:
    base_models = build_history_models(
        seed,
        vocabulary,
        d_model=d_model,
        state_dim=state_dim,
        num_heads=num_heads,
        device=device,
    )
    torch.manual_seed(seed + 30_000)
    reference_head = nn.Bilinear(d_model, d_model, 2, bias=True)
    shared_head = copy.deepcopy(reference_head.state_dict())
    models = {}
    for name, base in base_models.items():
        model = BilinearRelationModel(base, d_model)
        model.relation_head.load_state_dict(shared_head)
        models[name] = model.to(device)
    return models


def action_query_indices(
    example: MoveDeltaExample,
    vocabulary: Vocabulary,
) -> Tuple[int, int]:
    eos_positions = tuple(
        index
        for index, token_id in enumerate(example.prompt_ids)
        if token_id == vocabulary.eos_id
    )
    if len(eos_positions) != 2 or min(eos_positions) == 0:
        raise ValueError("SG8 prompt must contain exactly two non-leading EOS markers")
    return eos_positions[0] - 1, eos_positions[1] - 1


def _bilinear_batch_tensors(
    examples: Sequence[MoveDeltaExample],
    indices: Sequence[int],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = tuple(examples[index] for index in indices)
    lengths = {len(example.prompt_ids) for example in selected}
    queries = {action_query_indices(example, vocabulary) for example in selected}
    if len(lengths) != 1 or len(queries) != 1:
        raise ValueError("SG8 batch requires equal prompt and action query positions")
    input_ids = torch.tensor(
        [example.prompt_ids for example in selected],
        dtype=torch.long,
        device=device,
    )
    query_indices = torch.tensor(
        next(iter(queries)), dtype=torch.long, device=device
    )
    label_to_index = {
        vocabulary.token_id(label): index for index, label in enumerate(LABELS)
    }
    targets = torch.tensor(
        [label_to_index[example.target_ids[0]] for example in selected],
        dtype=torch.long,
        device=device,
    )
    return input_ids, query_indices, targets


def _query_hidden(
    language_model: Any,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    use_eligibility: bool,
    detach_state: bool,
) -> Tuple[torch.Tensor, Any]:
    embedded = language_model.input_dropout(language_model.embedding(input_ids))
    if use_eligibility:
        if not isinstance(language_model.core, E3GatedTraceScanCore):
            raise TypeError("eligibility queries require a gated trace SNN core")
        result = language_model.core.forward_multi_query_eligibility(
            embedded,
            query_indices,
            None,
            detach_state=detach_state,
            _unchecked=True,
        )
        sequence = result.sequence
    else:
        result = language_model.core(
            embedded,
            None,
            detach_state=detach_state,
        )
        sequence = result.sequence.index_select(1, query_indices)
    hidden = language_model.output_norm(
        language_model.output_dropout(sequence)
    )
    return hidden, result.state


def bilinear_logits(
    model: BilinearRelationModel,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
    *,
    use_eligibility: bool,
    detach_state: bool,
) -> Tuple[torch.Tensor, Any]:
    hidden, state = _query_hidden(
        model.language_model,
        input_ids,
        query_indices,
        use_eligibility=use_eligibility,
        detach_state=detach_state,
    )
    if hidden.shape[1] != 2:
        raise ValueError("bilinear relation head requires exactly two hidden queries")
    return model.relation_head(hidden[:, 0], hidden[:, 1]), state


def evaluate_bilinear(
    model: BilinearRelationModel,
    examples: Sequence[MoveDeltaExample],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    include_records: bool,
    batch_size: int = 64,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    relation_loss: Dict[str, float] = defaultdict(float)
    relation_correct: Counter[str] = Counter()
    relation_count: Counter[str] = Counter()
    predictions_by_id: Dict[str, Tuple[int, float]] = {}
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(examples))))
            input_ids, query_indices, targets = _bilinear_batch_tensors(
                examples, indices, vocabulary, device=device
            )
            logits, _state = bilinear_logits(
                model,
                input_ids,
                query_indices,
                use_eligibility=False,
                detach_state=True,
            )
            losses = F.cross_entropy(logits, targets, reduction="none")
            predictions = logits.argmax(dim=-1)
            correct = predictions == targets
            total_loss += float(losses.sum().item())
            total_correct += int(correct.sum().item())
            total_count += len(indices)
            margins = logits.gather(1, targets[:, None])[:, 0] - logits.gather(
                1, (1 - targets)[:, None]
            )[:, 0]
            for offset, example_index in enumerate(indices):
                example = examples[example_index]
                relation = vocabulary.decode(example.target_ids)[0]
                relation_loss[relation] += float(losses[offset].item())
                relation_correct[relation] += int(correct[offset].item())
                relation_count[relation] += 1
                predictions_by_id[example.example_id] = (
                    int(predictions[offset].item()),
                    float(margins[offset].item()),
                )

    timings = []
    state_sizes = []
    step_correct: Dict[str, list[bool]] = defaultdict(list)
    records = []
    with torch.inference_mode():
        for index, example in enumerate(examples):
            input_ids, query_indices, targets = _bilinear_batch_tensors(
                examples, (index,), vocabulary, device=device
            )
            _sync(device)
            started = time.perf_counter_ns()
            logits, state = bilinear_logits(
                model,
                input_ids,
                query_indices,
                use_eligibility=False,
                detach_state=True,
            )
            logits.sum().item()
            _sync(device)
            timings.append((time.perf_counter_ns() - started) / 1e6)
            state_sizes.append(state_nbytes(state))
            predicted = int(logits.argmax(dim=-1).item())
            target = int(targets.item())
            correct = predicted == target
            step_correct[example.step_group_id].append(correct)
            if include_records:
                _, margin = predictions_by_id[example.example_id]
                records.append(
                    {
                        "example_id": example.example_id,
                        "step_group_id": example.step_group_id,
                        "source": example.source,
                        "previous_action": example.previous_action,
                        "candidate_action": example.candidate_action,
                        "target_label": LABELS[target],
                        "predicted_label": LABELS[predicted],
                        "correct": correct,
                        "margin": margin,
                    }
                )
    step_consistency = sum(all(values) for values in step_correct.values())
    return {
        "binary_nll": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "step_consistency": step_consistency / len(step_correct),
        "example_count": total_count,
        "step_group_count": len(step_correct),
        "relations": {
            relation: {
                "binary_nll": relation_loss[relation] / relation_count[relation],
                "accuracy": relation_correct[relation] / relation_count[relation],
                "example_count": relation_count[relation],
            }
            for relation in LABELS
        },
        "timing": {
            **_sample_summary(timings, 1),
            "p99_ms": _percentile(timings, 0.99),
            "state_bytes_max": max(state_sizes),
            "state_bytes_mean": sg0._mean(state_sizes),
        },
        "records": records if include_records else None,
    }


def train_bilinear(
    name: str,
    model: BilinearRelationModel,
    examples: Sequence[MoveDeltaExample],
    vocabulary: Vocabulary,
    schedule: Sequence[Sequence[int]],
    *,
    epochs: int,
    batches_per_epoch: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.train(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-3, weight_decay=0.01, fused=True
    )
    batch_timings = []
    example_timings = []
    losses = []
    example_exposures = 0
    input_tokens = 0
    epoch_loss = 0.0
    epoch_examples = 0
    epoch_records = []
    started_all = time.perf_counter_ns()
    for update, indices in enumerate(schedule):
        input_ids, query_indices, targets = _bilinear_batch_tensors(
            examples, indices, vocabulary, device=device
        )
        _sync(device)
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits, _state = bilinear_logits(
            model,
            input_ids,
            query_indices,
            use_eligibility=name in ("snn_at1", "snn_ra0"),
            detach_state=True,
        )
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite SG8 loss for {name} at update {update + 1}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, foreach=True
        )
        optimizer.step()
        _sync(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        batch_size = len(indices)
        batch_timings.append(elapsed_ms)
        example_timings.append(elapsed_ms / batch_size)
        value = float(loss.detach().item())
        losses.append(value)
        example_exposures += batch_size
        input_tokens += input_ids.numel()
        epoch_loss += value * batch_size
        epoch_examples += batch_size
        if (update + 1) % batches_per_epoch == 0:
            epoch_records.append(
                {
                    "epoch": len(epoch_records) + 1,
                    "binary_nll": epoch_loss / epoch_examples,
                    "example_exposures": epoch_examples,
                    "last_gradient_norm": float(gradient_norm),
                }
            )
            epoch_loss = 0.0
            epoch_examples = 0
    elapsed_seconds = (time.perf_counter_ns() - started_all) / 1e9
    warmup_updates = len(batch_timings) // 5
    return {
        "epochs": epochs,
        "updates": len(schedule),
        "batches_per_epoch": batches_per_epoch,
        "batch_examples": len(schedule[0]),
        "example_exposures": example_exposures,
        "input_tokens": input_tokens,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_epoch": epoch_records[-1]["binary_nll"],
        "batch_timing": {
            **_sample_summary(batch_timings[warmup_updates:], 1),
            "warmup_updates_excluded": warmup_updates,
        },
        "example_equivalent_timing": {
            **_sample_summary(example_timings[warmup_updates:], 1),
            "warmup_updates_excluded": warmup_updates,
        },
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second_total": example_exposures / elapsed_seconds,
        "epoch_records": epoch_records,
    }


def _outer_features(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[1] != 2:
        raise ValueError("outer relation features require [batch, 2, hidden]")
    previous = hidden[:, 0]
    candidate = hidden[:, 1]
    outer = torch.einsum("bi,bj->bij", previous, candidate).flatten(1)
    ones = torch.ones(
        previous.shape[0], 1, dtype=previous.dtype, device=previous.device
    )
    return torch.cat((ones, previous, candidate, outer), dim=1)


def extract_reservoir_features(
    language_model: Any,
    examples: Sequence[MoveDeltaExample],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    language_model.eval()
    features = []
    targets = []
    group_ids = []
    started = time.perf_counter_ns()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(examples))))
            input_ids, query_indices, label_indices = _bilinear_batch_tensors(
                examples, indices, vocabulary, device=device
            )
            hidden, _state = _query_hidden(
                language_model,
                input_ids,
                query_indices,
                use_eligibility=False,
                detach_state=True,
            )
            features.append(_outer_features(hidden).to(dtype=torch.float64))
            targets.append(
                torch.where(
                    label_indices == 0,
                    torch.ones_like(label_indices, dtype=torch.float64),
                    -torch.ones_like(label_indices, dtype=torch.float64),
                )
            )
            group_ids.extend(examples[index].step_group_id for index in indices)
    _sync(device)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    return {
        "features": torch.cat(features, dim=0),
        "targets": torch.cat(targets, dim=0),
        "group_ids": tuple(group_ids),
        "elapsed_seconds": elapsed_seconds,
    }


def _ridge_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    group_ids: Sequence[str],
) -> Dict[str, Any]:
    predictions = torch.where(scores >= 0.0, 1.0, -1.0)
    correct = predictions == targets
    groups: Dict[str, list[bool]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(bool(correct[index].item()))
    target01 = (targets + 1.0) / 2.0
    return {
        "accuracy": float(correct.to(torch.float64).mean().item()),
        "step_consistency": sum(all(values) for values in groups.values())
        / len(groups),
        "mse": float(F.mse_loss(scores, targets).item()),
        "binary_nll": float(
            F.binary_cross_entropy_with_logits(scores, target01).item()
        ),
        "example_count": targets.numel(),
        "step_group_count": len(groups),
    }


def fit_closed_form_ridge(
    language_model: Any,
    examples: Mapping[str, Sequence[MoveDeltaExample]],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    lambdas: Sequence[float],
) -> Dict[str, Any]:
    extracted = {
        split: extract_reservoir_features(
            language_model, examples[split], vocabulary, device=device
        )
        for split in SPLITS
    }
    train_x = extracted["train"]["features"]
    train_y = extracted["train"]["targets"]
    mean = train_x[:, 1:].mean(dim=0)
    scale = train_x[:, 1:].std(dim=0, unbiased=False).clamp_min(1e-8)

    def transform(values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                values[:, :1],
                (values[:, 1:] - mean) / scale,
            ),
            dim=1,
        )

    x_train = transform(train_x)
    x_valid = transform(extracted["valid"]["features"])
    x_test = transform(extracted["test"]["features"])
    y_valid = extracted["valid"]["targets"]
    y_test = extracted["test"]["targets"]
    identity = torch.eye(x_train.shape[0], dtype=torch.float64, device=device)
    gram = x_train @ x_train.T
    candidates = []
    started_fit = time.perf_counter_ns()
    weights_by_lambda = {}
    for ridge_lambda in lambdas:
        alpha = torch.linalg.solve(
            gram + float(ridge_lambda) * identity,
            train_y,
        )
        weights = x_train.T @ alpha
        weights_by_lambda[float(ridge_lambda)] = weights
        valid_metrics = _ridge_metrics(
            x_valid @ weights,
            y_valid,
            extracted["valid"]["group_ids"],
        )
        candidates.append(
            {
                "lambda": float(ridge_lambda),
                "valid": valid_metrics,
            }
        )
    selected = min(
        candidates,
        key=lambda record: (
            -record["valid"]["accuracy"],
            record["valid"]["mse"],
            record["lambda"],
        ),
    )
    selected_lambda = selected["lambda"]
    weights = weights_by_lambda[selected_lambda]
    _sync(device)
    fit_seconds = (time.perf_counter_ns() - started_fit) / 1e9
    train_metrics = _ridge_metrics(
        x_train @ weights,
        train_y,
        extracted["train"]["group_ids"],
    )
    test_metrics = _ridge_metrics(
        x_test @ weights,
        y_test,
        extracted["test"]["group_ids"],
    )

    timings = []
    state_sizes = []
    with torch.inference_mode():
        for index, example in enumerate(examples["test"]):
            input_ids, query_indices, _targets = _bilinear_batch_tensors(
                examples["test"], (index,), vocabulary, device=device
            )
            _sync(device)
            started = time.perf_counter_ns()
            hidden, state = _query_hidden(
                language_model,
                input_ids,
                query_indices,
                use_eligibility=False,
                detach_state=True,
            )
            raw = _outer_features(hidden).to(dtype=torch.float64)
            transformed = transform(raw)
            score = transformed @ weights
            score.item()
            _sync(device)
            timings.append((time.perf_counter_ns() - started) / 1e6)
            state_sizes.append(state_nbytes(state))
    feature_seconds = sum(
        extracted[split]["elapsed_seconds"] for split in ("train", "valid")
    )
    return {
        "feature_dimension": x_train.shape[1],
        "readout_parameter_count": weights.numel(),
        "lambda_candidates": tuple(float(value) for value in lambdas),
        "selection_rule": (
            "max valid accuracy, then min valid MSE, then min lambda"
        ),
        "selected_lambda": selected_lambda,
        "validation_candidates": candidates,
        "train": train_metrics,
        "valid": selected["valid"],
        "test": test_metrics,
        "feature_extraction_seconds": {
            split: extracted[split]["elapsed_seconds"] for split in SPLITS
        },
        "fit_seconds": fit_seconds,
        "training_wall_seconds": feature_seconds + fit_seconds,
        "timing": {
            **_sample_summary(timings, 1),
            "p99_ms": _percentile(timings, 0.99),
            "state_bytes_max": max(state_sizes),
            "state_bytes_mean": sg0._mean(state_sizes),
        },
    }


def _decision(
    data_audit: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "task_gate": "SMOKE",
            "trainable_snn_quality_gate": "SMOKE",
            "ridge_quality_gate": "SMOKE",
            "trainable_speed_gate": "SMOKE",
            "ridge_speed_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg8_bilinear_closed_form",
        }

    mean_nll = {
        name: sg0._mean(seed["post"][name]["test"]["binary_nll"] for seed in seed_results)
        for name in MODEL_NAMES
    }
    mean_accuracy = {
        name: sg0._mean(seed["post"][name]["test"]["accuracy"] for seed in seed_results)
        for name in MODEL_NAMES
    }
    mean_step = {
        name: sg0._mean(
            seed["post"][name]["test"]["step_consistency"] for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    best_ann_accuracy = max(mean_accuracy["lstm"], mean_accuracy["transformer"])
    best_ann_step = max(mean_step["lstm"], mean_step["transformer"])
    best_ann_nll = min(mean_nll["lstm"], mean_nll["transformer"])
    task_pass = best_ann_accuracy >= 0.98 and best_ann_step >= 0.95
    nll_gap_bptt = abs(mean_nll["snn_ra0"] - mean_nll["snn_bptt"])
    nll_gap_at1 = abs(mean_nll["snn_ra0"] - mean_nll["snn_at1"])
    accuracy_gap_bptt = abs(
        mean_accuracy["snn_ra0"] - mean_accuracy["snn_bptt"]
    )
    accuracy_gap_at1 = abs(
        mean_accuracy["snn_ra0"] - mean_accuracy["snn_at1"]
    )
    trainable_quality = (
        mean_accuracy["snn_ra0"] >= best_ann_accuracy - 0.02
        and mean_accuracy["snn_ra0"] >= 0.98
        and mean_step["snn_ra0"] >= 0.95
        and mean_nll["snn_ra0"] <= best_ann_nll + 0.05
        and nll_gap_bptt <= 0.05
        and nll_gap_at1 <= 0.05
        and accuracy_gap_bptt <= 0.02
        and accuracy_gap_at1 <= 0.02
    )
    mean_ridge_accuracy = sg0._mean(
        seed["closed_form_ridge"]["test"]["accuracy"] for seed in seed_results
    )
    mean_ridge_step = sg0._mean(
        seed["closed_form_ridge"]["test"]["step_consistency"]
        for seed in seed_results
    )
    ridge_quality = mean_ridge_accuracy >= 0.98 and mean_ridge_step >= 0.95

    ann_qualified = tuple(
        name
        for name in ("lstm", "transformer")
        if mean_accuracy[name] >= 0.98 and mean_step[name] >= 0.95
    )
    if not ann_qualified:
        ann_qualified = ("lstm", "transformer")
    mean_example_p50 = {
        name: sg0._mean(
            seed["training"][name]["example_equivalent_timing"]["p50_ms"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_elapsed = {
        name: sg0._mean(
            seed["training"][name]["elapsed_seconds"] for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_response_p50 = {
        name: sg0._mean(seed["post"][name]["test"]["timing"]["p50_ms"] for seed in seed_results)
        for name in MODEL_NAMES
    }
    mean_response_p95 = {
        name: sg0._mean(seed["post"][name]["test"]["timing"]["p95_ms"] for seed in seed_results)
        for name in MODEL_NAMES
    }
    ann_example_p50 = min(mean_example_p50[name] for name in ann_qualified)
    ann_elapsed = min(mean_elapsed[name] for name in ann_qualified)
    ann_response_p50 = min(mean_response_p50[name] for name in ann_qualified)
    ann_response_p95 = min(mean_response_p95[name] for name in ann_qualified)
    at1_speedup = mean_example_p50["snn_at1"] / mean_example_p50["snn_ra0"]
    bptt_speedup = mean_example_p50["snn_bptt"] / mean_example_p50["snn_ra0"]
    trainable_speed = (
        at1_speedup >= 1.25
        and bptt_speedup >= 1.25
        and mean_example_p50["snn_ra0"] <= ann_example_p50
        and mean_elapsed["snn_ra0"] <= ann_elapsed
        and mean_response_p50["snn_ra0"] <= ann_response_p50
        and mean_response_p95["snn_ra0"] <= ann_response_p95
    )
    mean_ridge_wall = sg0._mean(
        seed["closed_form_ridge"]["training_wall_seconds"] for seed in seed_results
    )
    mean_ridge_p50 = sg0._mean(
        seed["closed_form_ridge"]["timing"]["p50_ms"] for seed in seed_results
    )
    mean_ridge_p95 = sg0._mean(
        seed["closed_form_ridge"]["timing"]["p95_ms"] for seed in seed_results
    )
    ridge_speed = (
        mean_ridge_wall <= ann_elapsed
        and mean_ridge_p50 <= ann_response_p50
        and mean_ridge_p95 <= ann_response_p95
    )
    candidate_pass = (
        trainable_quality and trainable_speed
    ) or (ridge_quality and ridge_speed)
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "task_gate": task_pass,
        "trainable_snn_quality_gate": trainable_quality,
        "ridge_quality_gate": ridge_quality,
        "trainable_speed_gate": trainable_speed,
        "ridge_speed_gate": ridge_speed,
    }
    overall = "PASS" if data_audit["passed"] and task_pass and candidate_pass else "FAIL"
    if ridge_quality:
        next_route = "sg9_recursive_closed_form_multichannel_delta_and_fused_stream"
    elif trainable_quality:
        next_route = "sg9_native_bilinear_spike_scan"
    else:
        next_route = "sg9_group_equivariant_direction_code"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "mean_test_binary_nll": mean_nll,
        "mean_test_accuracy": mean_accuracy,
        "mean_test_step_consistency": mean_step,
        "best_ann_nll": best_ann_nll,
        "best_ann_accuracy": best_ann_accuracy,
        "best_ann_step_consistency": best_ann_step,
        "mean_ridge_accuracy": mean_ridge_accuracy,
        "mean_ridge_step_consistency": mean_ridge_step,
        "mean_training_example_equivalent_p50_ms": mean_example_p50,
        "mean_training_elapsed_seconds": mean_elapsed,
        "mean_response_p50_ms": mean_response_p50,
        "mean_response_p95_ms": mean_response_p95,
        "mean_ridge_training_wall_seconds": mean_ridge_wall,
        "mean_ridge_response_p50_ms": mean_ridge_p50,
        "mean_ridge_response_p95_ms": mean_ridge_p95,
        "ra0_vs_at1_training_speedup": at1_speedup,
        "ra0_vs_bptt_training_speedup": bptt_speedup,
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
    expected_data_seeds = {
        "train": tuple(args.expected_train_seeds),
        "valid": tuple(args.expected_valid_seeds),
        "test": tuple(args.expected_test_seeds),
    }
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=expected_data_seeds
    )
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = build_move_delta_examples(corpus_root, corpus)
    expected_counts = dict(zip(SPLITS, args.expected_counts))
    expected_step_groups = dict(zip(SPLITS, args.expected_step_groups))
    data_audit = audit_move_delta_examples(
        examples,
        vocabulary,
        expected_counts=expected_counts,
        expected_step_groups=expected_step_groups,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG8 data audit failed; refusing experiment")
    if expected_step_groups["train"] % args.batch_groups:
        raise AssertionError("SG8 train groups must divide batch_groups")
    batches_per_epoch = expected_step_groups["train"] // args.batch_groups

    seed_results = []
    for seed in args.seeds:
        models = build_bilinear_models(
            10_000_000 + 100 * seed,
            vocabulary,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_heads=args.num_heads,
            device=device,
        )
        ridge_reservoir = copy.deepcopy(models["snn_ra0"].language_model)
        parameter_counts = {
            name: {
                "total": count_parameters(model),
                "core": count_parameters(model.language_model.core),
                "relation_head": count_parameters(model.relation_head),
            }
            for name, model in models.items()
        }
        totals = tuple(record["total"] for record in parameter_counts.values())
        parameter_spread = (max(totals) - min(totals)) / sg0._mean(totals)
        if parameter_spread > 0.03:
            raise AssertionError(f"SG8 parameter spread failed: {parameter_counts}")
        pre = {
            name: {
                split: evaluate_bilinear(
                    model,
                    examples[split],
                    vocabulary,
                    device=device,
                    include_records=False,
                )
                for split in ("valid", "test")
            }
            for name, model in models.items()
        }
        schedule = build_paired_batch_schedule(
            examples["train"],
            epochs=args.epochs,
            batch_groups=args.batch_groups,
            seed=10_001_000 + seed,
        )
        training = {
            name: train_bilinear(
                name,
                model,
                examples["train"],
                vocabulary,
                schedule,
                epochs=args.epochs,
                batches_per_epoch=batches_per_epoch,
                device=device,
            )
            for name, model in models.items()
        }
        post = {
            name: {
                split: evaluate_bilinear(
                    model,
                    examples[split],
                    vocabulary,
                    device=device,
                    include_records=split == "test",
                )
                for split in ("train", "valid", "test")
            }
            for name, model in models.items()
        }
        closed_form = fit_closed_form_ridge(
            ridge_reservoir,
            examples,
            vocabulary,
            device=device,
            lambdas=args.ridge_lambdas,
        )
        seed_results.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_relative_spread": parameter_spread,
                "pre": pre,
                "training": training,
                "post": post,
                "closed_form_ridge": closed_form,
            }
        )
    decision = _decision(data_audit, seed_results, quick=args.quick)
    return {
        "schema_version": 1,
        "experiment": "E3-SG8 bilinear spike binding and closed-form ridge",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "speculative new idea plus established solver",
            "statement": (
                "Explicit second-order event binding can supply the relation "
                "operation missing from first-order SNN traces, while a frozen "
                "reservoir may permit closed-form readout training."
            ),
            "what_if": (
                "What if multiplicative synapses are the native world-relation "
                "primitive and iterative temporal backpropagation is unnecessary "
                "for its readout?"
            ),
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "expected_data_seeds": expected_data_seeds,
            "expected_counts": expected_counts,
            "expected_step_groups": expected_step_groups,
            "epochs": args.epochs,
            "seeds": tuple(args.seeds),
            "threads": args.threads if device.type == "cpu" else None,
            "d_model": args.d_model,
            "state_dim": args.state_dim,
            "num_heads": args.num_heads,
            "batch_groups": args.batch_groups,
            "batch_examples": args.batch_groups * 2,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_updates_per_model": batches_per_epoch * args.epochs,
            "example_exposures_per_model": len(examples["train"]) * args.epochs,
            "trainable_head": "Bilinear(d_model,d_model,2)",
            "ridge_features": "[1,h_prev,h_candidate,vec(h_prev outer h_candidate)]",
            "ridge_lambdas": tuple(args.ridge_lambdas),
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "max_prompt_tokens": args.max_prompt_tokens,
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "corpus_provenance": tw0._corpus_provenance(corpus),
            "vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
                "source_split": "train",
                "source_fields": ("previous_move", "candidate_move", "label"),
                "tokenizer": corpus.tokenizer.metadata(),
            },
            "audit": data_audit,
        },
        "seeds": seed_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg8_bilinear_closed_form.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=32)
    parser.add_argument(
        "--ridge-lambdas", nargs="+", type=float, default=RIDGE_LAMBDAS
    )
    parser.add_argument(
        "--expected-counts",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_COUNTS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-step-groups",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_STEP_GROUPS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-train-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["train"],
    )
    parser.add_argument(
        "--expected-valid-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["valid"],
    )
    parser.add_argument(
        "--expected-test-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["test"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.epochs,
        args.threads,
        args.d_model,
        args.state_dim,
        args.num_heads,
        args.batch_groups,
        args.max_prompt_tokens,
        *args.ridge_lambdas,
        *args.expected_counts,
        *args.expected_step_groups,
        *args.expected_train_seeds,
        *args.expected_valid_seeds,
        *args.expected_test_seeds,
    ) <= 0:
        parser.error("all numeric experiment controls must be positive")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
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
