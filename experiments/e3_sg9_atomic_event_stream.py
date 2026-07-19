"""SG9 atomic-event bilinear world delta with cached streaming inference."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
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
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
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
from experiments.e3_sg8_bilinear_closed_form import (  # noqa: E402
    RIDGE_LAMBDAS,
    BilinearRelationModel,
    _outer_features,
    _query_hidden,
    _ridge_metrics,
    bilinear_logits,
    build_bilinear_models,
)
from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore,
    E3ScanState,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


_EVENT_COMPONENT = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class AtomicEventExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    candidate_index: int
    source: str
    step_group_id: str
    previous_action: str
    candidate_action: str
    previous_event_token: str
    candidate_event_token: str
    action_type: str
    prior_match_lags: Tuple[int, ...]
    prompt_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    prompt_unknowns: int

    @property
    def example_id(self) -> str:
        return (
            f"{self.split}:{self.game_seed}:{self.step_index}:"
            f"{self.candidate_index}"
        )

    @property
    def input_length(self) -> int:
        return len(self.prompt_ids)


def action_event_token(action: str) -> str:
    normalized = _EVENT_COMPONENT.sub("_", action.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("atomic action event cannot be empty")
    return f"<event_{normalized}>"


def build_atomic_event_examples(
    source_examples: Mapping[str, Sequence[MoveDeltaExample]],
    source_vocabulary: Vocabulary,
) -> Tuple[Dict[str, Tuple[AtomicEventExample, ...]], Vocabulary]:
    raw = {}
    for split in SPLITS:
        raw[split] = tuple(
            (
                example,
                action_event_token(example.previous_action),
                action_event_token(example.candidate_action),
                source_vocabulary.decode(example.target_ids)[0],
            )
            for example in source_examples[split]
        )
    vocabulary = Vocabulary.build(
        token
        for example, previous_event, candidate_event, target in raw["train"]
        for token in (previous_event, candidate_event, target)
    )
    encoded = {}
    for split in SPLITS:
        values = []
        for example, previous_event, candidate_event, target in raw[split]:
            prompt_ids = vocabulary.encode((previous_event, candidate_event))
            values.append(
                AtomicEventExample(
                    split=example.split,
                    episode_index=example.episode_index,
                    game_seed=example.game_seed,
                    step_index=example.step_index,
                    candidate_index=example.candidate_index,
                    source=example.source,
                    step_group_id=example.step_group_id,
                    previous_action=example.previous_action,
                    candidate_action=example.candidate_action,
                    previous_event_token=previous_event,
                    candidate_event_token=candidate_event,
                    action_type="move",
                    prior_match_lags=example.prior_match_lags,
                    prompt_ids=prompt_ids,
                    target_ids=vocabulary.encode((target,)),
                    prompt_unknowns=sum(
                        token_id == vocabulary.unk_id for token_id in prompt_ids
                    ),
                )
            )
        encoded[split] = tuple(values)
    return encoded, vocabulary


def audit_atomic_event_examples(
    examples: Mapping[str, Sequence[AtomicEventExample]],
    vocabulary: Vocabulary,
    *,
    expected_counts: Mapping[str, int],
    expected_step_groups: Mapping[str, int],
) -> Dict[str, Any]:
    splits = {}
    all_valid = True
    for split, values in examples.items():
        groups: Dict[str, list[AtomicEventExample]] = defaultdict(list)
        for example in values:
            groups[example.step_group_id].append(example)
        labels = Counter(vocabulary.decode(example.target_ids)[0] for example in values)
        event_mapping: Dict[str, set[str]] = defaultdict(set)
        for example in values:
            event_mapping[example.previous_event_token].add(example.previous_action)
            event_mapping[example.candidate_event_token].add(example.candidate_action)
        collision_count = sum(len(actions) > 1 for actions in event_mapping.values())
        group_valid = all(
            len(group) == 2
            and {example.candidate_index for example in group} == {0, 1}
            and {example.source for example in group}
            == {"factual", "counterfactual"}
            and len({example.candidate_event_token for example in group}) == 2
            and next(
                example for example in group if example.source == "factual"
            ).prior_match_lags
            == ()
            and next(
                example for example in group if example.source == "counterfactual"
            ).prior_match_lags
            == (1,)
            for group in groups.values()
        )
        record = {
            "example_count": len(values),
            "step_group_count": len(groups),
            "game_count": len({example.game_seed for example in values}),
            "label_counts": dict(sorted(labels.items())),
            "prompt_length": sg0._length_summary(
                [len(example.prompt_ids) for example in values]
            ),
            "prompt_unknown_count": sum(
                example.prompt_unknowns for example in values
            ),
            "unique_event_token_count": len(event_mapping),
            "event_token_collision_count": collision_count,
            "step_groups_valid": group_valid,
        }
        valid = (
            len(values) == expected_counts[split]
            and len(groups) == expected_step_groups[split]
            and labels[LABELS[0]] == labels[LABELS[1]] == len(values) // 2
            and record["prompt_length"]["min"]
            == record["prompt_length"]["max"]
            == 2
            and record["prompt_unknown_count"] == 0
            and collision_count == 0
            and group_valid
        )
        record["passed"] = valid
        splits[split] = record
        all_valid = all_valid and valid
    return {
        "task": "atomic real TextWorld move-event state delta",
        "label_source": (
            "unchanged normalized candidate next_obs membership in prior "
            "factual observations"
        ),
        "event_encoding": "one train-vocabulary token per real move action string",
        "expected_counts": dict(expected_counts),
        "expected_step_groups": dict(expected_step_groups),
        "splits": splits,
        "passed": all_valid,
    }


def _event_batch_tensors(
    examples: Sequence[AtomicEventExample],
    indices: Sequence[int],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = tuple(examples[index] for index in indices)
    if {len(example.prompt_ids) for example in selected} != {2}:
        raise ValueError("atomic event batch requires exactly two events")
    input_ids = torch.tensor(
        [example.prompt_ids for example in selected],
        dtype=torch.long,
        device=device,
    )
    query_indices = torch.tensor((0, 1), dtype=torch.long, device=device)
    label_to_index = {
        vocabulary.token_id(label): index for index, label in enumerate(LABELS)
    }
    targets = torch.tensor(
        [label_to_index[example.target_ids[0]] for example in selected],
        dtype=torch.long,
        device=device,
    )
    return input_ids, query_indices, targets


def evaluate_event_model(
    model: BilinearRelationModel,
    examples: Sequence[AtomicEventExample],
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
    predictions_by_id = {}
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(examples))))
            input_ids, query_indices, targets = _event_batch_tensors(
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
            total_loss += float(losses.sum().item())
            total_correct += int((predictions == targets).sum().item())
            total_count += len(indices)
            for offset, example_index in enumerate(indices):
                predictions_by_id[examples[example_index].example_id] = int(
                    predictions[offset].item()
                )

    timings = []
    state_sizes = []
    step_correct: Dict[str, list[bool]] = defaultdict(list)
    records = []
    with torch.inference_mode():
        for index, example in enumerate(examples):
            input_ids, query_indices, targets = _event_batch_tensors(
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
                records.append(
                    {
                        "example_id": example.example_id,
                        "step_group_id": example.step_group_id,
                        "previous_event": example.previous_event_token,
                        "candidate_event": example.candidate_event_token,
                        "target_label": LABELS[target],
                        "predicted_label": LABELS[predicted],
                        "correct": correct,
                    }
                )
    return {
        "binary_nll": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "step_consistency": sum(all(values) for values in step_correct.values())
        / len(step_correct),
        "example_count": total_count,
        "step_group_count": len(step_correct),
        "timing": {
            **_sample_summary(timings, 1),
            "p99_ms": _percentile(timings, 0.99),
            "state_bytes_max": max(state_sizes),
            "state_bytes_mean": sg0._mean(state_sizes),
        },
        "records": records if include_records else None,
    }


def train_event_model(
    name: str,
    model: BilinearRelationModel,
    examples: Sequence[AtomicEventExample],
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
    epoch_loss = 0.0
    epoch_examples = 0
    epoch_records = []
    started_all = time.perf_counter_ns()
    for update, indices in enumerate(schedule):
        input_ids, query_indices, targets = _event_batch_tensors(
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
                f"non-finite SG9 loss for {name} at update {update + 1}"
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
    warmup = len(batch_timings) // 5
    return {
        "epochs": epochs,
        "updates": len(schedule),
        "batches_per_epoch": batches_per_epoch,
        "batch_examples": len(schedule[0]),
        "example_exposures": example_exposures,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_epoch": epoch_records[-1]["binary_nll"],
        "batch_timing": {
            **_sample_summary(batch_timings[warmup:], 1),
            "warmup_updates_excluded": warmup,
        },
        "example_equivalent_timing": {
            **_sample_summary(example_timings[warmup:], 1),
            "warmup_updates_excluded": warmup,
        },
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second_total": example_exposures / elapsed_seconds,
        "epoch_records": epoch_records,
    }


def _prefill_previous_event(
    language_model: Any,
    previous_token_id: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Any]:
    input_ids = torch.tensor(
        [[previous_token_id]], dtype=torch.long, device=device
    )
    query = torch.tensor((0,), dtype=torch.long, device=device)
    hidden, state = _query_hidden(
        language_model,
        input_ids,
        query,
        use_eligibility=False,
        detach_state=True,
    )
    return hidden[:, 0], state


def _generic_candidate_hidden(
    language_model: Any,
    candidate_token_id: int,
    state: Any,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Any]:
    token = torch.tensor(
        [[candidate_token_id]], dtype=torch.long, device=device
    )
    embedded = language_model.input_dropout(language_model.embedding(token))
    result = language_model.core(embedded, state, detach_state=True)
    hidden = language_model.output_norm(
        language_model.output_dropout(result.sequence[:, -1])
    )
    return hidden, result.state


def _snn_cached_decay_candidate_hidden(
    language_model: Any,
    candidate_token_id: int,
    state: E3ScanState,
    decays: Tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, E3ScanState]:
    core = language_model.core
    if not isinstance(core, E3GatedTraceScanCore):
        raise TypeError("cached-decay event step requires E3 gated trace core")
    if len(state.layers) != 1:
        raise ValueError("cached-decay event step requires one SNN layer")
    token = torch.tensor([candidate_token_id], dtype=torch.long, device=device)
    embedded = language_model.input_dropout(language_model.embedding(token))
    layer = state.layers[0]
    output = core.forward_step_tensors_cached_decay(
        embedded,
        layer.excitatory,
        layer.inhibitory,
        decays[0],
        decays[1],
    )
    hidden = language_model.output_norm(
        language_model.output_dropout(output[0])
    )
    next_state = E3ScanState(
        layers=(type(layer)(excitatory=output[1], inhibitory=output[2]),)
    )
    return hidden, next_state


def evaluate_cached_stream(
    model: BilinearRelationModel,
    examples: Sequence[AtomicEventExample],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    use_cached_decay: bool,
    timing_repeats: int = 1,
    timing_warmup_repeats: int = 0,
) -> Dict[str, Any]:
    if timing_repeats <= 0 or timing_warmup_repeats < 0:
        raise ValueError("cached timing repeats must be positive and warmup nonnegative")
    model.eval()
    groups: Dict[str, list[AtomicEventExample]] = defaultdict(list)
    for example in examples:
        groups[example.step_group_id].append(example)
    prefix_timings = []
    candidate_timings = []
    state_sizes = []
    correct_by_group: Dict[str, list[bool]] = defaultdict(list)
    records = []
    max_logit_difference = 0.0
    decays = None
    if use_cached_decay:
        core = model.language_model.core
        if not isinstance(core, E3GatedTraceScanCore):
            raise TypeError("cached decay benchmark is SNN-only")
        decays = core.decays()
    with torch.inference_mode():
        for group_id in sorted(groups):
            group = sorted(groups[group_id], key=lambda value: value.candidate_index)
            previous_id = group[0].prompt_ids[0]
            if any(example.prompt_ids[0] != previous_id for example in group):
                raise ValueError("cached candidates must share previous event")
            _sync(device)
            prefix_started = time.perf_counter_ns()
            previous_hidden, prefix_state = _prefill_previous_event(
                model.language_model, previous_id, device=device
            )
            previous_hidden.sum().item()
            _sync(device)
            prefix_timings.append(
                (time.perf_counter_ns() - prefix_started) / 1e6
            )
            state_sizes.append(state_nbytes(prefix_state))
            for example in group:
                candidate_id = example.prompt_ids[1]

                def forward_candidate() -> torch.Tensor:
                    if use_cached_decay:
                        if not isinstance(prefix_state, E3ScanState):
                            raise TypeError("SNN prefix returned invalid state")
                        candidate_hidden, _next_state = (
                            _snn_cached_decay_candidate_hidden(
                                model.language_model,
                                candidate_id,
                                prefix_state,
                                decays,  # type: ignore[arg-type]
                                device=device,
                            )
                        )
                    else:
                        candidate_hidden, _next_state = _generic_candidate_hidden(
                            model.language_model,
                            candidate_id,
                            prefix_state,
                            device=device,
                        )
                    return model.relation_head(previous_hidden, candidate_hidden)

                logits = forward_candidate()
                for repeat in range(timing_warmup_repeats + timing_repeats):
                    _sync(device)
                    started = time.perf_counter_ns()
                    timed_logits = forward_candidate()
                    timed_logits.sum().item()
                    _sync(device)
                    if repeat >= timing_warmup_repeats:
                        candidate_timings.append(
                            (time.perf_counter_ns() - started) / 1e6
                        )
                target = 0 if vocabulary.decode(example.target_ids)[0] == LABELS[0] else 1
                predicted = int(logits.argmax(dim=-1).item())
                correct = predicted == target
                correct_by_group[group_id].append(correct)
                full_input, full_query, _full_targets = _event_batch_tensors(
                    examples,
                    (examples.index(example),),
                    vocabulary,
                    device=device,
                )
                full_logits, _state = bilinear_logits(
                    model,
                    full_input,
                    full_query,
                    use_eligibility=False,
                    detach_state=True,
                )
                difference = float((logits - full_logits).abs().max().item())
                max_logit_difference = max(max_logit_difference, difference)
                records.append(
                    {
                        "example_id": example.example_id,
                        "target_label": LABELS[target],
                        "predicted_label": LABELS[predicted],
                        "correct": correct,
                        "full_logit_max_abs_difference": difference,
                    }
                )
    total_correct = sum(
        sum(values) for values in correct_by_group.values()
    )
    total_count = sum(len(values) for values in correct_by_group.values())
    return {
        "mode": "snn_cached_decay" if use_cached_decay else "generic_cached_state",
        "accuracy": total_correct / total_count,
        "step_consistency": sum(all(values) for values in correct_by_group.values())
        / len(correct_by_group),
        "max_full_logit_abs_difference": max_logit_difference,
        "prefix_timing": {
            **_sample_summary(prefix_timings, 1),
            "p99_ms": _percentile(prefix_timings, 0.99),
        },
        "candidate_timing": {
            **_sample_summary(candidate_timings, 1),
            "p99_ms": _percentile(candidate_timings, 0.99),
        },
        "prefix_state_bytes_max": max(state_sizes),
        "timing_repeats_per_candidate": timing_repeats,
        "timing_warmup_repeats_per_candidate": timing_warmup_repeats,
        "candidate_timing_sample_count": len(candidate_timings),
        "records": records,
    }


def _extract_event_features(
    language_model: Any,
    examples: Sequence[AtomicEventExample],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    language_model.eval()
    features = []
    targets = []
    groups = []
    started = time.perf_counter_ns()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(examples))))
            input_ids, query_indices, labels = _event_batch_tensors(
                examples, indices, vocabulary, device=device
            )
            hidden, _state = _query_hidden(
                language_model,
                input_ids,
                query_indices,
                use_eligibility=False,
                detach_state=True,
            )
            features.append(_outer_features(hidden).to(torch.float64))
            targets.append(
                torch.where(
                    labels == 0,
                    torch.ones_like(labels, dtype=torch.float64),
                    -torch.ones_like(labels, dtype=torch.float64),
                )
            )
            groups.extend(examples[index].step_group_id for index in indices)
    _sync(device)
    return {
        "features": torch.cat(features),
        "targets": torch.cat(targets),
        "group_ids": tuple(groups),
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
    }


def fit_event_closed_form_ridge(
    language_model: Any,
    examples: Mapping[str, Sequence[AtomicEventExample]],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    lambdas: Sequence[float],
) -> Dict[str, Any]:
    extracted = {
        split: _extract_event_features(
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
            (values[:, :1], (values[:, 1:] - mean) / scale), dim=1
        )

    x_train = transform(train_x)
    x_valid = transform(extracted["valid"]["features"])
    x_test = transform(extracted["test"]["features"])
    y_valid = extracted["valid"]["targets"]
    y_test = extracted["test"]["targets"]
    identity = torch.eye(x_train.shape[0], dtype=torch.float64, device=device)
    gram = x_train @ x_train.T
    candidates = []
    weights_by_lambda = {}
    started_fit = time.perf_counter_ns()
    for ridge_lambda in lambdas:
        alpha = torch.linalg.solve(
            gram + float(ridge_lambda) * identity, train_y
        )
        weights = x_train.T @ alpha
        weights_by_lambda[float(ridge_lambda)] = weights
        candidates.append(
            {
                "lambda": float(ridge_lambda),
                "valid": _ridge_metrics(
                    x_valid @ weights,
                    y_valid,
                    extracted["valid"]["group_ids"],
                ),
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
    weights = weights_by_lambda[selected["lambda"]]
    _sync(device)
    fit_seconds = (time.perf_counter_ns() - started_fit) / 1e9
    feature_seconds = sum(
        extracted[split]["elapsed_seconds"] for split in ("train", "valid")
    )
    return {
        "feature_dimension": x_train.shape[1],
        "readout_parameter_count": weights.numel(),
        "selected_lambda": selected["lambda"],
        "lambda_candidates": tuple(float(value) for value in lambdas),
        "selection_rule": "max valid accuracy, min valid MSE, min lambda",
        "validation_candidates": candidates,
        "train": _ridge_metrics(
            x_train @ weights,
            train_y,
            extracted["train"]["group_ids"],
        ),
        "valid": selected["valid"],
        "test": _ridge_metrics(
            x_test @ weights,
            y_test,
            extracted["test"]["group_ids"],
        ),
        "feature_extraction_seconds": {
            split: extracted[split]["elapsed_seconds"] for split in SPLITS
        },
        "fit_seconds": fit_seconds,
        "training_wall_seconds": feature_seconds + fit_seconds,
    }


def _decision(
    data_audit: Mapping[str, Any],
    seeds: Sequence[Mapping[str, Any]],
    *,
    quick: bool,
    strict_per_seed_stream: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "quality_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "cached_stream_gate": "SMOKE",
            "ridge_quality_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg9_atomic_event_stream",
        }
    mean_accuracy = {
        name: sg0._mean(seed["post"][name]["test"]["accuracy"] for seed in seeds)
        for name in MODEL_NAMES
    }
    mean_step = {
        name: sg0._mean(
            seed["post"][name]["test"]["step_consistency"] for seed in seeds
        )
        for name in MODEL_NAMES
    }
    best_ann_accuracy = max(mean_accuracy["lstm"], mean_accuracy["transformer"])
    quality_pass = all(
        mean_accuracy[name] >= 0.98
        and mean_step[name] >= 0.95
        and mean_accuracy[name] >= best_ann_accuracy - 0.02
        for name in ("snn_bptt", "snn_at1", "snn_ra0")
    )
    ridge_accuracy = sg0._mean(
        seed["closed_form_ridge"]["test"]["accuracy"] for seed in seeds
    )
    ridge_step = sg0._mean(
        seed["closed_form_ridge"]["test"]["step_consistency"] for seed in seeds
    )
    ridge_quality = ridge_accuracy >= 0.98 and ridge_step >= 0.95
    mean_train_example_p50 = {
        name: sg0._mean(
            seed["training"][name]["example_equivalent_timing"]["p50_ms"]
            for seed in seeds
        )
        for name in MODEL_NAMES
    }
    mean_train_wall = {
        name: sg0._mean(seed["training"][name]["elapsed_seconds"] for seed in seeds)
        for name in MODEL_NAMES
    }
    fastest_ann_train_p50 = min(
        mean_train_example_p50["lstm"], mean_train_example_p50["transformer"]
    )
    fastest_ann_train_wall = min(
        mean_train_wall["lstm"], mean_train_wall["transformer"]
    )
    ridge_wall = sg0._mean(
        seed["closed_form_ridge"]["training_wall_seconds"] for seed in seeds
    )
    training_speed = (
        mean_train_example_p50["snn_ra0"] <= fastest_ann_train_p50
        and mean_train_wall["snn_ra0"] <= fastest_ann_train_wall
    ) or ridge_wall <= fastest_ann_train_wall
    mean_cached_p50 = {
        name: sg0._mean(
            seed["cached_stream"][name]["generic"]["candidate_timing"]["p50_ms"]
            for seed in seeds
        )
        for name in MODEL_NAMES
    }
    mean_cached_p95 = {
        name: sg0._mean(
            seed["cached_stream"][name]["generic"]["candidate_timing"]["p95_ms"]
            for seed in seeds
        )
        for name in MODEL_NAMES
    }
    ra0_cached_p50 = sg0._mean(
        seed["cached_stream"]["snn_ra0"]["cached_decay"]["candidate_timing"]["p50_ms"]
        for seed in seeds
    )
    ra0_cached_p95 = sg0._mean(
        seed["cached_stream"]["snn_ra0"]["cached_decay"]["candidate_timing"]["p95_ms"]
        for seed in seeds
    )
    ann_cached_p50 = min(mean_cached_p50["lstm"], mean_cached_p50["transformer"])
    ann_cached_p95 = min(mean_cached_p95["lstm"], mean_cached_p95["transformer"])
    cache_equivalence = all(
        seed["cached_stream"][name][mode]["accuracy"] == 1.0
        and seed["cached_stream"][name][mode]["max_full_logit_abs_difference"]
        <= 1e-5
        for seed in seeds
        for name in MODEL_NAMES
        for mode in (
            ("generic", "cached_decay")
            if name.startswith("snn_")
            else ("generic",)
        )
    )
    per_seed_stream = []
    for seed in seeds:
        ra0 = seed["cached_stream"]["snn_ra0"]["cached_decay"][
            "candidate_timing"
        ]
        ann_p50 = min(
            seed["cached_stream"][name]["generic"]["candidate_timing"][
                "p50_ms"
            ]
            for name in ("lstm", "transformer")
        )
        ann_p95 = min(
            seed["cached_stream"][name]["generic"]["candidate_timing"][
                "p95_ms"
            ]
            for name in ("lstm", "transformer")
        )
        per_seed_stream.append(
            {
                "seed": seed["seed"],
                "ra0_p50_ms": ra0["p50_ms"],
                "ra0_p95_ms": ra0["p95_ms"],
                "fastest_ann_p50_ms": ann_p50,
                "fastest_ann_p95_ms": ann_p95,
                "passed": ra0["p50_ms"] <= ann_p50
                and ra0["p95_ms"] <= ann_p95,
            }
        )
    cached_stream_pass = cache_equivalence and (
        all(record["passed"] for record in per_seed_stream)
        if strict_per_seed_stream
        else ra0_cached_p50 <= ann_cached_p50
        and ra0_cached_p95 <= ann_cached_p95
    )
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "quality_gate": quality_pass,
        "training_speed_gate": training_speed,
        "cached_stream_gate": cached_stream_pass,
        "ridge_quality_gate": ridge_quality,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "mean_accuracy": mean_accuracy,
        "mean_step_consistency": mean_step,
        "best_ann_accuracy": best_ann_accuracy,
        "mean_ridge_accuracy": ridge_accuracy,
        "mean_ridge_step_consistency": ridge_step,
        "mean_training_example_p50_ms": mean_train_example_p50,
        "mean_training_wall_seconds": mean_train_wall,
        "mean_ridge_training_wall_seconds": ridge_wall,
        "mean_generic_cached_candidate_p50_ms": mean_cached_p50,
        "mean_generic_cached_candidate_p95_ms": mean_cached_p95,
        "mean_ra0_cached_decay_candidate_p50_ms": ra0_cached_p50,
        "mean_ra0_cached_decay_candidate_p95_ms": ra0_cached_p95,
        "fastest_ann_cached_candidate_p50_ms": ann_cached_p50,
        "fastest_ann_cached_candidate_p95_ms": ann_cached_p95,
        "cache_equivalence": cache_equivalence,
        "stream_gate_mode": (
            "strict_per_seed" if strict_per_seed_stream else "cross_seed_mean"
        ),
        "per_seed_stream_comparison": per_seed_stream,
        "next_route": (
            "sg10_multichannel_delta_recursive_least_squares"
            if overall == "PASS"
            else "sg10_compiled_fused_event_step_or_multichannel_quality"
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
    source_examples, source_vocabulary = build_move_delta_examples(corpus_root, corpus)
    expected_counts = dict(zip(SPLITS, args.expected_counts))
    expected_step_groups = dict(zip(SPLITS, args.expected_step_groups))
    source_audit = audit_move_delta_examples(
        source_examples,
        source_vocabulary,
        expected_counts=expected_counts,
        expected_step_groups=expected_step_groups,
        max_prompt_tokens=32,
    )
    examples, vocabulary = build_atomic_event_examples(
        source_examples, source_vocabulary
    )
    data_audit = audit_atomic_event_examples(
        examples,
        vocabulary,
        expected_counts=expected_counts,
        expected_step_groups=expected_step_groups,
    )
    data_audit["source_move_delta_audit_passed"] = source_audit["passed"]
    data_audit["passed"] = bool(data_audit["passed"] and source_audit["passed"])
    if not data_audit["passed"]:
        raise AssertionError("SG9 atomic event data audit failed")
    if expected_step_groups["train"] % args.batch_groups:
        raise AssertionError("SG9 train groups must divide batch_groups")
    batches_per_epoch = expected_step_groups["train"] // args.batch_groups

    seed_results = []
    for seed in args.seeds:
        models = build_bilinear_models(
            10_100_000 + 100 * seed,
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
            raise AssertionError(f"SG9 parameter spread failed: {parameter_counts}")
        schedule = build_paired_batch_schedule(
            examples["train"],
            epochs=args.epochs,
            batch_groups=args.batch_groups,
            seed=10_101_000 + seed,
        )
        training = {
            name: train_event_model(
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
                split: evaluate_event_model(
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
        cached_stream = {}
        for name, model in models.items():
            cached_stream[name] = {
                "generic": evaluate_cached_stream(
                    model,
                    examples["test"],
                    vocabulary,
                    device=device,
                    use_cached_decay=False,
                    timing_repeats=args.timing_repeats,
                    timing_warmup_repeats=args.timing_warmup_repeats,
                )
            }
            if name.startswith("snn_"):
                cached_stream[name]["cached_decay"] = evaluate_cached_stream(
                    model,
                    examples["test"],
                    vocabulary,
                    device=device,
                    use_cached_decay=True,
                    timing_repeats=args.timing_repeats,
                    timing_warmup_repeats=args.timing_warmup_repeats,
                )
        closed_form = fit_event_closed_form_ridge(
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
                "training": training,
                "post": post,
                "cached_stream": cached_stream,
                "closed_form_ridge": closed_form,
            }
        )
    decision = _decision(
        data_audit,
        seed_results,
        quick=args.quick,
        strict_per_seed_stream=args.strict_per_seed_stream,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG9 atomic event cached bilinear stream",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "event-driven engineering hypothesis",
            "statement": (
                "A persistent SNN world state should process one typed event per "
                "response instead of repeatedly prefilling text formatting."
            ),
            "what_if": (
                "What if atomic multimodal events expose the constant-time SNN "
                "update advantage hidden by repeated language prefill?"
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
            "event_sequence_length": 2,
            "cached_candidate_event_length": 1,
            "timing_repeats_per_candidate": args.timing_repeats,
            "timing_warmup_repeats_per_candidate": args.timing_warmup_repeats,
            "strict_per_seed_stream_gate": args.strict_per_seed_stream,
            "ridge_lambdas": tuple(args.ridge_lambdas),
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "corpus_provenance": tw0._corpus_provenance(corpus),
            "source_move_delta_vocabulary": {
                "size": len(source_vocabulary),
                "fingerprint": source_vocabulary.fingerprint,
            },
            "vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
                "source_split": "train",
                "source_fields": ("previous_event", "candidate_event", "label"),
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
        default=Path("results/e3_scan/e3_sg9_atomic_event_stream.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--timing-warmup-repeats", type=int, default=0)
    parser.add_argument("--strict-per-seed-stream", action="store_true")
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
        args.timing_repeats,
        *args.ridge_lambdas,
        *args.expected_counts,
        *args.expected_step_groups,
        *args.expected_train_seeds,
        *args.expected_valid_seeds,
        *args.expected_test_seeds,
    ) <= 0:
        parser.error("all numeric experiment controls must be positive")
    if args.timing_warmup_repeats < 0:
        parser.error("timing-warmup-repeats must be nonnegative")
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
