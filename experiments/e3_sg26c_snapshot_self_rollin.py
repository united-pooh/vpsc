"""SG26C two-stage snapshot self-roll-in on the SG22R language task."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg22r_seventh_fresh_confirmation as sg22r  # noqa: E402
from experiments import e3_sg25e_bucketed_batch_graph as sg25e  # noqa: E402
from experiments import e3_sg25f_parallel_affine_graph as sg25f  # noqa: E402
from experiments import e3_sg26a_expanded_raw_language as sg26a  # noqa: E402
from experiments import e3_sg26b_length_mass_objective as sg26b  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from vpsc.world_model.triton_fused_gated_trace import (  # noqa: E402
    backend_audit as triton_fused_backend_audit,
    triton_fused_gated_trace,
)


RATES = (0.0, 0.25, 0.5, 1.0)
MODES = sg26a.MODES
BATCH_SIZE = sg26a.BATCH_SIZE
UPDATES_PER_EPOCH = sum(
    math.ceil(count / BATCH_SIZE) for count in sg26a.EXPECTED_BUCKET_COUNTS
)
ROLLIN_SEED = 26_150_003
EXPECTED_SG26B_SHA256 = (
    "76ad23327fddd50e6d60f583ee6adb1b4bb3bf3ab7fd88ab6f3dec321ddbe35f"
)
SG26B_SNN_ALPHA0_VALID_NLL = 0.6596352211074806
SG26B_SNN_ALPHA0_VALID_EDIT = 0.6196577924715371
SG25F_PARALLEL_EXAMPLES_PER_SECOND = 20133.046722635696


@contextmanager
def _local_rocm_triton_backend() -> Iterator[None]:
    """Install the four-bucket fused Triton route without CUDA extensions."""

    original_buckets = sg25e.BUCKET_CAPACITIES
    original_parallel_kernel = sg25f.fused_parallel_gated_trace
    sg25e.BUCKET_CAPACITIES = sg26a.BUCKET_CAPACITIES
    sg25f.fused_parallel_gated_trace = triton_fused_gated_trace
    try:
        yield
    finally:
        sg25f.fused_parallel_gated_trace = original_parallel_kernel
        sg25e.BUCKET_CAPACITIES = original_buckets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate_label(rate: float) -> str:
    return f"rate_{str(rate).replace('.', '_')}"


def _position_hash32(
    seed: int, epoch: int, example_index: int, history_index: int
) -> int:
    value = (
        seed
        + 0x9E3779B9 * (epoch + 1)
        + 0x85EBCA6B * (example_index + 1)
        + 0xC2B2AE35 * (history_index + 1)
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def _selected(
    rate: float, *, epoch: int, example_index: int, history_index: int
) -> bool:
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    threshold = int(rate * (1 << 32))
    return (
        _position_hash32(ROLLIN_SEED, epoch, example_index, history_index)
        < threshold
    )


def _clone_batch(batch: sg25e.DeviceBatch) -> sg25e.DeviceBatch:
    return sg25e.DeviceBatch(
        input_ids=batch.input_ids.clone(),
        query_indices=batch.query_indices.clone(),
        gather_indices=batch.gather_indices.clone(),
        targets=batch.targets.clone(),
        token_mask=batch.token_mask.clone(),
        example_mask=batch.example_mask.clone(),
        real_example_count=batch.real_example_count,
        input_token_count=batch.input_token_count,
        target_token_count=batch.target_token_count,
        example_indices=batch.example_indices,
    )


class EagerTrainer:
    """Shape-cached eager trainer used when HIP Graph capture is unavailable."""

    def __init__(
        self,
        mode: str,
        model: Any,
        batches: Sequence[sg25e.DeviceBatch],
        *,
        device: torch.device,
    ) -> None:
        self.mode = mode
        self.model = model
        self.device = device
        self.optimizer = sg25f._optimizer(model)
        self.parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        self._parameter_baseline = tuple(
            parameter.detach().clone() for parameter in self.parameters
        )
        representatives: Dict[Tuple[int, int, int], sg25e.DeviceBatch] = {}
        for batch in batches:
            representatives.setdefault(batch.key, _clone_batch(batch))
        if len(representatives) != len(sg25e.BUCKET_CAPACITIES):
            raise AssertionError("eager trainer requires every frozen bucket")
        self.entries = representatives
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter_ns()

        first = next(iter(self.entries.values()))
        sg25f._step_body(mode, model, self.optimizer, first)
        _sync(device)
        with torch.no_grad():
            for parameter, baseline in zip(
                self.parameters, self._parameter_baseline
            ):
                parameter.copy_(baseline)
            for state in self.optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        self.optimizer.zero_grad(set_to_none=False)
        self._optimizer_baseline = sg25f._tensor_state_snapshot(self.optimizer)
        self.reset()
        for entry in self.entries.values():
            sg25f._step_body(mode, model, self.optimizer, entry)
            _sync(device)
            self.reset()
        self.capture_audit = {
            "execution_mode": "uniform_eager",
            "graph_capture_attempted": False,
            "graph_capture_disabled_reason": (
                "HIP error 900: hipBLASLt operation not permitted during stream capture"
            ),
            "shape_count": len(self.entries),
            "keys": tuple(tuple(key) for key in sorted(self.entries)),
            "warmup_seconds": (time.perf_counter_ns() - started) / 1e9,
            "allocated_before_bytes": allocated_before,
            "allocated_after_bytes": torch.cuda.memory_allocated(device),
            "allocated_delta_bytes": max(
                0, torch.cuda.memory_allocated(device) - allocated_before
            ),
            "reserved_before_bytes": reserved_before,
            "reserved_after_bytes": torch.cuda.memory_reserved(device),
            "reserved_delta_bytes": max(
                0, torch.cuda.memory_reserved(device) - reserved_before
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    def reset(self) -> None:
        _sync(self.device)
        with torch.no_grad():
            for parameter, baseline in zip(
                self.parameters, self._parameter_baseline
            ):
                parameter.copy_(baseline)
            for parameter_index, parameter in enumerate(self.parameters):
                for name, baseline in self._optimizer_baseline[
                    parameter_index
                ].items():
                    self.optimizer.state[parameter][name].copy_(baseline)
        self.optimizer.zero_grad(set_to_none=False)
        _sync(self.device)

    def replay(
        self, batch: sg25e.DeviceBatch, *, timed: bool, inspect: bool
    ) -> Dict[str, Any]:
        entry = self.entries[batch.key]
        if timed:
            _sync(self.device)
            started = time.perf_counter_ns()
        entry.input_ids.copy_(batch.input_ids)
        entry.query_indices.copy_(batch.query_indices)
        entry.gather_indices.copy_(batch.gather_indices)
        entry.targets.copy_(batch.targets)
        entry.token_mask.copy_(batch.token_mask)
        entry.example_mask.copy_(batch.example_mask)
        loss, logits = sg25f._step_body(
            self.mode, self.model, self.optimizer, entry
        )
        _sync(self.device)
        result: Dict[str, Any] = {
            "wall_ms": (
                (time.perf_counter_ns() - started) / 1e6 if timed else None
            )
        }
        if inspect:
            result.update(
                {
                    "loss": float(loss.item()),
                    "predictions": logits.detach().argmax(dim=-1).clone(),
                }
            )
        return result


def collect_rollin_predictions(
    model: Any,
    train_examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
) -> Tuple[Dict[int, Tuple[int, ...]], Dict[str, Any]]:
    started = time.perf_counter_ns()
    generated = sg0.generate_model(
        model,
        train_examples,
        vocabulary,
        max_tokens=sg26b.VALID_GENERATION_TOKENS,
        device=device,
        include_records=True,
    )
    index_by_id = {
        example.example_id: index
        for index, example in enumerate(train_examples)
    }
    predictions: Dict[int, Tuple[int, ...]] = {}
    digest = hashlib.sha256()
    premature = 0
    for record in generated["records"]:
        example_index = index_by_id[record["example_id"]]
        values = tuple(vocabulary.encode(record["prediction_tokens"]))
        values = values + (vocabulary.eos_id,)
        predictions[example_index] = values
        target_history = len(train_examples[example_index].target_ids) - 1
        premature += int(len(values) - 1 < target_history)
        digest.update(example_index.to_bytes(4, "little", signed=False))
        for value in values:
            digest.update(int(value).to_bytes(4, "little", signed=False))
    if len(predictions) != len(train_examples):
        raise AssertionError("self-roll-in omitted training examples")
    summary = {key: value for key, value in generated.items() if key != "records"}
    return predictions, {
        "example_count": len(predictions),
        "premature_eos_example_count": premature,
        "prediction_sha256": digest.hexdigest(),
        "collection_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        "generation": summary,
    }


def apply_rollin_corruption(
    batches: Sequence[sg25e.DeviceBatch],
    train_examples: Sequence[Any],
    predictions: Mapping[int, Sequence[int]],
    *,
    rate: float,
    stage_start_epoch: int,
    eos_id: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("roll-in rate must be in [0, 1]")
    started = time.perf_counter_ns()
    if rate == 0.0:
        eligible_count = sum(
            len(train_examples[example_index].target_ids) - 1
            for batch in batches
            for example_index in batch.example_indices
        )
        return {
            "rate": rate,
            "eligible_history_token_count": eligible_count,
            "selected_history_token_count": 0,
            "selected_ratio": 0.0,
            "mismatched_selected_token_count": 0,
            "mismatched_selected_ratio": 0.0,
            "eos_injected_count": 0,
            "corruption_sha256": hashlib.sha256().hexdigest(),
            "corruption_wall_seconds": (time.perf_counter_ns() - started)
            / 1e9,
            "passed": True,
        }
    eligible_count = 0
    selected_count = 0
    mismatch_count = 0
    eos_injected_count = 0
    digest = hashlib.sha256()
    for batch_index, batch in enumerate(batches):
        epoch = stage_start_epoch + batch_index // UPDATES_PER_EPOCH
        rows = []
        columns = []
        values = []
        for row, example_index in enumerate(batch.example_indices):
            example = train_examples[example_index]
            prediction = predictions[example_index]
            history_count = len(example.target_ids) - 1
            eligible_count += history_count
            for history_index in range(history_count):
                if not _selected(
                    rate,
                    epoch=epoch,
                    example_index=example_index,
                    history_index=history_index,
                ):
                    continue
                replacement = int(
                    prediction[history_index]
                    if history_index < len(prediction)
                    else eos_id
                )
                gold = int(example.target_ids[history_index])
                rows.append(row)
                columns.append(len(example.prompt_ids) + history_index)
                values.append(replacement)
                selected_count += 1
                mismatch_count += int(replacement != gold)
                eos_injected_count += int(replacement == eos_id)
                digest.update(epoch.to_bytes(4, "little", signed=False))
                digest.update(example_index.to_bytes(4, "little", signed=False))
                digest.update(history_index.to_bytes(4, "little", signed=False))
                digest.update(replacement.to_bytes(4, "little", signed=False))
        if rows:
            row_tensor = torch.tensor(rows, dtype=torch.long, device=device)
            column_tensor = torch.tensor(
                columns, dtype=torch.long, device=device
            )
            value_tensor = torch.tensor(values, dtype=torch.long, device=device)
            batch.input_ids[row_tensor, column_tensor] = value_tensor
    if device.type == "cuda":
        _sync(device)
    return {
        "rate": rate,
        "eligible_history_token_count": eligible_count,
        "selected_history_token_count": selected_count,
        "selected_ratio": selected_count / max(eligible_count, 1),
        "mismatched_selected_token_count": mismatch_count,
        "mismatched_selected_ratio": mismatch_count / max(selected_count, 1),
        "eos_injected_count": eos_injected_count,
        "corruption_sha256": digest.hexdigest(),
        "corruption_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        "passed": (
            selected_count == 0
            if rate == 0.0
            else selected_count == eligible_count
            if rate == 1.0
            else abs(selected_count / eligible_count - rate) <= 0.02
        ),
    }


def _train_stage(
    trainer: sg25f.GraphTrainer,
    batches: Sequence[sg25e.DeviceBatch],
) -> Dict[str, Any]:
    trainer.model.train(True)
    samples = []
    losses = []
    all_finite = True
    started = time.perf_counter_ns()
    for batch in batches:
        record = trainer.replay(batch, timed=True, inspect=True)
        samples.append(record["wall_ms"])
        losses.append(record["loss"])
        all_finite = all_finite and math.isfinite(record["loss"])
    wall = (time.perf_counter_ns() - started) / 1e9
    warmup = len(samples) // 5
    return {
        "updates": len(batches),
        "all_losses_finite": all_finite,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "wall_seconds": wall,
        "timing": _sample_summary(samples[warmup:], 1),
    }


def _midpoint_snapshot(trainer: sg25f.GraphTrainer) -> Dict[str, Any]:
    return {
        "parameters": tuple(
            parameter.detach().clone() for parameter in trainer.parameters
        ),
        "optimizer": sg25f._tensor_state_snapshot(trainer.optimizer),
    }


def _restore_midpoint(
    trainer: sg25f.GraphTrainer,
    snapshot: Mapping[str, Any],
) -> None:
    _sync(trainer.device)
    with torch.no_grad():
        for parameter, baseline in zip(
            trainer.parameters, snapshot["parameters"]
        ):
            parameter.copy_(baseline)
        for parameter_index, parameter in enumerate(trainer.parameters):
            for name, baseline in snapshot["optimizer"][parameter_index].items():
                trainer.optimizer.state[parameter][name].copy_(baseline)
    trainer.optimizer.zero_grad(set_to_none=False)
    trainer.model.train(True)
    _sync(trainer.device)


def _candidate_quality(
    mode: str,
    rate: float,
    trainer: sg25f.GraphTrainer,
    stage_one: Mapping[str, Any],
    stage_two_batches: Sequence[sg25e.DeviceBatch],
    rollin_collection: Mapping[str, Any] | None,
    corruption: Mapping[str, Any],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    pre_valid: Mapping[str, Any],
    *,
    device: torch.device,
    total_epochs: int,
) -> Dict[str, Any]:
    stage_two = _train_stage(trainer, stage_two_batches)
    post = {
        split: sg0.evaluate_teacher(
            trainer.model, raw_examples[split], device=device
        )
        for split in ("train", "valid")
    }
    generation = sg0.generate_model(
        trainer.model,
        raw_examples["valid"],
        vocabulary,
        max_tokens=sg26b.VALID_GENERATION_TOKENS,
        device=device,
        include_records=True,
    )
    collection_wall = (
        rollin_collection["collection_wall_seconds"]
        if rate > 0.0 and rollin_collection is not None
        else 0.0
    )
    expected_updates = total_epochs * UPDATES_PER_EPOCH
    return {
        "mode": mode,
        "rate": rate,
        "epochs": total_epochs,
        "updates": stage_one["updates"] + stage_two["updates"],
        "expected_updates": expected_updates,
        "update_count_passed": (
            stage_one["updates"] + stage_two["updates"] == expected_updates
        ),
        "all_losses_finite": bool(
            stage_one["all_losses_finite"] and stage_two["all_losses_finite"]
        ),
        "stage_one": dict(stage_one),
        "stage_two": stage_two,
        "rollin_collection": (
            dict(rollin_collection) if rate > 0.0 else None
        ),
        "corruption": dict(corruption),
        "optimizer_update_wall_seconds": (
            stage_one["wall_seconds"] + stage_two["wall_seconds"]
        ),
        "end_to_end_training_wall_seconds": (
            stage_one["wall_seconds"]
            + collection_wall
            + corruption["corruption_wall_seconds"]
            + stage_two["wall_seconds"]
        ),
        "pre_teacher": {"valid": dict(pre_valid)},
        "post_teacher": post,
        "generation": generation,
        "model_test_teacher_calls": 0,
        "model_test_generation_calls": 0,
    }


def run_rate_sweep(
    mode: str,
    rates: Sequence[float],
    train_examples: Sequence[Any],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    benchmark_epochs: int,
    quality_epochs: int,
    equivalence_updates: int,
) -> Dict[str, Dict[str, Any]]:
    if quality_epochs % 2:
        raise ValueError("quality epochs must split evenly")
    stage_epochs = quality_epochs // 2
    stage_updates = stage_epochs * UPDATES_PER_EPOCH
    benchmark_batches = sg26a._build_schedule(
        train_examples,
        vocabulary,
        epochs=benchmark_epochs,
        seed=26_130_000,
        device=device,
    )
    quality_batches = sg26a._build_schedule(
        train_examples,
        vocabulary,
        epochs=quality_epochs,
        seed=26_140_000,
        device=device,
    )
    stage_one_batches = quality_batches[:stage_updates]
    stage_two_base = quality_batches[stage_updates:]
    equivalence_batches = sg26a._coverage_prefix(
        quality_batches, equivalence_updates
    )
    gc.collect()
    torch.cuda.empty_cache()
    with sg26b._patched_loss(0.0):
        model = sg25f._build_mode_model(mode, vocabulary, device=device)
        trainer_class = EagerTrainer if torch.version.hip is not None else sg25f.GraphTrainer
        trainer = trainer_class(mode, model, quality_batches, device=device)
        equivalence = sg25f.graph_equivalence(
            mode,
            trainer,
            equivalence_batches,
            vocabulary,
            device=device,
            updates=len(equivalence_batches),
        )
        benchmark = sg25e.graph_benchmark(trainer, benchmark_batches)
        trainer.reset()
        pre_valid = sg0.evaluate_teacher(
            model, raw_examples["valid"], device=device
        )
        stage_one = _train_stage(trainer, stage_one_batches)
        snapshot = _midpoint_snapshot(trainer)
        predictions = None
        rollin_collection = None
        if any(rate > 0.0 for rate in rates):
            predictions, rollin_collection = collect_rollin_predictions(
                model, train_examples, vocabulary, device=device
            )
        result = {}
        for rate in rates:
            _restore_midpoint(trainer, snapshot)
            stage_two_batches = tuple(
                _clone_batch(batch) for batch in stage_two_base
            )
            if rate > 0.0:
                if predictions is None:  # pragma: no cover
                    raise AssertionError("missing roll-in predictions")
                corruption = apply_rollin_corruption(
                    stage_two_batches,
                    train_examples,
                    predictions,
                    rate=rate,
                    stage_start_epoch=stage_epochs,
                    eos_id=vocabulary.eos_id,
                    device=device,
                )
            else:
                corruption = apply_rollin_corruption(
                    stage_two_batches,
                    train_examples,
                    {},
                    rate=0.0,
                    stage_start_epoch=stage_epochs,
                    eos_id=vocabulary.eos_id,
                    device=device,
                )
            quality = _candidate_quality(
                mode,
                rate,
                trainer,
                stage_one,
                stage_two_batches,
                rollin_collection,
                corruption,
                raw_examples,
                vocabulary,
                pre_valid,
                device=device,
                total_epochs=quality_epochs,
            )
            result[_rate_label(rate)] = {
                "mode": mode,
                "rate": rate,
                "capture": dict(trainer.capture_audit),
                "equivalence": dict(equivalence),
                "benchmark": dict(benchmark),
                "quality": quality,
            }
            del stage_two_batches
            gc.collect()
        del snapshot, predictions
    del (
        trainer,
        model,
        benchmark_batches,
        quality_batches,
        equivalence_batches,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def select_rate(
    sweep: Mapping[str, Mapping[str, Any]],
    valid_baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    control = sweep[_rate_label(0.0)]["quality"]
    control_nll = control["post_teacher"]["valid"]["nll"]
    eligible = {}
    for label, record in sweep.items():
        quality = record["quality"]
        generation = quality["generation"]
        eligible[label] = bool(
            quality["all_losses_finite"]
            and quality["update_count_passed"]
            and quality["corruption"]["passed"]
            and quality["post_teacher"]["valid"]["nll"]
            <= control_nll + 0.10
            and generation["paired_action_sensitivity"] >= 0.50
        )
    candidates = [label for label, passed in eligible.items() if passed]
    selected = max(
        candidates,
        key=lambda label: (
            sweep[label]["quality"]["generation"]["edit_similarity"],
            sweep[label]["quality"]["generation"]["room_accuracy"] or -1.0,
            -sweep[label]["rate"],
        ),
    )
    generation = sweep[selected]["quality"]["generation"]
    return {
        "eligible": eligible,
        "control_valid_nll": control_nll,
        "selected_label": selected,
        "selected_rate": sweep[selected]["rate"],
        "selected_valid_edit_similarity": generation["edit_similarity"],
        "selected_valid_room_accuracy": generation["room_accuracy"],
        "valid_task_edit_threshold": valid_baseline["task_edit_threshold"],
        "passed": generation["edit_similarity"]
        >= valid_baseline["task_edit_threshold"],
    }


def _decision(
    data_audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    snn_sweep: Mapping[str, Mapping[str, Any]],
    primary: Mapping[str, Mapping[str, Any]],
    *,
    quick: bool,
    local_rocm: bool,
) -> Dict[str, Any]:
    execution_protocol_pass = all(
        record["capture"]["shape_count"] == len(sg26a.BUCKET_CAPACITIES)
        and (
            record["capture"].get("execution_mode") == "uniform_eager"
            if local_rocm
            else "execution_mode" not in record["capture"]
        )
        for record in primary.values()
    )
    equivalence_pass = all(
        record["equivalence"]["passed"] for record in primary.values()
    )
    eps = {
        mode: record["benchmark"]["effective_examples_per_second"]
        for mode, record in primary.items()
    }
    target_tps = {
        mode: record["benchmark"]["effective_target_tokens_per_second"]
        for mode, record in primary.items()
    }
    p50 = {
        mode: record["benchmark"]["per_real_example_timing"]["p50_ms"]
        for mode, record in primary.items()
    }
    legacy_floor_pass = bool(
        local_rocm
        or eps["snn_parallel"] >= 0.75 * SG25F_PARALLEL_EXAMPLES_PER_SECOND
    )
    speed_pass = (
        legacy_floor_pass
        and eps["snn_parallel"] >= eps["lstm"]
        and eps["snn_parallel"] >= eps["transformer"]
        and target_tps["snn_parallel"] >= target_tps["lstm"]
        and target_tps["snn_parallel"] >= target_tps["transformer"]
        and p50["snn_parallel"] <= p50["lstm"]
        and p50["snn_parallel"] <= p50["transformer"]
    )
    quality = {mode: record["quality"] for mode, record in primary.items()}
    isolated = all(
        record["all_losses_finite"]
        and record["update_count_passed"]
        and record["model_test_teacher_calls"] == 0
        and record["model_test_generation_calls"] == 0
        for record in quality.values()
    )
    ann_improvement = all(
        quality[mode]["post_teacher"]["valid"]["nll"]
        <= quality[mode]["pre_teacher"]["valid"]["nll"] - 0.10
        for mode in ("lstm", "transformer")
    )
    best_ann_nll = min(
        quality[mode]["post_teacher"]["valid"]["nll"]
        for mode in ("lstm", "transformer")
    )
    best_ann_edit = max(
        quality[mode]["generation"]["edit_similarity"]
        for mode in ("lstm", "transformer")
    )
    snn = quality["snn_parallel"]
    cross_pass = (
        snn["post_teacher"]["valid"]["nll"] <= best_ann_nll + 0.10
        and snn["generation"]["edit_similarity"] >= best_ann_edit - 0.05
        and snn["generation"]["paired_action_sensitivity"] >= 0.50
    )
    quality_pass = isolated and ann_improvement and cross_pass
    task_threshold = selection["valid_task_edit_threshold"]
    best_neural_edit = max(
        record["generation"]["edit_similarity"] for record in quality.values()
    )
    task_pass = (
        snn["generation"]["edit_similarity"] >= task_threshold
        and best_neural_edit >= task_threshold
    )
    control = snn_sweep[_rate_label(0.0)]["quality"]
    control_reproduction = {
        "valid_nll_absolute_gap_to_sg26b": abs(
            control["post_teacher"]["valid"]["nll"]
            - SG26B_SNN_ALPHA0_VALID_NLL
        ),
        "valid_edit_absolute_gap_to_sg26b": abs(
            control["generation"]["edit_similarity"]
            - SG26B_SNN_ALPHA0_VALID_EDIT
        ),
    }
    infrastructure = {
        "data_gate": bool(data_audit["passed"]),
        "capture_gate": execution_protocol_pass,
        "equivalence_gate": equivalence_pass,
        "test_isolation_gate": isolated,
        "corruption_gate": all(
            record["quality"]["corruption"]["passed"]
            for record in snn_sweep.values()
        ),
    }
    diagnostics = {
        "selected_rate": selection["selected_rate"],
        "selected_valid_edit_similarity": selection[
            "selected_valid_edit_similarity"
        ],
        "valid_task_edit_threshold": task_threshold,
        "effective_examples_per_second": eps,
        "effective_target_tokens_per_second": target_tps,
        "per_real_example_p50_ms": p50,
        "legacy_v100_absolute_floor_applied": not local_rocm,
        "legacy_v100_absolute_floor_pass": legacy_floor_pass,
        "execution_protocol": (
            "uniform_eager" if local_rocm else "cuda_graph"
        ),
        "best_ann_valid_nll": best_ann_nll,
        "best_ann_valid_edit_similarity": best_ann_edit,
        "best_neural_valid_edit_similarity": best_neural_edit,
        "snn_valid_nll": snn["post_teacher"]["valid"]["nll"],
        "snn_valid_edit_similarity": snn["generation"]["edit_similarity"],
        "end_to_end_training_wall_seconds": {
            mode: record["end_to_end_training_wall_seconds"]
            for mode, record in quality.items()
        },
        "control_reproduction": control_reproduction,
    }
    if quick:
        return {
            **{
                name: "PASS" if passed else "FAIL"
                for name, passed in infrastructure.items()
            },
            "selection_gate": "SMOKE",
            "speed_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "task_validity_gate": "SMOKE",
            "overall": "SMOKE" if all(infrastructure.values()) else "FAIL",
            **diagnostics,
        }
    gates = {
        **infrastructure,
        "selection_gate": bool(selection["passed"]),
        "speed_gate": speed_pass,
        "quality_gate": quality_pass,
        "task_validity_gate": task_pass,
    }
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        **diagnostics,
        "ann_valid_nll_improvement_pass": ann_improvement,
        "snn_cross_architecture_quality_pass": cross_pass,
        "next_route": (
            "sg26d_fresh_corpus_confirmation"
            if all(gates.values())
            else "factorized_room_transition_objective"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG26C requires a CUDA/HIP PyTorch device")
    device = torch.device("cuda:0")
    local_rocm = torch.version.hip is not None
    device_name = torch.cuda.get_device_name(device)
    if local_rocm:
        if "7800 XT" not in device_name.upper():
            raise AssertionError("local SG26C is frozen to the RX 7800 XT")
        backend_context = _local_rocm_triton_backend
        serial_extension: Mapping[str, Any] = {
            "status": "not_loaded",
            "reason": "ROCm run uses portable serial oracle only",
        }
        parallel_extension: Mapping[str, Any] = {
            "status": "replaced",
            "backend": "triton_fused_gated_trace",
            **triton_fused_backend_audit(),
        }
    else:
        if "V100" not in device_name.upper():
            raise AssertionError("CUDA SG26C requires the frozen V100 backend")
        backend_context = sg26a._expanded_backend
        _, serial_extension = sg25f.load_serial_extension()
        _, parallel_extension = sg25f.load_parallel_extension()
    reference = ROOT / "results/e3_scan/e3_sg26b_length_mass_objective.json"
    reference_sha = _sha256(reference)
    if reference_sha != EXPECTED_SG26B_SHA256:
        raise AssertionError("SG26C SG26B reference hash mismatch")
    corpus_root = args.corpus_dir.expanduser().resolve()
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=sg22r.EXPECTED_SEEDS
    )
    corpus = sg0.load_event_corpus(corpus_root)
    raw_examples, vocabulary = sg0.build_counterfactual_examples(
        corpus_root, corpus
    )
    data_audit = sg26a.expanded_data_audit(raw_examples, vocabulary)
    bucket_audit = sg26a.expanded_bucket_audit(raw_examples["train"])
    valid_baseline = sg26b.valid_action_majority(raw_examples, vocabulary)
    if not data_audit["passed"] or not bucket_audit["passed"]:
        raise AssertionError("SG26C data/bucket audit failed")

    with backend_context():
        snn_sweep = run_rate_sweep(
            "snn_parallel",
            RATES,
            raw_examples["train"],
            raw_examples,
            vocabulary,
            device=device,
            benchmark_epochs=args.benchmark_epochs,
            quality_epochs=args.quality_epochs,
            equivalence_updates=args.equivalence_updates,
        )
        selection = select_rate(snn_sweep, valid_baseline)
        selected_label = selection["selected_label"]
        selected_rate = selection["selected_rate"]
        primary = {"snn_parallel": snn_sweep[selected_label]}
        for mode in ("lstm", "transformer"):
            record = run_rate_sweep(
                mode,
                (selected_rate,),
                raw_examples["train"],
                raw_examples,
                vocabulary,
                device=device,
                benchmark_epochs=args.benchmark_epochs,
                quality_epochs=args.quality_epochs,
                equivalence_updates=args.equivalence_updates,
            )
            primary[mode] = record[selected_label]

    decision = _decision(
        data_audit,
        selection,
        snn_sweep,
        primary,
        quick=args.quick,
        local_rocm=local_rocm,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG26C two-stage snapshot self-roll-in",
        "formal": not args.quick,
        "development_selection_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
            "execution_backend": (
                "rocm_triton_fused" if local_rocm else "cuda_extensions"
            ),
        },
        "configuration": {
            "rates": RATES,
            "modes": MODES,
            "batch_size": BATCH_SIZE,
            "bucket_capacities": sg26a.BUCKET_CAPACITIES,
            "benchmark_epochs": args.benchmark_epochs,
            "quality_epochs": args.quality_epochs,
            "stage_one_epochs": args.quality_epochs // 2,
            "stage_two_epochs": args.quality_epochs // 2,
            "equivalence_updates": args.equivalence_updates,
            "loss_alpha": 0.0,
            "rollin_seed": ROLLIN_SEED,
            "premature_rollout_fill": "eos",
            "optimizer": "AdamW(lr=1e-3,betas=.9/.999,wd=.01,fused,capturable)",
            "model_test_teacher_calls": 0,
            "model_test_generation_calls": 0,
            "device": "cuda:0",
            "hardware_frozen_to": device_name,
        },
        "provenance": {
            "sg26b_reference": str(reference),
            "sg26b_reference_sha256": reference_sha,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "manifest": manifest,
            "serial_extension": serial_extension,
            "parallel_extension": parallel_extension,
            "data_audit": data_audit,
            "bucket_audit": bucket_audit,
            "valid_action_majority": valid_baseline,
        },
        "snn_rate_sweep": snn_sweep,
        "selection": selection,
        "primary_selected_rate": primary,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir", type=Path, default=sg22r.DEFAULT_CORPUS
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg26c_snapshot_self_rollin.json"),
    )
    parser.add_argument("--benchmark-epochs", type=int, default=10)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--equivalence-updates", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(
        args.benchmark_epochs,
        args.quality_epochs,
        args.equivalence_updates,
    ) <= 0:
        parser.error("all counts must be positive")
    if args.quick:
        args.benchmark_epochs = 1
        args.quality_epochs = min(args.quality_epochs, 2)
        args.equivalence_updates = min(args.equivalence_updates, 8)
    if args.quality_epochs % 2:
        parser.error("quality epochs must be even")
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
