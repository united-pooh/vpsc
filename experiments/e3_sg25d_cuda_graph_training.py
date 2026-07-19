"""SG25D exact-shape CUDA Graph training comparison across SNN and ANN cores."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment, _sample_summary, _sync  # noqa: E402
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_ra0_reverse_adjoint import build_textworld_models  # noqa: E402
from vpsc.world_model.cores import E3GatedTraceScanCore  # noqa: E402
from vpsc.world_model.fused_gated_trace_cuda import load_extension  # noqa: E402


ARCHITECTURES = ("snn_ra0", "lstm", "transformer")
EXPECTED_SG25C_SHA256 = (
    "9657c17ca695fd3e4b310d2068d93eb231dbe1005b6360bfb84c99fd6a749f2b"
)
SG25C_SEED0_NLL = 2.6957537054423932
SG25C_SEED0_EDIT = 0.6513394872257832


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeviceExample:
    index: int
    input_ids: torch.Tensor
    query_indices: torch.Tensor
    targets: torch.Tensor

    @property
    def key(self) -> Tuple[int, Tuple[int, ...]]:
        return (
            int(self.input_ids.shape[1]),
            tuple(int(value) for value in self.query_indices.tolist()),
        )


@dataclass
class GraphEntry:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    query_indices: torch.Tensor
    targets: torch.Tensor
    loss: torch.Tensor
    logits: torch.Tensor


def _device_examples(
    examples: Sequence[Any], *, device: torch.device
) -> Tuple[DeviceExample, ...]:
    records = []
    for index, example in enumerate(examples):
        input_ids, query_indices, targets = sg0._example_tensors(
            example, device=device
        )
        records.append(
            DeviceExample(
                index=index,
                input_ids=input_ids,
                query_indices=query_indices,
                targets=targets,
            )
        )
    return tuple(records)


def _build_model(
    name: str, vocabulary: Any, *, device: torch.device
) -> Any:
    models = build_textworld_models(9_400_000, vocabulary, device=device)
    model = models[name]
    del models
    if name == "snn_ra0":
        if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
            raise TypeError("RA0 builder returned the wrong core")
        model.core.scan_math_mode = "cuda_fused"
    return model


def _optimizer(model: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=0.01,
        fused=True,
        capturable=True,
    )


def _forward_logits(
    name: str,
    model: Any,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    if name == "snn_ra0":
        logits, _ = tw0._sparse_forward(
            model,
            input_ids,
            query_indices,
            None,
            use_eligibility=True,
            detach_state=True,
        )
        return logits
    output = model(input_ids, None, detach_state=True)
    return output.logits.index_select(1, query_indices)


def _step_body(
    name: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=False)
    logits = _forward_logits(name, model, input_ids, query_indices)
    loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size), targets.reshape(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        1.0,
        foreach=True,
    )
    optimizer.step()
    return loss, logits


def _tensor_state_snapshot(
    optimizer: torch.optim.Optimizer,
) -> Dict[int, Dict[str, torch.Tensor]]:
    snapshot: Dict[int, Dict[str, torch.Tensor]] = {}
    for parameter_index, parameter in enumerate(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ):
        snapshot[parameter_index] = {
            name: value.detach().clone()
            for name, value in optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor)
        }
    return snapshot


class ExactShapeGraphTrainer:
    def __init__(
        self,
        name: str,
        model: Any,
        examples: Sequence[DeviceExample],
        *,
        device: torch.device,
    ) -> None:
        self.name = name
        self.model = model
        self.device = device
        self.optimizer = _optimizer(model)
        self.parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        optimizer_parameters = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        if len(optimizer_parameters) != len(self.parameters) or any(
            optimizer_parameter is not model_parameter
            for optimizer_parameter, model_parameter in zip(
                optimizer_parameters, self.parameters
            )
        ):
            raise AssertionError("optimizer/model parameter ordering mismatch")
        self._parameter_baseline = tuple(
            parameter.detach().clone() for parameter in self.parameters
        )
        self.entries: Dict[Tuple[int, Tuple[int, ...]], GraphEntry] = {}

        representatives: Dict[Tuple[int, Tuple[int, ...]], DeviceExample] = {}
        for example in examples:
            representatives.setdefault(example.key, example)
        self.shape_count = len(representatives)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter_ns()

        first = next(iter(representatives.values()))
        _step_body(
            name,
            model,
            self.optimizer,
            first.input_ids,
            first.query_indices,
            first.targets,
        )
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
        self._optimizer_baseline = _tensor_state_snapshot(self.optimizer)

        warmup_stream = torch.cuda.Stream(device=device)
        for key, example in representatives.items():
            static_input = example.input_ids.clone()
            static_targets = example.targets.clone()
            static_queries = example.query_indices.clone()
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream):
                _step_body(
                    name,
                    model,
                    self.optimizer,
                    static_input,
                    static_queries,
                    static_targets,
                )
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            _sync(device)
            self.reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                loss, logits = _step_body(
                    name,
                    model,
                    self.optimizer,
                    static_input,
                    static_queries,
                    static_targets,
                )
            _sync(device)
            self.entries[key] = GraphEntry(
                graph=graph,
                input_ids=static_input,
                query_indices=static_queries,
                targets=static_targets,
                loss=loss,
                logits=logits,
            )
            self.reset()
        restore_started = time.perf_counter_ns()
        self.reset()
        restore_wall_ms = (time.perf_counter_ns() - restore_started) / 1e6
        capture_seconds = (time.perf_counter_ns() - started) / 1e9
        cold_replay = self.replay(first, timed=True, inspect=True)
        cold_first_replay_wall_ms = cold_replay["wall_ms"]
        cold_first_replay_loss = cold_replay["loss"]
        self.reset()
        self.capture_audit = {
            "shape_count": self.shape_count,
            "capture_seconds": capture_seconds,
            "restore_wall_ms": restore_wall_ms,
            "cold_first_replay_wall_ms": cold_first_replay_wall_ms,
            "cold_first_replay_loss": cold_first_replay_loss,
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
            "peak_additional_allocated_bytes": max(
                0, torch.cuda.max_memory_allocated(device) - allocated_before
            ),
            "peak_additional_reserved_bytes": max(
                0, torch.cuda.max_memory_reserved(device) - reserved_before
            ),
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
        self, example: DeviceExample, *, timed: bool, inspect: bool
    ) -> Dict[str, Any]:
        entry = self.entries[example.key]
        if not torch.equal(entry.query_indices, example.query_indices):  # pragma: no cover
            raise AssertionError("graph query key mismatch")
        if timed:
            _sync(self.device)
            started = time.perf_counter_ns()
        entry.input_ids.copy_(example.input_ids)
        entry.targets.copy_(example.targets)
        entry.graph.replay()
        _sync(self.device)
        wall_ms = (
            (time.perf_counter_ns() - started) / 1e6 if timed else None
        )
        result: Dict[str, Any] = {"wall_ms": wall_ms}
        if inspect:
            result.update(
                {
                    "loss": float(entry.loss.item()),
                    "predictions": entry.logits.detach().argmax(dim=-1).clone(),
                }
            )
        return result


def _eager_update(
    name: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    example: DeviceExample,
    *,
    device: torch.device,
    timed: bool,
    inspect: bool,
) -> Dict[str, Any]:
    if timed:
        _sync(device)
        started = time.perf_counter_ns()
    loss, logits = _step_body(
        name,
        model,
        optimizer,
        example.input_ids,
        example.query_indices,
        example.targets,
    )
    _sync(device)
    wall_ms = (time.perf_counter_ns() - started) / 1e6 if timed else None
    result: Dict[str, Any] = {"wall_ms": wall_ms}
    if inspect:
        result.update(
            {
                "loss": float(loss.item()),
                "predictions": logits.detach().argmax(dim=-1).clone(),
            }
        )
    return result


def _schedule_examples(
    examples: Sequence[DeviceExample], schedule: Iterable[int]
) -> Tuple[DeviceExample, ...]:
    return tuple(examples[index] for index in schedule)


def graph_equivalence(
    name: str,
    trainer: ExactShapeGraphTrainer,
    examples: Sequence[DeviceExample],
    vocabulary: Any,
    *,
    device: torch.device,
    updates: int,
) -> Dict[str, Any]:
    trainer.reset()
    eager_model = _build_model(name, vocabulary, device=device)
    eager_optimizer = _optimizer(eager_model)
    schedule = sg0._training_schedule(
        len(examples), math.ceil(updates / len(examples)), 9_405_000
    )[:updates]
    loss_gaps = []
    prediction_disagreements = 0
    prediction_count = 0
    all_losses_finite = True
    for example in _schedule_examples(examples, schedule):
        eager = _eager_update(
            name,
            eager_model,
            eager_optimizer,
            example,
            device=device,
            timed=False,
            inspect=True,
        )
        graphed = trainer.replay(example, timed=False, inspect=True)
        all_losses_finite = all_losses_finite and math.isfinite(
            eager["loss"]
        ) and math.isfinite(graphed["loss"])
        loss_gaps.append(abs(graphed["loss"] - eager["loss"]))
        prediction_disagreements += int(
            (graphed["predictions"] != eager["predictions"]).sum().item()
        )
        prediction_count += eager["predictions"].numel()
    disagreement_rate = prediction_disagreements / prediction_count
    result = {
        "updates": len(schedule),
        "mean_loss_abs_gap": sum(loss_gaps) / len(loss_gaps),
        "last_loss_abs_gap": loss_gaps[-1],
        "maximum_loss_abs_gap": max(loss_gaps),
        "prediction_disagreements": prediction_disagreements,
        "prediction_count": prediction_count,
        "prediction_disagreement_rate": disagreement_rate,
        "all_losses_finite": all_losses_finite,
        "passed": all_losses_finite
        and sum(loss_gaps) / len(loss_gaps) <= 0.01
        and loss_gaps[-1] <= 0.02
        and disagreement_rate <= 0.01,
    }
    del eager_model, eager_optimizer
    trainer.reset()
    return result


def eager_benchmark(
    name: str,
    examples: Sequence[DeviceExample],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated(device)
    model = _build_model(name, vocabulary, device=device)
    optimizer = _optimizer(model)
    torch.cuda.reset_peak_memory_stats(device)
    schedule = sg0._training_schedule(len(examples), epochs, 9_406_000)
    samples = []
    for example in _schedule_examples(examples, schedule):
        result = _eager_update(
            name,
            model,
            optimizer,
            example,
            device=device,
            timed=True,
            inspect=False,
        )
        samples.append(result["wall_ms"])
    warmup = len(samples) // 5
    steady = samples[warmup:]
    record = {
        "epochs": epochs,
        "updates": len(schedule),
        "warmup_updates_excluded": warmup,
        "timing": _sample_summary(steady, 1),
        "examples_per_second_mean": 1000.0 / (sum(steady) / len(steady)),
        "allocated_total_delta_bytes": max(
            0, torch.cuda.max_memory_allocated(device) - allocated_before
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return record


def graph_benchmark(
    trainer: ExactShapeGraphTrainer,
    examples: Sequence[DeviceExample],
    *,
    epochs: int,
) -> Dict[str, Any]:
    trainer.reset()
    schedule = sg0._training_schedule(len(examples), epochs, 9_406_000)
    samples = []
    for example in _schedule_examples(examples, schedule):
        samples.append(
            trainer.replay(example, timed=True, inspect=False)["wall_ms"]
        )
    warmup = len(samples) // 5
    steady = samples[warmup:]
    result = {
        "epochs": epochs,
        "updates": len(schedule),
        "warmup_updates_excluded": warmup,
        "timing": _sample_summary(steady, 1),
        "examples_per_second_mean": 1000.0 / (sum(steady) / len(steady)),
    }
    trainer.reset()
    return result


def profiler_audit(
    trainer: ExactShapeGraphTrainer,
    example: DeviceExample,
) -> Dict[str, Any]:
    trainer.reset()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profile:
        trainer.replay(example, timed=False, inspect=False)
    events = profile.events()
    names = [str(event.name) for event in events]
    host_launches = sum(
        name in ("cudaLaunchKernel", "cudaGraphLaunch") for name in names
    )
    host_api = sum(
        name in ("cudaLaunchKernel", "cudaGraphLaunch", "cudaMemcpyAsync")
        for name in names
    )
    cuda_kernel_events = sum(
        "cuda" in str(getattr(event, "device_type", "")).lower()
        for event in events
    )
    trainer.reset()
    return {
        "host_launch_count": host_launches,
        "host_launch_and_copy_api_count": host_api,
        "cuda_kernel_event_count": cuda_kernel_events,
        "cuda_graph_launch_count": names.count("cudaGraphLaunch"),
        "cuda_launch_kernel_count": names.count("cudaLaunchKernel"),
        "cuda_memcpy_async_count": names.count("cudaMemcpyAsync"),
    }


def eager_profiler_audit(
    name: str,
    example: DeviceExample,
    vocabulary: Any,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    model = _build_model(name, vocabulary, device=device)
    optimizer = _optimizer(model)
    _eager_update(
        name,
        model,
        optimizer,
        example,
        device=device,
        timed=False,
        inspect=False,
    )
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profile:
        _eager_update(
            name,
            model,
            optimizer,
            example,
            device=device,
            timed=False,
            inspect=False,
        )
    events = profile.events()
    names = [str(event.name) for event in events]
    record = {
        "host_launch_count": sum(
            name in ("cudaLaunchKernel", "cudaGraphLaunch") for name in names
        ),
        "host_launch_and_copy_api_count": sum(
            name in ("cudaLaunchKernel", "cudaGraphLaunch", "cudaMemcpyAsync")
            for name in names
        ),
        "cuda_graph_launch_count": names.count("cudaGraphLaunch"),
        "cuda_launch_kernel_count": names.count("cudaLaunchKernel"),
        "cuda_memcpy_async_count": names.count("cudaMemcpyAsync"),
        "cuda_kernel_event_count": sum(
            "cuda" in str(getattr(event, "device_type", "")).lower()
            for event in events
        ),
    }
    del model, optimizer
    return record


def graph_quality(
    trainer: ExactShapeGraphTrainer,
    train_examples: Sequence[DeviceExample],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    trainer.reset()
    pre = {
        split: sg0.evaluate_teacher(
            trainer.model, raw_examples[split], device=device
        )
        for split in ("valid", "test")
    }
    trainer.model.train(True)
    schedule = sg0._training_schedule(len(train_examples), epochs, 9_401_000)
    samples = []
    losses = []
    for example in _schedule_examples(train_examples, schedule):
        result = trainer.replay(example, timed=True, inspect=True)
        samples.append(result["wall_ms"])
        losses.append(result["loss"])
    post = {
        split: sg0.evaluate_teacher(
            trainer.model, raw_examples[split], device=device
        )
        for split in ("train", "valid", "test")
    }
    generation = sg0.generate_model(
        trainer.model,
        raw_examples["test"],
        vocabulary,
        max_tokens=sg0.MAX_GENERATION_TOKENS,
        device=device,
        include_records=True,
    )
    if trainer.name == "snn_ra0":
        passed = (
            post["test"]["nll"] <= SG25C_SEED0_NLL + 0.10
            and generation["edit_similarity"] >= SG25C_SEED0_EDIT - 0.05
            and generation["paired_action_sensitivity"] >= 0.50
        )
    else:
        passed = post["test"]["nll"] <= pre["test"]["nll"] - 0.10
    return {
        "epochs": epochs,
        "updates": len(schedule),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "timing": _sample_summary(samples[len(samples) // 5 :], 1),
        "pre_teacher": pre,
        "post_teacher": post,
        "generation": generation,
        "passed": passed,
    }


def _decision(
    architectures: Mapping[str, Mapping[str, Any]],
    *,
    quality: Mapping[str, Mapping[str, Any]] | None,
    quick: bool,
) -> Dict[str, Any]:
    snn = architectures["snn_ra0"]
    equivalence_pass = all(
        record["equivalence"]["passed"] for record in architectures.values()
    )
    snn_eager_speedup = (
        snn["eager_benchmark"]["timing"]["p50_ms"]
        / snn["graph_benchmark"]["timing"]["p50_ms"]
    )
    graph_p50 = {
        name: record["graph_benchmark"]["timing"]["p50_ms"]
        for name, record in architectures.items()
    }
    graph_eps = {
        name: record["graph_benchmark"]["examples_per_second_mean"]
        for name, record in architectures.items()
    }
    ann_speed_pass = (
        graph_p50["snn_ra0"] <= graph_p50["lstm"]
        and graph_p50["snn_ra0"] <= graph_p50["transformer"]
        and graph_eps["snn_ra0"] >= graph_eps["lstm"]
        and graph_eps["snn_ra0"] >= graph_eps["transformer"]
    )
    launch_reduction = {}
    for name, record in architectures.items():
        eager_count = record["eager_profiler"]["host_launch_and_copy_api_count"]
        graph_count = record["graph_profiler"]["host_launch_and_copy_api_count"]
        launch_reduction[name] = (
            1.0 - graph_count / eager_count if eager_count > 0 else float("-inf")
        )
    launch_pass = all(value >= 0.50 for value in launch_reduction.values())
    memory_ratio = (
        snn["capture"]["allocated_delta_bytes"]
        / snn["eager_benchmark"]["allocated_total_delta_bytes"]
    )
    memory_pass = memory_ratio <= 4.0
    capture_pass = all(
        record["capture"]["shape_count"] == 35
        for record in architectures.values()
    )
    if quick:
        return {
            "capture_gate": "PASS" if capture_pass else "FAIL",
            "equivalence_gate": "PASS" if equivalence_pass else "FAIL",
            "speed_gate": "SMOKE",
            "graphed_ann_gate": "SMOKE",
            "launch_gate": "SMOKE",
            "memory_gate": "PASS" if memory_pass else "FAIL",
            "overall": "SMOKE" if capture_pass and equivalence_pass else "FAIL",
            "snn_eager_speedup": snn_eager_speedup,
            "graph_p50_ms": graph_p50,
            "graph_examples_per_second": graph_eps,
            "launch_reduction": launch_reduction,
        }
    quality_pass = quality is not None and all(
        record["passed"] for record in quality.values()
    )
    gates = {
        "capture_gate": capture_pass,
        "equivalence_gate": equivalence_pass,
        "speed_gate": snn_eager_speedup >= 1.5,
        "graphed_ann_gate": ann_speed_pass,
        "launch_gate": launch_pass,
        "memory_gate": memory_pass,
        "quality_gate": quality_pass,
    }
    overall = all(gates.values())
    capture_seconds = snn["capture"]["capture_seconds"]
    eager_ms = snn["eager_benchmark"]["timing"]["mean_ms"]
    graph_ms = snn["graph_benchmark"]["timing"]["mean_ms"]
    saved_seconds_per_update = max(0.0, (eager_ms - graph_ms) / 1000.0)
    break_even = (
        capture_seconds / saved_seconds_per_update
        if saved_seconds_per_update > 0.0
        else None
    )
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if overall else "FAIL",
        "snn_eager_speedup": snn_eager_speedup,
        "graph_p50_ms": graph_p50,
        "graph_examples_per_second": graph_eps,
        "launch_reduction": launch_reduction,
        "snn_graph_memory_to_eager_ratio": memory_ratio,
        "snn_capture_break_even_updates": break_even,
        "next_route": (
            "expanded_real_corpus" if overall else "padded_bucket_graph"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG25D requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG25D requires the frozen V100 backend")
    reference = ROOT / "results/e3_scan/e3_sg25c_native_fused_scan_cuda.json"
    reference_sha = _sha256(reference)
    if reference_sha != EXPECTED_SG25C_SHA256:
        raise AssertionError("SG25D SG25C reference hash mismatch")
    _, extension = load_extension()
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = sg0.load_event_corpus(corpus_root)
    raw_examples, vocabulary = sg0.build_counterfactual_examples(
        corpus_root, corpus
    )
    data_audit = sg0.audit_examples(raw_examples, vocabulary)
    if not data_audit["passed"]:
        raise AssertionError("SG25D data audit failed")
    device_examples = _device_examples(raw_examples["train"], device=device)
    unique_shapes = len({example.key for example in device_examples})
    if unique_shapes != 35:
        raise AssertionError(f"expected 35 graph shapes, got {unique_shapes}")

    architecture_results: Dict[str, Any] = {}
    trainers: Dict[str, ExactShapeGraphTrainer] = {}
    for name in ARCHITECTURES:
        eager = eager_benchmark(
            name,
            device_examples,
            vocabulary,
            device=device,
            epochs=args.benchmark_epochs,
        )
        eager_profiler = eager_profiler_audit(
            name, device_examples[0], vocabulary, device=device
        )
        gc.collect()
        torch.cuda.empty_cache()
        graph_clean_allocated = torch.cuda.memory_allocated(device)
        graph_clean_reserved = torch.cuda.memory_reserved(device)
        model = _build_model(name, vocabulary, device=device)
        trainer = ExactShapeGraphTrainer(
            name, model, device_examples, device=device
        )
        trainer.capture_audit.update(
            {
                "total_allocated_delta_from_clean_bytes": max(
                    0,
                    torch.cuda.memory_allocated(device)
                    - graph_clean_allocated,
                ),
                "total_reserved_delta_from_clean_bytes": max(
                    0,
                    torch.cuda.memory_reserved(device)
                    - graph_clean_reserved,
                ),
                "peak_total_allocated_delta_from_clean_bytes": max(
                    0,
                    trainer.capture_audit["peak_allocated_bytes"]
                    - graph_clean_allocated,
                ),
                "peak_total_reserved_delta_from_clean_bytes": max(
                    0,
                    trainer.capture_audit["peak_reserved_bytes"]
                    - graph_clean_reserved,
                ),
            }
        )
        trainers[name] = trainer
        equivalence = graph_equivalence(
            name,
            trainer,
            device_examples,
            vocabulary,
            device=device,
            updates=args.equivalence_updates,
        )
        graph = graph_benchmark(
            trainer,
            device_examples,
            epochs=args.benchmark_epochs,
        )
        graph_profiler = profiler_audit(trainer, device_examples[0])
        architecture_results[name] = {
            "capture": trainer.capture_audit,
            "equivalence": equivalence,
            "eager_benchmark": eager,
            "graph_benchmark": graph,
            "eager_profiler": eager_profiler,
            "graph_profiler": graph_profiler,
        }

    provisional = _decision(
        architecture_results, quality=None, quick=True
    )
    graph_p50 = provisional["graph_p50_ms"]
    graph_eps = provisional["graph_examples_per_second"]
    fair_speed_pass = (
        provisional["snn_eager_speedup"] >= 1.5
        and graph_p50["snn_ra0"] <= graph_p50["lstm"]
        and graph_p50["snn_ra0"] <= graph_p50["transformer"]
        and graph_eps["snn_ra0"] >= graph_eps["lstm"]
        and graph_eps["snn_ra0"] >= graph_eps["transformer"]
    )
    equivalence_pass = all(
        result["equivalence"]["passed"]
        for result in architecture_results.values()
    )
    quality = None
    if not args.quick and fair_speed_pass and equivalence_pass:
        quality = {
            name: graph_quality(
                trainers[name],
                device_examples,
                raw_examples,
                vocabulary,
                device=device,
                epochs=args.quality_epochs,
            )
            for name in ARCHITECTURES
        }
    decision = _decision(
        architecture_results,
        quality=quality,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG25D exact-shape CUDA Graph training",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "architectures": ARCHITECTURES,
            "unique_exact_shapes": unique_shapes,
            "equivalence_updates": args.equivalence_updates,
            "benchmark_epochs": args.benchmark_epochs,
            "quality_epochs": args.quality_epochs,
            "optimizer": "AdamW(fused=True,capturable=True)",
            "copy_included_in_graph_wall": True,
            "batch_size": 1,
            "device": "cuda:0",
        },
        "provenance": {
            "sg25c_reference": str(reference),
            "sg25c_reference_sha256": reference_sha,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "extension": extension,
            "data_audit": data_audit,
        },
        "architectures": architecture_results,
        "quality": quality,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg25d_cuda_graph_training.json"),
    )
    parser.add_argument("--equivalence-updates", type=int, default=20)
    parser.add_argument("--benchmark-epochs", type=int, default=10)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(
        args.equivalence_updates,
        args.benchmark_epochs,
        args.quality_epochs,
    ) <= 0:
        parser.error("all counts must be positive")
    if args.quick:
        args.equivalence_updates = min(args.equivalence_updates, 5)
        args.benchmark_epochs = 1
        args.quality_epochs = 2
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
