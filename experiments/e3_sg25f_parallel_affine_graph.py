"""SG25F block-parallel affine scan in the SG25E B16 CUDA Graph task."""

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
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments import e3_sg25e_bucketed_batch_graph as sg25e  # noqa: E402
from experiments.e3_sg25d_cuda_graph_training import (  # noqa: E402
    _build_model,
    _optimizer,
    _tensor_state_snapshot,
)
from vpsc.world_model.cores import E3GatedTraceScanCore  # noqa: E402
from vpsc.world_model.fused_batched_gated_trace_cuda import (  # noqa: E402
    fused_batched_gated_trace,
    load_extension as load_serial_extension,
)
from vpsc.world_model.fused_parallel_gated_trace_cuda import (  # noqa: E402
    fused_parallel_gated_trace,
    load_extension as load_parallel_extension,
)


MODES = ("snn_serial", "snn_parallel", "lstm", "transformer")
BATCH_SIZE = 16
EXPECTED_SG25E_SHA256 = (
    "87b531b8355e0294d7bdd069886908c2e37b39698bb9dee101bb988b0c202f73"
)
SG25E_LSTM_EXAMPLES_PER_SECOND = 20942.160305561825
SG25C_SEED0_NLL = 2.6957537054423932
SG25C_SEED0_EDIT = 0.6513394872257832


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_name(mode: str) -> str:
    return "snn_ra0" if mode.startswith("snn_") else mode


def _build_mode_model(mode: str, vocabulary: Any, *, device: torch.device) -> Any:
    return _build_model(_model_name(mode), vocabulary, device=device)


def _parallel_snn_logits(
    model: Any,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("parallel SNN path requires E3GatedTraceScanCore")
    embedded = model.input_dropout(model.embedding(input_ids))
    core = model.core
    state = core.initial_state(
        int(input_ids.shape[0]), device=embedded.device, dtype=embedded.dtype
    )
    layer = state.layers[0]
    drives = F.linear(
        embedded,
        core.input_event_projection.weight,
        core.input_event_projection.bias,
    )
    decay_e, decay_i = core.decays()
    raw, _, _ = fused_parallel_gated_trace(
        drives,
        query_indices,
        torch.stack((decay_e, decay_i), dim=0),
        layer.excitatory,
        layer.inhibitory,
        spike_threshold=core.spike_threshold,
        surrogate_scale=core.surrogate_scale,
    )
    sequence = core.output_projection(core.output_norm(raw))
    hidden = model.output_norm(model.output_dropout(sequence))
    return model.lm_head(hidden)


def _mode_logits(mode: str, model: Any, batch: sg25e.DeviceBatch) -> torch.Tensor:
    if mode == "snn_serial":
        return sg25e._snn_batched_logits(
            model, batch.input_ids, batch.query_indices
        )
    if mode == "snn_parallel":
        return _parallel_snn_logits(
            model, batch.input_ids, batch.query_indices
        )
    return sg25e._batched_logits(mode, model, batch)


def _step_body(
    mode: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: sg25e.DeviceBatch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=False)
    logits = _mode_logits(mode, model, batch)
    loss = sg25e._masked_example_mean_loss(
        logits,
        batch.targets,
        batch.token_mask,
        batch.example_mask,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        1.0,
        foreach=True,
    )
    optimizer.step()
    return loss, logits


@dataclass
class GraphEntry:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    query_indices: torch.Tensor
    gather_indices: torch.Tensor
    targets: torch.Tensor
    token_mask: torch.Tensor
    example_mask: torch.Tensor
    loss: torch.Tensor
    logits: torch.Tensor


def _static_batch(
    entry: GraphEntry, source: sg25e.DeviceBatch
) -> sg25e.DeviceBatch:
    return sg25e.DeviceBatch(
        input_ids=entry.input_ids,
        query_indices=entry.query_indices,
        gather_indices=entry.gather_indices,
        targets=entry.targets,
        token_mask=entry.token_mask,
        example_mask=entry.example_mask,
        real_example_count=source.real_example_count,
        input_token_count=source.input_token_count,
        target_token_count=source.target_token_count,
        example_indices=source.example_indices,
    )


class GraphTrainer:
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
            left is not right
            for left, right in zip(optimizer_parameters, self.parameters)
        ):
            raise AssertionError("optimizer/model parameter ordering mismatch")
        self._parameter_baseline = tuple(
            parameter.detach().clone() for parameter in self.parameters
        )
        representatives: Dict[Tuple[int, int, int], sg25e.DeviceBatch] = {}
        for batch in batches:
            representatives.setdefault(batch.key, batch)
        if len(representatives) != len(sg25e.BUCKET_CAPACITIES):
            raise AssertionError("SG25F requires exactly three bucket graphs")
        self.entries: Dict[Tuple[int, int, int], GraphEntry] = {}
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter_ns()

        first = next(iter(representatives.values()))
        _step_body(mode, model, self.optimizer, first)
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
        for key, source in representatives.items():
            entry = GraphEntry(
                graph=torch.cuda.CUDAGraph(),
                input_ids=source.input_ids.clone(),
                query_indices=source.query_indices.clone(),
                gather_indices=source.gather_indices.clone(),
                targets=source.targets.clone(),
                token_mask=source.token_mask.clone(),
                example_mask=source.example_mask.clone(),
                loss=torch.empty((), device=device),
                logits=torch.empty((), device=device),
            )
            batch = _static_batch(entry, source)
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream):
                _step_body(mode, model, self.optimizer, batch)
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            _sync(device)
            self.reset()
            with torch.cuda.graph(entry.graph):
                entry.loss, entry.logits = _step_body(
                    mode, model, self.optimizer, batch
                )
            _sync(device)
            self.entries[key] = entry
            self.reset()
        restore_started = time.perf_counter_ns()
        self.reset()
        restore_wall_ms = (time.perf_counter_ns() - restore_started) / 1e6
        capture_seconds = (time.perf_counter_ns() - started) / 1e9
        cold = self.replay(first, timed=True, inspect=True)
        self.reset()
        self.capture_audit = {
            "shape_count": len(self.entries),
            "keys": tuple(tuple(key) for key in sorted(self.entries)),
            "capture_seconds": capture_seconds,
            "restore_wall_ms": restore_wall_ms,
            "cold_first_replay_wall_ms": cold["wall_ms"],
            "cold_first_replay_loss": cold["loss"],
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
        entry.graph.replay()
        _sync(self.device)
        result: Dict[str, Any] = {
            "wall_ms": (
                (time.perf_counter_ns() - started) / 1e6 if timed else None
            )
        }
        if inspect:
            result.update(
                {
                    "loss": float(entry.loss.item()),
                    "predictions": entry.logits.detach().argmax(dim=-1).clone(),
                }
            )
        return result


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def kernel_equivalence_audit(*, device: torch.device) -> Dict[str, Any]:
    cases = []
    for batch_size in (1, 4, 16):
        for case_index, (time_steps, query_capacity) in enumerate(
            sg25e.BUCKET_CAPACITIES
        ):
            torch.manual_seed(25_900_100 + batch_size * 10 + case_index)
            state_dim = 9
            queries, valid_counts = sg25e._kernel_query_matrix(
                batch_size,
                time_steps,
                query_capacity,
                device=device,
            )
            serial_drives = torch.randn(
                batch_size,
                time_steps,
                4 * state_dim,
                device=device,
                requires_grad=True,
            )
            parallel_drives = serial_drives.detach().clone().requires_grad_(True)
            serial_decays = (
                0.5 + 0.45 * torch.rand(2, state_dim, device=device)
            ).requires_grad_(True)
            parallel_decays = serial_decays.detach().clone().requires_grad_(True)
            serial_initial_e = torch.rand(
                batch_size, state_dim, device=device, requires_grad=True
            )
            serial_initial_i = torch.rand(
                batch_size, state_dim, device=device, requires_grad=True
            )
            parallel_initial_e = (
                serial_initial_e.detach().clone().requires_grad_(True)
            )
            parallel_initial_i = (
                serial_initial_i.detach().clone().requires_grad_(True)
            )
            serial_raw, serial_final_e, serial_final_i = (
                fused_batched_gated_trace(
                    serial_drives,
                    queries,
                    serial_decays,
                    serial_initial_e,
                    serial_initial_i,
                    spike_threshold=0.5,
                    surrogate_scale=4.0,
                )
            )
            parallel_raw, parallel_final_e, parallel_final_i = (
                fused_parallel_gated_trace(
                    parallel_drives,
                    queries,
                    parallel_decays,
                    parallel_initial_e,
                    parallel_initial_i,
                    spike_threshold=0.5,
                    surrogate_scale=4.0,
                )
            )
            raw_probe = torch.randn_like(serial_raw)
            final_e_probe = torch.randn_like(serial_final_e)
            final_i_probe = torch.randn_like(serial_final_i)
            (
                (serial_raw * raw_probe).sum()
                + (serial_final_e * final_e_probe).sum()
                + (serial_final_i * final_i_probe).sum()
            ).backward()
            (
                (parallel_raw * raw_probe).sum()
                + (parallel_final_e * final_e_probe).sum()
                + (parallel_final_i * final_i_probe).sum()
            ).backward()
            valid_mask = queries.ge(0).unsqueeze(-1).expand_as(serial_raw)
            spike_width = 2 * state_dim
            spike_mask = queries.ge(0).unsqueeze(-1).expand(
                -1, -1, spike_width
            )
            errors = {
                "raw_max_abs": _max_abs(parallel_raw, serial_raw),
                "final_e_max_abs": _max_abs(parallel_final_e, serial_final_e),
                "final_i_max_abs": _max_abs(parallel_final_i, serial_final_i),
                "drive_grad_max_abs": _max_abs(
                    parallel_drives.grad, serial_drives.grad
                ),
                "decay_grad_max_abs": _max_abs(
                    parallel_decays.grad, serial_decays.grad
                ),
                "initial_e_grad_max_abs": _max_abs(
                    parallel_initial_e.grad, serial_initial_e.grad
                ),
                "initial_i_grad_max_abs": _max_abs(
                    parallel_initial_i.grad, serial_initial_i.grad
                ),
            }
            forward_pass = (
                torch.allclose(
                    parallel_raw, serial_raw, atol=2e-6, rtol=2e-5
                )
                and torch.allclose(
                    parallel_final_e,
                    serial_final_e,
                    atol=2e-6,
                    rtol=2e-5,
                )
                and torch.allclose(
                    parallel_final_i,
                    serial_final_i,
                    atol=2e-6,
                    rtol=2e-5,
                )
            )
            gradient_pass = all(
                torch.allclose(left, right, atol=3e-5, rtol=3e-4)
                for left, right in (
                    (parallel_drives.grad, serial_drives.grad),
                    (parallel_decays.grad, serial_decays.grad),
                    (parallel_initial_e.grad, serial_initial_e.grad),
                    (parallel_initial_i.grad, serial_initial_i.grad),
                )
            )
            padding_nonzero = int(
                torch.count_nonzero(parallel_raw[~valid_mask]).item()
            )
            spike_disagreements = int(
                (
                    parallel_raw[:, :, :spike_width][spike_mask]
                    != serial_raw[:, :, :spike_width][spike_mask]
                )
                .sum()
                .item()
            )
            all_finite = all(
                bool(torch.isfinite(value).all())
                for value in (
                    parallel_raw,
                    parallel_final_e,
                    parallel_final_i,
                    parallel_drives.grad,
                    parallel_decays.grad,
                    parallel_initial_e.grad,
                    parallel_initial_i.grad,
                )
            )
            cases.append(
                {
                    "batch_size": batch_size,
                    "time_steps": time_steps,
                    "query_capacity": query_capacity,
                    "valid_query_counts": valid_counts,
                    "errors": errors,
                    "padding_nonzero": padding_nonzero,
                    "spike_disagreements": spike_disagreements,
                    "all_finite": all_finite,
                    "forward_passed": forward_pass,
                    "gradient_passed": gradient_pass,
                    "passed": forward_pass
                    and gradient_pass
                    and padding_nonzero == 0
                    and spike_disagreements == 0
                    and all_finite,
                }
            )
    return {"cases": cases, "passed": all(case["passed"] for case in cases)}


def _eager_update(
    mode: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: sg25e.DeviceBatch,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    loss, logits = _step_body(mode, model, optimizer, batch)
    _sync(device)
    return {
        "loss": float(loss.item()),
        "predictions": logits.detach().argmax(dim=-1).clone(),
    }


def short_stability_audit(
    batches: Sequence[sg25e.DeviceBatch],
    vocabulary: Any,
    *,
    device: torch.device,
    updates: int,
) -> Dict[str, Any]:
    serial_model = _build_mode_model("snn_serial", vocabulary, device=device)
    parallel_model = _build_mode_model("snn_parallel", vocabulary, device=device)
    serial_optimizer = _optimizer(serial_model)
    parallel_optimizer = _optimizer(parallel_model)
    loss_gaps = []
    disagreements = 0
    prediction_count = 0
    all_losses_finite = True
    serial_losses = []
    parallel_losses = []
    for batch in batches[:updates]:
        serial = _eager_update(
            "snn_serial",
            serial_model,
            serial_optimizer,
            batch,
            device=device,
        )
        parallel = _eager_update(
            "snn_parallel",
            parallel_model,
            parallel_optimizer,
            batch,
            device=device,
        )
        serial_losses.append(serial["loss"])
        parallel_losses.append(parallel["loss"])
        all_losses_finite = (
            all_losses_finite
            and math.isfinite(serial["loss"])
            and math.isfinite(parallel["loss"])
        )
        loss_gaps.append(abs(serial["loss"] - parallel["loss"]))
        mask = batch.token_mask.bool()
        disagreements += int(
            (
                serial["predictions"][mask]
                != parallel["predictions"][mask]
            )
            .sum()
            .item()
        )
        prediction_count += int(mask.sum().item())
    tail = loss_gaps[-min(20, len(loss_gaps)) :]
    parameter_max_abs = max(
        _max_abs(parallel, serial)
        for parallel, serial in zip(
            parallel_model.parameters(), serial_model.parameters()
        )
    )
    disagreement_rate = disagreements / prediction_count
    mean_gap = sum(loss_gaps) / len(loss_gaps)
    tail_gap = sum(tail) / len(tail)
    result = {
        "updates": len(loss_gaps),
        "all_losses_finite": all_losses_finite,
        "serial_loss_first": serial_losses[0],
        "serial_loss_last": serial_losses[-1],
        "parallel_loss_first": parallel_losses[0],
        "parallel_loss_last": parallel_losses[-1],
        "mean_loss_abs_gap": mean_gap,
        "last_20_mean_loss_abs_gap": tail_gap,
        "maximum_loss_abs_gap": max(loss_gaps),
        "prediction_disagreements": disagreements,
        "prediction_count": prediction_count,
        "prediction_disagreement_rate": disagreement_rate,
        "final_parameter_max_abs": parameter_max_abs,
        "passed": all_losses_finite
        and mean_gap <= 0.02
        and tail_gap <= 0.02
        and disagreement_rate <= 0.02,
    }
    del serial_model, parallel_model, serial_optimizer, parallel_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _profile_counts(events: Sequence[Any]) -> Dict[str, int]:
    names = [str(event.name) for event in events]
    return {
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


def graph_profile(trainer: GraphTrainer, batch: sg25e.DeviceBatch) -> Dict[str, Any]:
    trainer.reset()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profile:
        trainer.replay(batch, timed=False, inspect=False)
    result: Dict[str, Any] = _profile_counts(profile.events())
    result["real_example_count"] = batch.real_example_count
    result["host_api_per_real_example"] = (
        result["host_launch_and_copy_api_count"] / batch.real_example_count
    )
    trainer.reset()
    return result


def graph_equivalence(
    mode: str,
    trainer: GraphTrainer,
    batches: Sequence[sg25e.DeviceBatch],
    vocabulary: Any,
    *,
    device: torch.device,
    updates: int,
) -> Dict[str, Any]:
    trainer.reset()
    eager_model = _build_mode_model(mode, vocabulary, device=device)
    eager_optimizer = _optimizer(eager_model)
    loss_gaps = []
    disagreements = 0
    prediction_count = 0
    all_losses_finite = True
    for batch in batches[:updates]:
        eager = _eager_update(
            mode,
            eager_model,
            eager_optimizer,
            batch,
            device=device,
        )
        graphed = trainer.replay(batch, timed=False, inspect=True)
        all_losses_finite = (
            all_losses_finite
            and math.isfinite(eager["loss"])
            and math.isfinite(graphed["loss"])
        )
        loss_gaps.append(abs(eager["loss"] - graphed["loss"]))
        mask = batch.token_mask.bool()
        disagreements += int(
            (
                eager["predictions"][mask]
                != graphed["predictions"][mask]
            )
            .sum()
            .item()
        )
        prediction_count += int(mask.sum().item())
    mean_gap = sum(loss_gaps) / len(loss_gaps)
    disagreement_rate = disagreements / prediction_count
    result = {
        "updates": len(loss_gaps),
        "all_losses_finite": all_losses_finite,
        "mean_loss_abs_gap": mean_gap,
        "last_loss_abs_gap": loss_gaps[-1],
        "maximum_loss_abs_gap": max(loss_gaps),
        "prediction_disagreements": disagreements,
        "prediction_count": prediction_count,
        "prediction_disagreement_rate": disagreement_rate,
        "passed": all_losses_finite
        and mean_gap <= 0.01
        and loss_gaps[-1] <= 0.02
        and disagreement_rate <= 0.01,
    }
    del eager_model, eager_optimizer
    trainer.reset()
    return result


def speed_audit(
    mode: str,
    batches: Sequence[sg25e.DeviceBatch],
    vocabulary: Any,
    *,
    device: torch.device,
    equivalence_updates: int,
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    clean_allocated = torch.cuda.memory_allocated(device)
    clean_reserved = torch.cuda.memory_reserved(device)
    model = _build_mode_model(mode, vocabulary, device=device)
    trainer = GraphTrainer(mode, model, batches, device=device)
    trainer.capture_audit.update(
        {
            "total_allocated_delta_from_clean_bytes": max(
                0, torch.cuda.memory_allocated(device) - clean_allocated
            ),
            "total_reserved_delta_from_clean_bytes": max(
                0, torch.cuda.memory_reserved(device) - clean_reserved
            ),
        }
    )
    equivalence = graph_equivalence(
        mode,
        trainer,
        batches,
        vocabulary,
        device=device,
        updates=equivalence_updates,
    )
    benchmark = sg25e.graph_benchmark(trainer, batches)
    fullest = sg25e._fullest_batch(batches)
    profiler = graph_profile(trainer, fullest)
    result = {
        "benchmark": benchmark,
        "capture": dict(trainer.capture_audit),
        "equivalence": equivalence,
        "profiler": profiler,
    }
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def quality_audit(
    mode: str,
    train_examples: Sequence[Any],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    batches = sg25e.build_batch_schedule(
        train_examples,
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        seed=25_920_000,
        device=device,
    )
    model = _build_mode_model(mode, vocabulary, device=device)
    trainer = GraphTrainer(mode, model, batches, device=device)
    trainer.reset()
    pre = {
        split: sg0.evaluate_teacher(model, raw_examples[split], device=device)
        for split in ("valid", "test")
    }
    model.train(True)
    samples = []
    losses = []
    started = time.perf_counter_ns()
    for batch in batches:
        record = trainer.replay(batch, timed=True, inspect=True)
        samples.append(record["wall_ms"])
        losses.append(record["loss"])
    train_wall_seconds = (time.perf_counter_ns() - started) / 1e9
    post = {
        split: sg0.evaluate_teacher(model, raw_examples[split], device=device)
        for split in ("train", "valid", "test")
    }
    generation = sg0.generate_model(
        model,
        raw_examples["test"],
        vocabulary,
        max_tokens=sg0.MAX_GENERATION_TOKENS,
        device=device,
        include_records=True,
    )
    if mode.startswith("snn_"):
        basic_pass = (
            post["test"]["nll"] <= SG25C_SEED0_NLL + 0.10
            and generation["edit_similarity"] >= SG25C_SEED0_EDIT - 0.05
            and generation["paired_action_sensitivity"] >= 0.50
        )
    else:
        basic_pass = post["test"]["nll"] <= pre["test"]["nll"] - 0.10
    warmup = len(samples) // 5
    result = {
        "mode": mode,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "updates": len(batches),
        "train_wall_seconds": train_wall_seconds,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "timing": _sample_summary(samples[warmup:], 1),
        "pre_teacher": pre,
        "post_teacher": post,
        "generation": generation,
        "basic_passed": basic_pass,
    }
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _speed_decision(speed: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    eps = {
        mode: record["benchmark"]["effective_examples_per_second"]
        for mode, record in speed.items()
    }
    target_tps = {
        mode: record["benchmark"]["effective_target_tokens_per_second"]
        for mode, record in speed.items()
    }
    p50 = {
        mode: record["benchmark"]["per_real_example_timing"]["p50_ms"]
        for mode, record in speed.items()
    }
    parallel_speedup = eps["snn_parallel"] / eps["snn_serial"]
    ann_pass = (
        eps["snn_parallel"] >= eps["lstm"]
        and eps["snn_parallel"] >= eps["transformer"]
        and target_tps["snn_parallel"] >= target_tps["lstm"]
        and target_tps["snn_parallel"] >= target_tps["transformer"]
        and p50["snn_parallel"] <= p50["lstm"]
        and p50["snn_parallel"] <= p50["transformer"]
    )
    return {
        "effective_examples_per_second": eps,
        "effective_target_tokens_per_second": target_tps,
        "per_real_example_p50_ms": p50,
        "parallel_over_serial_speedup": parallel_speedup,
        "parallel_over_canonical_sg25e_lstm": eps["snn_parallel"]
        / SG25E_LSTM_EXAMPLES_PER_SECOND,
        "five_percent_serial_gate": parallel_speedup >= 1.05,
        "ann_gate": ann_pass,
        "passed": parallel_speedup >= 1.05 and ann_pass,
    }


def _decision(
    kernel: Mapping[str, Any],
    stability: Mapping[str, Any],
    speed_records: Mapping[str, Mapping[str, Any]],
    speed: Mapping[str, Any],
    canonical_serial_memory: Mapping[str, Any],
    quality: Mapping[str, Mapping[str, Any]] | None,
    *,
    quick: bool,
) -> Dict[str, Any]:
    capture_pass = all(
        record["capture"]["shape_count"] == 3
        for record in speed_records.values()
    )
    equivalence_pass = all(
        record["equivalence"]["passed"] for record in speed_records.values()
    )
    event_pass = (
        speed_records["snn_parallel"]["profiler"][
            "host_launch_and_copy_api_count"
        ]
        <= speed_records["snn_serial"]["profiler"][
            "host_launch_and_copy_api_count"
        ]
    )
    serial_capture = speed_records["snn_serial"]["capture"]
    parallel_capture = speed_records["snn_parallel"]["capture"]
    same_run_allocated_ratio = (
        parallel_capture["allocated_delta_bytes"]
        / serial_capture["allocated_delta_bytes"]
    )
    same_run_peak_ratio = (
        parallel_capture["peak_additional_allocated_bytes"]
        / serial_capture["peak_additional_allocated_bytes"]
    )
    allocated_ratio = (
        parallel_capture["allocated_delta_bytes"]
        / canonical_serial_memory["allocated_delta_bytes"]
    )
    peak_ratio = (
        parallel_capture["peak_additional_allocated_bytes"]
        / canonical_serial_memory["peak_additional_allocated_bytes"]
    )
    memory_pass = allocated_ratio <= 1.25 and peak_ratio <= 1.25
    if quick:
        gates = {
            "kernel_gate": bool(kernel["passed"]),
            "stability_gate": bool(stability["passed"]),
            "capture_gate": capture_pass,
            "equivalence_gate": equivalence_pass,
            "event_gate": event_pass,
            "memory_gate": memory_pass,
        }
        return {
            **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
            "speed_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "overall": "SMOKE" if all(gates.values()) else "FAIL",
            "speed": dict(speed),
            "parallel_to_serial_allocated_ratio": allocated_ratio,
            "parallel_to_serial_peak_ratio": peak_ratio,
            "same_run_order_sensitive_allocated_ratio": same_run_allocated_ratio,
            "same_run_order_sensitive_peak_ratio": same_run_peak_ratio,
        }
    basic_quality_pass = quality is not None and all(
        record["basic_passed"] for record in quality.values()
    )
    cross_quality_pass = False
    if quality is not None:
        parallel = quality["snn_parallel"]
        best_ann_nll = min(
            quality[name]["post_teacher"]["test"]["nll"]
            for name in ("lstm", "transformer")
        )
        best_ann_edit = max(
            quality[name]["generation"]["edit_similarity"]
            for name in ("lstm", "transformer")
        )
        cross_quality_pass = (
            parallel["post_teacher"]["test"]["nll"] <= best_ann_nll + 0.10
            and parallel["generation"]["edit_similarity"] >= best_ann_edit - 0.05
        )
    quality_pass = basic_quality_pass and cross_quality_pass
    gates = {
        "kernel_gate": bool(kernel["passed"]),
        "stability_gate": bool(stability["passed"]),
        "capture_gate": capture_pass,
        "equivalence_gate": equivalence_pass,
        "speed_gate": bool(speed["passed"]),
        "event_gate": event_pass,
        "memory_gate": memory_pass,
        "quality_gate": quality_pass,
    }
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "speed": dict(speed),
        "parallel_to_serial_allocated_ratio": allocated_ratio,
        "parallel_to_serial_peak_ratio": peak_ratio,
        "same_run_order_sensitive_allocated_ratio": same_run_allocated_ratio,
        "same_run_order_sensitive_peak_ratio": same_run_peak_ratio,
        "basic_quality_pass": basic_quality_pass,
        "cross_architecture_quality_pass": cross_quality_pass,
        "next_route": (
            "expanded_real_corpus" if all(gates.values()) else "projection_readout_fusion"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG25F requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG25F requires the frozen V100 backend")
    reference = ROOT / "results/e3_scan/e3_sg25e_bucketed_batch_graph.json"
    reference_sha = _sha256(reference)
    if reference_sha != EXPECTED_SG25E_SHA256:
        raise AssertionError("SG25F SG25E reference hash mismatch")
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    canonical_serial_memory = reference_payload["selected_batch_audits"][
        "snn_ra0"
    ]["capture"]
    _, serial_extension = load_serial_extension()
    _, parallel_extension = load_parallel_extension()
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = sg0.load_event_corpus(corpus_root)
    raw_examples, vocabulary = sg0.build_counterfactual_examples(
        corpus_root, corpus
    )
    data_audit = sg0.audit_examples(raw_examples, vocabulary)
    bucket_audit = sg25e._bucket_audit(raw_examples["train"])
    if not data_audit["passed"] or not bucket_audit["passed"]:
        raise AssertionError("SG25F data/bucket audit failed")

    kernel = kernel_equivalence_audit(device=device)
    stability_epochs = max(2, math.ceil(args.stability_updates / 3))
    stability_batches = sg25e.build_batch_schedule(
        raw_examples["train"],
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=stability_epochs,
        seed=25_930_000,
        device=device,
    )
    stability = short_stability_audit(
        stability_batches,
        vocabulary,
        device=device,
        updates=args.stability_updates,
    )
    del stability_batches
    gc.collect()
    torch.cuda.empty_cache()

    benchmark_batches = sg25e.build_batch_schedule(
        raw_examples["train"],
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=args.benchmark_epochs,
        seed=25_940_000,
        device=device,
    )
    speed_records = {
        mode: speed_audit(
            mode,
            benchmark_batches,
            vocabulary,
            device=device,
            equivalence_updates=args.equivalence_updates,
        )
        for mode in MODES
    }
    speed = _speed_decision(speed_records)
    preliminary = _decision(
        kernel,
        stability,
        speed_records,
        speed,
        canonical_serial_memory,
        quality=None,
        quick=True,
    )
    quality = None
    if (
        not args.quick
        and kernel["passed"]
        and stability["passed"]
        and speed["passed"]
        and all(
            record["equivalence"]["passed"]
            for record in speed_records.values()
        )
    ):
        quality = {
            mode: quality_audit(
                mode,
                raw_examples["train"],
                raw_examples,
                vocabulary,
                device=device,
                epochs=args.quality_epochs,
            )
            for mode in MODES
        }
    decision = _decision(
        kernel,
        stability,
        speed_records,
        speed,
        canonical_serial_memory,
        quality,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG25F block-parallel affine CUDA Graph training",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "modes": MODES,
            "batch_size": BATCH_SIZE,
            "bucket_capacities": sg25e.BUCKET_CAPACITIES,
            "stability_updates": args.stability_updates,
            "equivalence_updates": args.equivalence_updates,
            "benchmark_epochs": args.benchmark_epochs,
            "quality_epochs": args.quality_epochs,
            "time_scan": "128-thread block shared-memory affine inclusive scan",
            "optimizer": "AdamW(lr=1e-3,wd=.01,fused=True,capturable=True)",
            "copy_and_padding_included": True,
            "device": "cuda:0",
        },
        "provenance": {
            "sg25e_reference": str(reference),
            "sg25e_reference_sha256": reference_sha,
            "canonical_serial_memory": canonical_serial_memory,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "serial_extension": serial_extension,
            "parallel_extension": parallel_extension,
            "data_audit": data_audit,
            "bucket_audit": bucket_audit,
        },
        "kernel_equivalence": kernel,
        "short_stability": stability,
        "speed_records": speed_records,
        "preliminary": preliminary,
        "quality": quality,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg25f_parallel_affine_graph.json"),
    )
    parser.add_argument("--stability-updates", type=int, default=100)
    parser.add_argument("--equivalence-updates", type=int, default=20)
    parser.add_argument("--benchmark-epochs", type=int, default=10)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(
        args.stability_updates,
        args.equivalence_updates,
        args.benchmark_epochs,
        args.quality_epochs,
    ) <= 0:
        parser.error("all counts must be positive")
    if args.quick:
        args.stability_updates = min(args.stability_updates, 10)
        args.equivalence_updates = min(args.equivalence_updates, 5)
        args.benchmark_epochs = 1
        args.quality_epochs = min(args.quality_epochs, 2)
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
