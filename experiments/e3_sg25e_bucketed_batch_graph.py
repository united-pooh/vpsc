"""SG25E bucketed batched-query CUDA Graph training comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

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
from experiments.e3_sg25d_cuda_graph_training import (  # noqa: E402
    _build_model,
    _optimizer,
    _tensor_state_snapshot,
)
from vpsc.world_model.cores import E3GatedTraceScanCore  # noqa: E402
from vpsc.world_model.fused_batched_gated_trace_cuda import (  # noqa: E402
    fused_batched_gated_trace,
    load_extension,
)
from vpsc.world_model.fused_gated_trace_cuda import (  # noqa: E402
    fused_gated_trace,
)


ARCHITECTURES = ("snn_ra0", "lstm", "transformer")
BATCH_SIZES = (1, 2, 4, 8, 16)
BUCKET_CAPACITIES = ((64, 6), (96, 41), (128, 65))
EXPECTED_BUCKET_COUNTS = (14, 14, 12)
EXPECTED_SG25C_SHA256 = (
    "9657c17ca695fd3e4b310d2068d93eb231dbe1005b6360bfb84c99fd6a749f2b"
)
EXPECTED_SG25D_SHA256 = (
    "fce3b99df2996e2ef884108d0e93ee94bb416440a862da5f2c936c81cd1bc595"
)
SG25D_SNN_EXAMPLES_PER_SECOND = 2260.0809161232346
SG25C_SEED0_NLL = 2.6957537054423932
SG25C_SEED0_EDIT = 0.6513394872257832


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeviceBatch:
    input_ids: torch.Tensor
    query_indices: torch.Tensor
    gather_indices: torch.Tensor
    targets: torch.Tensor
    token_mask: torch.Tensor
    example_mask: torch.Tensor
    real_example_count: int
    input_token_count: int
    target_token_count: int
    example_indices: Tuple[int, ...]

    @property
    def key(self) -> Tuple[int, int, int]:
        return (
            int(self.input_ids.shape[0]),
            int(self.input_ids.shape[1]),
            int(self.query_indices.shape[1]),
        )

    @property
    def padded_input_slots(self) -> int:
        return int(self.input_ids.numel())

    @property
    def padded_target_slots(self) -> int:
        return int(self.targets.numel())


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


def _bucket_index(example: Any) -> int:
    input_length = len(example.prompt_ids) + len(example.target_ids) - 1
    target_length = len(example.target_ids)
    for index, (time_capacity, query_capacity) in enumerate(BUCKET_CAPACITIES):
        if input_length <= time_capacity and target_length <= query_capacity:
            return index
    raise AssertionError(
        f"example ({input_length}, {target_length}) exceeds frozen buckets"
    )


def _bucket_audit(examples: Sequence[Any]) -> Dict[str, Any]:
    counts = [0 for _ in BUCKET_CAPACITIES]
    real_input_tokens = [0 for _ in BUCKET_CAPACITIES]
    real_target_tokens = [0 for _ in BUCKET_CAPACITIES]
    for example in examples:
        index = _bucket_index(example)
        counts[index] += 1
        real_input_tokens[index] += (
            len(example.prompt_ids) + len(example.target_ids) - 1
        )
        real_target_tokens[index] += len(example.target_ids)
    records = []
    for index, (time_capacity, query_capacity) in enumerate(BUCKET_CAPACITIES):
        records.append(
            {
                "time_capacity": time_capacity,
                "query_capacity": query_capacity,
                "example_count": counts[index],
                "real_input_tokens": real_input_tokens[index],
                "real_target_tokens": real_target_tokens[index],
                "input_utilization_if_unbatched": real_input_tokens[index]
                / (counts[index] * time_capacity),
                "target_utilization_if_unbatched": real_target_tokens[index]
                / (counts[index] * query_capacity),
            }
        )
    return {
        "records": records,
        "counts": tuple(counts),
        "passed": tuple(counts) == EXPECTED_BUCKET_COUNTS,
    }


def _make_device_batch(
    examples: Sequence[Any],
    indices: Sequence[int],
    *,
    bucket_index: int,
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> DeviceBatch:
    time_capacity, query_capacity = BUCKET_CAPACITIES[bucket_index]
    input_ids = torch.full(
        (batch_size, time_capacity), pad_id, dtype=torch.long, device=device
    )
    query_indices = torch.full(
        (batch_size, query_capacity), -1, dtype=torch.long, device=device
    )
    gather_indices = torch.zeros(
        (batch_size, query_capacity), dtype=torch.long, device=device
    )
    targets = torch.full(
        (batch_size, query_capacity), pad_id, dtype=torch.long, device=device
    )
    token_mask = torch.zeros(
        (batch_size, query_capacity), dtype=torch.float32, device=device
    )
    example_mask = torch.zeros(batch_size, dtype=torch.float32, device=device)
    input_token_count = 0
    target_token_count = 0
    for row, example_index in enumerate(indices):
        example = examples[example_index]
        sequence = example.prompt_ids + example.target_ids
        shifted_input = sequence[:-1]
        query_count = len(example.target_ids)
        first_query = len(example.prompt_ids) - 1
        query_values = torch.arange(
            first_query,
            first_query + query_count,
            dtype=torch.long,
            device=device,
        )
        input_ids[row, : len(shifted_input)] = torch.tensor(
            shifted_input, dtype=torch.long, device=device
        )
        query_indices[row, :query_count] = query_values
        gather_indices[row, :query_count] = query_values
        targets[row, :query_count] = torch.tensor(
            example.target_ids, dtype=torch.long, device=device
        )
        token_mask[row, :query_count] = 1.0
        example_mask[row] = 1.0
        input_token_count += len(shifted_input)
        target_token_count += query_count
    return DeviceBatch(
        input_ids=input_ids,
        query_indices=query_indices,
        gather_indices=gather_indices,
        targets=targets,
        token_mask=token_mask,
        example_mask=example_mask,
        real_example_count=len(indices),
        input_token_count=input_token_count,
        target_token_count=target_token_count,
        example_indices=tuple(indices),
    )


def build_epoch_batches(
    examples: Sequence[Any],
    vocabulary: Any,
    *,
    batch_size: int,
    epoch: int,
    seed: int,
    device: torch.device,
) -> Tuple[DeviceBatch, ...]:
    grouped = [[] for _ in BUCKET_CAPACITIES]
    for example_index, example in enumerate(examples):
        grouped[_bucket_index(example)].append(example_index)
    generator = random.Random(seed + epoch)
    batches = []
    for bucket_index, indices in enumerate(grouped):
        generator.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append(
                _make_device_batch(
                    examples,
                    indices[start : start + batch_size],
                    bucket_index=bucket_index,
                    batch_size=batch_size,
                    pad_id=vocabulary.pad_id,
                    device=device,
                )
            )
    generator.shuffle(batches)
    return tuple(batches)


def build_batch_schedule(
    examples: Sequence[Any],
    vocabulary: Any,
    *,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> Tuple[DeviceBatch, ...]:
    schedule = []
    for epoch in range(epochs):
        schedule.extend(
            build_epoch_batches(
                examples,
                vocabulary,
                batch_size=batch_size,
                epoch=epoch,
                seed=seed,
                device=device,
            )
        )
    return tuple(schedule)


def _snn_batched_logits(
    model: Any,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("SG25E SNN path requires E3GatedTraceScanCore")
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
    raw, _, _ = fused_batched_gated_trace(
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


def _batched_logits(name: str, model: Any, batch: DeviceBatch) -> torch.Tensor:
    if name == "snn_ra0":
        return _snn_batched_logits(
            model, batch.input_ids, batch.query_indices
        )
    output = model(batch.input_ids, None, detach_state=True)
    gather = batch.gather_indices.unsqueeze(-1).expand(
        -1, -1, model.vocab_size
    )
    return output.logits.gather(1, gather)


def _masked_example_mean_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_mask: torch.Tensor,
    example_mask: torch.Tensor,
) -> torch.Tensor:
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    per_example = (token_losses * token_mask).sum(dim=1) / token_mask.sum(
        dim=1
    ).clamp_min(1.0)
    return (per_example * example_mask).sum() / example_mask.sum().clamp_min(1.0)


def _step_body(
    name: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: DeviceBatch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=False)
    logits = _batched_logits(name, model, batch)
    loss = _masked_example_mean_loss(
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


def _entry_batch(entry: GraphEntry, source: DeviceBatch) -> DeviceBatch:
    return DeviceBatch(
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


class BucketGraphTrainer:
    def __init__(
        self,
        name: str,
        model: Any,
        batches: Sequence[DeviceBatch],
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
            left is not right
            for left, right in zip(optimizer_parameters, self.parameters)
        ):
            raise AssertionError("optimizer/model parameter ordering mismatch")
        self._parameter_baseline = tuple(
            parameter.detach().clone() for parameter in self.parameters
        )
        representatives: Dict[Tuple[int, int, int], DeviceBatch] = {}
        for batch in batches:
            representatives.setdefault(batch.key, batch)
        if len(representatives) != len(BUCKET_CAPACITIES):
            raise AssertionError(
                f"expected {len(BUCKET_CAPACITIES)} bucket graphs, "
                f"got {len(representatives)}"
            )
        self.entries: Dict[Tuple[int, int, int], GraphEntry] = {}
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter_ns()

        first = next(iter(representatives.values()))
        _step_body(name, model, self.optimizer, first)
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
            static_batch = _entry_batch(entry, source)
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream):
                _step_body(name, model, self.optimizer, static_batch)
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            _sync(device)
            self.reset()
            with torch.cuda.graph(entry.graph):
                entry.loss, entry.logits = _step_body(
                    name, model, self.optimizer, static_batch
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
            "keys": tuple(tuple(value) for value in sorted(self.entries)),
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
        self, batch: DeviceBatch, *, timed: bool, inspect: bool
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


def _eager_update(
    name: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: DeviceBatch,
    *,
    device: torch.device,
    timed: bool,
    inspect: bool,
) -> Dict[str, Any]:
    if timed:
        _sync(device)
        started = time.perf_counter_ns()
    loss, logits = _step_body(name, model, optimizer, batch)
    _sync(device)
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


def _maximum_absolute(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _normalized_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(right.detach().abs().max().item()) + 1e-12
    return _maximum_absolute(left, right) / denominator


def _kernel_query_matrix(
    batch_size: int,
    time_steps: int,
    query_capacity: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    matrix = torch.full(
        (batch_size, query_capacity), -1, dtype=torch.long, device=device
    )
    counts = []
    for row in range(batch_size):
        fraction = (row + 1) / batch_size
        count = max(1, min(query_capacity, round(query_capacity * fraction)))
        positions = torch.linspace(
            0, time_steps - 1, count, device=device
        ).round().to(dtype=torch.long)
        if positions.numel() > 1 and not bool(
            torch.all(positions[1:] > positions[:-1])
        ):
            raise AssertionError("kernel audit produced duplicate queries")
        matrix[row, :count] = positions
        counts.append(count)
    return matrix, tuple(counts)


def kernel_equivalence_audit(*, device: torch.device) -> Dict[str, Any]:
    cases = []
    for batch_size in (1, 2, 4, 8):
        for case_index, (time_steps, query_capacity) in enumerate(
            BUCKET_CAPACITIES
        ):
            torch.manual_seed(25_800_100 + batch_size * 10 + case_index)
            state_dim = 9
            queries, valid_counts = _kernel_query_matrix(
                batch_size,
                time_steps,
                query_capacity,
                device=device,
            )
            batched_drives = torch.randn(
                batch_size,
                time_steps,
                4 * state_dim,
                device=device,
                requires_grad=True,
            )
            reference_drives = (
                batched_drives.detach().clone().requires_grad_(True)
            )
            batched_decays = (
                0.5 + 0.45 * torch.rand(2, state_dim, device=device)
            ).requires_grad_(True)
            reference_decays = (
                batched_decays.detach().clone().requires_grad_(True)
            )
            batched_initial_e = torch.rand(
                batch_size, state_dim, device=device, requires_grad=True
            )
            batched_initial_i = torch.rand(
                batch_size, state_dim, device=device, requires_grad=True
            )
            reference_initial_e = (
                batched_initial_e.detach().clone().requires_grad_(True)
            )
            reference_initial_i = (
                batched_initial_i.detach().clone().requires_grad_(True)
            )
            batched_raw, batched_final_e, batched_final_i = (
                fused_batched_gated_trace(
                    batched_drives,
                    queries,
                    batched_decays,
                    batched_initial_e,
                    batched_initial_i,
                    spike_threshold=0.5,
                    surrogate_scale=4.0,
                )
            )
            reference_raw_rows = []
            reference_final_e_rows = []
            reference_final_i_rows = []
            for row, valid_count in enumerate(valid_counts):
                raw, final_e, final_i = fused_gated_trace(
                    reference_drives[row : row + 1],
                    queries[row, :valid_count],
                    reference_decays,
                    reference_initial_e[row : row + 1],
                    reference_initial_i[row : row + 1],
                    spike_threshold=0.5,
                    surrogate_scale=4.0,
                )
                padding = torch.zeros(
                    1,
                    query_capacity - valid_count,
                    4 * state_dim,
                    device=device,
                )
                reference_raw_rows.append(torch.cat((raw, padding), dim=1))
                reference_final_e_rows.append(final_e)
                reference_final_i_rows.append(final_i)
            reference_raw = torch.cat(reference_raw_rows, dim=0)
            reference_final_e = torch.cat(reference_final_e_rows, dim=0)
            reference_final_i = torch.cat(reference_final_i_rows, dim=0)
            raw_probe = torch.randn_like(batched_raw)
            final_e_probe = torch.randn_like(batched_final_e)
            final_i_probe = torch.randn_like(batched_final_i)
            (
                (batched_raw * raw_probe).sum()
                + (batched_final_e * final_e_probe).sum()
                + (batched_final_i * final_i_probe).sum()
            ).backward()
            (
                (reference_raw * raw_probe).sum()
                + (reference_final_e * final_e_probe).sum()
                + (reference_final_i * final_i_probe).sum()
            ).backward()
            valid_mask = queries.ge(0).unsqueeze(-1).expand_as(batched_raw)
            spike_width = 2 * state_dim
            spike_mask = queries.ge(0).unsqueeze(-1).expand(
                -1, -1, spike_width
            )
            spike_disagreements = int(
                (
                    batched_raw[:, :, :spike_width][spike_mask]
                    != reference_raw[:, :, :spike_width][spike_mask]
                )
                .sum()
                .item()
            )
            padding_nonzero = int(
                torch.count_nonzero(batched_raw[~valid_mask]).item()
            )
            errors = {
                "raw_max_abs": _maximum_absolute(
                    batched_raw, reference_raw
                ),
                "final_e_max_abs": _maximum_absolute(
                    batched_final_e, reference_final_e
                ),
                "final_i_max_abs": _maximum_absolute(
                    batched_final_i, reference_final_i
                ),
                "drive_grad_max_abs": _maximum_absolute(
                    batched_drives.grad, reference_drives.grad
                ),
                "decay_grad_max_abs": _maximum_absolute(
                    batched_decays.grad, reference_decays.grad
                ),
                "initial_e_grad_max_abs": _maximum_absolute(
                    batched_initial_e.grad, reference_initial_e.grad
                ),
                "initial_i_grad_max_abs": _maximum_absolute(
                    batched_initial_i.grad, reference_initial_i.grad
                ),
                "drive_grad_normalized_error": _normalized_error(
                    batched_drives.grad, reference_drives.grad
                ),
                "decay_grad_normalized_error": _normalized_error(
                    batched_decays.grad, reference_decays.grad
                ),
            }
            forward_pass = (
                torch.allclose(
                    batched_raw, reference_raw, atol=2e-6, rtol=2e-5
                )
                and torch.allclose(
                    batched_final_e,
                    reference_final_e,
                    atol=2e-6,
                    rtol=2e-5,
                )
                and torch.allclose(
                    batched_final_i,
                    reference_final_i,
                    atol=2e-6,
                    rtol=2e-5,
                )
            )
            gradient_pass = all(
                torch.allclose(left, right, atol=3e-5, rtol=3e-4)
                for left, right in (
                    (batched_drives.grad, reference_drives.grad),
                    (batched_decays.grad, reference_decays.grad),
                    (batched_initial_e.grad, reference_initial_e.grad),
                    (batched_initial_i.grad, reference_initial_i.grad),
                )
            )
            cases.append(
                {
                    "batch_size": batch_size,
                    "time_steps": time_steps,
                    "query_capacity": query_capacity,
                    "valid_query_counts": valid_counts,
                    "padding_nonzero": padding_nonzero,
                    "spike_disagreements": spike_disagreements,
                    "errors": errors,
                    "forward_passed": forward_pass,
                    "gradient_passed": gradient_pass,
                    "passed": forward_pass
                    and gradient_pass
                    and padding_nonzero == 0
                    and spike_disagreements == 0,
                }
            )
            del (
                batched_drives,
                reference_drives,
                batched_decays,
                reference_decays,
                batched_initial_e,
                batched_initial_i,
                reference_initial_e,
                reference_initial_i,
                batched_raw,
                reference_raw,
            )
    return {"cases": cases, "passed": all(case["passed"] for case in cases)}


def graph_benchmark(
    trainer: BucketGraphTrainer,
    batches: Sequence[DeviceBatch],
) -> Dict[str, Any]:
    trainer.reset()
    samples = []
    for batch in batches:
        samples.append(trainer.replay(batch, timed=True, inspect=False)["wall_ms"])
    warmup = len(samples) // 5
    steady_samples = samples[warmup:]
    steady_batches = batches[warmup:]
    real_examples = sum(batch.real_example_count for batch in steady_batches)
    input_tokens = sum(batch.input_token_count for batch in steady_batches)
    target_tokens = sum(batch.target_token_count for batch in steady_batches)
    padded_input_slots = sum(batch.padded_input_slots for batch in steady_batches)
    padded_target_slots = sum(
        batch.padded_target_slots for batch in steady_batches
    )
    total_ms = sum(steady_samples)
    per_example = [
        wall_ms / batch.real_example_count
        for wall_ms, batch in zip(steady_samples, steady_batches)
    ]
    result = {
        "updates": len(samples),
        "warmup_updates_excluded": warmup,
        "steady_real_examples": real_examples,
        "steady_input_tokens": input_tokens,
        "steady_target_tokens": target_tokens,
        "step_timing": _sample_summary(steady_samples, 1),
        "per_real_example_timing": _sample_summary(per_example, 1),
        "effective_examples_per_second": real_examples * 1000.0 / total_ms,
        "effective_input_tokens_per_second": input_tokens * 1000.0 / total_ms,
        "effective_target_tokens_per_second": target_tokens * 1000.0 / total_ms,
        "input_padding_utilization": input_tokens / padded_input_slots,
        "target_padding_utilization": target_tokens / padded_target_slots,
        "mean_real_examples_per_update": real_examples / len(steady_batches),
    }
    trainer.reset()
    return result


def graph_equivalence(
    name: str,
    trainer: BucketGraphTrainer,
    batches: Sequence[DeviceBatch],
    vocabulary: Any,
    *,
    device: torch.device,
    updates: int,
) -> Dict[str, Any]:
    trainer.reset()
    eager_model = _build_model(name, vocabulary, device=device)
    eager_optimizer = _optimizer(eager_model)
    loss_gaps = []
    disagreements = 0
    prediction_count = 0
    all_losses_finite = True
    for batch in batches[:updates]:
        eager = _eager_update(
            name,
            eager_model,
            eager_optimizer,
            batch,
            device=device,
            timed=False,
            inspect=True,
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
    disagreement_rate = disagreements / prediction_count
    mean_gap = sum(loss_gaps) / len(loss_gaps)
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


def _profile_event_counts(events: Sequence[Any]) -> Dict[str, int]:
    names = [str(event.name) for event in events]
    return {
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


def graph_profiler_audit(
    trainer: BucketGraphTrainer, batch: DeviceBatch
) -> Dict[str, Any]:
    trainer.reset()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profile:
        trainer.replay(batch, timed=False, inspect=False)
    result: Dict[str, Any] = _profile_event_counts(profile.events())
    result["real_example_count"] = batch.real_example_count
    result["host_api_per_real_example"] = (
        result["host_launch_and_copy_api_count"] / batch.real_example_count
    )
    trainer.reset()
    return result


def eager_runtime_audit(
    name: str,
    batch: DeviceBatch,
    vocabulary: Any,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = _build_model(name, vocabulary, device=device)
    optimizer = _optimizer(model)
    _eager_update(
        name,
        model,
        optimizer,
        batch,
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
            batch,
            device=device,
            timed=False,
            inspect=False,
        )
    result: Dict[str, Any] = _profile_event_counts(profile.events())
    result.update(
        {
            "real_example_count": batch.real_example_count,
            "host_api_per_real_example": result[
                "host_launch_and_copy_api_count"
            ]
            / batch.real_example_count,
            "allocated_total_delta_bytes": max(
                0, torch.cuda.max_memory_allocated(device) - allocated_before
            ),
            "reserved_total_delta_bytes": max(
                0, torch.cuda.max_memory_reserved(device) - reserved_before
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    )
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _fullest_batch(batches: Sequence[DeviceBatch]) -> DeviceBatch:
    return max(
        batches,
        key=lambda batch: (
            batch.real_example_count,
            batch.input_token_count,
            batch.target_token_count,
        ),
    )


def selected_batch_audit(
    name: str,
    batch_size: int,
    train_examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
    equivalence_updates: int,
) -> Dict[str, Any]:
    epochs = max(2, math.ceil(equivalence_updates / 3))
    batches = build_batch_schedule(
        train_examples,
        vocabulary,
        batch_size=batch_size,
        epochs=epochs,
        seed=25_810_000,
        device=device,
    )
    gc.collect()
    torch.cuda.empty_cache()
    clean_allocated = torch.cuda.memory_allocated(device)
    clean_reserved = torch.cuda.memory_reserved(device)
    model = _build_model(name, vocabulary, device=device)
    trainer = BucketGraphTrainer(name, model, batches, device=device)
    trainer.capture_audit.update(
        {
            "total_allocated_delta_from_clean_bytes": max(
                0, torch.cuda.memory_allocated(device) - clean_allocated
            ),
            "total_reserved_delta_from_clean_bytes": max(
                0, torch.cuda.memory_reserved(device) - clean_reserved
            ),
            "peak_total_allocated_delta_from_clean_bytes": max(
                0,
                trainer.capture_audit["peak_allocated_bytes"]
                - clean_allocated,
            ),
            "peak_total_reserved_delta_from_clean_bytes": max(
                0,
                trainer.capture_audit["peak_reserved_bytes"]
                - clean_reserved,
            ),
        }
    )
    equivalence = graph_equivalence(
        name,
        trainer,
        batches,
        vocabulary,
        device=device,
        updates=equivalence_updates,
    )
    fullest = _fullest_batch(batches)
    graph_profile = graph_profiler_audit(trainer, fullest)
    capture = dict(trainer.capture_audit)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    eager = eager_runtime_audit(
        name, fullest, vocabulary, device=device
    )
    return {
        "batch_size": batch_size,
        "equivalence": equivalence,
        "capture": capture,
        "graph_profiler": graph_profile,
        "eager_runtime": eager,
    }


def graph_quality(
    name: str,
    batch_size: int,
    train_examples: Sequence[Any],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    batches = build_batch_schedule(
        train_examples,
        vocabulary,
        batch_size=batch_size,
        epochs=epochs,
        seed=25_820_000,
        device=device,
    )
    model = _build_model(name, vocabulary, device=device)
    trainer = BucketGraphTrainer(name, model, batches, device=device)
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
        result = trainer.replay(batch, timed=True, inspect=True)
        samples.append(result["wall_ms"])
        losses.append(result["loss"])
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
    if name == "snn_ra0":
        basic_pass = (
            post["test"]["nll"] <= SG25C_SEED0_NLL + 0.10
            and generation["edit_similarity"] >= SG25C_SEED0_EDIT - 0.05
            and generation["paired_action_sensitivity"] >= 0.50
        )
    else:
        basic_pass = post["test"]["nll"] <= pre["test"]["nll"] - 0.10
    warmup = len(samples) // 5
    result = {
        "batch_size": batch_size,
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


def _select_best_batches(
    sweep: Mapping[str, Mapping[str, Any]]
) -> Dict[str, int]:
    selected = {}
    for name in ARCHITECTURES:
        best_batch = max(
            BATCH_SIZES,
            key=lambda batch_size: (
                sweep[name][str(batch_size)]["benchmark"][
                    "effective_examples_per_second"
                ],
                -batch_size,
            ),
        )
        selected[name] = best_batch
    return selected


def _speed_decision(
    sweep: Mapping[str, Mapping[str, Any]], selected: Mapping[str, int]
) -> Dict[str, Any]:
    best_eps = {
        name: sweep[name][str(selected[name])]["benchmark"][
            "effective_examples_per_second"
        ]
        for name in ARCHITECTURES
    }
    best_target_tps = {
        name: sweep[name][str(selected[name])]["benchmark"][
            "effective_target_tokens_per_second"
        ]
        for name in ARCHITECTURES
    }
    snn_batch = selected["snn_ra0"]
    same_batch_per_example_p50 = {
        name: sweep[name][str(snn_batch)]["benchmark"][
            "per_real_example_timing"
        ]["p50_ms"]
        for name in ARCHITECTURES
    }
    relative_to_sg25d = best_eps["snn_ra0"] / SG25D_SNN_EXAMPLES_PER_SECOND
    optimized_ann_pass = (
        best_eps["snn_ra0"] >= best_eps["lstm"]
        and best_eps["snn_ra0"] >= best_eps["transformer"]
        and best_target_tps["snn_ra0"] >= best_target_tps["lstm"]
        and best_target_tps["snn_ra0"] >= best_target_tps["transformer"]
    )
    same_batch_p50_pass = (
        same_batch_per_example_p50["snn_ra0"]
        <= same_batch_per_example_p50["lstm"]
        and same_batch_per_example_p50["snn_ra0"]
        <= same_batch_per_example_p50["transformer"]
    )
    return {
        "selected_batch_sizes": dict(selected),
        "best_effective_examples_per_second": best_eps,
        "best_effective_target_tokens_per_second": best_target_tps,
        "snn_selected_same_batch_per_example_p50_ms": same_batch_per_example_p50,
        "snn_speedup_over_sg25d_exact_graph": relative_to_sg25d,
        "snn_1p5x_gate": relative_to_sg25d >= 1.5,
        "optimized_ann_gate": optimized_ann_pass,
        "same_batch_p50_gate": same_batch_p50_pass,
        "passed": relative_to_sg25d >= 1.5
        and optimized_ann_pass
        and same_batch_p50_pass,
    }


def _final_decision(
    kernel: Mapping[str, Any],
    sweep: Mapping[str, Mapping[str, Any]],
    speed: Mapping[str, Any],
    selected_audits: Mapping[str, Mapping[str, Any]],
    quality: Mapping[str, Mapping[str, Any]] | None,
    *,
    quick: bool,
) -> Dict[str, Any]:
    capture_pass = all(
        record["capture"]["shape_count"] == 3
        for architecture in sweep.values()
        for record in architecture.values()
    ) and all(
        record["capture"]["shape_count"] == 3
        for record in selected_audits.values()
    )
    equivalence_pass = all(
        record["equivalence"]["passed"]
        for record in selected_audits.values()
    )
    launch_reduction = {}
    for name, record in selected_audits.items():
        eager_count = record["eager_runtime"][
            "host_launch_and_copy_api_count"
        ]
        graph_count = record["graph_profiler"][
            "host_launch_and_copy_api_count"
        ]
        launch_reduction[name] = 1.0 - graph_count / eager_count
    launch_pass = all(value >= 0.50 for value in launch_reduction.values()) and all(
        record["graph_profiler"]["host_api_per_real_example"] <= 9.0
        for record in selected_audits.values()
    )
    snn_audit = selected_audits["snn_ra0"]
    memory_ratio = (
        snn_audit["capture"]["allocated_delta_bytes"]
        / snn_audit["eager_runtime"]["allocated_total_delta_bytes"]
    )
    memory_pass = memory_ratio <= 4.0
    if quick:
        gates = {
            "kernel_gate": bool(kernel["passed"]),
            "capture_gate": capture_pass,
            "equivalence_gate": equivalence_pass,
            "launch_gate": launch_pass,
            "memory_gate": memory_pass,
        }
        return {
            **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
            "speed_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "overall": "SMOKE" if all(gates.values()) else "FAIL",
            "speed": dict(speed),
            "launch_reduction": launch_reduction,
            "snn_graph_memory_to_eager_ratio": memory_ratio,
        }
    basic_quality_pass = quality is not None and all(
        record["basic_passed"] for record in quality.values()
    )
    cross_architecture_quality_pass = False
    if quality is not None:
        snn_quality = quality["snn_ra0"]
        best_ann_nll = min(
            quality[name]["post_teacher"]["test"]["nll"]
            for name in ("lstm", "transformer")
        )
        best_ann_edit = max(
            quality[name]["generation"]["edit_similarity"]
            for name in ("lstm", "transformer")
        )
        cross_architecture_quality_pass = (
            snn_quality["post_teacher"]["test"]["nll"]
            <= best_ann_nll + 0.10
            and snn_quality["generation"]["edit_similarity"]
            >= best_ann_edit - 0.05
        )
    quality_pass = basic_quality_pass and cross_architecture_quality_pass
    gates = {
        "kernel_gate": bool(kernel["passed"]),
        "capture_gate": capture_pass,
        "equivalence_gate": equivalence_pass,
        "speed_gate": bool(speed["passed"]),
        "launch_gate": launch_pass,
        "memory_gate": memory_pass,
        "quality_gate": quality_pass,
    }
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "speed": dict(speed),
        "launch_reduction": launch_reduction,
        "snn_graph_memory_to_eager_ratio": memory_ratio,
        "basic_quality_pass": basic_quality_pass,
        "cross_architecture_quality_pass": cross_architecture_quality_pass,
        "next_route": (
            "expanded_real_corpus" if all(gates.values()) else "segmented_persistent_kernel"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG25E requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG25E requires the frozen V100 backend")
    sg25c_reference = ROOT / "results/e3_scan/e3_sg25c_native_fused_scan_cuda.json"
    sg25d_reference = ROOT / "results/e3_scan/e3_sg25d_cuda_graph_training.json"
    sg25c_sha = _sha256(sg25c_reference)
    sg25d_sha = _sha256(sg25d_reference)
    if sg25c_sha != EXPECTED_SG25C_SHA256:
        raise AssertionError("SG25E SG25C reference hash mismatch")
    if sg25d_sha != EXPECTED_SG25D_SHA256:
        raise AssertionError("SG25E SG25D reference hash mismatch")
    _, extension = load_extension()
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = sg0.load_event_corpus(corpus_root)
    raw_examples, vocabulary = sg0.build_counterfactual_examples(
        corpus_root, corpus
    )
    data_audit = sg0.audit_examples(raw_examples, vocabulary)
    bucket_audit = _bucket_audit(raw_examples["train"])
    if not data_audit["passed"] or not bucket_audit["passed"]:
        raise AssertionError("SG25E data/bucket audit failed")

    kernel = kernel_equivalence_audit(device=device)
    sweep: Dict[str, Dict[str, Any]] = {name: {} for name in ARCHITECTURES}
    for name in ARCHITECTURES:
        for batch_size in BATCH_SIZES:
            batches = build_batch_schedule(
                raw_examples["train"],
                vocabulary,
                batch_size=batch_size,
                epochs=args.benchmark_epochs,
                seed=25_830_000,
                device=device,
            )
            gc.collect()
            torch.cuda.empty_cache()
            clean_allocated = torch.cuda.memory_allocated(device)
            clean_reserved = torch.cuda.memory_reserved(device)
            model = _build_model(name, vocabulary, device=device)
            trainer = BucketGraphTrainer(name, model, batches, device=device)
            trainer.capture_audit.update(
                {
                    "total_allocated_delta_from_clean_bytes": max(
                        0,
                        torch.cuda.memory_allocated(device) - clean_allocated,
                    ),
                    "total_reserved_delta_from_clean_bytes": max(
                        0, torch.cuda.memory_reserved(device) - clean_reserved
                    ),
                }
            )
            benchmark = graph_benchmark(trainer, batches)
            sweep[name][str(batch_size)] = {
                "batch_size": batch_size,
                "benchmark": benchmark,
                "capture": dict(trainer.capture_audit),
            }
            del trainer, model, batches
            gc.collect()
            torch.cuda.empty_cache()

    selected = _select_best_batches(sweep)
    speed = _speed_decision(sweep, selected)
    selected_audits = {
        name: selected_batch_audit(
            name,
            selected[name],
            raw_examples["train"],
            vocabulary,
            device=device,
            equivalence_updates=args.equivalence_updates,
        )
        for name in ARCHITECTURES
    }
    preliminary = _final_decision(
        kernel,
        sweep,
        speed,
        selected_audits,
        quality=None,
        quick=True,
    )
    quality = None
    if (
        not args.quick
        and kernel["passed"]
        and speed["passed"]
        and all(
            record["equivalence"]["passed"]
            for record in selected_audits.values()
        )
    ):
        quality = {
            name: graph_quality(
                name,
                selected[name],
                raw_examples["train"],
                raw_examples,
                vocabulary,
                device=device,
                epochs=args.quality_epochs,
            )
            for name in ARCHITECTURES
        }
    decision = _final_decision(
        kernel,
        sweep,
        speed,
        selected_audits,
        quality,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG25E bucketed batched-query CUDA Graph training",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "architectures": ARCHITECTURES,
            "batch_sizes": BATCH_SIZES,
            "bucket_capacities": BUCKET_CAPACITIES,
            "benchmark_epochs": args.benchmark_epochs,
            "equivalence_updates": args.equivalence_updates,
            "quality_epochs": args.quality_epochs,
            "optimizer": "AdamW(lr=1e-3,wd=.01,fused=True,capturable=True)",
            "loss": "mean(valid-token CE per real example), then mean(real examples)",
            "copy_included_in_graph_wall": True,
            "padding_compute_included": True,
            "device": "cuda:0",
        },
        "provenance": {
            "sg25c_reference": str(sg25c_reference),
            "sg25c_reference_sha256": sg25c_sha,
            "sg25d_reference": str(sg25d_reference),
            "sg25d_reference_sha256": sg25d_sha,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "extension": extension,
            "data_audit": data_audit,
            "bucket_audit": bucket_audit,
        },
        "kernel_equivalence": kernel,
        "sweep": sweep,
        "selected_batch_audits": selected_audits,
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
        default=Path("results/e3_scan/e3_sg25e_bucketed_batch_graph.json"),
    )
    parser.add_argument("--benchmark-epochs", type=int, default=10)
    parser.add_argument("--equivalence-updates", type=int, default=20)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(
        args.benchmark_epochs,
        args.equivalence_updates,
        args.quality_epochs,
    ) <= 0:
        parser.error("all counts must be positive")
    if args.quick:
        args.benchmark_epochs = 1
        args.equivalence_updates = min(args.equivalence_updates, 5)
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
