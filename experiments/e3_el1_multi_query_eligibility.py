"""Exact multi-query IC0 eligibility: equivalence, scaling, speed, and quality."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _autograd_node_count,
    _environment,
    _interleaved_samples,
    _sample_summary,
    _sync,
)
from experiments.e3_el0_terminal_eligibility import (  # noqa: E402
    _SavedTensorCounter,
    _dataset_hash,
    _gradient_record,
    _max_abs,
    _parameter_max_abs,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3InputCodedScanCore,
    E3LayerState,
    E3ScanState,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
)


D_MODEL = 32
STATE_DIM = 42
PAYLOAD_VOCAB = 16
DISTRACTOR_TOKEN = 0
WRITE_BASE = 1
QUERY_TOKEN = WRITE_BASE + PAYLOAD_VOCAB
INPUT_VOCAB = QUERY_TOKEN + 1
SEQUENCE_LENGTH = 32
WRITE_INDICES = (0, 8, 16, 24)
QUERY_INDICES = (4, 12, 20, 28)
EL1_ATOL = 2e-6
EL1_RTOL = 1e-5
Runner = Callable[[], None]


def _query_tensor(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(values), dtype=torch.long, device=device)


def _even_queries(time_steps: int, count: int, device: torch.device) -> torch.Tensor:
    if count <= 0 or count > time_steps:
        raise ValueError("query count must lie in [1, time_steps]")
    if count == 1:
        values = (time_steps - 1,)
    else:
        values = tuple(
            round(index * (time_steps - 1) / (count - 1))
            for index in range(count)
        )
    if len(set(values)) != count:  # pragma: no cover - protected by count <= T
        raise AssertionError(f"query construction produced duplicates: {values}")
    return _query_tensor(values, device)


def _raw_from_trace(
    traces: Sequence[Any], query_indices: torch.Tensor
) -> torch.Tensor:
    trace = traces[0]
    return torch.cat(
        (
            trace.excitatory_spikes.index_select(1, query_indices),
            -trace.inhibitory_spikes.index_select(1, query_indices),
            trace.excitatory_residuals.index_select(1, query_indices),
            -trace.inhibitory_residuals.index_select(1, query_indices),
        ),
        dim=-1,
    )


def _equivalence_case(
    *,
    batch: int,
    time_steps: int,
    queries: Sequence[int],
    input_gradient: bool,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    reference = E3InputCodedScanCore(4, 6, state_dim=5).to(device)
    candidate = E3InputCodedScanCore(4, 6, state_dim=5).to(device)
    candidate.load_state_dict(reference.state_dict())
    reference_input = torch.randn(
        batch, time_steps, 4, device=device, requires_grad=input_gradient
    )
    candidate_input = reference_input.detach().clone().requires_grad_(input_gradient)
    initial_e = torch.rand(batch, 5, device=device, requires_grad=True)
    initial_i = torch.rand(batch, 5, device=device, requires_grad=True)
    reference_state = E3ScanState(
        layers=(E3LayerState(excitatory=initial_e, inhibitory=initial_i),)
    )
    candidate_state = E3ScanState(
        layers=(
            E3LayerState(
                excitatory=initial_e.detach().clone().requires_grad_(True),
                inhibitory=initial_i.detach().clone().requires_grad_(True),
            ),
        )
    )
    query_indices = _query_tensor(queries, device)

    reference_result, reference_traces = reference.forward_dynamics(
        reference_input, reference_state
    )
    captured: Dict[str, torch.Tensor] = {}

    def capture_raw(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        captured["raw"] = inputs[0]

    handle = candidate.output_norm.register_forward_pre_hook(capture_raw)
    try:
        candidate_result = candidate.forward_multi_query_eligibility(
            candidate_input, query_indices, candidate_state
        )
    finally:
        handle.remove()
    reference_queries = reference_result.sequence.index_select(1, query_indices)
    reference_raw = _raw_from_trace(reference_traces, query_indices)
    candidate_raw = captured["raw"]

    probe = torch.linspace(
        -0.8,
        0.7,
        reference_queries.numel(),
        device=device,
        dtype=reference_queries.dtype,
    ).reshape_as(reference_queries)
    reference_loss = (reference_queries * probe).mean() + 0.17 * (
        reference_result.state.layers[0].excitatory.mean()
        - reference_result.state.layers[0].inhibitory.mean()
    )
    candidate_loss = (candidate_result.sequence * probe).mean() + 0.17 * (
        candidate_result.state.layers[0].excitatory.mean()
        - candidate_result.state.layers[0].inhibitory.mean()
    )
    reference_loss.backward()
    candidate_loss.backward()

    hidden = reference_raw.shape[-1] // 4
    candidate_spike_e = candidate_raw[:, :, :hidden]
    candidate_spike_i = -candidate_raw[:, :, hidden : 2 * hidden]
    forward = {
        "sequence": {
            "passed": bool(
                torch.allclose(
                    candidate_result.sequence,
                    reference_queries,
                    atol=EL1_ATOL,
                    rtol=EL1_RTOL,
                )
            ),
            "max_abs": _max_abs(candidate_result.sequence, reference_queries),
        },
        "raw_query": {
            "passed": bool(torch.equal(candidate_raw, reference_raw)),
            "max_abs": _max_abs(candidate_raw, reference_raw),
        },
        "state_e": {
            "passed": bool(
                torch.equal(
                    candidate_result.state.layers[0].excitatory,
                    reference_result.state.layers[0].excitatory,
                )
            ),
            "max_abs": _max_abs(
                candidate_result.state.layers[0].excitatory,
                reference_result.state.layers[0].excitatory,
            ),
        },
        "state_i": {
            "passed": bool(
                torch.equal(
                    candidate_result.state.layers[0].inhibitory,
                    reference_result.state.layers[0].inhibitory,
                )
            ),
            "max_abs": _max_abs(
                candidate_result.state.layers[0].inhibitory,
                reference_result.state.layers[0].inhibitory,
            ),
        },
        "binary_spikes": {
            "passed": bool(
                torch.all((candidate_spike_e == 0.0) | (candidate_spike_e == 1.0))
                and torch.all(
                    (candidate_spike_i == 0.0) | (candidate_spike_i == 1.0)
                )
            ),
            "max_abs": None,
        },
    }
    gradients: Dict[str, Dict[str, Any]] = {
        "input": _gradient_record(candidate_input.grad, reference_input.grad),
        "initial_e": _gradient_record(
            candidate_state.layers[0].excitatory.grad,
            reference_state.layers[0].excitatory.grad,
        ),
        "initial_i": _gradient_record(
            candidate_state.layers[0].inhibitory.grad,
            reference_state.layers[0].inhibitory.grad,
        ),
    }
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        gradients[f"parameter:{name}"] = _gradient_record(
            parameter.grad, reference_parameters[name].grad
        )
    passed = all(value["passed"] for value in forward.values()) and all(
        value["passed"] for value in gradients.values()
    )
    return {
        "batch": batch,
        "time": time_steps,
        "queries": tuple(queries),
        "input_gradient": input_gradient,
        "forward": forward,
        "gradients": gradients,
        "passed": passed,
    }


def _validation_checks(device: torch.device) -> Dict[str, Any]:
    core = E3InputCodedScanCore(4, 6, state_dim=5).to(device)
    sequence = torch.randn(2, 8, 4, device=device)
    cases: Sequence[Tuple[str, Any, type[BaseException], str]] = (
        ("non_tensor", [], TypeError, "torch.Tensor"),
        (
            "two_dimensional",
            torch.tensor([[0, 1]], device=device),
            ValueError,
            "one-dimensional",
        ),
        (
            "wrong_dtype",
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            ValueError,
            "torch.long",
        ),
        ("empty", torch.empty(0, dtype=torch.long, device=device), ValueError, "non-empty"),
        ("negative", torch.tensor([-1, 2], device=device), ValueError, "lie in"),
        ("past_end", torch.tensor([0, 8], device=device), ValueError, "lie in"),
        (
            "duplicate",
            torch.tensor([1, 1], device=device),
            ValueError,
            "strictly increasing",
        ),
        (
            "descending",
            torch.tensor([3, 2], device=device),
            ValueError,
            "strictly increasing",
        ),
    )
    records = []
    for name, indices, error_type, message in cases:
        try:
            core.forward_multi_query_eligibility(sequence, indices)
        except Exception as error:  # noqa: PERF203 - explicit validation evidence
            passed = isinstance(error, error_type) and message in str(error)
            records.append(
                {
                    "name": name,
                    "passed": passed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            records.append({"name": name, "passed": False, "error": None})
    return {"cases": records, "passed": all(record["passed"] for record in records)}


def run_equivalence(device: torch.device) -> Dict[str, Any]:
    specifications = (
        (1, 1, (0,), True),
        (2, 32, (0, 7, 18, 31), True),
        (1, 512, (0, 1, 63, 127, 255, 383, 510, 511), False),
    )
    cases = [
        _equivalence_case(
            batch=batch,
            time_steps=time_steps,
            queries=queries,
            input_gradient=input_gradient,
            device=device,
            seed=8_900_000 + index,
        )
        for index, (batch, time_steps, queries, input_gradient) in enumerate(
            specifications
        )
    ]
    validation = _validation_checks(device)
    return {
        "cases": cases,
        "validation": validation,
        "passed": all(case["passed"] for case in cases) and validation["passed"],
    }


def _query_result(
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    if name.startswith("el1"):
        if not isinstance(core, E3InputCodedScanCore):  # pragma: no cover
            raise TypeError("EL1 runner requires E3InputCodedScanCore")
        return core.forward_multi_query_eligibility(value, query_indices).sequence
    return core(value).sequence.index_select(1, query_indices)


def _saved_tensor_measurement(
    *,
    name: str,
    time_steps: int,
    query_count: int,
    input_gradient: bool,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    core = E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM).to(device)
    value = torch.randn(
        1, time_steps, D_MODEL, device=device, requires_grad=input_gradient
    )
    query_indices = _even_queries(time_steps, query_count, device)
    counter = _SavedTensorCounter()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        queries = _query_result(name, core, value, query_indices)
        queries.square().mean().backward()
    return {
        "mode": name,
        "time": time_steps,
        "query_count": query_count,
        "query_indices": tuple(int(value) for value in query_indices.tolist()),
        "input_gradient": input_gradient,
        "logical_saved_bytes": counter.logical_bytes,
        "unique_storage_bytes": counter.unique_storage_bytes,
        "saved_tensor_count": counter.tensor_count,
        "autograd_nodes": _autograd_node_count(queries),
    }


def benchmark_saved_tensors(device: torch.device) -> Dict[str, Any]:
    length_records = []
    for time_steps in (128, 512, 2048):
        for name, input_gradient in (
            ("bptt_core_only", False),
            ("el1_core_only", False),
            ("bptt_input_grad", True),
            ("el1_input_grad", True),
        ):
            length_records.append(
                _saved_tensor_measurement(
                    name=name,
                    time_steps=time_steps,
                    query_count=4,
                    input_gradient=input_gradient,
                    device=device,
                    seed=8_910_000 + time_steps,
                )
            )
    lookup = {
        (record["mode"], record["time"]): record for record in length_records
    }
    bptt_2048 = lookup[("bptt_core_only", 2048)]["unique_storage_bytes"]
    el1_128 = lookup[("el1_core_only", 128)]["unique_storage_bytes"]
    el1_2048 = lookup[("el1_core_only", 2048)]["unique_storage_bytes"]
    checks = {
        "t2048_ratio_le_25pct": el1_2048 <= 0.25 * bptt_2048,
        "el1_t128_to_t2048_growth_le_1_25x": el1_2048 <= 1.25 * el1_128,
    }

    query_scale_records = []
    for query_count in (1, 4, 16, 32):
        for name in ("bptt_core_only", "el1_core_only"):
            query_scale_records.append(
                _saved_tensor_measurement(
                    name=name,
                    time_steps=512,
                    query_count=query_count,
                    input_gradient=False,
                    device=device,
                    seed=8_911_000 + query_count,
                )
            )
    query_lookup = {
        (record["mode"], record["query_count"]): record
        for record in query_scale_records
    }
    query_scale = []
    for query_count in (1, 4, 16, 32):
        bptt = query_lookup[("bptt_core_only", query_count)]
        el1 = query_lookup[("el1_core_only", query_count)]
        query_scale.append(
            {
                "query_count": query_count,
                "bptt_unique_storage_bytes": bptt["unique_storage_bytes"],
                "el1_unique_storage_bytes": el1["unique_storage_bytes"],
                "el1_to_bptt_ratio": el1["unique_storage_bytes"]
                / bptt["unique_storage_bytes"],
                "el1_autograd_nodes": el1["autograd_nodes"],
                "bptt_autograd_nodes": bptt["autograd_nodes"],
            }
        )
    return {
        "length_records": length_records,
        "query_scale_records": query_scale_records,
        "query_scale": query_scale,
        "checks": checks,
        "t2048_unique_storage_ratio": el1_2048 / bptt_2048,
        "el1_growth_t128_to_t2048": el1_2048 / el1_128,
        "passed": all(checks.values()),
    }


def _benchmark_suite(
    *, device: torch.device, time_steps: int, seed: int
) -> Dict[str, TemporalCore[Any]]:
    torch.manual_seed(seed)
    bptt = E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    el1 = E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    el1.load_state_dict(bptt.state_dict())
    return {
        "ic0_bptt": bptt.to(device).train(True),
        "el1_core_only": el1.to(device).train(True),
        "lstm": StatefulLSTMCore(D_MODEL, D_MODEL).to(device).train(True),
        "transformer": CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=time_steps,
        )
        .to(device)
        .train(True),
    }


def _query_training_runner(
    *,
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> Runner:
    def run() -> None:
        core.zero_grad(set_to_none=True)
        queries = _query_result(name, core, value, query_indices)
        queries.square().mean().backward()

    return run


def benchmark_speed(
    *,
    time_steps: Iterable[int],
    threads: Sequence[int],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for length in time_steps:
            cores = _benchmark_suite(
                device=device,
                time_steps=length,
                seed=8_920_000 + thread_count * 10000 + length,
            )
            value = torch.randn(1, length, D_MODEL, device=device)
            query_indices = _even_queries(length, 4, device)
            runners = {
                name: _query_training_runner(
                    name=name,
                    core=core,
                    value=value,
                    query_indices=query_indices,
                )
                for name, core in cores.items()
            }
            nodes = {
                name: _autograd_node_count(
                    _query_result(name, core, value, query_indices)
                )
                for name, core in cores.items()
            }
            for core in cores.values():
                core.zero_grad(set_to_none=True)
            samples = _interleaved_samples(
                runners,
                warmup=warmup,
                repeats=repeats,
                device=device,
                seed=8_921_000 + thread_count * 10000 + length,
            )
            models = {
                name: {
                    **_sample_summary(sample, length),
                    "parameters": count_parameters(cores[name]),
                    "autograd_nodes": nodes[name],
                }
                for name, sample in samples.items()
            }
            speedup = models["ic0_bptt"]["p50_ms"] / models["el1_core_only"][
                "p50_ms"
            ]
            passed = (
                speedup >= 1.25
                and models["el1_core_only"]["p50_ms"]
                <= models["lstm"]["p50_ms"]
            )
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "time": length,
                    "query_count": 4,
                    "query_indices": tuple(
                        int(value) for value in query_indices.tolist()
                    ),
                    "models": models,
                    "el1_vs_ic0_speedup": speedup,
                    "passed": passed,
                }
            )
    return {"records": records, "passed": any(record["passed"] for record in records)}


def generate_register_batch(
    *, seed: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    payloads = torch.randint(
        PAYLOAD_VOCAB,
        (batch_size, len(QUERY_INDICES)),
        generator=generator,
    )
    tokens = torch.full(
        (batch_size, SEQUENCE_LENGTH), DISTRACTOR_TOKEN, dtype=torch.long
    )
    for segment, (write_index, query_index) in enumerate(
        zip(WRITE_INDICES, QUERY_INDICES)
    ):
        tokens[:, write_index] = WRITE_BASE + payloads[:, segment]
        tokens[:, query_index] = QUERY_TOKEN
    return tokens, payloads


class MultiQueryTokenModel(nn.Module):
    def __init__(
        self,
        core: TemporalCore[Any],
        *,
        use_multi_query_eligibility: bool = False,
    ) -> None:
        super().__init__()
        self.core = core
        self.use_multi_query_eligibility = bool(use_multi_query_eligibility)
        self.embedding = nn.Embedding(INPUT_VOCAB, D_MODEL)
        self.embedding.weight.requires_grad_(False)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.decoder = nn.Linear(D_MODEL, PAYLOAD_VOCAB)
        self.register_buffer(
            "query_indices",
            torch.tensor(QUERY_INDICES, dtype=torch.long),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        if self.use_multi_query_eligibility and self.training:
            if not isinstance(self.core, E3InputCodedScanCore):  # pragma: no cover
                raise TypeError("multi-query eligibility requires IC0")
            sequence = self.core.forward_multi_query_eligibility(
                embedded, self.query_indices
            ).sequence
        else:
            sequence = self.core(embedded).sequence.index_select(
                1, self.query_indices
            )
        return self.decoder(self.output_norm(sequence))


def _shared_wrapper_state(seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    embedding = torch.zeros(INPUT_VOCAB, D_MODEL)
    embedding[:, :INPUT_VOCAB] = torch.eye(INPUT_VOCAB)
    return {
        "embedding": embedding,
        "norm_weight": torch.ones(D_MODEL),
        "norm_bias": torch.zeros(D_MODEL),
        "decoder_weight": torch.randn(
            PAYLOAD_VOCAB, D_MODEL, generator=generator
        )
        * 0.02,
        "decoder_bias": torch.zeros(PAYLOAD_VOCAB),
    }


def _initialise_shared_wrapper(
    model: MultiQueryTokenModel, shared: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        model.embedding.weight.copy_(shared["embedding"])
        model.output_norm.weight.copy_(shared["norm_weight"])
        model.output_norm.bias.copy_(shared["norm_bias"])
        model.decoder.weight.copy_(shared["decoder_weight"])
        model.decoder.bias.copy_(shared["decoder_bias"])


def build_quality_models(seed: int, device: torch.device) -> Dict[str, MultiQueryTokenModel]:
    shared = _shared_wrapper_state(8_930_001)
    torch.manual_seed(seed)
    bptt = MultiQueryTokenModel(
        E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    )
    _initialise_shared_wrapper(bptt, shared)
    el1 = copy.deepcopy(bptt)
    el1.use_multi_query_eligibility = True
    torch.manual_seed(seed + 1)
    lstm = MultiQueryTokenModel(StatefulLSTMCore(D_MODEL, D_MODEL))
    _initialise_shared_wrapper(lstm, shared)
    torch.manual_seed(seed + 2)
    transformer = MultiQueryTokenModel(
        CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=SEQUENCE_LENGTH,
        )
    )
    _initialise_shared_wrapper(transformer, shared)
    return {
        "ic0_bptt": bptt.to(device),
        "el1": el1.to(device),
        "lstm": lstm.to(device),
        "transformer": transformer.to(device),
    }


def _train_model(
    model: MultiQueryTokenModel,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    timing_warmup: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    timings = []
    checkpoints = []
    for update, (cpu_tokens, cpu_targets) in enumerate(batches, start=1):
        tokens = cpu_tokens.to(device)
        targets = cpu_targets.to(device)
        _sync(device)
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, PAYLOAD_VOCAB), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite multi-query loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sync(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if update > timing_warmup:
            timings.append(elapsed_ms)
        losses.append(float(loss.detach().item()))
        if update == 1 or update % 100 == 0 or update == len(batches):
            checkpoints.append(
                {
                    "update": update,
                    "loss": losses[-1],
                    "gradient_norm": float(gradient_norm),
                }
            )
    return {
        "updates": len(batches),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_100_mean": sum(losses[-100:]) / min(100, len(losses)),
        "losses": losses,
        "timing": _sample_summary(timings, batches[0][0].numel()),
        "checkpoints": checkpoints,
    }


def _evaluate_model(
    model: MultiQueryTokenModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    nll = 0.0
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            batch_tokens = tokens[start : start + batch_size].to(device)
            batch_targets = targets[start : start + batch_size].to(device)
            logits = model(batch_tokens)
            losses = F.cross_entropy(
                logits.reshape(-1, PAYLOAD_VOCAB),
                batch_targets.reshape(-1),
                reduction="none",
            )
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == batch_targets).sum().item())
            total += batch_targets.numel()
            nll += float(losses.sum().item())
    return {"accuracy": correct / total, "nll": nll / total, "count": total}


def _snn_event_diagnostics(
    model: MultiQueryTokenModel,
    tokens: torch.Tensor,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    if not isinstance(model.core, E3InputCodedScanCore):  # pragma: no cover
        raise TypeError("SNN diagnostics require IC0")
    model.eval()
    with torch.inference_mode():
        embedded = model.embedding(tokens.to(device))
        event_e, event_i = model.core.input_events(embedded)
        _result, traces = model.core.forward_dynamics(embedded)
        spike_e = traces[0].excitatory_spikes
        spike_i = traces[0].inhibitory_spikes
    values = {
        "input_event_e": event_e,
        "input_event_i": event_i,
        "output_spike_e": spike_e,
        "output_spike_i": spike_i,
    }
    return {
        name: {
            "binary": bool(torch.all((value == 0.0) | (value == 1.0))),
            "rate": float(value.float().mean().item()),
        }
        for name, value in values.items()
    }


def run_quality(*, quick: bool, device: torch.device) -> Dict[str, Any]:
    seeds = (0,) if quick else (0, 1, 2)
    updates = 3 if quick else 600
    train_batch_size = 4 if quick else 32
    test_count = 64 if quick else 4096
    seed_records = []
    for seed in seeds:
        train_batches = tuple(
            generate_register_batch(
                seed=8_930_000 + 10_000 * seed + update,
                batch_size=train_batch_size,
            )
            for update in range(updates)
        )
        test_tokens, test_targets = generate_register_batch(
            seed=8_990_000 + seed,
            batch_size=test_count,
        )
        models = build_quality_models(8_940_000 + 100 * seed, device)
        parameter_counts = {
            name: count_parameters(model) for name, model in models.items()
        }
        lstm_count = parameter_counts["lstm"]
        fairness = {
            name: abs(count - lstm_count) / lstm_count <= 0.02
            for name, count in parameter_counts.items()
        }
        if not all(fairness.values()):
            raise AssertionError(f"multi-query parameter fairness failed: {parameter_counts}")

        model_results = {}
        for name in ("ic0_bptt", "el1", "lstm", "transformer"):
            model_results[name] = {
                "train": _train_model(
                    models[name],
                    train_batches,
                    timing_warmup=min(100, updates - 1),
                    device=device,
                ),
                "test": _evaluate_model(
                    models[name],
                    test_tokens,
                    test_targets,
                    batch_size=256,
                    device=device,
                ),
            }
        bptt_losses = model_results["ic0_bptt"]["train"]["losses"]
        el1_losses = model_results["el1"]["train"]["losses"]
        loss_max_abs = max(
            abs(left - right) for left, right in zip(bptt_losses, el1_losses)
        )
        parameter_max_abs, parameter_errors = _parameter_max_abs(
            models["el1"], models["ic0_bptt"]
        )
        for result in model_results.values():
            result["train"].pop("losses")
        seed_records.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_fairness": fairness,
                "train_data_sha256": _dataset_hash(train_batches),
                "models": model_results,
                "el1_vs_bptt_loss_max_abs": loss_max_abs,
                "el1_vs_bptt_parameter_max_abs": parameter_max_abs,
                "el1_vs_bptt_parameter_errors": parameter_errors,
                "event_diagnostics": _snn_event_diagnostics(
                    models["el1"], test_tokens[:256], device=device
                ),
            }
        )

    if quick:
        task_valid = False
        quality_pass = False
    else:
        task_valid = all(
            record["models"][name]["test"]["accuracy"] >= 0.99
            for record in seed_records
            for name in ("lstm", "transformer")
        )
        quality_pass = task_valid and all(
            record["models"][name]["test"]["accuracy"] >= 0.99
            for record in seed_records
            for name in ("ic0_bptt", "el1")
        ) and all(
            record["el1_vs_bptt_loss_max_abs"] <= 1e-4
            and record["el1_vs_bptt_parameter_max_abs"] <= 1e-4
            for record in seed_records
        )
    return {
        "formal": not quick,
        "task": {
            "sequence_length": SEQUENCE_LENGTH,
            "write_indices": WRITE_INDICES,
            "query_indices": QUERY_INDICES,
            "delay": 4,
            "payload_classes": PAYLOAD_VOCAB,
            "frozen_orthogonal_embedding": True,
            "updates": updates,
            "train_batch_size": train_batch_size,
            "test_sequences_per_seed": test_count,
            "test_query_targets_per_seed": test_count * len(QUERY_INDICES),
        },
        "seeds": seed_records,
        "task_validation": "PASS" if task_valid else "NOT_RUN" if quick else "FAIL",
        "passed": quality_pass,
    }


def _decision(
    *,
    equivalence: Mapping[str, Any],
    memory: Mapping[str, Any],
    speed: Mapping[str, Any],
    quality: Mapping[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
            "memory_gate": "PASS" if memory["passed"] else "FAIL",
            "speed_gate": "SMOKE",
            "quality_gate": "NOT_RUN",
            "overall": "SMOKE",
            "run_textworld_k_query_next": False,
        }
    binary_gates = {
        "equivalence_gate": bool(equivalence["passed"]),
        "memory_gate": bool(memory["passed"]),
        "speed_gate": bool(speed["passed"]),
    }
    task_invalid = quality.get("task_validation") == "FAIL"
    quality_gate = "INVALID" if task_invalid else "PASS" if quality["passed"] else "FAIL"
    all_binary_pass = all(binary_gates.values())
    overall = (
        "PASS"
        if all_binary_pass and quality_gate == "PASS"
        else "MIXED"
        if all_binary_pass and quality_gate == "INVALID"
        else "FAIL"
    )
    return {
        **{
            name: "PASS" if value else "FAIL"
            for name, value in binary_gates.items()
        },
        "quality_gate": quality_gate,
        "overall": overall,
        "run_textworld_k_query_next": overall == "PASS",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_el1_multi_query_eligibility.json"),
    )
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.threads = args.threads[:1]
        args.warmup = 1
        args.repeats = 1
    return args


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    threads = tuple(dict.fromkeys(args.threads))
    if device.type == "cpu":
        torch.set_num_threads(threads[0])
    equivalence = run_equivalence(device)
    if not equivalence["passed"]:
        memory = {"records": [], "passed": False, "not_run": "equivalence failed"}
        speed = {"records": [], "passed": False, "not_run": "equivalence failed"}
        quality = {"passed": False, "not_run": "equivalence failed"}
    else:
        memory = benchmark_saved_tensors(device)
        speed = benchmark_speed(
            time_steps=(512,) if args.quick else (512, 2048),
            threads=threads,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        if device.type == "cpu":
            torch.set_num_threads(4 if not args.quick else threads[0])
        quality = run_quality(quick=args.quick, device=device)
    decision = _decision(
        equivalence=equivalence,
        memory=memory,
        speed=speed,
        quality=quality,
        quick=args.quick,
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-EL1 exact multi-query eligibility",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "threads": threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "atol": EL1_ATOL,
            "rtol": EL1_RTOL,
        },
        "equivalence": equivalence,
        "saved_tensors": memory,
        "speed": speed,
        "quality": quality,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
