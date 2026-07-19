"""AT0 gated synaptic-trace SNN: exact scan, memory quality, and ANN speed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _autograd_node_count,
    _environment,
    _percentile,
    _sample_summary,
    _sync,
)
from experiments.e3_el0_terminal_eligibility import (  # noqa: E402
    _gradient_record,
    _max_abs,
)
from experiments.e3_el1_multi_query_eligibility import (  # noqa: E402
    D_MODEL,
    PAYLOAD_VOCAB,
    MultiQueryTokenModel,
    _dataset_hash,
    _evaluate_model,
    _initialise_shared_wrapper,
    _shared_wrapper_state,
    _train_model,
    generate_register_batch,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3GatedTraceScanCore,
    E3InputCodedScanCore,
    E3LayerState,
    E3ScanState,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
    state_nbytes,
)


STATE_DIM = 31
AT0_ATOL = 2e-5
AT0_RTOL = 1e-4
QUERY_COUNT = 4
Runner = Callable[[], None]


def _even_queries(time_steps: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        tuple(
            round(index * (time_steps - 1) / (QUERY_COUNT - 1))
            for index in range(QUERY_COUNT)
        ),
        dtype=torch.long,
        device=device,
    )


def _equivalence_case(
    *,
    batch: int,
    time_steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    serial = E3GatedTraceScanCore(
        4, 6, state_dim=5, execution_mode="serial"
    ).to(device)
    scan = E3GatedTraceScanCore(
        4, 6, state_dim=5, execution_mode="scan"
    ).to(device)
    scan.load_state_dict(serial.state_dict())
    serial_input = torch.randn(batch, time_steps, 4, device=device, requires_grad=True)
    scan_input = serial_input.detach().clone().requires_grad_(True)
    initial_e = torch.rand(batch, 5, device=device, requires_grad=True)
    initial_i = torch.rand(batch, 5, device=device, requires_grad=True)
    serial_state = E3ScanState(
        layers=(E3LayerState(excitatory=initial_e, inhibitory=initial_i),)
    )
    scan_state = E3ScanState(
        layers=(
            E3LayerState(
                excitatory=initial_e.detach().clone().requires_grad_(True),
                inhibitory=initial_i.detach().clone().requires_grad_(True),
            ),
        )
    )
    serial_result, serial_trace = serial.forward_dynamics(serial_input, serial_state)
    scan_result, scan_trace = scan.forward_dynamics(scan_input, scan_state)

    continuous_names = ("excitatory_traces", "inhibitory_traces")
    binary_names = (
        "excitatory_content",
        "inhibitory_content",
        "excitatory_gate",
        "inhibitory_gate",
        "excitatory_writes",
        "inhibitory_writes",
        "excitatory_spikes",
        "inhibitory_spikes",
    )
    forward: Dict[str, Dict[str, Any]] = {
        "sequence": {
            "passed": bool(
                torch.allclose(
                    scan_result.sequence,
                    serial_result.sequence,
                    atol=AT0_ATOL,
                    rtol=AT0_RTOL,
                )
            ),
            "max_abs": _max_abs(scan_result.sequence, serial_result.sequence),
        },
        "state_e": {
            "passed": bool(
                torch.allclose(
                    scan_result.state.layers[0].excitatory,
                    serial_result.state.layers[0].excitatory,
                    atol=AT0_ATOL,
                    rtol=AT0_RTOL,
                )
            ),
            "max_abs": _max_abs(
                scan_result.state.layers[0].excitatory,
                serial_result.state.layers[0].excitatory,
            ),
        },
        "state_i": {
            "passed": bool(
                torch.allclose(
                    scan_result.state.layers[0].inhibitory,
                    serial_result.state.layers[0].inhibitory,
                    atol=AT0_ATOL,
                    rtol=AT0_RTOL,
                )
            ),
            "max_abs": _max_abs(
                scan_result.state.layers[0].inhibitory,
                serial_result.state.layers[0].inhibitory,
            ),
        },
    }
    for name in binary_names:
        candidate = getattr(scan_trace, name)
        reference = getattr(serial_trace, name)
        forward[name] = {
            "passed": bool(
                torch.equal(candidate, reference)
                and torch.all((candidate == 0.0) | (candidate == 1.0))
            ),
            "max_abs": _max_abs(candidate, reference),
        }
    for name in continuous_names:
        candidate = getattr(scan_trace, name)
        reference = getattr(serial_trace, name)
        forward[name] = {
            "passed": bool(
                torch.allclose(
                    candidate, reference, atol=AT0_ATOL, rtol=AT0_RTOL
                )
                and torch.all((candidate >= 0.0) & (candidate <= 1.0))
            ),
            "max_abs": _max_abs(candidate, reference),
        }

    probe = torch.linspace(
        -0.6,
        0.8,
        serial_result.sequence.numel(),
        device=device,
        dtype=serial_result.sequence.dtype,
    ).reshape_as(serial_result.sequence)
    serial_loss = (serial_result.sequence * probe).mean() + 0.11 * (
        serial_result.state.layers[0].excitatory.mean()
        - serial_result.state.layers[0].inhibitory.mean()
    )
    scan_loss = (scan_result.sequence * probe).mean() + 0.11 * (
        scan_result.state.layers[0].excitatory.mean()
        - scan_result.state.layers[0].inhibitory.mean()
    )
    serial_loss.backward()
    scan_loss.backward()
    gradients: Dict[str, Dict[str, Any]] = {
        "input": _gradient_record(scan_input.grad, serial_input.grad),
        "initial_e": _gradient_record(
            scan_state.layers[0].excitatory.grad,
            serial_state.layers[0].excitatory.grad,
        ),
        "initial_i": _gradient_record(
            scan_state.layers[0].inhibitory.grad,
            serial_state.layers[0].inhibitory.grad,
        ),
    }
    serial_parameters = dict(serial.named_parameters())
    for name, parameter in scan.named_parameters():
        gradients[f"parameter:{name}"] = _gradient_record(
            parameter.grad, serial_parameters[name].grad
        )
    passed = all(record["passed"] for record in forward.values()) and all(
        record["passed"] for record in gradients.values()
    )
    return {
        "batch": batch,
        "time": time_steps,
        "forward": forward,
        "gradients": gradients,
        "passed": passed,
    }


def _stream_equivalence(device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(9_010_000)
    core = E3GatedTraceScanCore(4, 6, state_dim=5).to(device).eval()
    tokens = torch.randn(3, 64, 4, device=device)
    initial = E3ScanState(
        layers=(
            E3LayerState(
                excitatory=torch.rand(3, 5, device=device),
                inhibitory=torch.rand(3, 5, device=device),
            ),
        )
    )
    with torch.inference_mode():
        full = core(tokens, initial)
        state = initial
        pieces = []
        for index in range(tokens.shape[1]):
            step = core.step(tokens[:, index], state)
            pieces.append(step.sequence)
            state = step.state
        streamed = torch.cat(pieces, dim=1)
    records = {
        "sequence": {
            "passed": bool(
                torch.allclose(streamed, full.sequence, atol=AT0_ATOL, rtol=AT0_RTOL)
            ),
            "max_abs": _max_abs(streamed, full.sequence),
        },
        "state_e": {
            "passed": bool(
                torch.allclose(
                    state.layers[0].excitatory,
                    full.state.layers[0].excitatory,
                    atol=AT0_ATOL,
                    rtol=AT0_RTOL,
                )
            ),
            "max_abs": _max_abs(
                state.layers[0].excitatory, full.state.layers[0].excitatory
            ),
        },
        "state_i": {
            "passed": bool(
                torch.allclose(
                    state.layers[0].inhibitory,
                    full.state.layers[0].inhibitory,
                    atol=AT0_ATOL,
                    rtol=AT0_RTOL,
                )
            ),
            "max_abs": _max_abs(
                state.layers[0].inhibitory, full.state.layers[0].inhibitory
            ),
        },
    }
    return {"components": records, "passed": all(v["passed"] for v in records.values())}


def run_equivalence(device: torch.device) -> Dict[str, Any]:
    cases = [
        _equivalence_case(
            batch=batch,
            time_steps=time_steps,
            device=device,
            seed=9_000_000 + index,
        )
        for index, (batch, time_steps) in enumerate(((1, 1), (4, 32), (1, 512)))
    ]
    streaming = _stream_equivalence(device)
    budget_core = E3GatedTraceScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM).to(device)
    budget = {
        "parameters": count_parameters(budget_core),
        "state_bytes": state_nbytes(budget_core.initial_state(1, device=device)),
        "lstm_parameters": count_parameters(
            StatefulLSTMCore(D_MODEL, D_MODEL).to(device)
        ),
        "lstm_state_bytes": state_nbytes(
            StatefulLSTMCore(D_MODEL, D_MODEL).to(device).initial_state(
                1, device=device
            )
        ),
    }
    budget["parameter_fair"] = (
        abs(budget["parameters"] - budget["lstm_parameters"])
        / budget["lstm_parameters"]
        <= 0.02
    )
    return {
        "cases": cases,
        "streaming": streaming,
        "budget": budget,
        "passed": all(case["passed"] for case in cases)
        and streaming["passed"]
        and budget["parameter_fair"],
    }


def _training_output(
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    if name == "ic0_el1":
        if not isinstance(core, E3InputCodedScanCore):  # pragma: no cover
            raise TypeError("ic0_el1 requires E3InputCodedScanCore")
        return core.forward_multi_query_eligibility(value, query_indices).sequence
    return core(value).sequence.index_select(1, query_indices)


def _training_runner(
    *,
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> Runner:
    def run() -> None:
        core.zero_grad(set_to_none=True)
        output = _training_output(name, core, value, query_indices)
        output.square().mean().backward()

    return run


def _interleaved_training_samples(
    runners: Mapping[str, Runner],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> Dict[str, list[float]]:
    for _ in range(warmup):
        for runner in runners.values():
            runner()
    samples: Dict[str, list[float]] = {name: [] for name in runners}
    names = list(runners)
    generator = random.Random(seed)
    for _ in range(repeats):
        generator.shuffle(names)
        for name in names:
            _sync(device)
            started = time.perf_counter_ns()
            runners[name]()
            _sync(device)
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    return samples


def benchmark_training(
    *,
    threads: Sequence[int],
    lengths: Sequence[int],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for length in lengths:
            torch.manual_seed(9_020_000 + thread_count * 10000 + length)
            at0_serial = E3GatedTraceScanCore(
                D_MODEL, D_MODEL, state_dim=STATE_DIM, execution_mode="serial"
            ).to(device)
            at0_scan = E3GatedTraceScanCore(
                D_MODEL, D_MODEL, state_dim=STATE_DIM, execution_mode="scan"
            ).to(device)
            at0_scan.load_state_dict(at0_serial.state_dict())
            ic0_el1 = E3InputCodedScanCore(
                D_MODEL, D_MODEL, state_dim=42
            ).to(device)
            lstm = StatefulLSTMCore(D_MODEL, D_MODEL).to(device)
            transformer = CausalTransformerCore(
                D_MODEL,
                D_MODEL,
                num_layers=1,
                num_heads=4,
                mlp_ratio=2.0,
                dropout=0.0,
                max_cache_tokens=length,
            ).to(device)
            cores: Dict[str, TemporalCore[Any]] = {
                "at0_serial": at0_serial,
                "at0_scan": at0_scan,
                "ic0_el1": ic0_el1,
                "lstm": lstm,
                "transformer": transformer,
            }
            value = torch.randn(1, length, D_MODEL, device=device)
            query_indices = _even_queries(length, device)
            runners = {
                name: _training_runner(
                    name=name,
                    core=core,
                    value=value,
                    query_indices=query_indices,
                )
                for name, core in cores.items()
            }
            nodes = {
                name: _autograd_node_count(
                    _training_output(name, core, value, query_indices)
                )
                for name, core in cores.items()
            }
            for core in cores.values():
                core.zero_grad(set_to_none=True)
            samples = _interleaved_training_samples(
                runners,
                warmup=warmup,
                repeats=repeats,
                device=device,
                seed=9_021_000 + thread_count * 10000 + length,
            )
            models = {
                name: {
                    **_sample_summary(sample, length),
                    "parameters": count_parameters(cores[name]),
                    "autograd_nodes": nodes[name],
                }
                for name, sample in samples.items()
            }
            serial_to_scan = (
                models["at0_serial"]["p50_ms"] / models["at0_scan"]["p50_ms"]
            )
            node_ratio = (
                models["at0_scan"]["autograd_nodes"]
                / models["at0_serial"]["autograd_nodes"]
            )
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "time": length,
                    "query_count": QUERY_COUNT,
                    "models": models,
                    "serial_to_scan_speedup": serial_to_scan,
                    "scan_to_serial_node_ratio": node_ratio,
                    "parallel_pass": serial_to_scan >= 5.0 and node_ratio <= 0.25,
                    "ann_train_pass": models["at0_scan"]["p50_ms"]
                    <= models["lstm"]["p50_ms"],
                }
            )
    return {
        "records": records,
        "parallel_passed": any(record["parallel_pass"] for record in records),
    }


class _CoreStepRunner:
    def __init__(self, core: TemporalCore[Any], tokens: torch.Tensor) -> None:
        self.core = core
        self.tokens = tokens
        with torch.inference_mode():
            self.state = core.initial_state(tokens.shape[1], device=tokens.device)
        self.index = 0

    def __call__(self) -> Tuple[torch.Tensor, ...]:
        result = self.core.step(self.tokens[self.index], self.state)
        self.state = result.state
        self.index += 1
        return (result.sequence,)


class _IC0TensorStepRunner:
    def __init__(self, core: E3InputCodedScanCore, tokens: torch.Tensor) -> None:
        self.core = core
        self.tokens = tokens
        with torch.inference_mode():
            state = core.initial_state(tokens.shape[1], device=tokens.device)
        self.excitatory = state.layers[0].excitatory
        self.inhibitory = state.layers[0].inhibitory
        self.index = 0

    def __call__(self) -> Tuple[torch.Tensor, ...]:
        output = self.core.forward_step_tensors(
            self.tokens[self.index], self.excitatory, self.inhibitory
        )
        self.excitatory = output[1]
        self.inhibitory = output[2]
        self.index += 1
        return output


class _AT0TensorStepRunner:
    def __init__(self, core: E3GatedTraceScanCore, tokens: torch.Tensor) -> None:
        self.core = core
        self.tokens = tokens
        with torch.inference_mode():
            state = core.initial_state(tokens.shape[1], device=tokens.device)
        self.excitatory = state.layers[0].excitatory
        self.inhibitory = state.layers[0].inhibitory
        self.index = 0

    def __call__(self) -> Tuple[torch.Tensor, ...]:
        output = self.core.forward_step_tensors(
            self.tokens[self.index], self.excitatory, self.inhibitory
        )
        self.excitatory = output[1]
        self.inhibitory = output[2]
        self.index += 1
        return output


def _time_step(runner: Any, device: torch.device) -> float:
    _sync(device)
    started = time.perf_counter_ns()
    output = runner()
    output[0].sum().item()
    _sync(device)
    return (time.perf_counter_ns() - started) / 1e6


def _stream_state_bytes(runner: Any) -> int:
    if isinstance(runner, (_AT0TensorStepRunner, _IC0TensorStepRunner)):
        return sum(
            value.numel() * value.element_size()
            for value in (runner.excitatory, runner.inhibitory)
        )
    return state_nbytes(runner.state)


def benchmark_streaming(
    *,
    threads: Sequence[int],
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        seed = 9_030_000 + thread_count
        torch.manual_seed(seed)
        at0 = E3GatedTraceScanCore(
            D_MODEL, D_MODEL, state_dim=STATE_DIM
        ).to(device).eval()
        ic0 = E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=42).to(device).eval()
        lstm = StatefulLSTMCore(D_MODEL, D_MODEL).to(device).eval()
        transformer = CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=warmup_steps + measured_steps,
        ).to(device).eval()
        with torch.inference_mode():
            tokens = torch.randn(
                warmup_steps + measured_steps,
                1,
                D_MODEL,
                device=device,
            )
        runners: Dict[str, Any] = {
            "at0_tensor_step": _AT0TensorStepRunner(at0, tokens),
            "ic0_tensor_step": _IC0TensorStepRunner(ic0, tokens),
            "lstm_step": _CoreStepRunner(lstm, tokens),
            "transformer_step": _CoreStepRunner(transformer, tokens),
        }
        with torch.inference_mode():
            for _ in range(warmup_steps):
                for runner in runners.values():
                    runner()
            samples: Dict[str, list[float]] = {name: [] for name in runners}
            names = list(runners)
            generator = random.Random(seed + 1)
            for _ in range(measured_steps):
                generator.shuffle(names)
                for name in names:
                    samples[name].append(_time_step(runners[name], device))
        core_lookup: Dict[str, nn.Module] = {
            "at0_tensor_step": at0,
            "ic0_tensor_step": ic0,
            "lstm_step": lstm,
            "transformer_step": transformer,
        }
        models = {
            name: {
                **_sample_summary(sample, 1),
                "p99_ms": _percentile(sample, 0.99),
                "parameters": count_parameters(core_lookup[name]),
                "state_bytes_after_stream": _stream_state_bytes(runners[name]),
            }
            for name, sample in samples.items()
        }
        at0_metrics = models["at0_tensor_step"]
        lstm_metrics = models["lstm_step"]
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "models": models,
                "at0_to_lstm_p50_ratio": at0_metrics["p50_ms"]
                / lstm_metrics["p50_ms"],
                "at0_to_lstm_p95_ratio": at0_metrics["p95_ms"]
                / lstm_metrics["p95_ms"],
                "passed": at0_metrics["p50_ms"] <= lstm_metrics["p50_ms"]
                and at0_metrics["p95_ms"] <= lstm_metrics["p95_ms"],
            }
        )
    return {"records": records}


def _build_quality_models(
    seed: int, device: torch.device
) -> Dict[str, MultiQueryTokenModel]:
    shared = _shared_wrapper_state(8_930_001)
    torch.manual_seed(seed)
    at0 = MultiQueryTokenModel(
        E3GatedTraceScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    )
    _initialise_shared_wrapper(at0, shared)
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
            max_cache_tokens=32,
        )
    )
    _initialise_shared_wrapper(transformer, shared)
    return {
        "at0": at0.to(device),
        "lstm": lstm.to(device),
        "transformer": transformer.to(device),
    }


def _event_diagnostics(
    model: MultiQueryTokenModel,
    tokens: torch.Tensor,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("AT0 diagnostics require E3GatedTraceScanCore")
    model.eval()
    with torch.inference_mode():
        embedded = model.embedding(tokens.to(device))
        _result, trace = model.core.forward_dynamics(embedded)
        decay_e, decay_i = model.core.decays()
    binary_names = (
        "excitatory_content",
        "inhibitory_content",
        "excitatory_gate",
        "inhibitory_gate",
        "excitatory_writes",
        "inhibitory_writes",
        "excitatory_spikes",
        "inhibitory_spikes",
    )
    return {
        "binary_events": {
            name: {
                "binary": bool(
                    torch.all(
                        (getattr(trace, name) == 0.0)
                        | (getattr(trace, name) == 1.0)
                    )
                ),
                "rate": float(getattr(trace, name).float().mean().item()),
            }
            for name in binary_names
        },
        "trace_range": {
            "minimum": min(
                float(trace.excitatory_traces.min().item()),
                float(trace.inhibitory_traces.min().item()),
            ),
            "maximum": max(
                float(trace.excitatory_traces.max().item()),
                float(trace.inhibitory_traces.max().item()),
            ),
        },
        "decay_range": {
            "minimum": min(float(decay_e.min().item()), float(decay_i.min().item())),
            "maximum": max(float(decay_e.max().item()), float(decay_i.max().item())),
        },
    }


def run_quality(*, quick: bool, device: torch.device) -> Dict[str, Any]:
    seeds = (0,) if quick else (0, 1, 2)
    updates = 3 if quick else 600
    train_batch_size = 4 if quick else 32
    test_count = 64 if quick else 4096
    records = []
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
        models = _build_quality_models(9_040_000 + 100 * seed, device)
        parameter_counts = {
            name: count_parameters(model) for name, model in models.items()
        }
        lstm_count = parameter_counts["lstm"]
        fairness = {
            name: abs(count - lstm_count) / lstm_count <= 0.02
            for name, count in parameter_counts.items()
        }
        if not all(fairness.values()):
            raise AssertionError(f"AT0 parameter fairness failed: {parameter_counts}")
        model_results = {}
        for name in ("at0", "lstm", "transformer"):
            train = _train_model(
                models[name],
                train_batches,
                timing_warmup=min(100, updates - 1),
                device=device,
            )
            train.pop("losses")
            model_results[name] = {
                "train": train,
                "test": _evaluate_model(
                    models[name],
                    test_tokens,
                    test_targets,
                    batch_size=256,
                    device=device,
                ),
            }
        records.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_fairness": fairness,
                "train_data_sha256": _dataset_hash(train_batches),
                "models": model_results,
                "event_diagnostics": _event_diagnostics(
                    models["at0"], test_tokens[:256], device=device
                ),
            }
        )
    if quick:
        task_valid = False
        quality_pass = False
    else:
        task_valid = all(
            record["models"][name]["test"]["accuracy"] >= 0.99
            for record in records
            for name in ("lstm", "transformer")
        )
        quality_pass = task_valid and all(
            record["models"]["at0"]["test"]["accuracy"] >= 0.99
            for record in records
        )
    return {
        "formal": not quick,
        "seeds": records,
        "task_validation": "PASS" if task_valid else "NOT_RUN" if quick else "FAIL",
        "passed": quality_pass,
    }


def _decision(
    *,
    equivalence: Mapping[str, Any],
    training: Mapping[str, Any],
    streaming: Mapping[str, Any],
    quality: Mapping[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
            "parallel_gate": "SMOKE",
            "quality_gate": "NOT_RUN",
            "ann_gate": "SMOKE",
            "overall": "SMOKE",
            "run_textworld_next": False,
        }
    parallel_pass = bool(training["parallel_passed"])
    thread_stream_pass = {
        record["threads"]: record["passed"] for record in streaming["records"]
    }
    ann_pass = any(
        record["ann_train_pass"]
        and thread_stream_pass.get(record["threads"], False)
        for record in training["records"]
    )
    task_invalid = quality.get("task_validation") == "FAIL"
    quality_gate = "INVALID" if task_invalid else "PASS" if quality["passed"] else "FAIL"
    gates = {
        "equivalence_gate": bool(equivalence["passed"]),
        "parallel_gate": parallel_pass,
        "ann_gate": ann_pass,
    }
    overall = "PASS" if all(gates.values()) and quality_gate == "PASS" else "FAIL"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "quality_gate": quality_gate,
        "overall": overall,
        "run_textworld_next": overall == "PASS",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_at0_gated_trace.json"),
    )
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--stream-warmup", type=int, default=64)
    parser.add_argument("--stream-steps", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.threads = args.threads[:1]
        args.warmup = 1
        args.repeats = 1
        args.stream_warmup = 4
        args.stream_steps = 32
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
    if equivalence["passed"]:
        training = benchmark_training(
            threads=threads,
            lengths=(512,) if args.quick else (512, 2048),
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        streaming = benchmark_streaming(
            threads=threads,
            warmup_steps=args.stream_warmup,
            measured_steps=args.stream_steps,
            device=device,
        )
        if device.type == "cpu":
            torch.set_num_threads(4 if not args.quick else threads[0])
        quality = run_quality(quick=args.quick, device=device)
    else:
        training = {"records": [], "parallel_passed": False, "not_run": "EQ failed"}
        streaming = {"records": [], "not_run": "EQ failed"}
        quality = {"passed": False, "not_run": "EQ failed"}
    decision = _decision(
        equivalence=equivalence,
        training=training,
        streaming=streaming,
        quality=quality,
        quick=args.quick,
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-AT0 gated synaptic trace exact scan",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "threads": threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "stream_warmup": args.stream_warmup,
            "stream_steps": args.stream_steps,
            "atol": AT0_ATOL,
            "rtol": AT0_RTOL,
        },
        "equivalence": equivalence,
        "training": training,
        "streaming": streaming,
        "quality": quality,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
