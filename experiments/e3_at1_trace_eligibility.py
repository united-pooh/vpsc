"""AT1 exact gated-trace eligibility and cached-decay streaming experiment."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
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
    _percentile,
    _sample_summary,
    _sync,
)
from experiments.e3_at0_gated_trace import (  # noqa: E402
    STATE_DIM,
    _AT0TensorStepRunner,
    _CoreStepRunner,
    _IC0TensorStepRunner,
    _event_diagnostics,
)
from experiments.e3_el0_terminal_eligibility import (  # noqa: E402
    _SavedTensorCounter,
    _dataset_hash,
    _max_abs,
    _parameter_max_abs,
)
from experiments.e3_el1_multi_query_eligibility import (  # noqa: E402
    D_MODEL,
    INPUT_VOCAB,
    PAYLOAD_VOCAB,
    QUERY_INDICES,
    SEQUENCE_LENGTH,
    MultiQueryTokenModel,
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


AT1_ATOL = 2e-5
AT1_RTOL = 1e-4
QUERY_COUNT = 4
Runner = Callable[[], None]


def _query_tensor(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(values), dtype=torch.long, device=device)


def _even_queries(
    time_steps: int, count: int, device: torch.device
) -> torch.Tensor:
    if count <= 0 or count > time_steps:
        raise ValueError("query count must lie in [1, time_steps]")
    if count == 1:
        values = (time_steps - 1,)
    else:
        values = tuple(
            round(index * (time_steps - 1) / (count - 1))
            for index in range(count)
        )
    if len(set(values)) != count:  # pragma: no cover
        raise AssertionError(f"query construction produced duplicates: {values}")
    return _query_tensor(values, device)


def _gradient_record(
    candidate: Optional[torch.Tensor], reference: Optional[torch.Tensor]
) -> Dict[str, Any]:
    if candidate is None or reference is None:
        return {
            "passed": candidate is None and reference is None,
            "max_abs": None,
        }
    return {
        "passed": bool(
            torch.allclose(
                candidate, reference, atol=AT1_ATOL, rtol=AT1_RTOL
            )
        ),
        "max_abs": _max_abs(candidate, reference),
    }


def _raw_from_trace(trace: Any, query_indices: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            trace.excitatory_spikes.index_select(1, query_indices),
            -trace.inhibitory_spikes.index_select(1, query_indices),
            trace.excitatory_traces.index_select(1, query_indices),
            -trace.inhibitory_traces.index_select(1, query_indices),
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
    eligibility_forward_mode: str = "segment",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    reference = E3GatedTraceScanCore(
        4, 6, state_dim=5, execution_mode="scan"
    ).to(device)
    candidate = E3GatedTraceScanCore(
        4,
        6,
        state_dim=5,
        execution_mode="scan",
        eligibility_forward_mode=eligibility_forward_mode,  # type: ignore[arg-type]
    ).to(device)
    candidate.load_state_dict(reference.state_dict())
    reference_input = torch.randn(
        batch, time_steps, 4, device=device, requires_grad=input_gradient
    )
    candidate_input = (
        reference_input.detach().clone().requires_grad_(input_gradient)
    )
    reference_e = torch.rand(batch, 5, device=device, requires_grad=True)
    reference_i = torch.rand(batch, 5, device=device, requires_grad=True)
    candidate_e = reference_e.detach().clone().requires_grad_(True)
    candidate_i = reference_i.detach().clone().requires_grad_(True)
    reference_state = E3ScanState(
        layers=(
            E3LayerState(excitatory=reference_e, inhibitory=reference_i),
        )
    )
    candidate_state = E3ScanState(
        layers=(
            E3LayerState(excitatory=candidate_e, inhibitory=candidate_i),
        )
    )
    query_indices = _query_tensor(queries, device)

    reference_result, reference_trace = reference.forward_dynamics(
        reference_input, reference_state
    )
    captured_raw: list[torch.Tensor] = []
    handle = candidate.output_norm.register_forward_pre_hook(
        lambda _module, arguments: captured_raw.append(arguments[0].detach())
    )
    try:
        candidate_result = candidate.forward_multi_query_eligibility(
            candidate_input, query_indices, candidate_state
        )
    finally:
        handle.remove()
    candidate_raw = captured_raw[0]
    reference_raw = _raw_from_trace(reference_trace, query_indices)
    reference_queries = reference_result.sequence.index_select(1, query_indices)

    state_records = {
        "sequence": {
            "passed": bool(
                torch.allclose(
                    candidate_result.sequence,
                    reference_queries,
                    atol=AT1_ATOL,
                    rtol=AT1_RTOL,
                )
            ),
            "max_abs": _max_abs(candidate_result.sequence, reference_queries),
        },
        "raw_query": {
            "passed": bool(
                torch.allclose(
                    candidate_raw,
                    reference_raw,
                    atol=AT1_ATOL,
                    rtol=AT1_RTOL,
                )
            ),
            "max_abs": _max_abs(candidate_raw, reference_raw),
        },
        "state_e": {
            "passed": bool(
                torch.allclose(
                    candidate_result.state.layers[0].excitatory,
                    reference_result.state.layers[0].excitatory,
                    atol=AT1_ATOL,
                    rtol=AT1_RTOL,
                )
            ),
            "max_abs": _max_abs(
                candidate_result.state.layers[0].excitatory,
                reference_result.state.layers[0].excitatory,
            ),
        },
        "state_i": {
            "passed": bool(
                torch.allclose(
                    candidate_result.state.layers[0].inhibitory,
                    reference_result.state.layers[0].inhibitory,
                    atol=AT1_ATOL,
                    rtol=AT1_RTOL,
                )
            ),
            "max_abs": _max_abs(
                candidate_result.state.layers[0].inhibitory,
                reference_result.state.layers[0].inhibitory,
            ),
        },
        "hard_query_events": {
            "passed": bool(
                torch.equal(
                    candidate_raw[:, :, : 2 * reference.state_dim],
                    reference_raw[:, :, : 2 * reference.state_dim],
                )
            ),
            "max_abs": _max_abs(
                candidate_raw[:, :, : 2 * reference.state_dim],
                reference_raw[:, :, : 2 * reference.state_dim],
            ),
        },
    }

    probe = torch.linspace(
        -0.7,
        0.9,
        reference_queries.numel(),
        device=device,
        dtype=reference_queries.dtype,
    ).reshape_as(reference_queries)
    reference_loss = (reference_queries * probe).mean() + 0.13 * (
        reference_result.state.layers[0].excitatory.square().mean()
        - reference_result.state.layers[0].inhibitory.square().mean()
    )
    candidate_loss = (candidate_result.sequence * probe).mean() + 0.13 * (
        candidate_result.state.layers[0].excitatory.square().mean()
        - candidate_result.state.layers[0].inhibitory.square().mean()
    )
    reference_loss.backward()
    candidate_loss.backward()
    gradients: Dict[str, Dict[str, Any]] = {
        "input": _gradient_record(candidate_input.grad, reference_input.grad),
        "initial_e": _gradient_record(candidate_e.grad, reference_e.grad),
        "initial_i": _gradient_record(candidate_i.grad, reference_i.grad),
    }
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        gradients[f"parameter:{name}"] = _gradient_record(
            parameter.grad, reference_parameters[name].grad
        )
    return {
        "batch": batch,
        "time": time_steps,
        "query_count": len(queries),
        "query_indices": tuple(queries),
        "input_gradient": input_gradient,
        "forward": state_records,
        "gradients": gradients,
        "passed": all(record["passed"] for record in state_records.values())
        and all(record["passed"] for record in gradients.values()),
    }


def _validation_checks(device: torch.device) -> Dict[str, Any]:
    core = E3GatedTraceScanCore(4, 4).to(device)
    value = torch.randn(1, 4, 4, device=device)
    invalid = (
        ("not_tensor", []),
        ("empty", torch.tensor([], dtype=torch.long, device=device)),
        ("rank", torch.tensor([[0]], dtype=torch.long, device=device)),
        ("dtype", torch.tensor([0.0], device=device)),
        ("negative", torch.tensor([-1], dtype=torch.long, device=device)),
        ("overflow", torch.tensor([4], dtype=torch.long, device=device)),
        ("duplicate", torch.tensor([1, 1], dtype=torch.long, device=device)),
        ("descending", torch.tensor([2, 1], dtype=torch.long, device=device)),
    )
    records = []
    for name, query in invalid:
        try:
            core.forward_multi_query_eligibility(value, query)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            records.append(
                {"name": name, "passed": True, "error": str(error)}
            )
        else:
            records.append({"name": name, "passed": False, "error": None})
    return {"cases": records, "passed": all(item["passed"] for item in records)}


def _cached_step_equivalence(device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(9_110_000)
    core = E3GatedTraceScanCore(4, 6, state_dim=5).to(device).eval()
    tokens = torch.randn(3, 64, 4, device=device)
    initial_e = torch.rand(3, 5, device=device)
    initial_i = torch.rand(3, 5, device=device)
    initial_state = E3ScanState(
        layers=(
            E3LayerState(
                excitatory=initial_e.clone(), inhibitory=initial_i.clone()
            ),
        )
    )
    with torch.inference_mode():
        full, diagnostics = core.forward_dynamics(tokens, initial_state)
        decay_e, decay_i = core.decays()
        cached_e, cached_i = initial_e, initial_i
        uncached_e, uncached_i = initial_e, initial_i
        cached_outputs = []
        cached_spikes_e = []
        cached_spikes_i = []
        cached_writes_e = []
        cached_writes_i = []
        uncached_outputs = []
        for index in range(tokens.shape[1]):
            cached = core.forward_step_tensors_cached_decay(
                tokens[:, index], cached_e, cached_i, decay_e, decay_i
            )
            uncached = core.forward_step_tensors(
                tokens[:, index], uncached_e, uncached_i
            )
            cached_outputs.append(cached[0])
            uncached_outputs.append(uncached[0])
            cached_e, cached_i = cached[1], cached[2]
            uncached_e, uncached_i = uncached[1], uncached[2]
            cached_spikes_e.append(cached[3])
            cached_spikes_i.append(cached[4])
            cached_writes_e.append(cached[5])
            cached_writes_i.append(cached[6])
    records = {
        "cached_vs_uncached_output": {
            "passed": bool(
                torch.equal(
                    torch.stack(cached_outputs, dim=1),
                    torch.stack(uncached_outputs, dim=1),
                )
            ),
            "max_abs": _max_abs(
                torch.stack(cached_outputs, dim=1),
                torch.stack(uncached_outputs, dim=1),
            ),
        },
        "cached_vs_full_output": {
            "passed": bool(
                torch.allclose(
                    torch.stack(cached_outputs, dim=1),
                    full.sequence,
                    atol=AT1_ATOL,
                    rtol=AT1_RTOL,
                )
            ),
            "max_abs": _max_abs(
                torch.stack(cached_outputs, dim=1), full.sequence
            ),
        },
        "state_e": {
            "passed": bool(
                torch.allclose(
                    cached_e,
                    full.state.layers[0].excitatory,
                    atol=AT1_ATOL,
                    rtol=AT1_RTOL,
                )
            ),
            "max_abs": _max_abs(cached_e, full.state.layers[0].excitatory),
        },
        "state_i": {
            "passed": bool(
                torch.allclose(
                    cached_i,
                    full.state.layers[0].inhibitory,
                    atol=AT1_ATOL,
                    rtol=AT1_RTOL,
                )
            ),
            "max_abs": _max_abs(cached_i, full.state.layers[0].inhibitory),
        },
        "spike_e": {
            "passed": bool(
                torch.equal(
                    torch.stack(cached_spikes_e, dim=1),
                    diagnostics.excitatory_spikes,
                )
            ),
            "max_abs": _max_abs(
                torch.stack(cached_spikes_e, dim=1),
                diagnostics.excitatory_spikes,
            ),
        },
        "spike_i": {
            "passed": bool(
                torch.equal(
                    torch.stack(cached_spikes_i, dim=1),
                    diagnostics.inhibitory_spikes,
                )
            ),
            "max_abs": _max_abs(
                torch.stack(cached_spikes_i, dim=1),
                diagnostics.inhibitory_spikes,
            ),
        },
        "write_e": {
            "passed": bool(
                torch.equal(
                    torch.stack(cached_writes_e, dim=1),
                    diagnostics.excitatory_writes,
                )
            ),
            "max_abs": _max_abs(
                torch.stack(cached_writes_e, dim=1),
                diagnostics.excitatory_writes,
            ),
        },
        "write_i": {
            "passed": bool(
                torch.equal(
                    torch.stack(cached_writes_i, dim=1),
                    diagnostics.inhibitory_writes,
                )
            ),
            "max_abs": _max_abs(
                torch.stack(cached_writes_i, dim=1),
                diagnostics.inhibitory_writes,
            ),
        },
    }
    return {"components": records, "passed": all(v["passed"] for v in records.values())}


def run_equivalence(
    device: torch.device, *, eligibility_forward_mode: str = "segment"
) -> Dict[str, Any]:
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
            seed=9_100_000 + index,
            eligibility_forward_mode=eligibility_forward_mode,
        )
        for index, (batch, time_steps, queries, input_gradient) in enumerate(
            specifications
        )
    ]
    validation = _validation_checks(device)
    cached_step = _cached_step_equivalence(device)
    return {
        "cases": cases,
        "validation": validation,
        "cached_step": cached_step,
        "passed": all(case["passed"] for case in cases)
        and validation["passed"]
        and cached_step["passed"],
    }


def _query_output(
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    if name == "at1_eligibility":
        if not isinstance(core, E3GatedTraceScanCore):  # pragma: no cover
            raise TypeError("AT1 requires E3GatedTraceScanCore")
        return core.forward_multi_query_eligibility(
            value, query_indices
        ).sequence
    if name == "ic0_el1":
        if not isinstance(core, E3InputCodedScanCore):  # pragma: no cover
            raise TypeError("IC0-EL1 requires E3InputCodedScanCore")
        return core.forward_multi_query_eligibility(
            value, query_indices
        ).sequence
    return core(value).sequence.index_select(1, query_indices)


def _saved_tensor_measurement(
    *,
    name: str,
    time_steps: int,
    query_count: int,
    input_gradient: bool,
    device: torch.device,
    seed: int,
    eligibility_forward_mode: str = "segment",
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    core = E3GatedTraceScanCore(
        D_MODEL,
        D_MODEL,
        state_dim=STATE_DIM,
        execution_mode="scan",
        eligibility_forward_mode=eligibility_forward_mode,  # type: ignore[arg-type]
    ).to(device)
    value = torch.randn(
        1, time_steps, D_MODEL, device=device, requires_grad=input_gradient
    )
    query_indices = _even_queries(time_steps, query_count, device)
    counter = _SavedTensorCounter()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        queries = _query_output(name, core, value, query_indices)
        queries.square().mean().backward()
    return {
        "mode": name,
        "time": time_steps,
        "query_count": query_count,
        "query_indices": tuple(int(v) for v in query_indices.tolist()),
        "input_gradient": input_gradient,
        "logical_saved_bytes": counter.logical_bytes,
        "unique_storage_bytes": counter.unique_storage_bytes,
        "saved_tensor_count": counter.tensor_count,
        "autograd_nodes": _autograd_node_count(queries),
    }


def benchmark_saved_tensors(
    device: torch.device, *, eligibility_forward_mode: str = "segment"
) -> Dict[str, Any]:
    length_records = []
    for time_steps in (128, 512, 2048):
        for name, input_gradient in (
            ("at0_bptt", False),
            ("at1_eligibility", False),
            ("at0_bptt", True),
            ("at1_eligibility", True),
        ):
            length_records.append(
                _saved_tensor_measurement(
                    name=name,
                    time_steps=time_steps,
                    query_count=QUERY_COUNT,
                    input_gradient=input_gradient,
                    device=device,
                    seed=9_120_000
                    + time_steps * 10
                    + int(input_gradient),
                    eligibility_forward_mode=eligibility_forward_mode,
                )
            )
    sweep_records = []
    for query_count in (1, 4, 16, 32):
        for name in ("at0_bptt", "at1_eligibility"):
            sweep_records.append(
                _saved_tensor_measurement(
                    name=name,
                    time_steps=512,
                    query_count=query_count,
                    input_gradient=False,
                    device=device,
                    seed=9_121_000 + query_count,
                    eligibility_forward_mode=eligibility_forward_mode,
                )
            )
    lookup = {
        (record["mode"], record["time"], record["input_gradient"]): record
        for record in length_records
    }
    at0_long = lookup[("at0_bptt", 2048, False)]["unique_storage_bytes"]
    at1_long = lookup[("at1_eligibility", 2048, False)]["unique_storage_bytes"]
    at1_short = lookup[("at1_eligibility", 128, False)]["unique_storage_bytes"]
    ratio = at1_long / at0_long
    growth = at1_long / at1_short
    return {
        "length_records": length_records,
        "query_sweep_records": sweep_records,
        "at1_to_at0_t2048_ratio": ratio,
        "at1_t128_to_t2048_growth": growth,
        "passed": ratio <= 0.25 and growth <= 1.25,
    }


def _training_runner(
    *,
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> Runner:
    def run() -> None:
        core.zero_grad(set_to_none=True)
        output = _query_output(name, core, value, query_indices)
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
    lengths: Iterable[int],
    warmup: int,
    repeats: int,
    device: torch.device,
    eligibility_forward_mode: str = "segment",
) -> Dict[str, Any]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for length in lengths:
            torch.manual_seed(9_130_000 + thread_count * 10000 + length)
            at0 = E3GatedTraceScanCore(
                D_MODEL, D_MODEL, state_dim=STATE_DIM, execution_mode="scan"
            ).to(device)
            at1 = E3GatedTraceScanCore(
                D_MODEL,
                D_MODEL,
                state_dim=STATE_DIM,
                execution_mode="scan",
                eligibility_forward_mode=eligibility_forward_mode,  # type: ignore[arg-type]
            ).to(device)
            at1.load_state_dict(at0.state_dict())
            cores: Dict[str, TemporalCore[Any]] = {
                "at0_bptt": at0,
                "at1_eligibility": at1,
                "ic0_el1": E3InputCodedScanCore(
                    D_MODEL, D_MODEL, state_dim=42
                ).to(device),
                "lstm": StatefulLSTMCore(D_MODEL, D_MODEL).to(device),
                "transformer": CausalTransformerCore(
                    D_MODEL,
                    D_MODEL,
                    num_layers=1,
                    num_heads=4,
                    mlp_ratio=2.0,
                    dropout=0.0,
                    max_cache_tokens=length,
                ).to(device),
            }
            value = torch.randn(1, length, D_MODEL, device=device)
            query_indices = _even_queries(length, QUERY_COUNT, device)
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
                    _query_output(name, core, value, query_indices)
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
                seed=9_131_000 + thread_count * 10000 + length,
            )
            models = {
                name: {
                    **_sample_summary(sample, length),
                    "parameters": count_parameters(cores[name]),
                    "autograd_nodes": nodes[name],
                }
                for name, sample in samples.items()
            }
            speedup = (
                models["at0_bptt"]["p50_ms"]
                / models["at1_eligibility"]["p50_ms"]
            )
            passed = (
                speedup >= 1.25
                and models["at1_eligibility"]["p50_ms"]
                <= models["lstm"]["p50_ms"]
            )
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "time": length,
                    "query_count": QUERY_COUNT,
                    "models": models,
                    "at1_vs_at0_speedup": speedup,
                    "passed": passed,
                }
            )
    return {"records": records, "passed": any(item["passed"] for item in records)}


class _AT1CachedStepRunner:
    def __init__(self, core: E3GatedTraceScanCore, tokens: torch.Tensor) -> None:
        self.core = core
        self.tokens = tokens
        with torch.inference_mode():
            state = core.initial_state(tokens.shape[1], device=tokens.device)
            self.decay_e, self.decay_i = core.decays()
        self.excitatory = state.layers[0].excitatory
        self.inhibitory = state.layers[0].inhibitory
        self.index = 0

    def __call__(self) -> Tuple[torch.Tensor, ...]:
        output = self.core.forward_step_tensors_cached_decay(
            self.tokens[self.index],
            self.excitatory,
            self.inhibitory,
            self.decay_e,
            self.decay_i,
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
    if isinstance(
        runner,
        (_AT1CachedStepRunner, _AT0TensorStepRunner, _IC0TensorStepRunner),
    ):
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
        seed = 9_140_000 + thread_count
        torch.manual_seed(seed)
        at1 = E3GatedTraceScanCore(
            D_MODEL, D_MODEL, state_dim=STATE_DIM
        ).to(device).eval()
        ic0 = E3InputCodedScanCore(
            D_MODEL, D_MODEL, state_dim=42
        ).to(device).eval()
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
                warmup_steps + measured_steps, 1, D_MODEL, device=device
            )
        runners: Dict[str, Any] = {
            "at1_cached_step": _AT1CachedStepRunner(at1, tokens),
            "at0_uncached_step": _AT0TensorStepRunner(at1, tokens),
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
            "at1_cached_step": at1,
            "at0_uncached_step": at1,
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
        cached = models["at1_cached_step"]
        lstm_metrics = models["lstm_step"]
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "models": models,
                "at1_to_lstm_p50_ratio": cached["p50_ms"]
                / lstm_metrics["p50_ms"],
                "at1_to_lstm_p95_ratio": cached["p95_ms"]
                / lstm_metrics["p95_ms"],
                "passed": cached["p50_ms"] <= lstm_metrics["p50_ms"]
                and cached["p95_ms"] <= lstm_metrics["p95_ms"],
            }
        )
    return {"records": records, "passed": any(item["passed"] for item in records)}


class AT1TokenModel(nn.Module):
    def __init__(
        self,
        core: TemporalCore[Any],
        *,
        use_trace_eligibility: bool = False,
    ) -> None:
        super().__init__()
        self.core = core
        self.use_trace_eligibility = bool(use_trace_eligibility)
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
        if self.use_trace_eligibility and self.training:
            if not isinstance(self.core, E3GatedTraceScanCore):  # pragma: no cover
                raise TypeError("trace eligibility requires the gated-trace core")
            sequence = self.core.forward_multi_query_eligibility(
                embedded, self.query_indices
            ).sequence
        else:
            sequence = self.core(embedded).sequence.index_select(
                1, self.query_indices
            )
        return self.decoder(self.output_norm(sequence))


def _build_quality_models(
    seed: int, device: torch.device
) -> Dict[str, nn.Module]:
    shared = _shared_wrapper_state(8_930_001)
    torch.manual_seed(seed)
    at0 = AT1TokenModel(
        E3GatedTraceScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    )
    _initialise_shared_wrapper(at0, shared)  # type: ignore[arg-type]
    at1 = copy.deepcopy(at0)
    at1.use_trace_eligibility = True
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
        "at0_bptt": at0.to(device),
        "at1_eligibility": at1.to(device),
        "lstm": lstm.to(device),
        "transformer": transformer.to(device),
    }


def _evaluate_ablation(
    model: AT1TokenModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("ablation requires gated-trace core")
    if mode not in ("spike_only", "trace_only"):
        raise ValueError(f"unknown ablation mode: {mode}")
    model.eval()
    correct = 0
    total = 0
    nll = 0.0
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            batch_tokens = tokens[start : start + batch_size].to(device)
            batch_targets = targets[start : start + batch_size].to(device)
            embedded = model.embedding(batch_tokens)
            _result, trace = model.core.forward_dynamics(embedded)
            zeros_e = torch.zeros_like(trace.excitatory_spikes)
            zeros_i = torch.zeros_like(trace.inhibitory_spikes)
            if mode == "spike_only":
                raw = torch.cat(
                    (
                        trace.excitatory_spikes,
                        -trace.inhibitory_spikes,
                        torch.zeros_like(trace.excitatory_traces),
                        torch.zeros_like(trace.inhibitory_traces),
                    ),
                    dim=-1,
                )
            else:
                raw = torch.cat(
                    (
                        zeros_e,
                        -zeros_i,
                        trace.excitatory_traces,
                        -trace.inhibitory_traces,
                    ),
                    dim=-1,
                )
            sequence = model.core.output_projection(model.core.output_norm(raw))
            queries = sequence.index_select(1, model.query_indices)
            logits = model.decoder(model.output_norm(queries))
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
            seed=8_990_000 + seed, batch_size=test_count
        )
        models = _build_quality_models(9_150_000 + 100 * seed, device)
        parameter_counts = {
            name: count_parameters(model) for name, model in models.items()
        }
        lstm_count = parameter_counts["lstm"]
        fairness = {
            name: abs(count - lstm_count) / lstm_count <= 0.02
            for name, count in parameter_counts.items()
        }
        if not all(fairness.values()):
            raise AssertionError(f"AT1 parameter fairness failed: {parameter_counts}")
        model_results: Dict[str, Dict[str, Any]] = {}
        for name in ("at0_bptt", "at1_eligibility", "lstm", "transformer"):
            train = _train_model(
                models[name],  # type: ignore[arg-type]
                train_batches,
                timing_warmup=min(100, updates - 1),
                device=device,
            )
            model_results[name] = {
                "train": train,
                "test": _evaluate_model(
                    models[name],  # type: ignore[arg-type]
                    test_tokens,
                    test_targets,
                    batch_size=256,
                    device=device,
                ),
            }
        at0_losses = model_results["at0_bptt"]["train"]["losses"]
        at1_losses = model_results["at1_eligibility"]["train"]["losses"]
        loss_max_abs = max(
            abs(left - right) for left, right in zip(at0_losses, at1_losses)
        )
        parameter_max_abs, parameter_errors = _parameter_max_abs(
            models["at1_eligibility"], models["at0_bptt"]
        )
        for result in model_results.values():
            result["train"].pop("losses")
        at1_model = models["at1_eligibility"]
        if not isinstance(at1_model, AT1TokenModel):  # pragma: no cover
            raise TypeError("AT1 quality model mismatch")
        ablations = {
            mode: _evaluate_ablation(
                at1_model,
                test_tokens,
                test_targets,
                mode=mode,
                batch_size=256,
                device=device,
            )
            for mode in ("spike_only", "trace_only")
        }
        records.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_fairness": fairness,
                "train_data_sha256": _dataset_hash(train_batches),
                "models": model_results,
                "at1_vs_at0_loss_max_abs": loss_max_abs,
                "at1_vs_at0_parameter_max_abs": parameter_max_abs,
                "at1_vs_at0_parameter_errors": parameter_errors,
                "event_diagnostics": _event_diagnostics(
                    at1_model, test_tokens[:256], device=device  # type: ignore[arg-type]
                ),
                "mechanism_ablation": ablations,
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
            record["models"][name]["test"]["accuracy"] == 1.0
            for record in records
            for name in ("at0_bptt", "at1_eligibility")
        ) and all(
            record["at1_vs_at0_loss_max_abs"] <= 1e-3
            and record["at1_vs_at0_parameter_max_abs"] <= 5e-3
            for record in records
        )
    return {
        "formal": not quick,
        "task": {
            "sequence_length": SEQUENCE_LENGTH,
            "query_indices": QUERY_INDICES,
            "delay": 4,
            "payload_classes": PAYLOAD_VOCAB,
            "frozen_orthogonal_embedding": True,
            "updates": updates,
            "train_batch_size": train_batch_size,
            "test_sequences_per_seed": test_count,
            "test_query_targets_per_seed": test_count * len(QUERY_INDICES),
        },
        "seeds": records,
        "task_validation": "PASS" if task_valid else "NOT_RUN" if quick else "FAIL",
        "passed": quality_pass,
    }


def _decision(
    *,
    equivalence: Mapping[str, Any],
    memory: Mapping[str, Any],
    training: Mapping[str, Any],
    streaming: Mapping[str, Any],
    quality: Mapping[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
            "memory_gate": "PASS" if memory["passed"] else "FAIL",
            "speed_gate": "SMOKE",
            "stream_gate": "SMOKE",
            "quality_gate": "NOT_RUN",
            "ann_gate": "SMOKE",
            "overall": "SMOKE",
            "run_textworld_next": False,
        }
    stream_by_thread = {
        record["threads"]: record["passed"] for record in streaming["records"]
    }
    ann_pass = any(
        record["passed"] and stream_by_thread.get(record["threads"], False)
        for record in training["records"]
    )
    gates = {
        "equivalence_gate": bool(equivalence["passed"]),
        "memory_gate": bool(memory["passed"]),
        "speed_gate": bool(training["passed"]),
        "stream_gate": bool(streaming["passed"]),
        "quality_gate": bool(quality["passed"]),
        "ann_gate": ann_pass,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "run_textworld_next": overall == "PASS",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_at1_trace_eligibility.json"),
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
        memory = benchmark_saved_tensors(device)
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
            torch.set_num_threads(threads[0] if args.quick else 4)
        quality = run_quality(quick=args.quick, device=device)
    else:
        memory = {"records": [], "passed": False, "not_run": "EQ failed"}
        training = {"records": [], "passed": False, "not_run": "EQ failed"}
        streaming = {"records": [], "passed": False, "not_run": "EQ failed"}
        quality = {"passed": False, "not_run": "EQ failed"}
    decision = _decision(
        equivalence=equivalence,
        memory=memory,
        training=training,
        streaming=streaming,
        quality=quality,
        quick=args.quick,
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-AT1 exact gated-trace eligibility and cached decay",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "query_count": QUERY_COUNT,
            "threads": threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "stream_warmup": args.stream_warmup,
            "stream_steps": args.stream_steps,
            "atol": AT1_ATOL,
            "rtol": AT1_RTOL,
        },
        "equivalence": equivalence,
        "saved_tensors": memory,
        "training": training,
        "streaming": streaming,
        "quality": quality,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
