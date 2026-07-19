"""Exact IC0 tensor-step and torch.compile streaming benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _percentile,
    _sample_summary,
    _sync,
)
from vpsc.world_model.cores import (  # noqa: E402
    E3InputCodedScanCore,
    E3LayerState,
    E3ScanState,
    StatefulLSTMCore,
    count_parameters,
    state_nbytes,
)


D_MODEL = 32
STATE_DIM = 42
EK0_ATOL = 2e-6
EK0_RTOL = 1e-5
TensorStep = Callable[[], Tuple[torch.Tensor, ...]]


class IC0TensorCell(nn.Module):
    """Tensor-only wrapper used as the fullgraph compilation boundary."""

    def __init__(self, core: E3InputCodedScanCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        x_t: torch.Tensor,
        excitatory: torch.Tensor,
        inhibitory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.core._forward_step_tensors_unchecked(x_t, excitatory, inhibitory)


def _compile_cell(
    core: E3InputCodedScanCore,
    *,
    batch_size: int,
    device: torch.device,
) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    cell = IC0TensorCell(core).eval()
    with torch.inference_mode():
        x_t = torch.zeros(batch_size, D_MODEL, device=device)
        state = core.initial_state(batch_size, device=device)
    started = time.perf_counter_ns()
    try:
        compiled = torch.compile(cell, fullgraph=True, mode="reduce-overhead")
        with torch.inference_mode():
            compiled(
                x_t,
                state.layers[0].excitatory,
                state.layers[0].inhibitory,
            )
        _sync(device)
    except Exception as error:  # pragma: no cover - environment dependent
        return None, {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "first_call_ms": (time.perf_counter_ns() - started) / 1e6,
        }
    return compiled, {
        "status": "PASS",
        "fullgraph": True,
        "mode": "reduce-overhead",
        "first_call_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def _run_tensor_stream(
    cell: nn.Module,
    tokens: torch.Tensor,
    initial_e: torch.Tensor,
    initial_i: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    excitatory = initial_e
    inhibitory = initial_i
    outputs = []
    spikes_e = []
    spikes_i = []
    with torch.inference_mode():
        for index in range(tokens.shape[1]):
            output, excitatory, inhibitory, spike_e, spike_i = cell(
                tokens[:, index], excitatory, inhibitory
            )
            outputs.append(output)
            spikes_e.append(spike_e)
            spikes_i.append(spike_i)
    return (
        torch.stack(outputs, dim=1),
        excitatory,
        inhibitory,
        torch.stack(spikes_e, dim=1),
        torch.stack(spikes_i, dim=1),
    )


def _run_generic_stream(
    core: E3InputCodedScanCore,
    tokens: torch.Tensor,
    initial_e: torch.Tensor,
    initial_i: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = E3ScanState(
        layers=(
            E3LayerState(
                excitatory=initial_e,
                inhibitory=initial_i,
            ),
        )
    )
    outputs = []
    spikes_e = []
    spikes_i = []
    with torch.inference_mode():
        for index in range(tokens.shape[1]):
            result, traces = core.forward_dynamics(tokens[:, index : index + 1], state)
            outputs.append(result.sequence[:, 0])
            spikes_e.append(traces[0].excitatory_spikes[:, 0])
            spikes_i.append(traces[0].inhibitory_spikes[:, 0])
            state = result.state
    return (
        torch.stack(outputs, dim=1),
        state.layers[0].excitatory,
        state.layers[0].inhibitory,
        torch.stack(spikes_e, dim=1),
        torch.stack(spikes_i, dim=1),
    )


def _comparison(
    candidate: Tuple[torch.Tensor, ...], reference: Tuple[torch.Tensor, ...]
) -> Dict[str, Any]:
    names = ("sequence", "state_e", "state_i", "spikes_e", "spikes_i")
    records = {}
    for index, name in enumerate(names):
        left = candidate[index]
        right = reference[index]
        bit_exact = name != "sequence"
        records[name] = {
            "passed": bool(
                torch.equal(left, right)
                if bit_exact
                else torch.allclose(left, right, atol=EK0_ATOL, rtol=EK0_RTOL)
            ),
            "max_abs": float((left - right).abs().max().item()),
        }
    return {
        "components": records,
        "passed": all(record["passed"] for record in records.values()),
    }


def run_equivalence(
    *, device: torch.device, time_steps: int, batches: Sequence[int]
) -> Dict[str, Any]:
    cases = []
    compile_records = []
    for case_index, batch_size in enumerate(batches):
        torch.manual_seed(8_800_000 + case_index)
        generic_core = E3InputCodedScanCore(
            D_MODEL, D_MODEL, state_dim=STATE_DIM
        ).to(device)
        tensor_core = E3InputCodedScanCore(
            D_MODEL, D_MODEL, state_dim=STATE_DIM
        ).to(device)
        compiled_core = E3InputCodedScanCore(
            D_MODEL, D_MODEL, state_dim=STATE_DIM
        ).to(device)
        tensor_core.load_state_dict(generic_core.state_dict())
        compiled_core.load_state_dict(generic_core.state_dict())
        tensor_cell = IC0TensorCell(tensor_core).eval()
        compiled_cell, compile_record = _compile_cell(
            compiled_core, batch_size=batch_size, device=device
        )
        compile_records.append({"batch": batch_size, **compile_record})
        if compiled_cell is None:
            cases.append(
                {
                    "batch": batch_size,
                    "time": time_steps,
                    "passed": False,
                    "not_run": "fullgraph compilation failed",
                }
            )
            continue
        with torch.inference_mode():
            tokens = torch.randn(batch_size, time_steps, D_MODEL, device=device)
            initial_e = torch.rand(batch_size, STATE_DIM, device=device)
            initial_i = torch.rand(batch_size, STATE_DIM, device=device)
        generic = _run_generic_stream(
            generic_core,
            tokens,
            initial_e.clone(),
            initial_i.clone(),
        )
        eager = _run_tensor_stream(
            tensor_cell,
            tokens,
            initial_e.clone(),
            initial_i.clone(),
        )
        compiled = _run_tensor_stream(
            compiled_cell,
            tokens,
            initial_e.clone(),
            initial_i.clone(),
        )
        eager_comparison = _comparison(eager, generic)
        compiled_comparison = _comparison(compiled, generic)
        binary = bool(
            torch.all((compiled[3] == 0.0) | (compiled[3] == 1.0))
            and torch.all((compiled[4] == 0.0) | (compiled[4] == 1.0))
        )
        bounded = bool(
            torch.all((compiled[1] >= 0.0) & (compiled[1] < 1.0))
            and torch.all((compiled[2] >= 0.0) & (compiled[2] < 1.0))
        )
        cases.append(
            {
                "batch": batch_size,
                "time": time_steps,
                "tensor_eager_vs_generic": eager_comparison,
                "compiled_vs_generic": compiled_comparison,
                "binary_spikes": binary,
                "bounded_residuals": bounded,
                "passed": eager_comparison["passed"]
                and compiled_comparison["passed"]
                and binary
                and bounded,
            }
        )
    return {
        "compile": compile_records,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


class _GenericIC0Runner:
    def __init__(self, core: E3InputCodedScanCore, tokens: torch.Tensor) -> None:
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


class _TensorIC0Runner:
    def __init__(
        self,
        cell: nn.Module,
        core: E3InputCodedScanCore,
        tokens: torch.Tensor,
    ) -> None:
        self.cell = cell
        self.tokens = tokens
        with torch.inference_mode():
            state = core.initial_state(tokens.shape[1], device=tokens.device)
        self.excitatory = state.layers[0].excitatory
        self.inhibitory = state.layers[0].inhibitory
        self.index = 0

    def __call__(self) -> Tuple[torch.Tensor, ...]:
        output = self.cell(
            self.tokens[self.index], self.excitatory, self.inhibitory
        )
        self.excitatory = output[1]
        self.inhibitory = output[2]
        self.index += 1
        return output


class _LSTMRunner:
    def __init__(self, core: StatefulLSTMCore, tokens: torch.Tensor) -> None:
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


def _time_call(runner: Callable[[], Tuple[torch.Tensor, ...]], device: torch.device) -> float:
    _sync(device)
    started = time.perf_counter_ns()
    output = runner()
    if output:
        output[0].sum().item()
    _sync(device)
    return (time.perf_counter_ns() - started) / 1e6


def _latency_summary(samples: Sequence[float]) -> Dict[str, float]:
    summary = _sample_summary(samples, 1)
    summary["p99_ms"] = _percentile(samples, 0.99)
    return summary


def _state_bytes(runner: Any) -> int:
    if isinstance(runner, _GenericIC0Runner):
        return state_nbytes(runner.state)
    if isinstance(runner, _TensorIC0Runner):
        return sum(
            value.numel() * value.element_size()
            for value in (runner.excitatory, runner.inhibitory)
        )
    if isinstance(runner, _LSTMRunner):
        return state_nbytes(runner.state)
    raise TypeError(type(runner))  # pragma: no cover


def benchmark_streaming(
    *,
    threads: Sequence[int],
    batches: Sequence[int],
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    records = []
    compile_records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for batch_size in batches:
            seed = 8_810_000 + thread_count * 1000 + batch_size
            torch.manual_seed(seed)
            generic_core = E3InputCodedScanCore(
                D_MODEL, D_MODEL, state_dim=STATE_DIM
            ).to(device).eval()
            tensor_core = E3InputCodedScanCore(
                D_MODEL, D_MODEL, state_dim=STATE_DIM
            ).to(device).eval()
            compiled_core = E3InputCodedScanCore(
                D_MODEL, D_MODEL, state_dim=STATE_DIM
            ).to(device).eval()
            tensor_core.load_state_dict(generic_core.state_dict())
            compiled_core.load_state_dict(generic_core.state_dict())
            lstm = StatefulLSTMCore(D_MODEL, D_MODEL).to(device).eval()
            compiled_cell, compile_record = _compile_cell(
                compiled_core, batch_size=batch_size, device=device
            )
            compile_records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "batch": batch_size,
                    **compile_record,
                }
            )
            if compiled_cell is None:
                records.append(
                    {
                        "threads": thread_count,
                        "batch": batch_size,
                        "passed": False,
                        "not_run": "fullgraph compilation failed",
                    }
                )
                continue
            with torch.inference_mode():
                tokens = torch.randn(
                    warmup_steps + measured_steps,
                    batch_size,
                    D_MODEL,
                    device=device,
                )
            runners: Dict[str, Any] = {
                "ic0_generic": _GenericIC0Runner(generic_core, tokens),
                "ic0_tensor_eager": _TensorIC0Runner(
                    IC0TensorCell(tensor_core).eval(), tensor_core, tokens
                ),
                "ic0_tensor_compiled": _TensorIC0Runner(
                    compiled_cell, compiled_core, tokens
                ),
                "lstm_fused_eager": _LSTMRunner(lstm, tokens),
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
                        samples[name].append(_time_call(runners[name], device))
            models = {
                name: {
                    **_latency_summary(sample),
                    "state_bytes": _state_bytes(runners[name]),
                    "parameters": count_parameters(
                        generic_core
                        if name == "ic0_generic"
                        else tensor_core
                        if name == "ic0_tensor_eager"
                        else compiled_core
                        if name == "ic0_tensor_compiled"
                        else lstm
                    ),
                }
                for name, sample in samples.items()
            }
            compiled_metrics = models["ic0_tensor_compiled"]
            lstm_metrics = models["lstm_fused_eager"]
            passed = (
                batch_size == 1
                and compiled_metrics["p50_ms"] <= lstm_metrics["p50_ms"]
                and compiled_metrics["p95_ms"] <= lstm_metrics["p95_ms"]
            )
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "batch": batch_size,
                    "models": models,
                    "generic_to_tensor_eager_p50_speedup": models["ic0_generic"][
                        "p50_ms"
                    ]
                    / models["ic0_tensor_eager"]["p50_ms"],
                    "tensor_eager_to_compiled_p50_speedup": models[
                        "ic0_tensor_eager"
                    ]["p50_ms"]
                    / compiled_metrics["p50_ms"],
                    "compiled_to_lstm_p50_ratio": compiled_metrics["p50_ms"]
                    / lstm_metrics["p50_ms"],
                    "compiled_to_lstm_p95_ratio": compiled_metrics["p95_ms"]
                    / lstm_metrics["p95_ms"],
                    "passed": passed,
                }
            )
    return {
        "compile": compile_records,
        "records": records,
        "passed": any(record["passed"] for record in records),
    }


def _decision(
    equivalence: Dict[str, Any], streaming: Dict[str, Any]
) -> Dict[str, Any]:
    gates = {
        "equivalence_gate": bool(equivalence["passed"]),
        "realtime_gate": bool(streaming["passed"]),
    }
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_ek0_compiled_streaming.json"),
    )
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--measured-steps", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.threads = args.threads[:1]
        args.warmup_steps = 4
        args.measured_steps = 32
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
    batches = (1,) if args.quick else (1, 8)
    equivalence = run_equivalence(
        device=device,
        time_steps=32 if args.quick else 512,
        batches=batches,
    )
    if equivalence["passed"]:
        streaming = benchmark_streaming(
            threads=threads,
            batches=batches,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            device=device,
        )
    else:
        streaming = {
            "compile": [],
            "records": [],
            "passed": False,
            "not_run": "equivalence failed",
        }
    decision = _decision(equivalence, streaming)
    result = {
        "schema_version": 1,
        "experiment": "E3-EK0 exact compiled tensor streaming",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "threads": threads,
            "batches": batches,
            "equivalence_steps": 32 if args.quick else 512,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "atol": EK0_ATOL,
            "rtol": EK0_RTOL,
        },
        "equivalence": equivalence,
        "streaming": streaming,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
