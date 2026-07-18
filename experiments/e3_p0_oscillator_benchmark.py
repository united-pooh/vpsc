"""P0 selective complex oscillator scan equivalence, speed, and streaming gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _autograd_node_count,
    _core_training_runner,
    _cuda_peak_bytes,
    _environment,
    _interleaved_samples,
    _max_abs,
    _sample_summary,
    _sync,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3CumulativeScanCore,
    E3OscillatorState,
    E3OscillatoryScanCore,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
    state_nbytes,
)


ATOL = 3e-5
RTOL = 1e-4
STATE_DIM = 31
SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (1, 32, 32),
    (8, 32, 32),
    (1, 512, 32),
    (1, 2048, 32),
)


def _close(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))


def _grad_close(
    left: torch.Tensor | None, right: torch.Tensor | None
) -> Tuple[bool, float]:
    if left is None or right is None:
        return left is None and right is None, 0.0
    return _close(left, right), _max_abs(left, right)


def _equivalence_case(
    *, batch: int, time_steps: int, seed: int, device: torch.device
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    serial = E3OscillatoryScanCore(
        4, 6, state_dim=5, execution_mode="serial"
    ).to(device)
    scan = E3OscillatoryScanCore(4, 6, state_dim=5, execution_mode="scan").to(device)
    scan.load_state_dict(serial.state_dict())
    serial_input = torch.randn(batch, time_steps, 4, device=device, requires_grad=True)
    scan_input = serial_input.detach().clone().requires_grad_(True)
    real = 0.1 * torch.randn(batch, 5, device=device)
    imag = 0.1 * torch.randn(batch, 5, device=device)
    initial = torch.complex(real, imag)
    serial_state = E3OscillatorState(
        value=initial.detach().clone().requires_grad_(True)
    )
    scan_state = E3OscillatorState(value=initial.detach().clone().requires_grad_(True))
    serial_result, serial_trace = serial.forward_dynamics(serial_input, serial_state)
    scan_result, scan_trace = scan.forward_dynamics(scan_input, scan_state)
    checks = {
        "sequence": _close(scan_result.sequence, serial_result.sequence),
        "state": _close(scan_result.state.value, serial_result.state.value),
        "e_spikes_exact": bool(
            torch.equal(scan_trace.excitatory_spikes, serial_trace.excitatory_spikes)
        ),
        "i_spikes_exact": bool(
            torch.equal(scan_trace.inhibitory_spikes, serial_trace.inhibitory_spikes)
        ),
        "parameter_count": count_parameters(scan) == count_parameters(serial),
        "state_bytes": state_nbytes(scan_result.state) == state_nbytes(serial_result.state),
    }
    errors = {
        "sequence": _max_abs(scan_result.sequence, serial_result.sequence),
        "state": _max_abs(scan_result.state.value, serial_result.state.value),
        "values": _max_abs(scan_trace.values, serial_trace.values),
    }
    probe = torch.linspace(
        -0.7, 0.9, serial_result.sequence.numel(), device=device
    ).reshape_as(serial_result.sequence)
    serial_loss = (serial_result.sequence * probe).mean() + 0.03 * (
        serial_result.state.value.real.mean() - serial_result.state.value.imag.mean()
    )
    scan_loss = (scan_result.sequence * probe).mean() + 0.03 * (
        scan_result.state.value.real.mean() - scan_result.state.value.imag.mean()
    )
    serial_loss.backward()
    scan_loss.backward()
    checks["input_gradient"], errors["input_gradient"] = _grad_close(
        scan_input.grad, serial_input.grad
    )
    checks["state_gradient"], errors["state_gradient"] = _grad_close(
        scan_state.value.grad, serial_state.value.grad
    )
    serial_parameters = dict(serial.named_parameters())
    parameter_errors = {}
    for name, parameter in scan.named_parameters():
        passed, error = _grad_close(parameter.grad, serial_parameters[name].grad)
        checks[f"parameter_gradient_{name}"] = passed
        parameter_errors[name] = error
    errors["parameter_gradient_max"] = max(parameter_errors.values())

    with torch.no_grad():
        full, _ = scan.forward_dynamics(scan_input.detach(), scan_state)
        stream_state: E3OscillatorState | None = scan_state
        pieces = []
        for index in range(time_steps):
            stepped = scan.step(scan_input.detach()[:, index], stream_state)
            pieces.append(stepped.sequence)
            stream_state = stepped.state
        streamed = torch.cat(pieces, dim=1)
    checks["streaming_sequence"] = _close(streamed, full.sequence)
    errors["streaming_sequence"] = _max_abs(streamed, full.sequence)
    assert stream_state is not None
    checks["streaming_state"] = _close(stream_state.value, full.state.value)
    errors["streaming_state"] = _max_abs(stream_state.value, full.state.value)
    return {
        "batch": batch,
        "time": time_steps,
        "seed": seed,
        "checks": checks,
        "max_abs_errors": errors,
        "parameter_gradient_max_abs": parameter_errors,
        "passed": all(checks.values()),
    }


def run_equivalence(device: torch.device) -> Dict[str, Any]:
    cases = tuple(
        _equivalence_case(
            batch=batch,
            time_steps=time_steps,
            seed=13_000 + index,
            device=device,
        )
        for index, (batch, time_steps) in enumerate(((1, 1), (4, 32), (1, 512)))
    )
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def _suite(
    dimension: int, time_steps: int, device: torch.device, seed: int
) -> Dict[str, TemporalCore[Any]]:
    torch.manual_seed(seed)
    serial = E3OscillatoryScanCore(
        dimension,
        dimension,
        state_dim=STATE_DIM,
        execution_mode="serial",
    )
    scan = E3OscillatoryScanCore(
        dimension,
        dimension,
        state_dim=STATE_DIM,
        execution_mode="scan",
    )
    scan.load_state_dict(serial.state_dict())
    cores: Dict[str, TemporalCore[Any]] = {
        "p0_serial": serial,
        "p0_scan": scan,
        "s0_scan": E3CumulativeScanCore(
            dimension,
            dimension,
            state_dim=27,
            num_layers=2,
            execution_mode="scan",
        ),
        "lstm": StatefulLSTMCore(dimension, dimension),
        "transformer": CausalTransformerCore(
            dimension,
            dimension,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=max(128, time_steps),
        ),
    }
    return {name: core.to(device).train(True) for name, core in cores.items()}


def benchmark_training(
    *,
    shapes: Iterable[Tuple[int, int, int]],
    threads: Sequence[int],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for shape_index, (batch, time_steps, dimension) in enumerate(shapes):
            cores = _suite(dimension, time_steps, device, 14_000 + shape_index)
            base = torch.randn(batch, time_steps, dimension, device=device)
            values = {
                name: base.detach().clone().requires_grad_(True) for name in cores
            }
            runners = {
                name: _core_training_runner(core, values[name])
                for name, core in cores.items()
            }
            nodes = {}
            for name, core in cores.items():
                output = core(values[name])
                nodes[name] = _autograd_node_count(output.sequence)
                core.zero_grad(set_to_none=True)
                values[name].grad = None
            samples = _interleaved_samples(
                runners,
                warmup=warmup,
                repeats=repeats,
                device=device,
                seed=15_000 + thread_count * 100 + shape_index,
            )
            summaries = {
                name: {
                    **_sample_summary(sample, batch * time_steps),
                    "parameters": count_parameters(cores[name]),
                    "state_bytes": state_nbytes(cores[name].initial_state(batch)),
                    "autograd_nodes": nodes[name],
                    "peak_memory_bytes": _cuda_peak_bytes(runners[name], device),
                }
                for name, sample in samples.items()
            }
            serial_p50 = summaries["p0_serial"]["p50_ms"]
            for summary in summaries.values():
                summary["versus_p0_serial_speedup"] = serial_p50 / summary["p50_ms"]
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "batch": batch,
                    "time": time_steps,
                    "dimension": dimension,
                    "models": summaries,
                }
            )
    return records


def benchmark_streaming(
    *,
    threads: Sequence[int],
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        cores = _suite(32, 128, device, 16_000 + thread_count)
        for core in cores.values():
            core.eval()
        tokens = torch.randn(warmup_steps + measured_steps, 1, 32, device=device)
        states: Dict[str, Any] = {name: None for name in cores}
        samples: Dict[str, list[float]] = {name: [] for name in cores}
        generator = random.Random(17_000 + thread_count)
        with torch.inference_mode():
            for index in range(warmup_steps):
                for name, core in cores.items():
                    states[name] = core.step(tokens[index], states[name]).state
            names = list(cores)
            for index in range(warmup_steps, warmup_steps + measured_steps):
                generator.shuffle(names)
                for name in names:
                    _sync(device)
                    started = time.perf_counter_ns()
                    result = cores[name].step(tokens[index], states[name])
                    _sync(device)
                    samples[name].append((time.perf_counter_ns() - started) / 1e6)
                    states[name] = result.state
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "models": {
                    name: {
                        **_sample_summary(sample, 1),
                        "state_bytes": state_nbytes(states[name]),
                    }
                    for name, sample in samples.items()
                },
            }
        )
    return records


def _decision(
    equivalence: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    streaming: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    t512 = [
        record
        for record in training
        if (record["batch"], record["time"], record["dimension"]) == (1, 512, 32)
    ]
    parallel = {}
    for record in t512:
        serial = record["models"]["p0_serial"]
        scan = record["models"]["p0_scan"]
        parallel[str(record["threads"])] = {
            "speedup": scan["versus_p0_serial_speedup"],
            "node_ratio": scan["autograd_nodes"] / serial["autograd_nodes"],
            "passed": scan["versus_p0_serial_speedup"] >= 10.0
            and scan["autograd_nodes"] <= 0.25 * serial["autograd_nodes"],
        }
    parallel_pass = bool(equivalence["passed"]) and any(
        value["passed"] for value in parallel.values()
    )
    stream_by_thread = {str(record["threads"]): record for record in streaming}
    ann = {}
    for record in t512:
        key = str(record["threads"])
        stream = stream_by_thread[key]
        ann[key] = {
            "train_p0_ms": record["models"]["p0_scan"]["p50_ms"],
            "train_lstm_ms": record["models"]["lstm"]["p50_ms"],
            "stream_p0_p95_ms": stream["models"]["p0_scan"]["p95_ms"],
            "stream_lstm_p95_ms": stream["models"]["lstm"]["p95_ms"],
        }
        ann[key]["passed"] = (
            ann[key]["train_p0_ms"] <= ann[key]["train_lstm_ms"]
            and ann[key]["stream_p0_p95_ms"] <= ann[key]["stream_lstm_p95_ms"]
        )
    ann_pass = bool(equivalence["passed"]) and any(value["passed"] for value in ann.values())
    return {
        "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
        "parallel_training_gate": "PASS" if parallel_pass else "FAIL",
        "ann_train_and_inference_speed_gate": "PASS" if ann_pass else "FAIL",
        "parallel_checks": parallel,
        "ann_checks": ann,
        "run_a0_next": bool(equivalence["passed"] and parallel_pass),
        "boundary": "P0 is reset-free even if every measured gate passes.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_p0_oscillator_benchmark.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--streaming-warmup", type=int, default=16)
    parser.add_argument("--streaming-steps", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (
            *args.threads,
            args.warmup,
            args.repeats,
            args.streaming_warmup,
            args.streaming_steps,
        )
    ):
        parser.error("threads, warmup, repeats, and streaming steps must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    resolved = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(resolved)
    threads = tuple(dict.fromkeys(args.threads))
    shapes = ((1, 16, 32),) if args.quick else SHAPES
    if args.quick:
        threads = threads[:1]
    if device.type == "cpu":
        torch.set_num_threads(threads[0])
    equivalence = run_equivalence(device)
    training = benchmark_training(
        shapes=shapes,
        threads=threads,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    streaming = benchmark_streaming(
        threads=threads,
        warmup_steps=args.streaming_warmup,
        measured_steps=args.streaming_steps,
        device=device,
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-P0 selective complex oscillator scan",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "state_dim": STATE_DIM,
            "threads": threads,
            "shapes": shapes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "streaming_warmup": args.streaming_warmup,
            "streaming_steps": args.streaming_steps,
        },
        "equivalence": equivalence,
        "training_forward_backward": training,
        "streaming_inference": streaming,
        "decision": _decision(equivalence, training, streaming),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
