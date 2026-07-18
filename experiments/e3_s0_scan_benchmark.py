"""Formal E3-S0 equivalence, time-parallel speed, and streaming benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
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
    E2SignedCore,
    E3CumulativeScanCore,
    E3LayerState,
    E3ScanState,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
    state_nbytes,
)


Tensor = torch.Tensor
ATOL = 2e-6
RTOL = 1e-5
STATE_DIM = 27
FORMAL_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (1, 32, 32),
    (8, 32, 32),
    (1, 512, 32),
    (1, 2048, 32),
)


def _close(left: Tensor, right: Tensor) -> bool:
    return bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))


def _grad_close(left: Tensor | None, right: Tensor | None) -> Tuple[bool, float]:
    if left is None or right is None:
        return left is None and right is None, 0.0
    return _close(left, right), _max_abs(left, right)


def _quantised_state(
    *, batch: int, state_dim: int, layers: int, device: torch.device
) -> E3ScanState:
    return E3ScanState(
        layers=tuple(
            E3LayerState(
                excitatory=(
                    torch.randint(4096, (batch, state_dim), device=device).float() / 4096
                ).requires_grad_(True),
                inhibitory=(
                    torch.randint(4096, (batch, state_dim), device=device).float() / 4096
                ).requires_grad_(True),
            )
            for _ in range(layers)
        )
    )


def _clone_state(state: E3ScanState) -> E3ScanState:
    return E3ScanState(
        layers=tuple(
            E3LayerState(
                excitatory=layer.excitatory.detach().clone().requires_grad_(True),
                inhibitory=layer.inhibitory.detach().clone().requires_grad_(True),
            )
            for layer in state.layers
        )
    )


def _equivalence_case(
    *,
    batch: int,
    time_steps: int,
    layers: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    serial = E3CumulativeScanCore(
        4,
        6,
        state_dim=5,
        num_layers=layers,
        execution_mode="serial",
    ).to(device)
    scan = E3CumulativeScanCore(
        4,
        6,
        state_dim=5,
        num_layers=layers,
        execution_mode="scan",
    ).to(device)
    scan.load_state_dict(serial.state_dict())
    serial_input = torch.randn(batch, time_steps, 4, device=device, requires_grad=True)
    scan_input = serial_input.detach().clone().requires_grad_(True)
    serial_state = _quantised_state(
        batch=batch, state_dim=5, layers=layers, device=device
    )
    scan_state = _clone_state(serial_state)
    serial_result, serial_traces = serial.forward_dynamics(serial_input, serial_state)
    scan_result, scan_traces = scan.forward_dynamics(scan_input, scan_state)

    checks: Dict[str, bool] = {
        "sequence": _close(scan_result.sequence, serial_result.sequence),
        "trace_count": len(scan_traces) == len(serial_traces) == layers,
        "parameter_count": count_parameters(scan) == count_parameters(serial),
        "state_bytes": state_nbytes(scan_result.state) == state_nbytes(serial_result.state),
    }
    errors: Dict[str, float] = {
        "sequence": _max_abs(scan_result.sequence, serial_result.sequence)
    }
    for index, (scan_trace, serial_trace) in enumerate(zip(scan_traces, serial_traces)):
        for population, scan_spikes, serial_spikes in (
            (
                "e",
                scan_trace.excitatory_spikes,
                serial_trace.excitatory_spikes,
            ),
            (
                "i",
                scan_trace.inhibitory_spikes,
                serial_trace.inhibitory_spikes,
            ),
        ):
            checks[f"layer_{index}_{population}_spikes_exact"] = bool(
                torch.equal(scan_spikes, serial_spikes)
            )
            checks[f"layer_{index}_{population}_spikes_binary"] = bool(
                torch.all((scan_spikes == 0.0) | (scan_spikes == 1.0))
            )
        for population, scan_residuals, serial_residuals in (
            (
                "e",
                scan_trace.excitatory_residuals,
                serial_trace.excitatory_residuals,
            ),
            (
                "i",
                scan_trace.inhibitory_residuals,
                serial_trace.inhibitory_residuals,
            ),
        ):
            name = f"layer_{index}_{population}_residuals"
            checks[name] = _close(scan_residuals, serial_residuals)
            checks[f"{name}_bounded"] = bool(
                torch.all(scan_residuals >= 0.0) and torch.all(scan_residuals < 1.0)
            )
            errors[name] = _max_abs(scan_residuals, serial_residuals)

    for index, (scan_layer, serial_layer) in enumerate(
        zip(scan_result.state.layers, serial_result.state.layers)
    ):
        for population, scan_value, serial_value in (
            ("e", scan_layer.excitatory, serial_layer.excitatory),
            ("i", scan_layer.inhibitory, serial_layer.inhibitory),
        ):
            name = f"state_{index}_{population}"
            checks[name] = _close(scan_value, serial_value)
            errors[name] = _max_abs(scan_value, serial_value)

    probe = torch.linspace(
        -0.8, 1.2, serial_result.sequence.numel(), device=device
    ).reshape_as(serial_result.sequence)
    serial_loss = (serial_result.sequence * probe).mean()
    scan_loss = (scan_result.sequence * probe).mean()
    for serial_layer, scan_layer in zip(
        serial_result.state.layers, scan_result.state.layers
    ):
        serial_loss = serial_loss + 0.11 * (
            serial_layer.excitatory.mean() - serial_layer.inhibitory.mean()
        )
        scan_loss = scan_loss + 0.11 * (
            scan_layer.excitatory.mean() - scan_layer.inhibitory.mean()
        )
    serial_loss.backward()
    scan_loss.backward()
    checks["input_gradient"], errors["input_gradient"] = _grad_close(
        scan_input.grad, serial_input.grad
    )
    for index, (scan_layer, serial_layer) in enumerate(
        zip(scan_state.layers, serial_state.layers)
    ):
        for population, scan_gradient, serial_gradient in (
            ("e", scan_layer.excitatory.grad, serial_layer.excitatory.grad),
            ("i", scan_layer.inhibitory.grad, serial_layer.inhibitory.grad),
        ):
            name = f"initial_state_gradient_{index}_{population}"
            checks[name], errors[name] = _grad_close(scan_gradient, serial_gradient)
    serial_parameters = dict(serial.named_parameters())
    parameter_errors: Dict[str, float] = {}
    for name, scan_parameter in scan.named_parameters():
        passed, error = _grad_close(scan_parameter.grad, serial_parameters[name].grad)
        checks[f"parameter_gradient_{name}"] = passed
        parameter_errors[name] = error
    errors["parameter_gradient_max"] = max(parameter_errors.values())

    with torch.no_grad():
        full, _ = scan.forward_dynamics(scan_input.detach(), scan_state)
        stream_state: E3ScanState | None = scan_state
        pieces = []
        for index in range(time_steps):
            stepped = scan.step(scan_input.detach()[:, index], stream_state)
            pieces.append(stepped.sequence)
            stream_state = stepped.state
        streamed_sequence = torch.cat(pieces, dim=1)
    checks["scan_streaming_sequence"] = _close(streamed_sequence, full.sequence)
    errors["scan_streaming_sequence"] = _max_abs(streamed_sequence, full.sequence)
    assert stream_state is not None
    for index, (stream_layer, full_layer) in enumerate(
        zip(stream_state.layers, full.state.layers)
    ):
        for population, stream_value, full_value in (
            ("e", stream_layer.excitatory, full_layer.excitatory),
            ("i", stream_layer.inhibitory, full_layer.inhibitory),
        ):
            name = f"streaming_state_{index}_{population}"
            checks[name] = bool(torch.equal(stream_value, full_value))
            errors[name] = _max_abs(stream_value, full_value)

    return {
        "batch": batch,
        "time": time_steps,
        "layers": layers,
        "seed": seed,
        "checks": checks,
        "max_abs_errors": errors,
        "parameter_gradient_max_abs": parameter_errors,
        "passed": all(checks.values()),
    }


def run_equivalence(device: torch.device, seed: int) -> Dict[str, Any]:
    cases = tuple(
        _equivalence_case(
            batch=batch,
            time_steps=time_steps,
            layers=layers,
            seed=seed + index,
            device=device,
        )
        for index, (batch, time_steps, layers) in enumerate(
            ((1, 1, 1), (4, 32, 1), (4, 32, 2), (1, 512, 2))
        )
    )
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "case_count": len(cases),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def _core_suite(
    dimension: int, time_steps: int, device: torch.device, seed: int
) -> Dict[str, TemporalCore[Any]]:
    torch.manual_seed(seed)
    serial = E3CumulativeScanCore(
        dimension,
        dimension,
        state_dim=STATE_DIM,
        num_layers=2,
        execution_mode="serial",
    )
    scan = E3CumulativeScanCore(
        dimension,
        dimension,
        state_dim=STATE_DIM,
        num_layers=2,
        execution_mode="scan",
    )
    scan.load_state_dict(serial.state_dict())
    return {
        name: core.to(device).train(True)
        for name, core in {
            "e3_serial": serial,
            "e3_scan": scan,
            "e2_fused": E2SignedCore(
                dimension,
                dimension,
                policy="hybrid",
                positive_factor=0.8,
                execution_mode="fused",
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
        }.items()
    }


def benchmark_training(
    *,
    shapes: Iterable[Tuple[int, int, int]],
    threads: Sequence[int],
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for shape_index, (batch, time_steps, dimension) in enumerate(shapes):
            cores = _core_suite(dimension, time_steps, device, seed + shape_index)
            base = torch.randn(batch, time_steps, dimension, device=device)
            inputs = {
                name: base.detach().clone().requires_grad_(True) for name in cores
            }
            runners = {
                name: _core_training_runner(core, inputs[name])
                for name, core in cores.items()
            }
            nodes: Dict[str, int] = {}
            for name, core in cores.items():
                result = core(inputs[name])
                nodes[name] = _autograd_node_count(result.sequence)
                core.zero_grad(set_to_none=True)
                inputs[name].grad = None
            samples = _interleaved_samples(
                runners,
                warmup=warmup,
                repeats=repeats,
                device=device,
                seed=seed + thread_count * 1000 + shape_index,
            )
            summaries = {
                name: {
                    **_sample_summary(values, batch * time_steps),
                    "parameters": count_parameters(cores[name]),
                    "state_bytes": state_nbytes(cores[name].initial_state(batch)),
                    "autograd_nodes": nodes[name],
                    "peak_memory_bytes": _cuda_peak_bytes(runners[name], device),
                }
                for name, values in samples.items()
            }
            serial_p50 = summaries["e3_serial"]["p50_ms"]
            for summary in summaries.values():
                summary["versus_e3_serial_speedup"] = serial_p50 / summary["p50_ms"]
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "batch": batch,
                    "time": time_steps,
                    "dimension": dimension,
                    "e3_state_dim": STATE_DIM,
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
    seed: int,
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        cores = _core_suite(32, 128, device, seed + thread_count)
        for core in cores.values():
            core.eval()
        tokens = torch.randn(warmup_steps + measured_steps, 1, 32, device=device)
        states: Dict[str, Any] = {name: None for name in cores}
        samples: Dict[str, list[float]] = {name: [] for name in cores}
        generator = random.Random(seed + 2000 * thread_count)
        with torch.inference_mode():
            for index in range(warmup_steps):
                for name, core in cores.items():
                    states[name] = core.step(tokens[index], states[name]).state
            names = list(cores)
            for index in range(warmup_steps, warmup_steps + measured_steps):
                generator.shuffle(names)
                for name in names:
                    _sync(device)
                    started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    ended = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    if started is not None and ended is not None:
                        started.record()
                        result = cores[name].step(tokens[index], states[name])
                        ended.record()
                        ended.synchronize()
                        elapsed_ms = float(started.elapsed_time(ended))
                    else:
                        import time

                        began = time.perf_counter_ns()
                        result = cores[name].step(tokens[index], states[name])
                        elapsed_ms = (time.perf_counter_ns() - began) / 1e6
                    samples[name].append(elapsed_ms)
                    states[name] = result.state
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "warmup_steps": warmup_steps,
                "measured_steps": measured_steps,
                "models": {
                    name: {
                        **_sample_summary(values, 1),
                        "state_bytes": state_nbytes(states[name]),
                    }
                    for name, values in samples.items()
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
    parallel_checks = {}
    for record in t512:
        serial = record["models"]["e3_serial"]
        scan = record["models"]["e3_scan"]
        parallel_checks[str(record["threads"])] = {
            "speedup": scan["versus_e3_serial_speedup"],
            "node_ratio": scan["autograd_nodes"] / serial["autograd_nodes"],
            "passed": scan["versus_e3_serial_speedup"] >= 10.0
            and scan["autograd_nodes"] <= 0.25 * serial["autograd_nodes"],
        }
    parallel_pass = bool(equivalence["passed"]) and any(
        check["passed"] for check in parallel_checks.values()
    )

    streaming_by_thread = {str(record["threads"]): record for record in streaming}
    ann_checks = {}
    for record in t512:
        thread = str(record["threads"])
        stream = streaming_by_thread[thread]
        train_scan = record["models"]["e3_scan"]["p50_ms"]
        train_lstm = record["models"]["lstm"]["p50_ms"]
        infer_scan = stream["models"]["e3_scan"]["p95_ms"]
        infer_lstm = stream["models"]["lstm"]["p95_ms"]
        ann_checks[thread] = {
            "train_scan_ms": train_scan,
            "train_lstm_ms": train_lstm,
            "stream_scan_p95_ms": infer_scan,
            "stream_lstm_p95_ms": infer_lstm,
            "passed": train_scan <= train_lstm and infer_scan <= infer_lstm,
        }
    ann_pass = bool(equivalence["passed"]) and any(
        check["passed"] for check in ann_checks.values()
    )
    return {
        "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
        "parallel_training_gate": "PASS" if parallel_pass else "FAIL",
        "ann_train_and_inference_speed_gate": "PASS" if ann_pass else "FAIL",
        "parallel_checks_by_threads": parallel_checks,
        "ann_checks_by_threads": ann_checks,
        "run_synthetic_memory_next": bool(equivalence["passed"] and (parallel_pass or ann_pass)),
        "boundary": (
            "S0 is a strict spike/reset additive IF scan, but has no leak or same-layer "
            "recurrence. Speed does not establish memory-task or world-model quality."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_s0_scan_benchmark.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--streaming-warmup", type=int, default=16)
    parser.add_argument("--streaming-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9300)
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
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    resolved_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(resolved_device)
    threads = tuple(dict.fromkeys(args.threads))
    shapes = ((1, 16, 32),) if args.quick else FORMAL_SHAPES
    if args.quick:
        threads = threads[:1]
    if device.type == "cpu":
        torch.set_num_threads(threads[0])
    equivalence = run_equivalence(device, args.seed)
    training = benchmark_training(
        shapes=shapes,
        threads=threads,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
        seed=args.seed,
    )
    streaming = benchmark_streaming(
        threads=threads,
        warmup_steps=args.streaming_warmup,
        measured_steps=args.streaming_steps,
        device=device,
        seed=args.seed,
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-S0 exact-reset cumulative-charge scan",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "seed": args.seed,
            "threads": threads,
            "shapes": shapes,
            "e3_state_dim": STATE_DIM,
            "e3_layers": 2,
            "drive_levels": 1024,
            "charge_levels": 4096,
            "max_charge": 0.95,
            "surrogate_scale": 5.0,
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
