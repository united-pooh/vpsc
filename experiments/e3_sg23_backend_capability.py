"""SG23H multicore and DirectML backend capability benchmark.

The runner intentionally has no dependency on the rest of the VPSC experiment
stack so that it can execute inside an isolated Windows torch-directml virtual
environment.  It compares the same PyTorch build on CPU and DirectML and keeps
device-resident (scalar synchronization) timings separate from full host/device
transfer timings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg23_backend_capability.json")
DEFAULT_SIZES = (443, 1024, 2048, 4096)
DEFAULT_THREAD_SWEEP = (1, 2, 4, 8, 16)
FEATURE_WIDTH = 256
OUTPUT_WIDTH = 19
FROZEN_SEED = 230719
CORRECTNESS_ABS_TOLERANCE = 1e-4


def timing_summary(samples_seconds: Sequence[float]) -> Dict[str, float]:
    """Return deterministic summaries without interpolating tail samples."""

    if not samples_seconds:
        raise ValueError("timing samples cannot be empty")
    ordered = sorted(float(value) for value in samples_seconds)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "minimum_seconds": ordered[0],
        "median_seconds": float(statistics.median(ordered)),
        "p95_seconds": ordered[p95_index],
        "maximum_seconds": ordered[-1],
        "sample_count": len(ordered),
    }


def _consume_scalar(value: torch.Tensor) -> float:
    """Force asynchronous backends to finish while copying only one scalar."""

    return float(value.reshape(-1)[0].detach().cpu().item())


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (actual.detach().cpu().to(torch.float64) - expected.to(torch.float64))
        .abs()
        .max()
        .item()
    )


def _benchmark_operation(
    *,
    name: str,
    operation: Callable[..., torch.Tensor],
    cpu_inputs: Sequence[torch.Tensor],
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> Dict[str, Any]:
    expected = operation(*cpu_inputs).detach().cpu()
    device_inputs = tuple(value.to(device) for value in cpu_inputs)
    warning_messages = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual_device = operation(*device_inputs)
            actual = actual_device.detach().cpu()
        warning_messages.extend(str(item.message) for item in caught)
    except Exception as error:  # capability probe must preserve unsupported ops
        return {
            "name": name,
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "warnings": warning_messages,
        }

    for _ in range(warmups):
        _consume_scalar(operation(*device_inputs))

    resident_samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        _consume_scalar(operation(*device_inputs))
        resident_samples.append((time.perf_counter_ns() - started) / 1e9)

    transfer_samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        transferred = tuple(value.to(device, copy=True) for value in cpu_inputs)
        result = operation(*transferred).detach().cpu()
        _ = float(result.reshape(-1)[0].item())
        transfer_samples.append((time.perf_counter_ns() - started) / 1e9)

    return {
        "name": name,
        "supported": True,
        "output_shape": tuple(int(value) for value in actual.shape),
        "output_device": str(actual_device.device),
        "max_abs_error_vs_cpu": _max_abs_error(actual, expected),
        "warnings": tuple(warning_messages),
        "resident_scalar_sync": timing_summary(resident_samples),
        "full_input_output_transfer": timing_summary(transfer_samples),
    }


def _probe_cholesky(
    device: torch.device, *, size: int = 128
) -> Dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(FROZEN_SEED + 99)
    raw = torch.randn(size, 48, generator=generator, dtype=torch.float32)
    system = raw @ raw.T / 48.0 + 0.25 * torch.eye(size)
    expected = torch.linalg.cholesky(system)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual_device = torch.linalg.cholesky(system.to(device))
            actual = actual_device.detach().cpu()
        messages = tuple(str(item.message) for item in caught)
        return {
            "supported": True,
            "output_device": str(actual_device.device),
            "max_abs_error_vs_cpu": _max_abs_error(actual, expected),
            "warnings": messages,
            "silent_or_warned_cpu_fallback": bool(
                actual_device.device.type == "cpu"
                or any("fallback" in message.lower() for message in messages)
            ),
        }
    except Exception as error:
        return {
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _probe_float64_matmul(device: torch.device) -> Dict[str, Any]:
    left = torch.arange(64, dtype=torch.float64).reshape(8, 8) / 64.0
    right = left.T.contiguous()
    expected = left @ right
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual_device = left.to(device) @ right.to(device)
            actual = actual_device.detach().cpu()
        messages = tuple(str(item.message) for item in caught)
        return {
            "supported": True,
            "output_device": str(actual_device.device),
            "max_abs_error_vs_cpu": _max_abs_error(actual, expected),
            "warnings": messages,
        }
    except Exception as error:
        return {
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _operation_inputs(size: int) -> Dict[str, Sequence[torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(FROZEN_SEED + size)
    features = torch.randn(
        size, FEATURE_WIDTH, generator=generator, dtype=torch.float32
    )
    readout = torch.randn(
        FEATURE_WIDTH, OUTPUT_WIDTH, generator=generator, dtype=torch.float32
    )
    right_hand_sides = torch.randn(
        size, OUTPUT_WIDTH, generator=generator, dtype=torch.float32
    )
    return {
        "readout": (features, readout),
        "gram": (features,),
        "matrix_free_normal": (features, right_hand_sides),
    }


OPERATIONS: Mapping[str, Callable[..., torch.Tensor]] = {
    "readout": lambda features, weights: features @ weights,
    "gram": lambda features: features @ features.T,
    "matrix_free_normal": lambda features, rhs: features @ (
        features.T @ rhs
    ),
}


def benchmark_device(
    *,
    label: str,
    device: torch.device,
    sizes: Sequence[int],
    warmups: int,
    repetitions: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for size in sizes:
        inputs = _operation_inputs(size)
        results[str(size)] = {
            name: _benchmark_operation(
                name=name,
                operation=OPERATIONS[name],
                cpu_inputs=inputs[name],
                device=device,
                warmups=warmups,
                repetitions=repetitions,
            )
            for name in OPERATIONS
        }
    add_generator = torch.Generator(device="cpu").manual_seed(FROZEN_SEED)
    left = torch.randn(1_000_000, generator=add_generator)
    right = torch.randn(1_000_000, generator=add_generator)
    return {
        "label": label,
        "device": str(device),
        "sizes": results,
        "vector_add": _benchmark_operation(
            name="vector_add",
            operation=lambda first, second: first + second,
            cpu_inputs=(left, right),
            device=device,
            warmups=warmups,
            repetitions=repetitions,
        ),
        "float64_matmul": _probe_float64_matmul(device),
        "cholesky_fp32": _probe_cholesky(device),
    }


def benchmark_cpu_thread_sweep(
    *,
    thread_counts: Sequence[int],
    size: int,
    warmups: int,
    repetitions: int,
) -> Dict[str, Any]:
    """Measure the largest representative kernels across CPU thread counts."""

    inputs = _operation_inputs(size)
    results: Dict[str, Any] = {}
    for thread_count in thread_counts:
        torch.set_num_threads(thread_count)
        results[str(thread_count)] = {
            name: _benchmark_operation(
                name=name,
                operation=OPERATIONS[name],
                cpu_inputs=inputs[name],
                device=torch.device("cpu"),
                warmups=warmups,
                repetitions=repetitions,
            )
            for name in OPERATIONS
        }
    return {
        "size": size,
        "thread_counts": tuple(thread_counts),
        "results": results,
    }


def _run_wsl(script: str) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            ["wsl.exe", "-e", "bash", "-lc", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "available": True,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _wsl_rocm_gate(probe: Mapping[str, Any]) -> bool:
    combined = "\n".join(
        str(probe.get(name, {}).get("stdout", ""))
        for name in ("torch", "runtime", "opencl")
    ).lower()
    return bool(
        ("rx 7800 xt" in combined or "gfx1101" in combined)
        and '"hip": null' not in combined
        and '"cuda_available": false' not in combined
    )


def probe_wsl_rocm() -> Dict[str, Any]:
    torch_probe = _run_wsl(
        "cd /mnt/d/projects/vpsc && "
        ".venv-wsl/bin/python -c 'import json,torch; "
        "print(json.dumps({\"torch\":torch.__version__,"
        "\"hip\":torch.version.hip,"
        "\"cuda_available\":torch.cuda.is_available(),"
        "\"device_count\":torch.cuda.device_count()}))'"
    )
    runtime_probe = _run_wsl(
        "printf 'kernel='; uname -r; "
        "printf 'dxg='; test -e /dev/dxg && echo yes || echo no; "
        "rocminfo 2>&1 | grep -E 'Name:|Marketing Name:|gfx[0-9]+' | head -30"
    )
    opencl_probe = _run_wsl(
        "clinfo 2>&1 | grep -E 'Number of devices|Device Name' | head -20"
    )
    probe = {
        "torch": torch_probe,
        "runtime": runtime_probe,
        "opencl": opencl_probe,
    }
    probe["usable_rocm_pytorch_gpu"] = _wsl_rocm_gate(probe)
    return probe


def compare_backends(
    cpu: Mapping[str, Any], directml: Mapping[str, Any]
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    useful_resident = False
    useful_transfer = False
    for size, cpu_operations in cpu["sizes"].items():
        comparisons[size] = {}
        for name, cpu_record in cpu_operations.items():
            dml_record = directml["sizes"][size][name]
            if not cpu_record["supported"] or not dml_record["supported"]:
                comparisons[size][name] = {"comparable": False}
                continue
            cpu_resident = cpu_record["resident_scalar_sync"]["median_seconds"]
            dml_resident = dml_record["resident_scalar_sync"]["median_seconds"]
            cpu_transfer = cpu_record["full_input_output_transfer"][
                "median_seconds"
            ]
            dml_transfer = dml_record["full_input_output_transfer"][
                "median_seconds"
            ]
            resident_speedup = cpu_resident / dml_resident
            transfer_speedup = cpu_transfer / dml_transfer
            useful_resident = useful_resident or resident_speedup > 1.0
            useful_transfer = useful_transfer or transfer_speedup > 1.0
            comparisons[size][name] = {
                "comparable": True,
                "cpu_over_directml_resident_speedup": resident_speedup,
                "cpu_over_directml_full_transfer_speedup": transfer_speedup,
            }
    return {
        "by_size": comparisons,
        "any_resident_speedup": useful_resident,
        "any_full_transfer_speedup": useful_transfer,
    }


def make_decision(
    *,
    adapter_names: Sequence[str],
    directml: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> Dict[str, Any]:
    required = [directml["vector_add"]]
    required.extend(
        operations[name]
        for operations in directml["sizes"].values()
        for name in ("readout", "gram", "matrix_free_normal")
    )
    correctness = all(
        bool(record["supported"])
        and float(record["max_abs_error_vs_cpu"])
        <= CORRECTNESS_ABS_TOLERANCE
        for record in required
    )
    adapter_gate = any("RX 7800 XT" in name for name in adapter_names)
    backend_available = adapter_gate and correctness
    useful_speed = bool(comparison["any_resident_speedup"])
    return {
        "adapter_gate": adapter_gate,
        "fp32_correctness_gate": correctness,
        "backend_available": backend_available,
        "useful_device_resident_speed_gate": useful_speed,
        "useful_full_transfer_speed_gate": bool(
            comparison["any_full_transfer_speedup"]
        ),
        "overall": "PASS" if backend_available and useful_speed else "FAIL",
        "deployment_boundary": (
            "directml_batch_or_matrix_free_candidate"
            if backend_available and useful_speed
            else "cpu_multicore_only_until_backend_changes"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(args.threads)
    try:
        import torch_directml  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "SG23H formal run requires the isolated torch-directml environment"
        ) from error

    adapter_names = tuple(
        torch_directml.device_name(index).rstrip("\x00")
        for index in range(torch_directml.device_count())
    )
    if args.adapter_index >= len(adapter_names):
        raise ValueError(
            f"adapter index {args.adapter_index} not in {adapter_names}"
        )
    dml_device = torch_directml.device(args.adapter_index)
    cpu = benchmark_device(
        label=f"cpu-{args.threads}-threads",
        device=torch.device("cpu"),
        sizes=args.sizes,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    directml = benchmark_device(
        label=f"directml-adapter-{args.adapter_index}",
        device=dml_device,
        sizes=args.sizes,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    cpu_thread_sweep = benchmark_cpu_thread_sweep(
        thread_counts=args.thread_sweep,
        size=max(args.sizes),
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    torch.set_num_threads(args.threads)
    comparison = compare_backends(cpu, directml)
    wsl_rocm = probe_wsl_rocm()
    decision = make_decision(
        adapter_names=adapter_names,
        directml=directml,
        comparison=comparison,
    )
    decision["rocm_available"] = bool(wsl_rocm["usable_rocm_pytorch_gpu"])
    protocol = {
        "seed": FROZEN_SEED,
        "sizes": tuple(args.sizes),
        "feature_width": FEATURE_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cpu_threads": args.threads,
        "cpu_thread_sweep": tuple(args.thread_sweep),
        "correctness_abs_tolerance": CORRECTNESS_ABS_TOLERANCE,
    }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()
    return {
        "experiment": "E3-SG23H multicore and DirectML backend capability",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torch_directml": importlib.metadata.version("torch-directml"),
            "logical_cpu_count": os.cpu_count(),
            "torch_intraop_threads": torch.get_num_threads(),
            "directml_adapter_names": adapter_names,
            "selected_adapter_index": args.adapter_index,
        },
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "cpu": cpu,
        "cpu_thread_sweep": cpu_thread_sweep,
        "directml": directml,
        "wsl_rocm": wsl_rocm,
        "comparison": comparison,
        "decision": decision,
    }


def _parse_sizes(raw: str) -> tuple[int, ...]:
    sizes = tuple(int(value) for value in raw.split(",") if value.strip())
    if not sizes or any(value <= 0 for value in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma values")
    return sizes


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--adapter-index", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated row counts",
    )
    parser.add_argument(
        "--thread-sweep",
        type=_parse_sizes,
        default=DEFAULT_THREAD_SWEEP,
        help="comma-separated CPU intra-op thread counts",
    )
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.warmups < 0 or args.repetitions <= 0:
        parser.error("threads/repetitions must be positive and warmups nonnegative")
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
