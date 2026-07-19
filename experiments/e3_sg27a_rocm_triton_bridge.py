"""SG27A ROCm benchmark for serial, tensor-tree, and Triton SNN scans."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vpsc.world_model.portable_gated_trace import (  # noqa: E402
    portable_parallel_gated_trace,
    serial_gated_trace_reference,
)
from vpsc.world_model.triton_affine_scan import (  # noqa: E402
    backend_audit,
    triton_composed_gated_trace,
)


STATE_DIM = 31
BUCKETS = ((64, 27), (96, 55), (128, 71))
Route = Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_summary(samples_ms: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in samples_ms)
    return {
        "count": float(len(ordered)),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _inputs(
    *,
    batch_size: int,
    time_steps: int,
    query_count: int,
    device: torch.device,
    seed: int,
    requires_grad: bool,
) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    base = torch.randn(batch_size, time_steps, 4 * STATE_DIM, device=device)
    drives = (base.sign() * (base.abs() + 0.2)).requires_grad_(requires_grad)
    positions = torch.linspace(
        0,
        time_steps - 1,
        query_count,
        device=device,
    ).round().to(torch.long)
    queries = positions.unsqueeze(0).expand(batch_size, -1).contiguous()
    decays = (
        0.55 + 0.35 * torch.rand(2, STATE_DIM, device=device)
    ).requires_grad_(requires_grad)
    initial_e = (
        0.05 + 0.2 * torch.rand(batch_size, STATE_DIM, device=device)
    ).requires_grad_(requires_grad)
    initial_i = (
        0.05 + 0.2 * torch.rand(batch_size, STATE_DIM, device=device)
    ).requires_grad_(requires_grad)
    return drives, queries, decays, initial_e, initial_i


def _route_call(route: Route, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    keyword: Dict[str, Any] = {
        "spike_threshold": 0.43,
        "surrogate_scale": 4.0,
    }
    if route in (serial_gated_trace_reference, portable_parallel_gated_trace):
        keyword["_unchecked"] = True
    return route(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        **keyword,
    )


def _loss(outputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    return (
        outputs[0].square().mean()
        + outputs[1].square().mean()
        + outputs[2].square().mean()
    )


def _clear_gradients(inputs: Tuple[torch.Tensor, ...]) -> None:
    for value in (inputs[0], inputs[2], inputs[3], inputs[4]):
        value.grad = None


def _timings(
    route: Route,
    inputs: Tuple[torch.Tensor, ...],
    *,
    warmups: int,
    repeats: int,
    training: bool,
) -> Dict[str, float]:
    samples = []
    for iteration in range(warmups + repeats):
        if training:
            _clear_gradients(inputs)
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        if training:
            _loss(_route_call(route, inputs)).backward()
        else:
            with torch.no_grad():
                _route_call(route, inputs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if iteration >= warmups:
            samples.append(elapsed_ms)
    return _sample_summary(samples)


def _memory(
    route: Route,
    inputs: Tuple[torch.Tensor, ...],
) -> Dict[str, int]:
    _clear_gradients(inputs)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    _loss(_route_call(route, inputs)).backward()
    torch.cuda.synchronize()
    return {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated_bytes": max(
            0, torch.cuda.max_memory_allocated() - baseline_allocated
        ),
        "incremental_peak_reserved_bytes": max(
            0, torch.cuda.max_memory_reserved() - baseline_reserved
        ),
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _equivalence(
    routes: Mapping[str, Route],
    *,
    device: torch.device,
    time_steps: int = 31,
    query_count: int = 11,
    seed: int = 27_500_001,
) -> Dict[str, Any]:
    base_inputs = _inputs(
        batch_size=3,
        time_steps=time_steps,
        query_count=query_count,
        device=device,
        seed=seed,
        requires_grad=False,
    )
    route_outputs: Dict[str, Tuple[torch.Tensor, ...]] = {}
    route_inputs: Dict[str, Tuple[torch.Tensor, ...]] = {}
    torch.manual_seed(27_500_002)
    raw_probe = torch.randn(3, query_count, 4 * STATE_DIM, device=device)
    final_e_probe = torch.randn(3, STATE_DIM, device=device)
    final_i_probe = torch.randn(3, STATE_DIM, device=device)
    for name, route in routes.items():
        inputs = (
            base_inputs[0].detach().clone().requires_grad_(True),
            base_inputs[1],
            base_inputs[2].detach().clone().requires_grad_(True),
            base_inputs[3].detach().clone().requires_grad_(True),
            base_inputs[4].detach().clone().requires_grad_(True),
        )
        outputs = _route_call(route, inputs)
        (
            (outputs[0] * raw_probe).sum()
            + (outputs[1] * final_e_probe).sum()
            + (outputs[2] * final_i_probe).sum()
        ).backward()
        route_inputs[name] = inputs
        route_outputs[name] = outputs
    reference_outputs = route_outputs["serial"]
    reference_inputs = route_inputs["serial"]
    records = {}
    for name in (route_name for route_name in routes if route_name != "serial"):
        outputs = route_outputs[name]
        inputs = route_inputs[name]
        errors = {
            "raw_max_abs": _max_abs(outputs[0], reference_outputs[0]),
            "final_e_max_abs": _max_abs(outputs[1], reference_outputs[1]),
            "final_i_max_abs": _max_abs(outputs[2], reference_outputs[2]),
            "drive_grad_max_abs": _max_abs(inputs[0].grad, reference_inputs[0].grad),
            "decay_grad_max_abs": _max_abs(inputs[2].grad, reference_inputs[2].grad),
            "initial_e_grad_max_abs": _max_abs(
                inputs[3].grad, reference_inputs[3].grad
            ),
            "initial_i_grad_max_abs": _max_abs(
                inputs[4].grad, reference_inputs[4].grad
            ),
        }
        spike_width = 2 * STATE_DIM
        spike_disagreements = int(
            (
                outputs[0][:, :, :spike_width]
                != reference_outputs[0][:, :, :spike_width]
            )
            .sum()
            .item()
        )
        records[name] = {
            "errors": errors,
            "spike_disagreements": spike_disagreements,
            "passed": spike_disagreements == 0
            and max(errors.values()) <= 2e-4,
        }
    return {
        "shape": {
            "batch": 3,
            "time": time_steps,
            "state": STATE_DIM,
            "query": query_count,
        },
        "routes": records,
        "passed": all(record["passed"] for record in records.values()),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("SG27A requires a ROCm PyTorch device")
    device = torch.device("cuda:0")
    if "7800 XT" not in torch.cuda.get_device_name(device):
        raise RuntimeError("SG27A is frozen to the local RX 7800 XT")
    routes: Mapping[str, Route] = {
        "serial": serial_gated_trace_reference,
        "tensor_tree": portable_parallel_gated_trace,
        "triton_composed": triton_composed_gated_trace,
    }
    equivalence = _equivalence(routes, device=device)
    buckets: Dict[str, Any] = {}
    for bucket_index, (time_steps, query_count) in enumerate(BUCKETS):
        records = {}
        for name, route in routes.items():
            inputs = _inputs(
                batch_size=args.batch_size,
                time_steps=time_steps,
                query_count=query_count,
                device=device,
                seed=27_510_000 + 100 * bucket_index,
                requires_grad=True,
            )
            records[name] = {
                "training": _timings(
                    route,
                    inputs,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    training=True,
                ),
                "inference": _timings(
                    route,
                    inputs,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    training=False,
                ),
                "memory": _memory(route, inputs),
            }
        buckets[f"t{time_steps}_q{query_count}"] = records
    speed_pass = all(
        records["triton_composed"][phase]["p50_ms"]
        <= records["serial"][phase]["p50_ms"]
        and records["triton_composed"][phase]["p50_ms"]
        <= records["tensor_tree"][phase]["p50_ms"]
        for records in buckets.values()
        for phase in ("training", "inference")
    )
    return {
        "experiment": "E3-SG27A ROCm Triton affine-scan bridge",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "batch_size": args.batch_size,
            "state_dim": STATE_DIM,
            "buckets": BUCKETS,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timing_scope": "forward or forward+backward with device synchronize",
            "same_hardware_only": True,
        },
        "environment": {
            **backend_audit(),
            "device_name": torch.cuda.get_device_name(device),
            "device_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "sources": {
            "portable_sha256": _sha256(
                ROOT / "vpsc/world_model/portable_gated_trace.py"
            ),
            "triton_sha256": _sha256(
                ROOT / "vpsc/world_model/triton_affine_scan.py"
            ),
        },
        "equivalence": equivalence,
        "buckets": buckets,
        "gates": {
            "equivalence": equivalence["passed"],
            "speed": speed_pass,
            "overall": equivalence["passed"] and speed_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg27a_rocm_triton_bridge.json"),
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmups < 0 or args.repeats <= 0:
        parser.error("batch/repeat values must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
