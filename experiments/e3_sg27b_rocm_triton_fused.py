"""SG27B formal benchmark for the fully fused ROCm Triton gated trace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e3_sg27a_rocm_triton_bridge import (  # noqa: E402
    STATE_DIM,
    Route,
    _equivalence,
    _inputs,
    _memory,
    _sha256,
    _timings,
)
from vpsc.world_model.portable_gated_trace import (  # noqa: E402
    portable_parallel_gated_trace,
    serial_gated_trace_reference,
)
from vpsc.world_model.triton_affine_scan import (  # noqa: E402
    backend_audit as affine_backend_audit,
    triton_composed_gated_trace,
)
from vpsc.world_model.triton_fused_gated_trace import (  # noqa: E402
    backend_audit as fused_backend_audit,
    triton_fused_gated_trace,
)


BUCKETS = ((64, 27), (96, 55), (128, 71), (160, 71))


def _speedups(records: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    fused = records["triton_fused"]
    result: Dict[str, Dict[str, float]] = {}
    for control in ("serial", "tensor_tree", "triton_composed"):
        result[control] = {
            phase: records[control][phase]["p50_ms"] / fused[phase]["p50_ms"]
            for phase in ("training", "inference")
        }
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("SG27B requires a ROCm PyTorch device")
    device = torch.device("cuda:0")
    if "7800 XT" not in torch.cuda.get_device_name(device):
        raise RuntimeError("SG27B is frozen to the local RX 7800 XT")
    routes: Mapping[str, Route] = {
        "serial": serial_gated_trace_reference,
        "tensor_tree": portable_parallel_gated_trace,
        "triton_composed": triton_composed_gated_trace,
        "triton_fused": triton_fused_gated_trace,
    }
    equivalence = {
        "non_power_of_two": _equivalence(
            routes,
            device=device,
            time_steps=31,
            query_count=11,
            seed=27_520_001,
        ),
        "long_fallback": _equivalence(
            routes,
            device=device,
            time_steps=160,
            query_count=71,
            seed=27_520_002,
        ),
    }
    equivalence_pass = all(case["passed"] for case in equivalence.values())
    buckets: Dict[str, Any] = {}
    for bucket_index, (time_steps, query_count) in enumerate(BUCKETS):
        records: Dict[str, Any] = {}
        for name, route in routes.items():
            inputs = _inputs(
                batch_size=args.batch_size,
                time_steps=time_steps,
                query_count=query_count,
                device=device,
                seed=27_530_000 + 100 * bucket_index,
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
        records["speedups_fused_over"] = _speedups(records)
        buckets[f"t{time_steps}_q{query_count}"] = records
    speed_pass = all(
        records["triton_fused"][phase]["p50_ms"]
        <= records[control][phase]["p50_ms"]
        for records in buckets.values()
        for phase in ("training", "inference")
        for control in ("serial", "tensor_tree", "triton_composed")
    )
    memory_pass = all(
        records["triton_fused"]["memory"]["incremental_peak_allocated_bytes"]
        <= records["triton_composed"]["memory"][
            "incremental_peak_allocated_bytes"
        ]
        for records in buckets.values()
    )
    return {
        "experiment": "E3-SG27B fully fused ROCm Triton gated trace",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "batch_size": args.batch_size,
            "state_dim": STATE_DIM,
            "buckets": BUCKETS,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timing_scope": "complete forward or complete forward+backward with device synchronize",
            "same_seed_per_route": True,
            "same_hardware_only": True,
        },
        "environment": {
            **affine_backend_audit(),
            "fused": fused_backend_audit(),
            "device_name": torch.cuda.get_device_name(device),
            "device_memory_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
        },
        "sources": {
            "runner_sha256": _sha256(Path(__file__)),
            "portable_sha256": _sha256(
                ROOT / "vpsc/world_model/portable_gated_trace.py"
            ),
            "affine_sha256": _sha256(
                ROOT / "vpsc/world_model/triton_affine_scan.py"
            ),
            "fused_sha256": _sha256(
                ROOT / "vpsc/world_model/triton_fused_gated_trace.py"
            ),
        },
        "equivalence": equivalence,
        "buckets": buckets,
        "gates": {
            "equivalence": equivalence_pass,
            "speed": speed_pass,
            "memory": memory_pass,
            "overall": equivalence_pass and speed_pass and memory_pass,
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
        default=Path("results/e3_scan/e3_sg27b_rocm_triton_fused.json"),
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
