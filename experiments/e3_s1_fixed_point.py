"""E3-S1 hard-reset fixed-point convergence gate against exact serial dynamics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment, _max_abs  # noqa: E402
from vpsc.world_model.cores import (  # noqa: E402
    E3FixedPointScanCore,
    E3LayerState,
    E3ScanState,
    count_parameters,
    state_nbytes,
)


ITERATIONS = (1, 2, 4, 8)
ATOL = 1e-3
RTOL = 1e-3


def _case(
    *,
    seed: int,
    batch: int,
    time_steps: int,
    input_scale: float,
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    serial = E3FixedPointScanCore(
        16,
        16,
        state_dim=16,
        execution_mode="serial",
    ).to(device)
    value = input_scale * torch.randn(batch, time_steps, 16, device=device)
    initial = E3ScanState(
        layers=(
            E3LayerState(
                excitatory=0.9 * torch.rand(batch, 16, device=device),
                inhibitory=0.9 * torch.rand(batch, 16, device=device),
            ),
        )
    )
    with torch.no_grad():
        serial_result, serial_trace = serial.forward_dynamics(value, initial)
    iteration_results = {}
    serial_spikes = torch.cat(
        (
            serial_trace.excitatory_spikes.reshape(-1),
            serial_trace.inhibitory_spikes.reshape(-1),
        )
    )
    for iterations in ITERATIONS:
        fixed = E3FixedPointScanCore(
            16,
            16,
            state_dim=16,
            execution_mode="fixed_point",
            fixed_point_iterations=iterations,
        ).to(device)
        fixed.load_state_dict(serial.state_dict())
        with torch.no_grad():
            fixed_result, fixed_trace = fixed.forward_dynamics(value, initial)
        fixed_spikes = torch.cat(
            (
                fixed_trace.excitatory_spikes.reshape(-1),
                fixed_trace.inhibitory_spikes.reshape(-1),
            )
        )
        mismatch = float((fixed_spikes != serial_spikes).float().mean().item())
        state_error = max(
            _max_abs(
                fixed_result.state.layers[0].excitatory,
                serial_result.state.layers[0].excitatory,
            ),
            _max_abs(
                fixed_result.state.layers[0].inhibitory,
                serial_result.state.layers[0].inhibitory,
            ),
        )
        output_error = _max_abs(fixed_result.sequence, serial_result.sequence)
        output_close = bool(
            torch.allclose(
                fixed_result.sequence,
                serial_result.sequence,
                atol=ATOL,
                rtol=RTOL,
            )
        )
        state_close = all(
            torch.allclose(fixed_value, serial_value, atol=ATOL, rtol=RTOL)
            for fixed_value, serial_value in (
                (
                    fixed_result.state.layers[0].excitatory,
                    serial_result.state.layers[0].excitatory,
                ),
                (
                    fixed_result.state.layers[0].inhibitory,
                    serial_result.state.layers[0].inhibitory,
                ),
            )
        )
        binary = bool(torch.all((fixed_spikes == 0.0) | (fixed_spikes == 1.0)))
        residual_bounded = all(
            bool(torch.all(value_ >= 0.0) and torch.all(value_ < 1.0))
            for value_ in (
                fixed_trace.excitatory_residuals,
                fixed_trace.inhibitory_residuals,
            )
        )
        passed = (
            mismatch <= 0.001
            and output_close
            and state_close
            and binary
            and residual_bounded
        )
        iteration_results[str(iterations)] = {
            "spike_mismatch_rate": mismatch,
            "output_max_abs": output_error,
            "state_max_abs": state_error,
            "output_close": output_close,
            "state_close": state_close,
            "binary_spikes": binary,
            "bounded_residuals": residual_bounded,
            "passed": passed,
        }
    return {
        "seed": seed,
        "batch": batch,
        "time": time_steps,
        "input_scale": input_scale,
        "serial_spike_rate": float(serial_spikes.mean().item()),
        "parameters": count_parameters(serial),
        "state_bytes": state_nbytes(serial_result.state),
        "iterations": iteration_results,
    }


def _decision(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = None
    by_iteration = {}
    for iterations in ITERATIONS:
        key = str(iterations)
        selected_cases = [case["iterations"][key] for case in cases]
        passed = all(case["passed"] for case in selected_cases)
        by_iteration[key] = {
            "all_cases_passed": passed,
            "worst_spike_mismatch_rate": max(
                case["spike_mismatch_rate"] for case in selected_cases
            ),
            "worst_output_max_abs": max(case["output_max_abs"] for case in selected_cases),
            "worst_state_max_abs": max(case["state_max_abs"] for case in selected_cases),
        }
        if selected is None and passed:
            selected = iterations
    return {
        "convergence_gate": "PASS" if selected is not None else "FAIL",
        "selected_iterations": selected,
        "by_iteration": by_iteration,
        "parallel_speed_gate": "NOT_RUN" if selected is None else "PENDING",
        "a0_quality_gate": "NOT_RUN" if selected is None else "PENDING",
        "next": (
            "benchmark selected K and run A0"
            if selected is not None
            else "PRF/reset-free oscillatory code or exact event segmentation"
        ),
        "boundary": (
            "Failure means K<=8 does not approximate exact serial hard reset at the frozen "
            "tolerance; it does not invalidate affine scans without hard reset."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_s1_fixed_point.json"),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    specifications: Tuple[Tuple[int, int, float], ...] = (
        ((1, 32, 0.25),) if args.quick else (
            (1, 32, 0.25),
            (4, 32, 1.0),
            (1, 512, 0.25),
            (1, 512, 1.0),
        )
    )
    cases = tuple(
        _case(
            seed=11_000 + index,
            batch=batch,
            time_steps=time_steps,
            input_scale=input_scale,
            device=device,
        )
        for index, (batch, time_steps, input_scale) in enumerate(specifications)
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-S1 fixed-point hard-reset convergence",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "iterations": ITERATIONS,
            "atol": ATOL,
            "rtol": RTOL,
            "spike_mismatch_threshold": 0.001,
            "threads": args.threads,
        },
        "cases": cases,
        "decision": _decision(cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
