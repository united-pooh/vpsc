#!/usr/bin/env python3
"""LDAA-1: exact-backward crossover over loss density and state size.

This runner compares four exact training paths for the same gated-trace
forward dynamics:

* ``bptt``: dense autograd through the full sequence;
* ``forward_eligibility``: K-query forward eligibility snapshots;
* ``reverse_adjoint``: dense T-step reverse scan (RA0);
* ``segmented_adjoint``: K-anchor reverse scan plus analytic segment fill.

The operator study measures exactness, unique saved storage, autograd nodes,
and interleaved forward+backward latency.  It does not claim model-level
quality or GPU performance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _autograd_node_count,
    _environment,
    _sync,
)
from experiments.e3_el0_terminal_eligibility import _SavedTensorCounter  # noqa: E402
from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore,
    E3LayerState,
    E3ScanState,
)
from vpsc.world_model.devices import choose_device, device_label  # noqa: E402


ATOL = 2e-5
RTOL = 1e-4
BACKENDS = (
    "bptt",
    "forward_eligibility",
    "reverse_adjoint",
    "segmented_adjoint",
)


@dataclass(frozen=True)
class MatrixConfig:
    lengths: Tuple[int, ...] = (512, 2048, 8192, 32768)
    loss_densities: Tuple[float, ...] = (
        1 / 1024,
        1 / 256,
        1 / 64,
        1 / 16,
        1 / 4,
        1.0,
    )
    state_dims: Tuple[int, ...] = (16, 64)
    input_gradients: Tuple[bool, ...] = (False, True)
    input_dim: int = 16
    hidden_dim: int = 16
    batch_size: int = 1
    threads: int = 4
    warmup: int = 1
    repeats: int = 5


@dataclass
class BackendCase:
    core: E3GatedTraceScanCore
    value: Tensor
    initial_e: Tensor
    initial_i: Tensor

    @property
    def state(self) -> E3ScanState:
        return E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=self.initial_e,
                    inhibitory=self.initial_i,
                ),
            )
        )

    def clear_gradients(self) -> None:
        self.core.zero_grad(set_to_none=True)
        self.value.grad = None
        self.initial_e.grad = None
        self.initial_i.grad = None


def query_indices_for_density(
    time_steps: int,
    density: float,
    *,
    device: torch.device,
) -> Tensor:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if not 0.0 < density <= 1.0:
        raise ValueError("density must lie inside (0, 1]")
    query_count = min(time_steps, max(1, int(round(time_steps * density))))
    if query_count == time_steps:
        return torch.arange(time_steps, device=device, dtype=torch.long)
    return torch.linspace(
        0,
        time_steps - 1,
        steps=query_count,
        device=device,
    ).round().to(dtype=torch.long).unique()


def _build_core(
    backend: str,
    *,
    input_dim: int,
    hidden_dim: int,
    state_dim: int,
    device: torch.device,
) -> E3GatedTraceScanCore:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")
    mode = "forward_eligibility" if backend == "bptt" else backend
    return E3GatedTraceScanCore(
        input_dim,
        hidden_dim,
        state_dim=state_dim,
        execution_mode="scan",
        eligibility_backward_mode=mode,
    ).to(device)


def _build_cases(
    *,
    seed: int,
    time_steps: int,
    state_dim: int,
    input_gradient: bool,
    config: MatrixConfig,
    device: torch.device,
) -> Dict[str, BackendCase]:
    torch.manual_seed(seed)
    reference = _build_core(
        "bptt",
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        state_dim=state_dim,
        device=device,
    )
    reference_state = reference.state_dict()
    source_value = torch.randn(
        config.batch_size,
        time_steps,
        config.input_dim,
        device=device,
    )
    source_e = torch.rand(config.batch_size, state_dim, device=device)
    source_i = torch.rand(config.batch_size, state_dim, device=device)
    cases: Dict[str, BackendCase] = {}
    for backend in BACKENDS:
        core = _build_core(
            backend,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            state_dim=state_dim,
            device=device,
        )
        core.load_state_dict(reference_state)
        cases[backend] = BackendCase(
            core=core,
            value=source_value.detach().clone().requires_grad_(input_gradient),
            initial_e=source_e.detach().clone().requires_grad_(True),
            initial_i=source_i.detach().clone().requires_grad_(True),
        )
    return cases


def _query_output(
    backend: str,
    case: BackendCase,
    query_indices: Tensor,
) -> Tuple[Tensor, E3ScanState]:
    if backend == "bptt":
        output = case.core(case.value, case.state)
        return output.sequence.index_select(1, query_indices), output.state
    output = case.core.forward_multi_query_eligibility(
        case.value,
        query_indices,
        case.state,
        _unchecked=True,
    )
    return output.sequence, output.state


def _loss(output: Tensor, state: E3ScanState) -> Tensor:
    probe = torch.linspace(
        -0.8,
        0.7,
        output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape_as(output)
    layer = state.layers[0]
    return (output * probe).mean() + 0.17 * (
        layer.excitatory.square().mean() - layer.inhibitory.square().mean()
    )


def _gradient_snapshot(case: BackendCase) -> Dict[str, Optional[Tensor]]:
    gradients: Dict[str, Optional[Tensor]] = {
        "input": None if case.value.grad is None else case.value.grad.detach().clone(),
        "initial_e": case.initial_e.grad.detach().clone(),
        "initial_i": case.initial_i.grad.detach().clone(),
    }
    for name, parameter in case.core.named_parameters():
        gradients[f"parameter:{name}"] = (
            None if parameter.grad is None else parameter.grad.detach().clone()
        )
    return gradients


def _max_abs(left: Optional[Tensor], right: Optional[Tensor]) -> Optional[float]:
    if left is None or right is None:
        return None if left is right else float("inf")
    return float((left - right).abs().max())


def _allclose(left: Optional[Tensor], right: Optional[Tensor]) -> bool:
    if left is None or right is None:
        return left is right
    return bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))


def measure_exactness_and_storage(
    backend: str,
    case: BackendCase,
    query_indices: Tensor,
) -> Dict[str, Any]:
    case.clear_gradients()
    counter = _SavedTensorCounter()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        output, state = _query_output(backend, case, query_indices)
        loss = _loss(output, state)
        nodes = _autograd_node_count(output)
        loss.backward()
    return {
        "output": output.detach().clone(),
        "final_e": state.layers[0].excitatory.detach().clone(),
        "final_i": state.layers[0].inhibitory.detach().clone(),
        "gradients": _gradient_snapshot(case),
        "logical_saved_bytes": counter.logical_bytes,
        "unique_storage_bytes": counter.unique_storage_bytes,
        "saved_tensor_count": counter.tensor_count,
        "autograd_nodes": nodes,
    }


def compare_to_reference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "output": _allclose(candidate["output"], reference["output"]),
        "final_e": _allclose(candidate["final_e"], reference["final_e"]),
        "final_i": _allclose(candidate["final_i"], reference["final_i"]),
    }
    max_abs = {
        "output": _max_abs(candidate["output"], reference["output"]),
        "final_e": _max_abs(candidate["final_e"], reference["final_e"]),
        "final_i": _max_abs(candidate["final_i"], reference["final_i"]),
    }
    gradient_checks = {}
    gradient_max_abs = {}
    for name, reference_gradient in reference["gradients"].items():
        candidate_gradient = candidate["gradients"][name]
        gradient_checks[name] = _allclose(candidate_gradient, reference_gradient)
        gradient_max_abs[name] = _max_abs(candidate_gradient, reference_gradient)
    checks["gradients"] = all(gradient_checks.values())
    finite_errors = [
        value
        for value in [*max_abs.values(), *gradient_max_abs.values()]
        if value is not None and math.isfinite(value)
    ]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "gradient_checks": gradient_checks,
        "max_abs": max(finite_errors, default=0.0),
        "component_max_abs": max_abs,
        "gradient_max_abs": gradient_max_abs,
    }


def _timed_runner(
    backend: str,
    case: BackendCase,
    query_indices: Tensor,
) -> None:
    case.clear_gradients()
    output, state = _query_output(backend, case, query_indices)
    _loss(output, state).backward()


def interleaved_latency(
    cases: Mapping[str, BackendCase],
    query_indices: Tensor,
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    for _ in range(warmup):
        for backend, case in cases.items():
            _timed_runner(backend, case, query_indices)
    samples: Dict[str, List[float]] = {backend: [] for backend in cases}
    order = list(cases)
    rng = random.Random(seed)
    for _ in range(repeats):
        rng.shuffle(order)
        for backend in order:
            _sync(device)
            started = time.perf_counter_ns()
            _timed_runner(backend, cases[backend], query_indices)
            _sync(device)
            samples[backend].append((time.perf_counter_ns() - started) / 1e6)
    return {
        backend: {
            "samples_ms": values,
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
        }
        for backend, values in samples.items()
    }


def frozen_dispatch_backend(actual_density: float) -> str:
    """Pre-result heuristic frozen for LDAA-1; later work may train a selector."""

    return "segmented_adjoint" if actual_density <= 1 / 32 else "reverse_adjoint"


def run_cell(
    *,
    time_steps: int,
    nominal_density: float,
    state_dim: int,
    input_gradient: bool,
    config: MatrixConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    query_indices = query_indices_for_density(
        time_steps, nominal_density, device=device
    )
    actual_density = int(query_indices.numel()) / time_steps
    cases = _build_cases(
        seed=seed,
        time_steps=time_steps,
        state_dim=state_dim,
        input_gradient=input_gradient,
        config=config,
        device=device,
    )
    measurements = {
        backend: measure_exactness_and_storage(backend, case, query_indices)
        for backend, case in cases.items()
    }
    reference = measurements["bptt"]
    exactness = {
        backend: (
            {
                "passed": True,
                "checks": {"self_reference": True},
                "max_abs": 0.0,
            }
            if backend == "bptt"
            else compare_to_reference(measurement, reference)
        )
        for backend, measurement in measurements.items()
    }
    latency = interleaved_latency(
        cases,
        query_indices,
        warmup=config.warmup,
        repeats=config.repeats,
        device=device,
        seed=seed + 17,
    )
    backend_records = {}
    for backend in BACKENDS:
        measurement = measurements[backend]
        backend_records[backend] = {
            "exactness": exactness[backend],
            "logical_saved_bytes": measurement["logical_saved_bytes"],
            "unique_storage_bytes": measurement["unique_storage_bytes"],
            "saved_tensor_count": measurement["saved_tensor_count"],
            "autograd_nodes": measurement["autograd_nodes"],
            **latency[backend],
            "speedup_vs_bptt": (
                latency["bptt"]["p50_ms"] / latency[backend]["p50_ms"]
            ),
            "storage_ratio_to_bptt": (
                measurement["unique_storage_bytes"]
                / max(1, reference["unique_storage_bytes"])
            ),
        }
    oracle = min(BACKENDS, key=lambda name: latency[name]["p50_ms"])
    selected = frozen_dispatch_backend(actual_density)
    return {
        "time_steps": time_steps,
        "nominal_density": nominal_density,
        "query_count": int(query_indices.numel()),
        "actual_density": actual_density,
        "state_dim": state_dim,
        "input_gradient": input_gradient,
        "backends": backend_records,
        "oracle_backend": oracle,
        "frozen_dispatch_backend": selected,
        "frozen_dispatch_regret": (
            latency[selected]["p50_ms"] / latency[oracle]["p50_ms"]
        ),
    }


def analyse_matrix(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    exactness_pass = all(
        backend["exactness"]["passed"]
        for cell in cells
        for backend in cell["backends"].values()
    )
    sparse_cells = [cell for cell in cells if cell["actual_density"] <= 1 / 32]
    sparse_pass = [
        cell["backends"]["segmented_adjoint"]["speedup_vs_bptt"] >= 1.5
        and cell["backends"]["segmented_adjoint"]["storage_ratio_to_bptt"] <= 0.25
        for cell in sparse_cells
    ]
    sparse_fraction = sum(sparse_pass) / max(1, len(sparse_pass))
    input_on_sparse = [cell for cell in sparse_cells if cell["input_gradient"]]
    input_on_fraction = sum(
        cell["backends"]["segmented_adjoint"]["speedup_vs_bptt"] >= 1.5
        and cell["backends"]["segmented_adjoint"]["storage_ratio_to_bptt"] <= 0.25
        for cell in input_on_sparse
    ) / max(1, len(input_on_sparse))
    regrets = [float(cell["frozen_dispatch_regret"]) for cell in cells]
    backend_oracle_counts = {
        backend: sum(cell["oracle_backend"] == backend for cell in cells)
        for backend in BACKENDS
    }
    segmented_faster_than_dense = [
        cell["backends"]["reverse_adjoint"]["p50_ms"]
        / cell["backends"]["segmented_adjoint"]["p50_ms"]
        for cell in sparse_cells
    ]
    gates = {
        "H1_exactness": exactness_pass,
        "H2_sparse_fraction": sparse_fraction >= 0.60,
        "H2_input_gradient_on": input_on_fraction >= 0.60,
        "RQ2_static_dispatch_all_within_1p10": all(regret <= 1.10 for regret in regrets),
    }
    if not gates["H1_exactness"]:
        verdict = "NO_GO_EXACTNESS"
    elif gates["H2_sparse_fraction"] and gates["H2_input_gradient_on"]:
        verdict = "OPERATOR_GO_MODEL_VALIDATION_REQUIRED"
    else:
        verdict = "NO_GO_SPARSE_CROSSOVER"
    return {
        "frozen_thresholds": {
            "atol": ATOL,
            "rtol": RTOL,
            "sparse_density_max": 1 / 32,
            "minimum_speedup_vs_bptt": 1.5,
            "maximum_storage_ratio_to_bptt": 0.25,
            "minimum_sparse_pass_fraction": 0.60,
            "maximum_dispatch_regret": 1.10,
        },
        "counts": {
            "cells": len(cells),
            "sparse_cells": len(sparse_cells),
            "input_gradient_on_sparse_cells": len(input_on_sparse),
        },
        "sparse_pass_fraction": sparse_fraction,
        "input_gradient_on_sparse_pass_fraction": input_on_fraction,
        "segmented_vs_dense_sparse_speedup": {
            "mean": float(np.mean(segmented_faster_than_dense)),
            "min": float(np.min(segmented_faster_than_dense)),
            "max": float(np.max(segmented_faster_than_dense)),
        },
        "frozen_dispatch_regret": {
            "mean": float(np.mean(regrets)),
            "p95": float(np.percentile(regrets, 95)),
            "max": float(np.max(regrets)),
            "within_1p10_fraction": sum(regret <= 1.10 for regret in regrets)
            / max(1, len(regrets)),
        },
        "oracle_backend_counts": backend_oracle_counts,
        "gates": gates,
        "verdict": verdict,
    }


def run_matrix(config: MatrixConfig, *, device: torch.device) -> Dict[str, Any]:
    if device.type == "cpu":
        torch.set_num_threads(config.threads)
    cells = []
    index = 0
    for time_steps in config.lengths:
        for density in config.loss_densities:
            for state_dim in config.state_dims:
                for input_gradient in config.input_gradients:
                    cell = run_cell(
                        time_steps=time_steps,
                        nominal_density=density,
                        state_dim=state_dim,
                        input_gradient=input_gradient,
                        config=config,
                        device=device,
                        seed=9_410_000 + index,
                    )
                    cells.append(cell)
                    index += 1
                    segmented = cell["backends"]["segmented_adjoint"]
                    print(
                        f"T={time_steps:5d} K={cell['query_count']:5d} "
                        f"S={state_dim:3d} input_grad={int(input_gradient)} "
                        f"seg/bptt={segmented['speedup_vs_bptt']:.2f}x "
                        f"mem={segmented['storage_ratio_to_bptt']:.3f} "
                        f"exact={int(segmented['exactness']['passed'])}"
                    )
    return {
        "experiment": "LDAA-1 exact operator crossover",
        "status": "FORMAL_OPERATOR_MATRIX",
        "device": device_label(device),
        "environment": _environment(device),
        "config": asdict(config),
        "backends": list(BACKENDS),
        "cells": cells,
        "analysis": analyse_matrix(cells),
    }


def _parse_bool_values(values: Iterable[str]) -> Tuple[bool, ...]:
    lookup = {"off": False, "on": True, "false": False, "true": True}
    parsed = []
    for value in values:
        if value.lower() not in lookup:
            raise ValueError(f"invalid input-gradient value: {value}")
        parsed.append(lookup[value.lower()])
    return tuple(parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--lengths", nargs="+", type=int, default=[512, 2048, 8192, 32768])
    parser.add_argument(
        "--loss-densities",
        nargs="+",
        type=float,
        default=[1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0],
    )
    parser.add_argument("--state-dims", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--input-gradients", nargs="+", default=["off", "on"])
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/e3_scan/e3_ldaa1_operator_crossover.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MatrixConfig(
        lengths=tuple(args.lengths),
        loss_densities=tuple(args.loss_densities),
        state_dims=tuple(args.state_dims),
        input_gradients=_parse_bool_values(args.input_gradients),
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        threads=args.threads,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    device = choose_device(args.device)
    payload = run_matrix(config, device=device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={args.out}")
    print(json.dumps(payload["analysis"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
