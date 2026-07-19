"""SG23C exact CUDA dual and effective-rank Woodbury experiments.

The runner keeps feature construction and deterministic pivot selection visible,
then benchmarks the declared linear-algebra stages on both the AutoDL CPU and
CUDA device.  CUDA timings are synchronized and split into transfer, cold, and
device-resident measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments import e3_sg23_spike_feature_solvers as sg23  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg23c_cuda_adaptive_krr.json")
DEFAULT_STRESS_SIZES = (1024, 2048, 4096)
DEFAULT_THREADS = 16
DEFAULT_WARMUPS = 2
DEFAULT_REPETITIONS = 7
DEFAULT_RANK_CAP = 1024
DTYPE = torch.float64
FEATURE_TOLERANCE = sg23.FEATURE_GRAM_TOLERANCE
SCORE_TOLERANCE = sg23.EXACT_SCORE_TOLERANCE
RANK_RECONSTRUCTION_TOLERANCE = 1e-10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def timing_summary(samples_seconds: Sequence[float]) -> Dict[str, float]:
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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(
    operation: Callable[[], Dict[str, torch.Tensor]], device: torch.device
) -> Tuple[Dict[str, torch.Tensor], float]:
    _synchronize(device)
    started = time.perf_counter_ns()
    result = operation()
    _synchronize(device)
    return result, (time.perf_counter_ns() - started) / 1e9


def benchmark_operation(
    operation: Callable[[], Dict[str, torch.Tensor]],
    *,
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    cold_result, cold_seconds = _timed_call(operation, device)
    del cold_result
    for _ in range(warmups):
        warm_result, _seconds = _timed_call(operation, device)
        del warm_result
    samples = []
    result: Optional[Dict[str, torch.Tensor]] = None
    for _ in range(repetitions):
        result, seconds = _timed_call(operation, device)
        samples.append(seconds)
    if result is None:
        raise AssertionError("benchmark produced no result")
    return result, {
        "cold_seconds": cold_seconds,
        "warmups": warmups,
        "repetitions": repetitions,
        "resident": timing_summary(samples),
    }


def _coo_on_device(
    features: sg23.ExplicitFeatureMatrix, device: torch.device
) -> Tuple[torch.Tensor, float]:
    started = time.perf_counter_ns()
    coo = features.matrix.to_sparse_coo().coalesce().to(
        device=device, dtype=DTYPE
    )
    _synchronize(device)
    return coo, (time.perf_counter_ns() - started) / 1e9


def _dense_dual_operation(
    feature_matrix: torch.Tensor,
    counts: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    sparse_kernel = torch.sparse.mm(
        feature_matrix, feature_matrix.transpose(0, 1)
    )
    kernel = sparse_kernel.to_dense()
    sqrt_counts = counts.sqrt()
    system = (
        sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
        + sg19.FROZEN_LAMBDA
        * torch.eye(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    )
    rhs = sqrt_counts[:, None] * targets
    factor = torch.linalg.cholesky(system)
    dual = torch.cholesky_solve(rhs, factor)
    coefficients = sqrt_counts[:, None] * dual
    scores = kernel @ coefficients
    residual = system @ dual - rhs
    relative_residual = torch.linalg.vector_norm(
        residual, dim=0
    ) / torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)
    return {
        "kernel": kernel,
        "coefficients": coefficients,
        "scores": scores,
        "maximum_relative_residual": relative_residual.max(),
    }


def _cpu_dual_from_kernel(
    kernel: torch.Tensor,
    counts: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if kernel.device.type != "cpu":
        raise ValueError("hybrid final solve requires a CPU kernel")
    sqrt_counts = counts.sqrt()
    system = (
        sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
        + sg19.FROZEN_LAMBDA
        * torch.eye(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    )
    rhs = sqrt_counts[:, None] * targets
    factor = torch.linalg.cholesky(system)
    dual = torch.cholesky_solve(rhs, factor)
    coefficients = sqrt_counts[:, None] * dual
    scores = kernel @ coefficients
    residual = system @ dual - rhs
    relative_residual = torch.linalg.vector_norm(
        residual, dim=0
    ) / torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)
    return {
        "kernel": kernel,
        "coefficients": coefficients,
        "scores": scores,
        "maximum_relative_residual": relative_residual.max(),
    }


def _hybrid_dense_dual_operation(
    feature_matrix_cuda: torch.Tensor,
    counts_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    *,
    canonical_grid_denominator: Optional[int] = None,
) -> Dict[str, Any]:
    torch.cuda.synchronize(feature_matrix_cuda.device)
    gram_started = time.perf_counter_ns()
    sparse_kernel = torch.sparse.mm(
        feature_matrix_cuda, feature_matrix_cuda.transpose(0, 1)
    )
    kernel_cuda = sparse_kernel.to_dense()
    torch.cuda.synchronize(feature_matrix_cuda.device)
    gram_seconds = (time.perf_counter_ns() - gram_started) / 1e9

    transfer_started = time.perf_counter_ns()
    kernel_cpu = kernel_cuda.cpu()
    device_to_host_seconds = (time.perf_counter_ns() - transfer_started) / 1e9
    canonicalization_max_abs_delta = 0.0
    if canonical_grid_denominator is not None:
        canonical_kernel = (
            torch.round(kernel_cpu * canonical_grid_denominator)
            / canonical_grid_denominator
        )
        canonicalization_max_abs_delta = float(
            (canonical_kernel - kernel_cpu).abs().max().item()
        )
        kernel_cpu = canonical_kernel

    solve_started = time.perf_counter_ns()
    result: Dict[str, Any] = _cpu_dual_from_kernel(
        kernel_cpu, counts_cpu, targets_cpu
    )
    cpu_solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    result.update(
        {
            "cuda_gram_seconds": gram_seconds,
            "device_to_host_seconds": device_to_host_seconds,
            "cpu_solve_seconds": cpu_solve_seconds,
            "canonical_grid_denominator": canonical_grid_denominator,
            "canonicalization_max_abs_delta": (
                canonicalization_max_abs_delta
            ),
        }
    )
    return result


def benchmark_hybrid_dense_dual(
    features: sg23.ExplicitFeatureMatrix,
    states: Mapping[str, Any],
    *,
    warmups: int,
    repetitions: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    feature_matrix, input_transfer_seconds = _coo_on_device(features, device)
    counts = states["counts"].to(dtype=DTYPE, device="cpu")
    targets = states["target_means"].to(dtype=DTYPE, device="cpu")
    operation = lambda: _hybrid_dense_dual_operation(
        feature_matrix,
        counts,
        targets,
        canonical_grid_denominator=sg23.AFFORDANCE_MASK_WIDTH,
    )
    cold_result, cold_seconds = _timed_call(operation, device)
    del cold_result
    for _ in range(warmups):
        warm_result, _seconds = _timed_call(operation, device)
        del warm_result
    totals = []
    gram_samples = []
    transfer_samples = []
    solve_samples = []
    result: Optional[Dict[str, Any]] = None
    for _ in range(repetitions):
        result, total_seconds = _timed_call(operation, device)
        totals.append(total_seconds)
        gram_samples.append(float(result["cuda_gram_seconds"]))
        transfer_samples.append(float(result["device_to_host_seconds"]))
        solve_samples.append(float(result["cpu_solve_seconds"]))
    if result is None:
        raise AssertionError("hybrid benchmark produced no result")
    row_count = features.row_count
    element_bytes = torch.empty((), dtype=DTYPE).element_size()
    metrics = {
        "device_contract": "cuda_sparse_gram_then_cpu_fp64_cholesky",
        "cuda_stage_device": str(device),
        "cpu_stage_device": "cpu",
        "dtype": str(DTYPE),
        "input_transfer_seconds": input_transfer_seconds,
        "cold_seconds": cold_seconds,
        "warmups": warmups,
        "repetitions": repetitions,
        "total": timing_summary(totals),
        "cuda_gram": timing_summary(gram_samples),
        "device_to_host": timing_summary(transfer_samples),
        "cpu_solve": timing_summary(solve_samples),
        "maximum_relative_residual": float(
            result["maximum_relative_residual"].detach().cpu().item()
        ),
        "canonical_grid_denominator": result["canonical_grid_denominator"],
        "canonicalization_max_abs_delta": result[
            "canonicalization_max_abs_delta"
        ],
        "kernel_plus_system_logical_bytes": (
            2 * row_count * row_count * element_bytes
        ),
        "cuda_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "cuda_peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }
    metrics["end_to_end_median_seconds_excluding_feature_build"] = float(
        input_transfer_seconds + metrics["total"]["median_seconds"]
    )
    return result, metrics


def benchmark_dense_dual(
    features: sg23.ExplicitFeatureMatrix,
    states: Mapping[str, Any],
    *,
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feature_matrix, transfer_seconds = _coo_on_device(features, device)
    transfer_started = time.perf_counter_ns()
    counts = states["counts"].to(device=device, dtype=DTYPE)
    targets = states["target_means"].to(device=device, dtype=DTYPE)
    _synchronize(device)
    state_transfer_seconds = (time.perf_counter_ns() - transfer_started) / 1e9

    result, timing = benchmark_operation(
        lambda: _dense_dual_operation(feature_matrix, counts, targets),
        device=device,
        warmups=warmups,
        repetitions=repetitions,
    )
    row_count = features.row_count
    element_bytes = torch.empty((), dtype=DTYPE).element_size()
    metrics: Dict[str, Any] = {
        "device": str(device),
        "dtype": str(DTYPE),
        "feature_transfer_seconds": transfer_seconds,
        "state_transfer_seconds": state_transfer_seconds,
        "transfer_seconds": transfer_seconds + state_transfer_seconds,
        **timing,
        "maximum_relative_residual": float(
            result["maximum_relative_residual"].detach().cpu().item()
        ),
        "kernel_logical_bytes": row_count * row_count * element_bytes,
        "kernel_plus_system_logical_bytes": (
            2 * row_count * row_count * element_bytes
        ),
        "kernel_system_factor_logical_bytes": (
            3 * row_count * row_count * element_bytes
        ),
    }
    metrics["end_to_end_median_seconds_excluding_feature_build"] = float(
        metrics["transfer_seconds"] + metrics["resident"]["median_seconds"]
    )
    if device.type == "cuda":
        metrics["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        metrics["cuda_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    return result, metrics


def _rank_operation(
    factor: torch.Tensor,
    counts: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    sqrt_counts = counts.sqrt()
    weighted_factor = sqrt_counts[:, None] * factor
    rhs = sqrt_counts[:, None] * targets
    small_system = (
        weighted_factor.T @ weighted_factor
        + sg19.FROZEN_LAMBDA
        * torch.eye(
            factor.shape[1], device=factor.device, dtype=factor.dtype
        )
    )
    projected_rhs = weighted_factor.T @ rhs
    cholesky = torch.linalg.cholesky(small_system)
    theta = torch.cholesky_solve(projected_rhs, cholesky)
    scores = factor @ theta
    residual = small_system @ theta - projected_rhs
    relative_residual = torch.linalg.vector_norm(
        residual, dim=0
    ) / torch.linalg.vector_norm(projected_rhs, dim=0).clamp_min(1e-30)
    return {
        "theta": theta,
        "scores": scores,
        "maximum_relative_residual": relative_residual.max(),
    }


def benchmark_effective_rank(
    factor_cpu: torch.Tensor,
    states: Mapping[str, Any],
    *,
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter_ns()
    factor = factor_cpu.to(device=device, dtype=DTYPE)
    counts = states["counts"].to(device=device, dtype=DTYPE)
    targets = states["target_means"].to(device=device, dtype=DTYPE)
    _synchronize(device)
    transfer_seconds = (time.perf_counter_ns() - started) / 1e9
    result, timing = benchmark_operation(
        lambda: _rank_operation(factor, counts, targets),
        device=device,
        warmups=warmups,
        repetitions=repetitions,
    )
    row_count, rank = factor_cpu.shape
    output_width = int(states["target_means"].shape[1])
    element_bytes = torch.empty((), dtype=DTYPE).element_size()
    logical_bytes = element_bytes * (
        row_count * rank
        + 2 * rank * rank
        + rank * output_width
        + row_count * output_width
    )
    metrics: Dict[str, Any] = {
        "device": str(device),
        "dtype": str(DTYPE),
        "rank": int(rank),
        "transfer_seconds": transfer_seconds,
        **timing,
        "maximum_relative_residual": float(
            result["maximum_relative_residual"].detach().cpu().item()
        ),
        "factor_system_solver_logical_bytes": int(logical_bytes),
    }
    metrics["end_to_end_median_seconds_excluding_factor_build"] = float(
        transfer_seconds + metrics["resident"]["median_seconds"]
    )
    if device.type == "cuda":
        metrics["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        metrics["cuda_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    return result, metrics


def _prediction_equivalence(
    first: torch.Tensor, second: torch.Tensor
) -> bool:
    return sg23._prediction_equivalence(
        first.detach().cpu(), second.detach().cpu()
    )


def _maximum_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        (first.detach().cpu() - second.detach().cpu()).abs().max().item()
    )


def _factor_model_coefficients(
    factor: torch.Tensor,
    pivots: Sequence[int],
    theta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    pivot_index = torch.tensor(tuple(pivots), dtype=torch.long)
    pivot_factor = factor.index_select(0, pivot_index)
    gamma = torch.linalg.solve_triangular(
        pivot_factor.T,
        theta,
        upper=True,
    )
    coefficients = torch.zeros(
        factor.shape[0], theta.shape[1], dtype=torch.float64
    )
    coefficients.index_copy_(0, pivot_index, gamma)
    return coefficients, gamma, {
        "gamma_all_finite": bool(torch.isfinite(gamma).all()),
        "minimum_abs_pivot_diagonal": float(
            torch.diagonal(pivot_factor).abs().min().item()
        ),
        "maximum_abs_gamma": float(gamma.abs().max().item()),
    }


def _benchmark_case(
    states: Mapping[str, Any],
    *,
    warmups: int,
    repetitions: int,
    rank_cap: int,
    include_quality_problem: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    features = sg23.build_explicit_features(states)
    cpu = torch.device("cpu")
    cuda = torch.device("cuda:0")
    cpu_dense_result, cpu_dense = benchmark_dense_dual(
        features,
        states,
        device=cpu,
        warmups=warmups,
        repetitions=repetitions,
    )
    gpu_dense_result, gpu_dense = benchmark_dense_dual(
        features,
        states,
        device=cuda,
        warmups=warmups,
        repetitions=repetitions,
    )
    hybrid_result, hybrid = benchmark_hybrid_dense_dual(
        features,
        states,
        warmups=warmups,
        repetitions=repetitions,
    )
    explicit_kernel = cpu_dense_result["kernel"].detach().cpu()
    analytic_coefficients, analytic_metrics, analytic_kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    analytic_scores = analytic_kernel @ analytic_coefficients
    feature_error = _maximum_difference(explicit_kernel, analytic_kernel)

    factor_started = time.perf_counter_ns()
    factor, pivots, _rank_times, pivot = sg23.pivoted_cholesky(
        explicit_kernel,
        maximum_rank=min(rank_cap, int(explicit_kernel.shape[0])),
    )
    factor_build_seconds = (time.perf_counter_ns() - factor_started) / 1e9
    cpu_rank_result, cpu_rank = benchmark_effective_rank(
        factor,
        states,
        device=cpu,
        warmups=warmups,
        repetitions=repetitions,
    )
    gpu_rank_result, gpu_rank = benchmark_effective_rank(
        factor,
        states,
        device=cuda,
        warmups=warmups,
        repetitions=repetitions,
    )

    dense_gpu_scores = gpu_dense_result["scores"]
    dense_cpu_scores = cpu_dense_result["scores"]
    hybrid_scores = hybrid_result["scores"]
    rank_gpu_scores = gpu_rank_result["scores"]
    rank_cpu_scores = cpu_rank_result["scores"]
    coefficients, gamma, coefficient_audit = _factor_model_coefficients(
        factor, pivots, gpu_rank_result["theta"].detach().cpu()
    )
    landmark_scores = explicit_kernel.index_select(
        1, torch.tensor(tuple(pivots), dtype=torch.long)
    ) @ gamma
    rank_reconstruction_error = float(
        (explicit_kernel - factor @ factor.T).abs().max().item()
    )
    dense_speedup = (
        cpu_dense["resident"]["median_seconds"]
        / gpu_dense["resident"]["median_seconds"]
    )
    hybrid_speedup = (
        cpu_dense["resident"]["median_seconds"]
        / hybrid["total"]["median_seconds"]
    )
    rank_speedup = (
        cpu_rank["resident"]["median_seconds"]
        / gpu_rank["resident"]["median_seconds"]
    )
    memory_ratio = (
        gpu_dense["kernel_plus_system_logical_bytes"]
        / gpu_rank["factor_system_solver_logical_bytes"]
    )
    result: Dict[str, Any] = {
        "shape": {
            "rows": features.row_count,
            "features": features.feature_count,
            "nnz": features.nnz,
            "feature_over_row_ratio": features.feature_count / features.row_count,
        },
        "feature_map": {
            "build_seconds": features.build_seconds,
            "logical_csr_bytes": features.logical_csr_bytes,
            "vocabulary_sha256": features.vocabulary_sha256,
            "full_gram_max_abs_error_vs_analytic": feature_error,
        },
        "analytic_dense_reference": analytic_metrics,
        "dense_dual": {
            "cpu": cpu_dense,
            "cuda": gpu_dense,
            "cuda_resident_speedup_over_same_host_cpu": dense_speedup,
            "cuda_vs_cpu_explicit_score_max_abs_difference": _maximum_difference(
                dense_gpu_scores, dense_cpu_scores
            ),
            "cuda_vs_analytic_score_max_abs_difference": _maximum_difference(
                dense_gpu_scores, analytic_scores
            ),
            "cuda_vs_analytic_prediction_equivalent": _prediction_equivalence(
                dense_gpu_scores, analytic_scores
            ),
        },
        "hybrid_exact": {
            "metrics": hybrid,
            "total_speedup_over_same_host_cpu": hybrid_speedup,
            "vs_cpu_explicit_score_max_abs_difference": _maximum_difference(
                hybrid_scores, dense_cpu_scores
            ),
            "vs_analytic_score_max_abs_difference": _maximum_difference(
                hybrid_scores, analytic_scores
            ),
            "vs_analytic_prediction_equivalent": _prediction_equivalence(
                hybrid_scores, analytic_scores
            ),
        },
        "effective_rank_woodbury": {
            "rank_cap": rank_cap,
            "effective_rank": int(factor.shape[1]),
            "factor_build_seconds_cpu": factor_build_seconds,
            "factor": pivot,
            "reconstruction_max_abs_error_vs_explicit": (
                rank_reconstruction_error
            ),
            "cpu": cpu_rank,
            "cuda": gpu_rank,
            "cuda_resident_speedup_over_same_host_cpu": rank_speedup,
            "dense_over_rank_logical_memory_ratio": memory_ratio,
            "cuda_vs_cpu_rank_score_max_abs_difference": _maximum_difference(
                rank_gpu_scores, rank_cpu_scores
            ),
            "cuda_vs_explicit_dense_score_max_abs_difference": _maximum_difference(
                rank_gpu_scores, dense_cpu_scores
            ),
            "cuda_vs_analytic_score_max_abs_difference": _maximum_difference(
                rank_gpu_scores, analytic_scores
            ),
            "cuda_vs_analytic_prediction_equivalent": _prediction_equivalence(
                rank_gpu_scores, analytic_scores
            ),
            "landmark_vs_direct_rank_score_max_abs_difference": (
                _maximum_difference(landmark_scores, rank_gpu_scores)
            ),
            "landmark_model": coefficient_audit,
        },
    }
    result["dense_dual"]["cuda"]["end_to_end_median_seconds"] = float(
        features.build_seconds
        + result["dense_dual"]["cuda"][
            "end_to_end_median_seconds_excluding_feature_build"
        ]
    )
    result["hybrid_exact"]["metrics"]["end_to_end_median_seconds"] = float(
        features.build_seconds
        + result["hybrid_exact"]["metrics"][
            "end_to_end_median_seconds_excluding_feature_build"
        ]
    )
    result["effective_rank_woodbury"]["cuda"][
        "end_to_end_median_seconds"
    ] = float(
        features.build_seconds
        + factor_build_seconds
        + result["effective_rank_woodbury"]["cuda"][
            "end_to_end_median_seconds_excluding_factor_build"
        ]
    )
    if include_quality_problem is not None:
        dense_coefficients = gpu_dense_result["coefficients"].detach().cpu()
        dense_quality = sg23._evaluate_exact_route(
            include_quality_problem, dense_coefficients
        )
        dense_quality["perfect_quality"] = sg23._quality_is_perfect(
            dense_quality
        )
        hybrid_quality = sg23._evaluate_exact_route(
            include_quality_problem,
            hybrid_result["coefficients"].detach().cpu(),
        )
        hybrid_quality["perfect_quality"] = sg23._quality_is_perfect(
            hybrid_quality
        )
        rank_quality = sg23._evaluate_exact_route(
            include_quality_problem, coefficients
        )
        rank_quality["perfect_quality"] = sg23._quality_is_perfect(
            rank_quality
        )
        result["quality"] = {
            "dense_cuda": dense_quality,
            "hybrid_exact": hybrid_quality,
            "effective_rank_cuda": rank_quality,
        }
    return result


def _command_output(command: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            tuple(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _environment() -> Dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": tuple(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_bytes": int(properties.total_memory),
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": _command_output(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            )
        ),
        "git_commit": _command_output(("git", "rev-parse", "HEAD")),
        "cpu_threads": torch.get_num_threads(),
        "pid": os.getpid(),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG23C requires a CUDA device")
    if torch.cuda.get_device_capability(0)[0] < 6:
        raise RuntimeError("SG23C formal FP64 route requires CUDA capability >= 6")
    torch.set_num_threads(args.threads)
    problem = sg23._load_problem(args)
    real = _benchmark_case(
        problem["unique"],
        warmups=args.warmups,
        repetitions=args.repetitions,
        rank_cap=args.rank_cap,
        include_quality_problem=problem,
    )
    stress = {}
    for size in args.stress_sizes:
        states, generation = sg23.generate_stress_states(problem["unique"], size)
        case = _benchmark_case(
            states,
            warmups=args.warmups,
            repetitions=args.repetitions,
            rank_cap=args.rank_cap,
        )
        case["generation"] = generation
        stress[str(size)] = case

    cases = (real, *stress.values())
    backend_gate = bool(
        all(
            case["dense_dual"]["cuda"]["device"] == "cuda:0"
            and case["effective_rank_woodbury"]["cuda"]["device"] == "cuda:0"
            and case["hybrid_exact"]["metrics"]["cuda_stage_device"]
            == "cuda:0"
            for case in cases
        )
    )
    feature_gate = bool(
        all(
            case["feature_map"]["full_gram_max_abs_error_vs_analytic"]
            <= FEATURE_TOLERANCE
            for case in cases
        )
    )
    dense_exact_gate = bool(
        all(
            case["dense_dual"][
                "cuda_vs_analytic_score_max_abs_difference"
            ]
            <= SCORE_TOLERANCE
            and case["dense_dual"][
                "cuda_vs_analytic_prediction_equivalent"
            ]
            for case in cases
        )
    )
    hybrid_exact_gate = bool(
        all(
            case["hybrid_exact"][
                "vs_analytic_score_max_abs_difference"
            ]
            <= SCORE_TOLERANCE
            and case["hybrid_exact"]["vs_analytic_prediction_equivalent"]
            for case in cases
        )
    )
    hybrid_grid_gate = bool(
        all(
            case["hybrid_exact"]["metrics"][
                "canonicalization_max_abs_delta"
            ]
            <= FEATURE_TOLERANCE
            for case in cases
        )
    )
    rank_exact_gate = bool(
        all(
            case["effective_rank_woodbury"][
                "reconstruction_max_abs_error_vs_explicit"
            ]
            <= RANK_RECONSTRUCTION_TOLERANCE
            and case["effective_rank_woodbury"][
                "cuda_vs_analytic_score_max_abs_difference"
            ]
            <= SCORE_TOLERANCE
            and case["effective_rank_woodbury"][
                "cuda_vs_analytic_prediction_equivalent"
            ]
            and case["effective_rank_woodbury"]["landmark_model"][
                "gamma_all_finite"
            ]
            for case in cases
        )
    )
    quality_gate = bool(
        real["quality"]["dense_cuda"]["perfect_quality"]
        and real["quality"]["hybrid_exact"]["perfect_quality"]
        and real["quality"]["effective_rank_cuda"]["perfect_quality"]
    )
    cuda_speed_gate = bool(
        any(
            case["dense_dual"][
                "cuda_resident_speedup_over_same_host_cpu"
            ]
            > 1.0
            or case["effective_rank_woodbury"][
                "cuda_resident_speedup_over_same_host_cpu"
            ]
            > 1.0
            or case["hybrid_exact"][
                "total_speedup_over_same_host_cpu"
            ]
            > 1.0
            for case in cases
        )
    )
    largest = stress[str(max(args.stress_sizes))]
    scale_memory_gate = bool(
        largest["effective_rank_woodbury"][
            "dense_over_rank_logical_memory_ratio"
        ]
        >= 2.0
        and largest["effective_rank_woodbury"][
            "cuda_vs_analytic_score_max_abs_difference"
        ]
        <= SCORE_TOLERANCE
    )
    provenance_gate = bool(
        problem["graph_audit"]["all_masks_match_exhaustive_cache"]
        and problem["graph_audit"]["all_binding_steps_precede_snapshot_root"]
        and problem["graph_audit"]["all_rooms_unique_and_present"]
        and problem["graph_audit"]["no_edge_conflicts"]
        and problem["constraint_audit"]["all_targets_match"]
    )
    overall = bool(
        provenance_gate
        and backend_gate
        and feature_gate
        and hybrid_grid_gate
        and hybrid_exact_gate
        and rank_exact_gate
        and quality_gate
        and cuda_speed_gate
        and scale_memory_gate
    )
    return {
        "experiment": "E3-SG23C AutoDL CUDA exact dual and effective-rank Woodbury",
        "references": {
            "sg22r_sha256": problem["reference_sha"],
            "cache_sha256": problem["cache_sha"],
            "sg23_source_sha256": _sha256_file(
                ROOT / "experiments" / "e3_sg23_spike_feature_solvers.py"
            ),
            "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "environment": _environment(),
        "protocol": {
            "threads": args.threads,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "stress_sizes": tuple(args.stress_sizes),
            "rank_cap": args.rank_cap,
            "dtype": str(DTYPE),
            "lambda": sg19.FROZEN_LAMBDA,
            "feature_tolerance": FEATURE_TOLERANCE,
            "score_tolerance": SCORE_TOLERANCE,
            "rank_reconstruction_tolerance": RANK_RECONSTRUCTION_TOLERANCE,
            "cuda_timing_contract": "synchronize_before_and_after",
        },
        "real_443": real,
        "stress": stress,
        "decision": {
            "provenance_gate": provenance_gate,
            "backend_gate": backend_gate,
            "feature_math_gate": feature_gate,
            "dense_cuda_exactness_gate": dense_exact_gate,
            "hybrid_exactness_gate": hybrid_exact_gate,
            "hybrid_grid_audit_gate": hybrid_grid_gate,
            "effective_rank_exactness_gate": rank_exact_gate,
            "real_quality_gate": quality_gate,
            "cuda_speed_gate": cuda_speed_gate,
            "scale_memory_gate": scale_memory_gate,
            "overall": "PASS" if overall else "FAIL",
            "next_route": (
                "sg24_same_v100_raw_language_ann_comparison"
                if overall
                else "sg23c_residual_correction_or_deflation"
            ),
        },
    }


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus-dir", type=Path, default=sg23.DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=sg23.DEFAULT_CACHE)
    parser.add_argument(
        "--sg22r-reference", type=Path, default=sg23.DEFAULT_SG22R_REFERENCE
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--rank-cap", type=int, default=DEFAULT_RANK_CAP)
    parser.add_argument(
        "--stress-sizes",
        type=_parse_int_tuple,
        default=DEFAULT_STRESS_SIZES,
    )
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.repetitions <= 0 or args.rank_cap <= 0:
        parser.error("threads, repetitions, and rank cap must be positive")
    if args.warmups < 0:
        parser.error("warmups must be non-negative")
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
