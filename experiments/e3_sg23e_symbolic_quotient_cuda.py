"""SG23E symbolic support quotient and identifiable CUDA solve."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments import e3_sg23_spike_feature_solvers as sg23  # noqa: E402
from experiments import e3_sg23c_cuda_adaptive_krr as sg23c  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg23e_symbolic_quotient_cuda.json")
DEFAULT_SG23C_REFERENCE = Path(
    "results/e3_scan/e3_sg23c_cuda_adaptive_krr.json"
)
SG23C_REFERENCE_SHA256 = (
    "17DDD84BCB1B010D13F19C567735BA9498534BE122B473E94407659FA6BA162B"
)
SG23C_EXPERIMENT = (
    "E3-SG23C AutoDL CUDA exact dual and effective-rank Woodbury"
)
DEFAULT_SG23D_REFERENCE = Path(
    "results/e3_scan/e3_sg23d_phase_block_cuda.json"
)
SG23D_REFERENCE_SHA256 = (
    "88074887CF2999E02EA75D2D1F366EA1FD2723CE3DEE3C0514D555F317BF06F8"
)
SG23D_EXPERIMENT = "E3-SG23D exact phase blocks and CUDA independent solves"
DEFAULT_STRESS_SIZES = sg23c.DEFAULT_STRESS_SIZES
DEFAULT_THREADS = sg23c.DEFAULT_THREADS
DEFAULT_WARMUPS = sg23c.DEFAULT_WARMUPS
DEFAULT_REPETITIONS = sg23c.DEFAULT_REPETITIONS
SYMBOLIC_ROUND_TOLERANCE = 1e-9
CPU_CUDA_SCORE_TOLERANCE = 1e-9
BACKWARD_ERROR_TOLERANCE = 1e-12


@dataclass
class SymbolicQuotient:
    support_matrix: torch.Tensor
    group_weights: torch.Tensor
    basis: torch.Tensor
    transform: torch.Tensor
    metric: torch.Tensor
    pivots: Tuple[int, ...]
    original_to_group: torch.Tensor
    original_amplitudes: torch.Tensor
    metrics: Dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _gf2_pivots(packed_supports: np.ndarray) -> Tuple[int, ...]:
    basis: Dict[int, int] = {}
    pivots = []
    for column, packed in enumerate(packed_supports):
        vector = int.from_bytes(packed.tobytes(), byteorder="big")
        while vector:
            bit = vector.bit_length() - 1
            previous = basis.get(bit)
            if previous is None:
                basis[bit] = vector
                pivots.append(column)
                break
            vector ^= previous
    return tuple(pivots)


def build_symbolic_quotient(
    features: sg23.ExplicitFeatureMatrix,
    *,
    coordinate_device: torch.device,
) -> SymbolicQuotient:
    total_started = time.perf_counter_ns()
    coo = features.matrix.to_sparse_coo().coalesce()
    indices = coo.indices().cpu().numpy()
    values = coo.values().cpu().numpy()
    row_count, feature_count = features.matrix.shape

    grouping_started = time.perf_counter_ns()
    supports = np.zeros((feature_count, row_count), dtype=np.uint8)
    supports[indices[1], indices[0]] = 1
    amplitudes = np.zeros(feature_count, dtype=np.float64)
    amplitudes[indices[1]] = values
    constant_error = float(
        np.max(np.abs(values - amplitudes[indices[1]]), initial=0.0)
    )
    packed = np.packbits(supports, axis=1, bitorder="big")
    unique_packed, inverse = np.unique(
        packed, axis=0, return_inverse=True
    )
    quotient_support = np.unpackbits(
        unique_packed, axis=1, count=row_count, bitorder="big"
    ).T
    raw_weights = np.bincount(
        inverse,
        weights=np.square(amplitudes),
        minlength=unique_packed.shape[0],
    )
    snapped_weights = np.round(raw_weights * sg23.AFFORDANCE_MASK_WIDTH) / (
        sg23.AFFORDANCE_MASK_WIDTH
    )
    weight_grid_error = float(
        np.max(np.abs(raw_weights - snapped_weights), initial=0.0)
    )
    grouping_seconds = (time.perf_counter_ns() - grouping_started) / 1e9

    elimination_started = time.perf_counter_ns()
    pivots = _gf2_pivots(unique_packed)
    elimination_seconds = (
        time.perf_counter_ns() - elimination_started
    ) / 1e9
    support_matrix = torch.from_numpy(quotient_support).to(torch.float64)
    group_weights = torch.from_numpy(snapped_weights).to(torch.float64)
    pivot_index = torch.tensor(pivots, dtype=torch.long)
    basis = support_matrix.index_select(1, pivot_index)

    coordinate_started = time.perf_counter_ns()
    basis_device = basis.to(device=coordinate_device)
    support_device = support_matrix.to(device=coordinate_device)
    sg23c._synchronize(coordinate_device)
    coordinate_float = torch.linalg.lstsq(
        basis_device, support_device, driver="gels"
    ).solution
    sg23c._synchronize(coordinate_device)
    coordinate_seconds = (time.perf_counter_ns() - coordinate_started) / 1e9
    coordinate_float_cpu = coordinate_float.detach().cpu()
    transform = coordinate_float_cpu.round()
    coordinate_round_error = float(
        (coordinate_float_cpu - transform).abs().max().item()
    )
    reconstruction = basis @ transform
    integer_reconstruction_error = float(
        (reconstruction - support_matrix).abs().max().item()
    )
    metric = (transform * group_weights[None, :]) @ transform.T
    original_to_group = torch.from_numpy(inverse.astype(np.int64))
    original_amplitudes = torch.from_numpy(amplitudes)
    total_seconds = (time.perf_counter_ns() - total_started) / 1e9
    return SymbolicQuotient(
        support_matrix=support_matrix,
        group_weights=group_weights,
        basis=basis,
        transform=transform,
        metric=metric,
        pivots=pivots,
        original_to_group=original_to_group,
        original_amplitudes=original_amplitudes,
        metrics={
            "rows": int(row_count),
            "original_feature_count": int(feature_count),
            "support_group_count": int(unique_packed.shape[0]),
            "gf2_rank": len(pivots),
            "constant_column_max_abs_error": constant_error,
            "weight_grid_max_abs_error": weight_grid_error,
            "coordinate_round_max_abs_error": coordinate_round_error,
            "integer_reconstruction_max_abs_error": (
                integer_reconstruction_error
            ),
            "maximum_abs_integer_coordinate": int(
                transform.abs().max().item()
            ),
            "grouping_seconds": grouping_seconds,
            "gf2_elimination_seconds": elimination_seconds,
            "cuda_coordinate_seconds": coordinate_seconds,
            "total_build_seconds": total_seconds,
        },
    )


def _symbolic_operation(
    basis: torch.Tensor,
    metric: torch.Tensor,
    counts: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    moment = basis.T @ (counts[:, None] * basis)
    target_moment = basis.T @ (counts[:, None] * targets)
    system = metric @ moment + sg19.FROZEN_LAMBDA * torch.eye(
        metric.shape[0], device=metric.device, dtype=metric.dtype
    )
    rhs = metric @ target_moment
    beta = torch.linalg.solve(system, rhs)
    scores = basis @ beta
    residual = system @ beta - rhs
    relative_residual = torch.linalg.vector_norm(
        residual, dim=0
    ) / torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)
    return {
        "beta": beta,
        "scores": scores,
        "maximum_small_system_relative_residual": relative_residual.max(),
    }


def benchmark_symbolic(
    quotient: SymbolicQuotient,
    states: Mapping[str, Any],
    *,
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats()
    transfer_started = time.perf_counter_ns()
    basis = quotient.basis.to(device=device, dtype=torch.float64)
    metric = quotient.metric.to(device=device, dtype=torch.float64)
    counts = states["counts"].to(device=device, dtype=torch.float64)
    targets = states["target_means"].to(device=device, dtype=torch.float64)
    sg23c._synchronize(device)
    transfer_seconds = (time.perf_counter_ns() - transfer_started) / 1e9
    result, timing = sg23c.benchmark_operation(
        lambda: _symbolic_operation(basis, metric, counts, targets),
        device=device,
        warmups=warmups,
        repetitions=repetitions,
    )
    row_count, rank = quotient.basis.shape
    group_count = int(quotient.support_matrix.shape[1])
    feature_count = int(quotient.original_to_group.numel())
    output_width = int(states["target_means"].shape[1])
    conservative_bytes = int(
        row_count * rank * 8
        + 2 * rank * rank * 8
        + rank * group_count * 2
        + feature_count * (4 + 8)
        + rank * output_width * 8
        + row_count * output_width * 8
        + feature_count * output_width * 8
    )
    metrics: Dict[str, Any] = {
        "device": str(device),
        "dtype": str(torch.float64),
        "transfer_seconds": transfer_seconds,
        **timing,
        "maximum_small_system_relative_residual": float(
            result["maximum_small_system_relative_residual"]
            .detach()
            .cpu()
            .item()
        ),
        "conservative_training_and_model_logical_bytes": conservative_bytes,
    }
    metrics["end_to_end_median_seconds_excluding_build"] = float(
        transfer_seconds + timing["resident"]["median_seconds"]
    )
    if device.type == "cuda":
        metrics["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated()
        )
        metrics["cuda_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved()
        )
    return result, metrics


def _recover_models(
    quotient: SymbolicQuotient,
    beta: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    beta = beta.detach().cpu().to(torch.float64)
    coordinate = torch.linalg.solve(quotient.metric, beta)
    gram = quotient.basis.T @ quotient.basis
    dual = quotient.basis @ torch.linalg.solve(gram, coordinate)
    group_model = quotient.group_weights[:, None] * (
        quotient.transform.T @ coordinate
    )
    group_index = quotient.original_to_group
    original_model = (
        quotient.original_amplitudes[:, None]
        / quotient.group_weights.index_select(0, group_index)[:, None]
        * group_model.index_select(0, group_index)
    )
    return {
        "coordinate": coordinate,
        "dual_prediction_equivalent": dual,
        "group_model": group_model,
        "original_feature_model": original_model,
    }


def _original_backward_error(
    states: Mapping[str, Any],
    kernel: torch.Tensor,
    basis: torch.Tensor,
    metric: torch.Tensor,
) -> float:
    sqrt_counts = states["counts"].sqrt()
    rhs = sqrt_counts[:, None] * states["target_means"]
    weighted_basis = sqrt_counts[:, None] * basis
    orthogonal, triangular = torch.linalg.qr(weighted_basis, mode="reduced")
    range_rhs = orthogonal.T @ rhs
    range_system = (
        triangular @ metric @ triangular.T
        + sg19.FROZEN_LAMBDA
        * torch.eye(triangular.shape[0], dtype=torch.float64)
    )
    range_solution = torch.linalg.solve(range_system, range_rhs)
    projected_rhs = orthogonal @ range_rhs
    complement = rhs - projected_rhs
    dual = (
        orthogonal @ range_solution
        + complement / sg19.FROZEN_LAMBDA
    )
    system = (
        weighted_basis @ metric @ weighted_basis.T
        + sg19.FROZEN_LAMBDA
        * torch.eye(kernel.shape[0], dtype=torch.float64)
    )
    range_residual = range_system @ range_solution - range_rhs
    residual = (
        orthogonal @ range_residual
        + projected_rhs
        + complement
        - rhs
    )
    system_norm = float(system.abs().sum(dim=1).max().item())
    denominator = (
        system_norm * dual.abs().max(dim=0).values
        + rhs.abs().max(dim=0).values
    ).clamp_min(1e-30)
    return float(
        (residual.abs().max(dim=0).values / denominator).max().item()
    )


def _benchmark_case(
    states: Mapping[str, Any],
    *,
    warmups: int,
    repetitions: int,
    sg23d_case: Mapping[str, Any],
    quality_problem: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    features = sg23.build_explicit_features(states)
    quotient = build_symbolic_quotient(
        features, coordinate_device=torch.device("cuda:0")
    )
    dense_coefficients, dense_metrics, kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    dense_scores = kernel @ dense_coefficients
    symbolic_kernel = quotient.basis @ quotient.metric @ quotient.basis.T
    symbolic_kernel_error = float(
        (symbolic_kernel - kernel).abs().max().item()
    )
    metric_eigenvalues = torch.linalg.eigvalsh(quotient.metric)
    cpu_result, cpu = benchmark_symbolic(
        quotient,
        states,
        device=torch.device("cpu"),
        warmups=warmups,
        repetitions=repetitions,
    )
    gpu_result, cuda = benchmark_symbolic(
        quotient,
        states,
        device=torch.device("cuda:0"),
        warmups=warmups,
        repetitions=repetitions,
    )
    cpu_scores = cpu_result["scores"].detach().cpu()
    gpu_scores = gpu_result["scores"].detach().cpu()
    models = _recover_models(quotient, gpu_result["beta"])
    dual_scores = kernel @ models["dual_prediction_equivalent"]
    feature_scores = torch.sparse.mm(
        features.matrix, models["original_feature_model"]
    )
    backward_error = _original_backward_error(
        states, kernel, quotient.basis, quotient.metric
    )
    full_bytes = int(3 * kernel.numel() * kernel.element_size())
    memory_ratio = (
        full_bytes / cuda["conservative_training_and_model_logical_bytes"]
    )
    cpu_gpu_difference = float((gpu_scores - cpu_scores).abs().max().item())
    legacy_difference = float((gpu_scores - dense_scores).abs().max().item())
    dual_difference = float((dual_scores - gpu_scores).abs().max().item())
    feature_difference = float((feature_scores - gpu_scores).abs().max().item())
    gpu_over_cpu = (
        cpu["resident"]["median_seconds"]
        / cuda["resident"]["median_seconds"]
    )
    build_seconds = features.build_seconds + quotient.metrics["total_build_seconds"]
    cuda_e2e = (
        build_seconds + cuda["end_to_end_median_seconds_excluding_build"]
    )
    result: Dict[str, Any] = {
        "rows": int(states["keys"].shape[0]),
        "feature_map": {
            "feature_count": features.feature_count,
            "nnz": features.nnz,
            "build_seconds": features.build_seconds,
            "vocabulary_sha256": features.vocabulary_sha256,
            "logical_csr_bytes": features.logical_csr_bytes,
        },
        "symbolic": {
            **quotient.metrics,
            "minimum_metric_eigenvalue": float(metric_eigenvalues.min().item()),
            "maximum_metric_eigenvalue": float(metric_eigenvalues.max().item()),
            "kernel_max_abs_error_vs_analytic": symbolic_kernel_error,
            "pivots_sha256": hashlib.sha256(
                json.dumps(quotient.pivots).encode("utf-8")
            ).hexdigest().upper(),
            "transform_nonzero_count": int(
                torch.count_nonzero(quotient.transform).item()
            ),
        },
        "dense_reference": dense_metrics,
        "cpu": cpu,
        "cuda": {
            **cuda,
            "resident_speedup_over_cpu": gpu_over_cpu,
            "end_to_end_median_seconds": cuda_e2e,
            "end_to_end_speedup_over_sg23d": (
                sg23d_case["cuda_blocks"]["end_to_end_median_seconds"]
                / cuda_e2e
            ),
        },
        "exactness": {
            "cpu_cuda_score_max_abs_difference": cpu_gpu_difference,
            "legacy_dense_score_max_abs_difference": legacy_difference,
            "legacy_prediction_equivalent": sg23._prediction_equivalence(
                gpu_scores, dense_scores
            ),
            "dual_recovery_score_max_abs_difference": dual_difference,
            "original_feature_model_score_max_abs_difference": (
                feature_difference
            ),
            "original_system_normalized_backward_error": backward_error,
        },
        "memory": {
            "full_dense_kernel_system_factor_logical_bytes": full_bytes,
            "symbolic_conservative_logical_bytes": cuda[
                "conservative_training_and_model_logical_bytes"
            ],
            "full_over_symbolic_ratio": memory_ratio,
        },
    }
    if quality_problem is not None:
        quality = sg23._evaluate_exact_route(
            quality_problem, models["dual_prediction_equivalent"]
        )
        quality["perfect_quality"] = sg23._quality_is_perfect(quality)
        result["quality"] = quality
    return result


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG23E requires CUDA")
    torch.set_num_threads(args.threads)
    sg23c_reference, sg23c_sha = sg23._load_frozen_json(
        args.sg23c_reference.expanduser().resolve(),
        SG23C_REFERENCE_SHA256,
        SG23C_EXPERIMENT,
    )
    sg23d_reference, sg23d_sha = sg23._load_frozen_json(
        args.sg23d_reference.expanduser().resolve(),
        SG23D_REFERENCE_SHA256,
        SG23D_EXPERIMENT,
    )
    problem = sg23._load_problem(args)
    real = _benchmark_case(
        problem["unique"],
        warmups=args.warmups,
        repetitions=args.repetitions,
        sg23d_case=sg23d_reference["real_443"],
        quality_problem=problem,
    )
    stress = {}
    for size in args.stress_sizes:
        states, generation = sg23.generate_stress_states(
            problem["unique"], size
        )
        case = _benchmark_case(
            states,
            warmups=args.warmups,
            repetitions=args.repetitions,
            sg23d_case=sg23d_reference["stress"][str(size)],
        )
        case["generation"] = generation
        stress[str(size)] = case
    cases = (real, *stress.values())
    provenance_gate = bool(
        problem["graph_audit"]["all_masks_match_exhaustive_cache"]
        and problem["graph_audit"]["all_binding_steps_precede_snapshot_root"]
        and problem["graph_audit"]["all_rooms_unique_and_present"]
        and problem["graph_audit"]["no_edge_conflicts"]
        and problem["constraint_audit"]["all_targets_match"]
    )
    symbolic_gate = bool(
        all(
            case["symbolic"]["constant_column_max_abs_error"] == 0.0
            and case["symbolic"]["weight_grid_max_abs_error"]
            <= sg23.FEATURE_GRAM_TOLERANCE
            and case["symbolic"]["coordinate_round_max_abs_error"]
            <= SYMBOLIC_ROUND_TOLERANCE
            and case["symbolic"]["integer_reconstruction_max_abs_error"]
            == 0.0
            and case["symbolic"]["kernel_max_abs_error_vs_analytic"] == 0.0
            and case["symbolic"]["minimum_metric_eigenvalue"] > 0.0
            for case in cases
        )
    )
    backend_gate = bool(all(case["cuda"]["device"] == "cuda:0" for case in cases))
    numerical_gate = bool(
        all(
            case["exactness"]["cpu_cuda_score_max_abs_difference"]
            <= CPU_CUDA_SCORE_TOLERANCE
            and case["exactness"]["original_system_normalized_backward_error"]
            <= BACKWARD_ERROR_TOLERANCE
            and case["exactness"]["dual_recovery_score_max_abs_difference"]
            <= CPU_CUDA_SCORE_TOLERANCE
            and case["exactness"][
                "original_feature_model_score_max_abs_difference"
            ]
            <= CPU_CUDA_SCORE_TOLERANCE
            for case in cases
        )
    )
    legacy_gate = bool(
        all(
            case["exactness"]["legacy_dense_score_max_abs_difference"]
            <= sg23.EXACT_SCORE_TOLERANCE
            and case["exactness"]["legacy_prediction_equivalent"]
            for case in cases
        )
    )
    quality_gate = bool(real["quality"]["perfect_quality"])
    largest = stress[str(max(args.stress_sizes))]
    memory_gate = bool(largest["memory"]["full_over_symbolic_ratio"] >= 2.0)
    speed_gate = bool(
        any(
            int(size) >= 2048
            and case["cuda"]["resident_speedup_over_cpu"] > 1.0
            and case["cuda"]["end_to_end_speedup_over_sg23d"] > 1.0
            for size, case in stress.items()
        )
    )
    engineering = bool(
        provenance_gate
        and symbolic_gate
        and backend_gate
        and numerical_gate
        and quality_gate
        and memory_gate
        and speed_gate
    )
    overall = bool(engineering and legacy_gate)
    return {
        "experiment": "E3-SG23E symbolic support quotient and identifiable CUDA solve",
        "references": {
            "sg22r_sha256": problem["reference_sha"],
            "cache_sha256": problem["cache_sha"],
            "sg23c_sha256": sg23c_sha,
            "sg23d_sha256": sg23d_sha,
            "sg23c_decision": sg23c_reference["decision"],
            "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "environment": sg23c._environment(),
        "protocol": {
            "threads": args.threads,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "stress_sizes": tuple(args.stress_sizes),
            "dtype": str(torch.float64),
            "lambda": sg19.FROZEN_LAMBDA,
            "symbolic_round_tolerance": SYMBOLIC_ROUND_TOLERANCE,
            "cpu_cuda_score_tolerance": CPU_CUDA_SCORE_TOLERANCE,
            "legacy_score_tolerance": sg23.EXACT_SCORE_TOLERANCE,
            "backward_error_tolerance": BACKWARD_ERROR_TOLERANCE,
        },
        "real_443": real,
        "stress": stress,
        "decision": {
            "provenance_gate": provenance_gate,
            "symbolic_gate": symbolic_gate,
            "backend_gate": backend_gate,
            "numerical_gate": numerical_gate,
            "legacy_cpu_trajectory_gate": legacy_gate,
            "real_quality_gate": quality_gate,
            "scale_memory_gate": memory_gate,
            "end_to_end_speed_gate": speed_gate,
            "engineering_substrate": "PASS" if engineering else "FAIL",
            "overall": "PASS" if overall else "FAIL",
            "next_route": "sg24_same_v100_raw_language_ann_comparison",
        },
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus-dir", type=Path, default=sg23.DEFAULT_CORPUS)
    parser.add_argument("--cache", type=Path, default=sg23.DEFAULT_CACHE)
    parser.add_argument(
        "--sg22r-reference", type=Path, default=sg23.DEFAULT_SG22R_REFERENCE
    )
    parser.add_argument(
        "--sg23c-reference", type=Path, default=DEFAULT_SG23C_REFERENCE
    )
    parser.add_argument(
        "--sg23d-reference", type=Path, default=DEFAULT_SG23D_REFERENCE
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--stress-sizes",
        type=sg23c._parse_int_tuple,
        default=DEFAULT_STRESS_SIZES,
    )
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.repetitions <= 0:
        parser.error("threads and repetitions must be positive")
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
