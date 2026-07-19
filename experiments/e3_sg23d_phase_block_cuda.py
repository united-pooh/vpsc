"""SG23D exact phase-block CPU/CUDA solver experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments import e3_sg23_spike_feature_solvers as sg23  # noqa: E402
from experiments import e3_sg23c_cuda_adaptive_krr as sg23c  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg23d_phase_block_cuda.json")
DEFAULT_SG23C_REFERENCE = Path(
    "results/e3_scan/e3_sg23c_cuda_adaptive_krr.json"
)
SG23C_REFERENCE_SHA256 = (
    "17DDD84BCB1B010D13F19C567735BA9498534BE122B473E94407659FA6BA162B"
)
SG23C_EXPERIMENT = (
    "E3-SG23C AutoDL CUDA exact dual and effective-rank Woodbury"
)
DEFAULT_STRESS_SIZES = sg23c.DEFAULT_STRESS_SIZES
DEFAULT_THREADS = sg23c.DEFAULT_THREADS
DEFAULT_WARMUPS = sg23c.DEFAULT_WARMUPS
DEFAULT_REPETITIONS = sg23c.DEFAULT_REPETITIONS
BACKWARD_ERROR_TOLERANCE = 1e-12


@dataclass
class PhaseBlock:
    phase: int
    indices: torch.Tensor
    states: Dict[str, Any]
    features: sg23.ExplicitFeatureMatrix


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def phase_groups(states: Mapping[str, Any]) -> Dict[int, Tuple[int, ...]]:
    groups: Dict[int, list[int]] = {}
    for index, value in enumerate(states["phases"].tolist()):
        groups.setdefault(int(value), []).append(index)
    return {phase: tuple(groups[phase]) for phase in sorted(groups)}


def build_phase_blocks(
    states: Mapping[str, Any]
) -> Tuple[Tuple[PhaseBlock, ...], Dict[str, Any]]:
    started = time.perf_counter_ns()
    blocks = []
    for phase, indices in phase_groups(states).items():
        subset = sg23._subset_states(states, indices)
        blocks.append(
            PhaseBlock(
                phase=phase,
                indices=torch.tensor(indices, dtype=torch.long),
                states=subset,
                features=sg23.build_explicit_features(subset),
            )
        )
    build_seconds = (time.perf_counter_ns() - started) / 1e9
    row_count = int(states["keys"].shape[0])
    sum_squared = sum(block.indices.numel() ** 2 for block in blocks)
    sum_cubed = sum(block.indices.numel() ** 3 for block in blocks)
    return tuple(blocks), {
        "block_count": len(blocks),
        "block_sizes": {
            str(block.phase): int(block.indices.numel()) for block in blocks
        },
        "build_seconds": build_seconds,
        "feature_build_seconds_sum": float(
            sum(block.features.build_seconds for block in blocks)
        ),
        "feature_count_sum": int(
            sum(block.features.feature_count for block in blocks)
        ),
        "nnz_sum": int(sum(block.features.nnz for block in blocks)),
        "csr_logical_bytes_sum": int(
            sum(block.features.logical_csr_bytes for block in blocks)
        ),
        "quadratic_ratio_full_over_blocks": (
            row_count * row_count / sum_squared
        ),
        "cubic_ratio_full_over_blocks": row_count**3 / sum_cubed,
    }


def cross_phase_max_abs(
    kernel: torch.Tensor,
    groups: Mapping[int, Sequence[int]],
) -> float:
    maximum = 0.0
    phases = tuple(sorted(groups))
    for left_position, left_phase in enumerate(phases):
        left = torch.tensor(groups[left_phase], dtype=torch.long)
        for right_phase in phases[left_position + 1 :]:
            right = torch.tensor(groups[right_phase], dtype=torch.long)
            value = float(
                kernel.index_select(0, left)
                .index_select(1, right)
                .abs()
                .max()
                .item()
            )
            maximum = max(maximum, value)
    return maximum


def _prepare_blocks(
    blocks: Sequence[PhaseBlock], device: torch.device
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats()
    prepared = []
    transfers = {}
    total_started = time.perf_counter_ns()
    for block in blocks:
        started = time.perf_counter_ns()
        feature_matrix = block.features.matrix.to_sparse_coo().coalesce().to(
            device=device, dtype=torch.float64
        )
        counts = block.states["counts"].to(
            device=device, dtype=torch.float64
        )
        targets = block.states["target_means"].to(
            device=device, dtype=torch.float64
        )
        indices = block.indices.to(device=device)
        sg23c._synchronize(device)
        seconds = (time.perf_counter_ns() - started) / 1e9
        transfers[str(block.phase)] = seconds
        prepared.append(
            {
                "phase": block.phase,
                "indices": indices,
                "feature_matrix": feature_matrix,
                "counts": counts,
                "targets": targets,
            }
        )
    return tuple(prepared), {
        "per_block_seconds": transfers,
        "total_seconds": (time.perf_counter_ns() - total_started) / 1e9,
    }


def _phase_block_operation(
    prepared: Sequence[Mapping[str, Any]],
    *,
    row_count: int,
    output_width: int,
    device: torch.device,
    collect_block_times: bool = False,
) -> Dict[str, Any]:
    coefficients = torch.zeros(
        row_count, output_width, device=device, dtype=torch.float64
    )
    scores = torch.zeros_like(coefficients)
    residuals = []
    block_seconds: Dict[str, float] = {}
    for block in prepared:
        if collect_block_times:
            sg23c._synchronize(device)
            started = time.perf_counter_ns()
        solved = sg23c._dense_dual_operation(
            block["feature_matrix"], block["counts"], block["targets"]
        )
        if collect_block_times:
            sg23c._synchronize(device)
            block_seconds[str(block["phase"])] = (
                time.perf_counter_ns() - started
            ) / 1e9
        coefficients.index_copy_(
            0, block["indices"], solved["coefficients"]
        )
        scores.index_copy_(0, block["indices"], solved["scores"])
        residuals.append(solved["maximum_relative_residual"])
    return {
        "coefficients": coefficients,
        "scores": scores,
        "maximum_relative_residual": torch.stack(residuals).max(),
        "per_block_seconds": block_seconds,
    }


def benchmark_phase_blocks(
    blocks: Sequence[PhaseBlock],
    states: Mapping[str, Any],
    *,
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prepared, transfer = _prepare_blocks(blocks, device)
    row_count = int(states["keys"].shape[0])
    output_width = int(states["target_means"].shape[1])
    operation = lambda: _phase_block_operation(
        prepared,
        row_count=row_count,
        output_width=output_width,
        device=device,
    )
    result, timing = sg23c.benchmark_operation(
        operation,
        device=device,
        warmups=warmups,
        repetitions=repetitions,
    )
    block_audit, _audit_seconds = sg23c._timed_call(
        lambda: _phase_block_operation(
            prepared,
            row_count=row_count,
            output_width=output_width,
            device=device,
            collect_block_times=True,
        ),
        device,
    )
    element_bytes = torch.empty((), dtype=torch.float64).element_size()
    sum_squared = sum(block.indices.numel() ** 2 for block in blocks)
    metrics: Dict[str, Any] = {
        "device": str(device),
        "dtype": str(torch.float64),
        "transfer": transfer,
        **timing,
        "per_block_seconds_single_audit": block_audit["per_block_seconds"],
        "maximum_relative_residual": float(
            result["maximum_relative_residual"].detach().cpu().item()
        ),
        "kernel_system_factor_logical_bytes": int(
            3 * sum_squared * element_bytes
        ),
    }
    metrics["end_to_end_median_seconds_excluding_feature_build"] = float(
        transfer["total_seconds"] + timing["resident"]["median_seconds"]
    )
    if device.type == "cuda":
        metrics["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated()
        )
        metrics["cuda_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved()
        )
    return result, metrics


def normalized_backward_error(
    states: Mapping[str, Any],
    groups: Mapping[int, Sequence[int]],
    coefficients: torch.Tensor,
) -> Dict[str, Any]:
    coefficients = coefficients.detach().cpu().to(torch.float64)
    per_phase = {}
    maximum = 0.0
    for phase, indices in groups.items():
        block = sg23._subset_states(states, indices)
        kernel = sg19.plan_edge_kernel(block, block)
        counts = block["counts"]
        sqrt_counts = counts.sqrt()
        system = (
            sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
            + sg19.FROZEN_LAMBDA
            * torch.eye(len(indices), dtype=torch.float64)
        )
        rhs = sqrt_counts[:, None] * block["target_means"]
        index = torch.tensor(tuple(indices), dtype=torch.long)
        dual = coefficients.index_select(0, index) / sqrt_counts[:, None]
        residual = system @ dual - rhs
        system_norm = float(system.abs().sum(dim=1).max().item())
        dual_norm = dual.abs().max(dim=0).values
        rhs_norm = rhs.abs().max(dim=0).values
        denominator = (system_norm * dual_norm + rhs_norm).clamp_min(1e-30)
        errors = residual.abs().max(dim=0).values / denominator
        error = float(errors.max().item())
        per_phase[str(phase)] = error
        maximum = max(maximum, error)
    return {"maximum": maximum, "per_phase": per_phase}


def _feature_kernel_audit(blocks: Sequence[PhaseBlock]) -> Dict[str, Any]:
    per_phase = {}
    maximum = 0.0
    for block in blocks:
        explicit, _seconds = sg23.explicit_dense_gram(block.features)
        analytic = sg19.plan_edge_kernel(block.states, block.states)
        error = float((explicit - analytic).abs().max().item())
        per_phase[str(block.phase)] = error
        maximum = max(maximum, error)
    return {"maximum": maximum, "per_phase": per_phase}


def _benchmark_case(
    states: Mapping[str, Any],
    *,
    warmups: int,
    repetitions: int,
    sg23c_case: Mapping[str, Any],
    quality_problem: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    groups = phase_groups(states)
    blocks, structure = build_phase_blocks(states)
    dense_coefficients, dense_metrics, dense_kernel, _system = (
        sg23.dense_weighted_cholesky(states)
    )
    dense_scores = dense_kernel @ dense_coefficients
    cross_max = cross_phase_max_abs(dense_kernel, groups)
    feature_audit = _feature_kernel_audit(blocks)
    cpu_result, cpu = benchmark_phase_blocks(
        blocks,
        states,
        device=torch.device("cpu"),
        warmups=warmups,
        repetitions=repetitions,
    )
    gpu_result, cuda = benchmark_phase_blocks(
        blocks,
        states,
        device=torch.device("cuda:0"),
        warmups=warmups,
        repetitions=repetitions,
    )
    cpu_scores = cpu_result["scores"].detach().cpu()
    gpu_scores = gpu_result["scores"].detach().cpu()
    cpu_coefficients = cpu_result["coefficients"].detach().cpu()
    gpu_coefficients = gpu_result["coefficients"].detach().cpu()
    cpu_backward = normalized_backward_error(
        states, groups, cpu_coefficients
    )
    gpu_backward = normalized_backward_error(
        states, groups, gpu_coefficients
    )
    memory_ratio = (
        3 * dense_kernel.numel() * dense_kernel.element_size()
        / cuda["kernel_system_factor_logical_bytes"]
    )
    gpu_over_cpu = (
        cpu["resident"]["median_seconds"]
        / cuda["resident"]["median_seconds"]
    )
    full_pure_cuda_seconds = sg23c_case["dense_dual"]["cuda"][
        "resident"
    ]["median_seconds"]
    full_hybrid_seconds = sg23c_case["hybrid_exact"]["metrics"]["total"][
        "median_seconds"
    ]
    result: Dict[str, Any] = {
        "rows": int(states["keys"].shape[0]),
        "structure": {
            **structure,
            "cross_phase_kernel_max_abs": cross_max,
            "feature_kernel_audit": feature_audit,
            "full_dense_kernel_system_factor_logical_bytes": int(
                3 * dense_kernel.numel() * dense_kernel.element_size()
            ),
            "full_over_block_logical_memory_ratio": memory_ratio,
        },
        "dense_reference": dense_metrics,
        "cpu_blocks": {
            **cpu,
            "legacy_score_max_abs_difference": float(
                (cpu_scores - dense_scores).abs().max().item()
            ),
            "legacy_prediction_equivalent": sg23._prediction_equivalence(
                cpu_scores, dense_scores
            ),
            "backward_error": cpu_backward,
        },
        "cuda_blocks": {
            **cuda,
            "legacy_score_max_abs_difference": float(
                (gpu_scores - dense_scores).abs().max().item()
            ),
            "legacy_prediction_equivalent": sg23._prediction_equivalence(
                gpu_scores, dense_scores
            ),
            "cpu_block_score_max_abs_difference": float(
                (gpu_scores - cpu_scores).abs().max().item()
            ),
            "backward_error": gpu_backward,
            "resident_speedup_over_cpu_blocks": gpu_over_cpu,
            "resident_speedup_over_sg23c_full_pure_cuda": (
                full_pure_cuda_seconds
                / cuda["resident"]["median_seconds"]
            ),
            "resident_speedup_over_sg23c_hybrid_pipeline": (
                full_hybrid_seconds / cuda["resident"]["median_seconds"]
            ),
        },
    }
    result["cpu_blocks"]["end_to_end_median_seconds"] = float(
        structure["build_seconds"]
        + cpu["end_to_end_median_seconds_excluding_feature_build"]
    )
    result["cuda_blocks"]["end_to_end_median_seconds"] = float(
        structure["build_seconds"]
        + cuda["end_to_end_median_seconds_excluding_feature_build"]
    )
    if quality_problem is not None:
        quality = sg23._evaluate_exact_route(
            quality_problem, gpu_coefficients
        )
        quality["perfect_quality"] = sg23._quality_is_perfect(quality)
        result["quality"] = quality
    return result


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG23D requires CUDA")
    torch.set_num_threads(args.threads)
    sg23c_reference, sg23c_sha = sg23._load_frozen_json(
        args.sg23c_reference.expanduser().resolve(),
        SG23C_REFERENCE_SHA256,
        SG23C_EXPERIMENT,
    )
    problem = sg23._load_problem(args)
    real = _benchmark_case(
        problem["unique"],
        warmups=args.warmups,
        repetitions=args.repetitions,
        sg23c_case=sg23c_reference["real_443"],
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
            sg23c_case=sg23c_reference["stress"][str(size)],
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
    structure_gate = bool(
        all(
            case["structure"]["cross_phase_kernel_max_abs"] == 0.0
            and case["structure"]["feature_kernel_audit"]["maximum"]
            <= sg23.FEATURE_GRAM_TOLERANCE
            for case in cases
        )
    )
    backend_gate = bool(
        all(case["cuda_blocks"]["device"] == "cuda:0" for case in cases)
    )
    legacy_gate = bool(
        all(
            case["cuda_blocks"]["legacy_score_max_abs_difference"]
            <= sg23.EXACT_SCORE_TOLERANCE
            and case["cuda_blocks"]["legacy_prediction_equivalent"]
            for case in cases
        )
    )
    backward_gate = bool(
        all(
            case["cpu_blocks"]["backward_error"]["maximum"]
            <= BACKWARD_ERROR_TOLERANCE
            and case["cuda_blocks"]["backward_error"]["maximum"]
            <= BACKWARD_ERROR_TOLERANCE
            for case in cases
        )
    )
    quality_gate = bool(real["quality"]["perfect_quality"])
    largest = stress[str(max(args.stress_sizes))]
    memory_gate = bool(
        largest["structure"]["full_over_block_logical_memory_ratio"] >= 2.0
    )
    speed_gate = bool(
        any(
            int(size) >= 2048
            and case["cuda_blocks"]["resident_speedup_over_cpu_blocks"] > 1.0
            and (
                case["cuda_blocks"][
                    "resident_speedup_over_sg23c_full_pure_cuda"
                ]
                > 1.0
                or case["cuda_blocks"][
                    "resident_speedup_over_sg23c_hybrid_pipeline"
                ]
                > 1.0
            )
            for size, case in stress.items()
        )
    )
    engineering = bool(
        provenance_gate
        and structure_gate
        and backend_gate
        and backward_gate
        and quality_gate
        and memory_gate
        and speed_gate
    )
    overall = bool(engineering and legacy_gate)
    return {
        "experiment": "E3-SG23D exact phase blocks and CUDA independent solves",
        "references": {
            "sg22r_sha256": problem["reference_sha"],
            "cache_sha256": problem["cache_sha"],
            "sg23c_sha256": sg23c_sha,
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
            "legacy_score_tolerance": sg23.EXACT_SCORE_TOLERANCE,
            "backward_error_tolerance": BACKWARD_ERROR_TOLERANCE,
            "group_key": "phase",
        },
        "real_443": real,
        "stress": stress,
        "decision": {
            "provenance_gate": provenance_gate,
            "structure_gate": structure_gate,
            "backend_gate": backend_gate,
            "legacy_cpu_trajectory_gate": legacy_gate,
            "backward_error_gate": backward_gate,
            "real_quality_gate": quality_gate,
            "scale_memory_gate": memory_gate,
            "cuda_speed_gate": speed_gate,
            "engineering_substrate": "PASS" if engineering else "FAIL",
            "overall": "PASS" if overall else "FAIL",
            "next_route": (
                "sg24_same_v100_raw_language_ann_comparison"
                if overall
                else (
                    "sg23e_symbolic_contrast_quotient"
                    if engineering
                    else "sg23d_phase_block_failure_diagnostic"
                )
            ),
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
