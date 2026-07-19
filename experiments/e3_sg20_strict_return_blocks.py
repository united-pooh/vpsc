"""SG20 strict return-state kernel with exact weighted block solves."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment, _sync  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments import e3_sg16_closed_loop_planner as sg16  # noqa: E402
from experiments import e3_sg17_two_step_rollout as sg17  # noqa: E402
from experiments import e3_sg18_affordance_weighted_krr as sg18  # noqa: E402
from experiments import e3_sg19_plan_edge_spikes as sg19  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    build_action_alphabet,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, file_sha256  # noqa: E402


DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg20_strict_return_blocks.json")
DEFAULT_SG19_REFERENCE = Path("results/e3_scan/e3_sg19_plan_edge_spikes.json")
SG19_REFERENCE_SHA256 = (
    "EA2C855CDC9ECA0D41B8345B3CF3918F4BD2BADCCCA0B35F17DFE1F104505C22"
)
SG19_EXPERIMENT = "E3-SG19 objective plan tape and visited-edge spikes"
KERNEL_NAME = "strict_return_block_affordance_phase_suffix_objective_plan"
THREAD_SWEEP = (1, 2, 4, 8, 16)


def strict_return_kernel(
    query: Mapping[str, torch.Tensor],
    prototypes: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """PSD product kernel with exactly orthogonal return-state blocks."""

    base = sg18.affordance_spike_kernel(
        query["keys"],
        prototypes["keys"],
        query["phases"],
        prototypes["phases"],
        query["masks"],
        prototypes["masks"],
        dtype=dtype,
    )
    return_equal = (
        query["return_edges"][:, None]
        == prototypes["return_edges"][None, :]
    ).to(dtype)
    current_equal = (
        query["plan_current"][:, None]
        == prototypes["plan_current"][None, :]
    ).to(dtype)
    next_equal = (
        query["plan_next"][:, None]
        == prototypes["plan_next"][None, :]
    ).to(dtype)
    return base * return_equal * (1.0 + current_equal + next_equal)


def _select_rows(
    tensors: Mapping[str, Any], indices: torch.Tensor
) -> Dict[str, Any]:
    row_count = int(tensors["keys"].shape[0])
    return {
        name: (
            value.index_select(0, indices)
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == row_count
            else value
        )
        for name, value in tensors.items()
    }


def _solve_one_return_block(
    label: int,
    unique: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    indices = torch.nonzero(
        unique["return_edges"] == label, as_tuple=False
    ).flatten()
    if indices.numel() == 0:
        raise ValueError(f"empty strict return block {label}")
    block = _select_rows(unique, indices)
    kernel_started = time.perf_counter_ns()
    kernel = strict_return_kernel(block, block)
    _sync(device)
    kernel_seconds = (time.perf_counter_ns() - kernel_started) / 1e9
    sqrt_counts = block["counts"].sqrt()
    system = (
        sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
        + sg19.FROZEN_LAMBDA
        * torch.eye(kernel.shape[0], dtype=torch.float64, device=device)
    )
    rhs = sqrt_counts[:, None] * block["target_means"]
    solve_started = time.perf_counter_ns()
    factor = torch.linalg.cholesky(system)
    coefficients = sqrt_counts[:, None] * torch.cholesky_solve(rhs, factor)
    _sync(device)
    solve_seconds = (time.perf_counter_ns() - solve_started) / 1e9
    return indices, coefficients, {
        "return_edge": label,
        "prototype_count": int(indices.numel()),
        "kernel_seconds": kernel_seconds,
        "cholesky_solve_seconds": solve_seconds,
    }


def solve_strict_return_blocks(
    unique: Mapping[str, Any],
    *,
    device: torch.device,
    block_workers: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    labels = tuple(
        sorted(int(value) for value in unique["return_edges"].unique().tolist())
    )
    if labels != (0, 1):
        raise ValueError(f"expected strict return blocks (0, 1), got {labels}")
    started = time.perf_counter_ns()
    if block_workers == 1:
        solved = [
            _solve_one_return_block(label, unique, device=device)
            for label in labels
        ]
    else:
        with ThreadPoolExecutor(max_workers=block_workers) as executor:
            solved = list(
                executor.map(
                    lambda label: _solve_one_return_block(
                        label, unique, device=device
                    ),
                    labels,
                )
            )
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
    coefficients = torch.zeros_like(unique["target_means"])
    reports = []
    for indices, block_coefficients, report in solved:
        coefficients.index_copy_(0, indices, block_coefficients)
        reports.append(report)
    return coefficients, {
        "block_workers": block_workers,
        "wall_seconds": wall_seconds,
        "summed_kernel_seconds": sum(
            float(report["kernel_seconds"]) for report in reports
        ),
        "summed_cholesky_solve_seconds": sum(
            float(report["cholesky_solve_seconds"]) for report in reports
        ),
        "blocks": tuple(sorted(reports, key=lambda report: report["return_edge"])),
    }


def _solve_dense_weighted(
    unique: Mapping[str, Any], *, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel = strict_return_kernel(unique, unique)
    sqrt_counts = unique["counts"].sqrt()
    system = (
        sqrt_counts[:, None] * kernel * sqrt_counts[None, :]
        + sg19.FROZEN_LAMBDA
        * torch.eye(kernel.shape[0], dtype=torch.float64, device=device)
    )
    rhs = sqrt_counts[:, None] * unique["target_means"]
    factor = torch.linalg.cholesky(system)
    coefficients = sqrt_counts[:, None] * torch.cholesky_solve(rhs, factor)
    return coefficients, kernel


def _prediction_equivalent(
    left_scores: torch.Tensor, right_scores: torch.Tensor
) -> bool:
    return bool(
        torch.equal(
            sg10._prediction_matrix(left_scores[:, : sg10.TOTAL_LOGITS]),
            sg10._prediction_matrix(right_scores[:, : sg10.TOTAL_LOGITS]),
        )
        and torch.equal(
            left_scores[:, sg18.NEXT_MASK_OFFSET :]
            > sg18.MASK_DECISION_THRESHOLD,
            right_scores[:, sg18.NEXT_MASK_OFFSET :]
            > sg18.MASK_DECISION_THRESHOLD,
        )
    )


def fit_weighted_strict_blocks(
    train: Mapping[str, Any],
    unique: Mapping[str, Any],
    *,
    device: torch.device,
    block_workers: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    coefficients, block_fit = solve_strict_return_blocks(
        unique, device=device, block_workers=block_workers
    )

    audit_started = time.perf_counter_ns()
    dense_coefficients, dense_kernel = _solve_dense_weighted(
        unique, device=device
    )
    dense_scores = strict_return_kernel(train, unique) @ dense_coefficients
    block_scores = strict_return_kernel(train, unique) @ coefficients
    block_dense_score_difference = float(
        (block_scores - dense_scores).abs().max().item()
    )
    block_dense_coefficient_difference = float(
        (coefficients - dense_coefficients).abs().max().item()
    )

    expanded_kernel = strict_return_kernel(train, train)
    expanded_factor = torch.linalg.cholesky(
        expanded_kernel
        + sg19.FROZEN_LAMBDA
        * torch.eye(
            expanded_kernel.shape[0], dtype=torch.float64, device=device
        )
    )
    expanded_alpha = torch.cholesky_solve(
        train["target_code"], expanded_factor
    )
    expanded_scores = expanded_kernel @ expanded_alpha
    weighted_expanded_difference = float(
        (block_scores - expanded_scores).abs().max().item()
    )

    return_zero = unique["return_edges"] == 0
    return_one = unique["return_edges"] == 1
    cross_block = dense_kernel[return_zero][:, return_one]
    cross_block_max = float(cross_block.abs().max().item())
    minimum_eigenvalue = float(
        torch.linalg.eigvalsh((dense_kernel + dense_kernel.T) * 0.5)
        .min()
        .item()
    )
    _sync(device)
    audit_seconds = (time.perf_counter_ns() - audit_started) / 1e9
    deployment_wall = (
        float(train["elapsed_seconds"])
        + float(train["extended_state_encoding_seconds"])
        + float(unique["elapsed_seconds"])
        + float(block_fit["wall_seconds"])
    )
    return coefficients, {
        "kernel": KERNEL_NAME,
        "ridge_lambda": sg19.FROZEN_LAMBDA,
        "mask_decision_threshold": sg18.MASK_DECISION_THRESHOLD,
        "expanded_example_count": int(train["keys"].shape[0]),
        "unique_prototype_count": int(unique["keys"].shape[0]),
        "compression_ratio": (
            unique["keys"].shape[0] / train["keys"].shape[0]
        ),
        "ambiguous_unique_key_count": unique["ambiguous_unique_key_count"],
        "single_pass_aggregation_seconds": unique["elapsed_seconds"],
        "block_fit": block_fit,
        "deployment_training_wall_seconds": deployment_wall,
        "dense_and_expanded_audit_seconds_excluded": audit_seconds,
        "strict_cross_return_kernel_max_abs": cross_block_max,
        "strict_kernel_minimum_eigenvalue": minimum_eigenvalue,
        "block_dense_coefficient_max_abs_difference": (
            block_dense_coefficient_difference
        ),
        "block_dense_train_score_max_abs_difference": (
            block_dense_score_difference
        ),
        "block_dense_prediction_equivalent": _prediction_equivalent(
            block_scores, dense_scores
        ),
        "weighted_expanded_train_score_max_abs_difference": (
            weighted_expanded_difference
        ),
        "weighted_expanded_prediction_equivalent": _prediction_equivalent(
            block_scores, expanded_scores
        ),
    }


def benchmark_thread_scaling(
    unique: Mapping[str, Any],
    *,
    device: torch.device,
    primary_threads: int,
    repetitions: int,
) -> Dict[str, Any]:
    records = []
    original_threads = torch.get_num_threads()
    try:
        for threads in THREAD_SWEEP:
            torch.set_num_threads(threads)
            solve_strict_return_blocks(unique, device=device, block_workers=1)
            samples = []
            for _ in range(repetitions):
                started = time.perf_counter_ns()
                solve_strict_return_blocks(
                    unique, device=device, block_workers=1
                )
                samples.append((time.perf_counter_ns() - started) / 1e9)
            records.append(
                {
                    "intraop_threads": threads,
                    "block_workers": 1,
                    "repetitions": repetitions,
                    "median_seconds": statistics.median(samples),
                    "minimum_seconds": min(samples),
                    "maximum_seconds": max(samples),
                }
            )

        parallel_intraop = max(1, primary_threads // 2)
        torch.set_num_threads(parallel_intraop)
        solve_strict_return_blocks(unique, device=device, block_workers=2)
        parallel_samples = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            solve_strict_return_blocks(unique, device=device, block_workers=2)
            parallel_samples.append((time.perf_counter_ns() - started) / 1e9)
        parallel_record = {
            "intraop_threads_per_process_setting": parallel_intraop,
            "block_workers": 2,
            "repetitions": repetitions,
            "median_seconds": statistics.median(parallel_samples),
            "minimum_seconds": min(parallel_samples),
            "maximum_seconds": max(parallel_samples),
        }
    finally:
        torch.set_num_threads(original_threads)
    one_thread = next(
        record["median_seconds"]
        for record in records
        if record["intraop_threads"] == 1
    )
    for record in records:
        record["speedup_vs_one_thread"] = (
            one_thread / record["median_seconds"]
        )
    fastest = min(records, key=lambda record: record["median_seconds"])
    return {
        "logical_cpu_count": os.cpu_count(),
        "primary_resource_matched_threads": primary_threads,
        "single_worker_thread_sweep": tuple(records),
        "two_worker_control": parallel_record,
        "fastest_single_worker_threads": fastest["intraop_threads"],
        "fastest_single_worker_median_seconds": fastest["median_seconds"],
        "gpu_backend": {
            "torch_build": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "hip_version": getattr(torch.version, "hip", None),
            "status": "CPU_ONLY_CURRENT_ENVIRONMENT",
        },
    }


def _error_partition(rollout: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "previous_room_total": int(rollout["previous_room_self_error_count"]),
        "immediate_return_edge": int(
            rollout["previous_room_return_edge_error_count"]
        ),
        "nonreturn_or_longer_visited_edge": int(
            rollout["previous_room_nonreturn_error_count"]
        ),
    }


def _decision(
    plan_audit: Mapping[str, Any],
    fit: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    rollout: Mapping[str, Any],
    sg19_reference: Mapping[str, Any],
    sg17_reference: Mapping[str, Any],
    logical_model_bytes: int,
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "language_state_gate": "SMOKE",
            "strict_block_math_gate": "SMOKE",
            "return_mechanism_gate": "SMOKE",
            "one_step_non_regression_gate": "SMOKE",
            "world_state_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "response_speed_gate": "SMOKE",
            "storage_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg20_strict_return_blocks",
        }
    language = (
        plan_audit["game_count"] == 48
        and plan_audit["all_plans_equal_walkthrough_for_audit"]
    )
    math_gate = (
        fit["strict_cross_return_kernel_max_abs"] == 0.0
        and fit["strict_kernel_minimum_eigenvalue"] >= -1e-8
        and fit["block_dense_train_score_max_abs_difference"] <= 1e-9
        and fit["block_dense_prediction_equivalent"]
        and fit["weighted_expanded_train_score_max_abs_difference"] <= 1e-6
        and fit["weighted_expanded_prediction_equivalent"]
    )
    errors = _error_partition(rollout)
    mechanism = (
        errors["immediate_return_edge"] == 0
        and errors["previous_room_total"] <= 2
        and rollout["teacher_forced_second"]["exact_vector_accuracy"] >= 0.995
        and rollout["self_rollout_second"]["exact_vector_accuracy"] >= 0.995
        and rollout["self_minus_teacher_exact"] >= -0.01
        and all(
            value >= 0.995
            for value in rollout["self_rollout_second"][
                "channel_accuracy"
            ].values()
        )
        and rollout["first_routing_accuracy"] == 1.0
        and rollout["premature_stop_first_branch_count"] == 0
    )
    sg19_test = sg19_reference["split_metrics"]["test"]
    non_regression = (
        test_metrics["delta"]["exact_vector_accuracy"] == 1.0
        and all(
            value == 1.0
            for value in test_metrics["delta"]["channel_accuracy"].values()
        )
        and test_metrics["next_affordance"]["bit_accuracy"]
        >= sg19_test["next_affordance"]["bit_accuracy"]
        and test_metrics["next_affordance"]["exact_mask_accuracy"]
        >= sg19_test["next_affordance"]["exact_mask_accuracy"]
    )
    world_state = (
        test_metrics["next_affordance"]["bit_accuracy"] >= 0.98
        and test_metrics["next_affordance"]["exact_mask_accuracy"] >= 0.95
    )
    ann_training = [
        replication["training"][name]["elapsed_seconds"]
        for replication in sg17_reference["replications"]
        for name in sg16.ANN_MODEL_NAMES
    ]
    training = fit["deployment_training_wall_seconds"] < min(ann_training)
    response_records = []
    for replication in sg17_reference["replications"]:
        for name in sg16.ANN_MODEL_NAMES:
            ann = replication["rollout"][name]["teacher_pair_timing"]
            record = {
                "seed": replication["seed"],
                "ann_model": name,
                "snn_p50_ms": rollout["teacher_pair_timing"]["p50_ms"],
                "ann_p50_ms": ann["p50_ms"],
                "snn_p95_ms": rollout["teacher_pair_timing"]["p95_ms"],
                "ann_p95_ms": ann["p95_ms"],
            }
            record["passed"] = (
                record["snn_p50_ms"] <= record["ann_p50_ms"]
                and record["snn_p95_ms"] <= record["ann_p95_ms"]
            )
            response_records.append(record)
    response = all(record["passed"] for record in response_records)
    min_ann_bytes = min(
        replication["parameter_counts"][name] * 4
        for replication in sg17_reference["replications"]
        for name in sg16.ANN_MODEL_NAMES
    )
    storage = logical_model_bytes <= min_ann_bytes
    gates = {
        "language_state_gate": language,
        "strict_block_math_gate": math_gate,
        "return_mechanism_gate": mechanism,
        "one_step_non_regression_gate": non_regression,
        "world_state_gate": world_state,
        "training_speed_gate": training,
        "response_speed_gate": response,
        "storage_gate": storage,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = "sg20r_matched_ann_and_fresh_game_confirmation"
    elif mechanism and not world_state:
        next_route = "sg21_visited_edges_and_deterministic_mask_state"
    elif errors["immediate_return_edge"] > 0:
        next_route = "sg21_episodic_graph_spikes"
    elif not math_gate or not training:
        next_route = "sg21_sparse_primal_or_matrix_free_pcg"
    else:
        next_route = "sg21_state_error_diagnostic"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "return_error_partition": errors,
        "response_comparisons": response_records,
        "next_route": next_route,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    sg19_reference, sg19_digest = sg16._load_reference(
        args.sg19_reference.expanduser().resolve(),
        SG19_REFERENCE_SHA256,
        SG19_EXPERIMENT,
    )
    sg18_reference, sg18_digest = sg16._load_reference(
        args.sg18_reference.expanduser().resolve(),
        sg19.SG18_REFERENCE_SHA256,
        sg19.SG18_EXPERIMENT,
    )
    sg17_reference, sg17_digest = sg16._load_reference(
        args.sg17_reference.expanduser().resolve(),
        sg19.SG17_REFERENCE_SHA256,
        sg19.SG17_EXPERIMENT,
    )
    cache_path = args.exhaustive_cache.expanduser().resolve()
    cache_digest = file_sha256(cache_path).upper()
    if cache_digest != sg19.EXHAUSTIVE_CACHE_SHA256:
        raise ValueError("SG20 exhaustive cache SHA mismatch")
    exhaustive = json.loads(cache_path.read_text(encoding="utf-8"))["exhaustive"]

    corpus = load_event_corpus(corpus_root)
    base_examples, vocabulary = sg10.build_multichannel_examples(
        corpus_root, corpus
    )
    alphabet = build_action_alphabet(base_examples)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    action_order = tuple(sg18_reference["configuration"]["action_order"])
    plans, plan_audit = sg19.load_objective_plans(corpus_root)
    if not plan_audit["all_plans_equal_walkthrough_for_audit"]:
        raise AssertionError("SG20 objective plan compiler audit failed")
    tensors = {
        split: sg19.tensorize_extended(
            exhaustive[split]["records"],
            plans[split],
            alphabet_index=alphabet_index,
            device=device,
        )
        for split in SPLITS
    }
    unique = sg19.compress_extended(tensors["train"])
    coefficients, fit = fit_weighted_strict_blocks(
        tensors["train"],
        unique,
        device=device,
        block_workers=args.block_workers,
    )
    split_metrics = {
        split: sg19.evaluate_split(
            tensors[split],
            unique,
            coefficients,
            action_order,
            kernel_fn=strict_return_kernel,
        )
        for split in SPLITS
    }
    repaired_tree, tree_repair_audit = sg17.repair_persistent_room_semantics(
        sg17_reference["branch_tree"]
    )
    rollout = sg19.evaluate_two_step(
        repaired_tree,
        exhaustive["test"]["records"],
        plans["test"],
        alphabet_index=alphabet_index,
        unique=unique,
        coefficients=coefficients,
        action_order=action_order,
        device=device,
        kernel_fn=strict_return_kernel,
    )
    if args.quick:
        hardware_scaling: Mapping[str, Any] = {
            "status": "SKIPPED_IN_QUICK_MODE"
        }
    else:
        hardware_scaling = benchmark_thread_scaling(
            unique,
            device=device,
            primary_threads=args.threads,
            repetitions=args.benchmark_repetitions,
        )
    logical_model_bytes = int(
        unique["keys"].shape[0]
        * (
            unique["keys"].shape[1]
            + 1
            + len(action_order)
            + 3
            + coefficients.shape[1] * 4
        )
    )
    decision = _decision(
        plan_audit,
        fit,
        split_metrics["test"],
        rollout,
        sg19_reference,
        sg17_reference,
        logical_model_bytes,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG20 strict return blocks and exact block solve",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "mechanism repair on observed fifth games",
            "statement": (
                "Orthogonal categorical return-state spikes eliminate "
                "cross-state negative transfer and expose exact independent "
                "ridge blocks."
            ),
            "what_if": (
                "What if one causal state partition fixes both rollout "
                "quality and cubic training structure?"
            ),
        },
        "references": {
            "sg19_plan_edge_negative": {
                "path": str(args.sg19_reference.expanduser().resolve()),
                "sha256": sg19_digest,
            },
            "sg18_affordance_negative": {
                "path": str(args.sg18_reference.expanduser().resolve()),
                "sha256": sg18_digest,
            },
            "sg17_rollout_tree": {
                "path": str(args.sg17_reference.expanduser().resolve()),
                "sha256": sg17_digest,
            },
            "exhaustive_cache": {
                "path": str(cache_path),
                "sha256": cache_digest,
            },
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "threads": args.threads,
            "block_workers": args.block_workers,
            "objective_plan_source": "public objective string only",
            "walkthrough_role": "audit equality only; never model input",
            "action_order": action_order,
            "kernel": KERNEL_NAME,
            "ridge_lambda": sg19.FROZEN_LAMBDA,
        },
        "dataset": {
            "base_vocabulary_fingerprint": vocabulary.fingerprint,
            "base_action_alphabet": alphabet,
            "objective_plan_audit": plan_audit,
            "persistent_room_tree_repair": tree_repair_audit,
            "exhaustive_counts": {
                split: exhaustive[split]["record_count"] for split in SPLITS
            },
        },
        "weighted_block_fit": fit,
        "split_metrics": split_metrics,
        "two_step_rollout": rollout,
        "hardware_scaling": hardware_scaling,
        "logical_model_storage_bytes": logical_model_bytes,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=sg19.DEFAULT_CORPUS)
    parser.add_argument(
        "--sg19-reference", type=Path, default=DEFAULT_SG19_REFERENCE
    )
    parser.add_argument(
        "--sg18-reference", type=Path, default=sg19.DEFAULT_SG18_REFERENCE
    )
    parser.add_argument(
        "--sg17-reference", type=Path, default=sg19.DEFAULT_SG17_REFERENCE
    )
    parser.add_argument(
        "--exhaustive-cache", type=Path, default=sg19.DEFAULT_EXHAUSTIVE_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--block-workers", type=int, default=1)
    parser.add_argument("--benchmark-repetitions", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.block_workers not in (1, 2):
        parser.error("--block-workers must be 1 or 2")
    if args.benchmark_repetitions <= 0:
        parser.error("--benchmark-repetitions must be positive")
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
