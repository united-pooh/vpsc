"""SG14 phase-bound hierarchical spike-kernel mechanism experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg10_multichannel_delta as sg10  # noqa: E402
from experiments.e3_sg8_bilinear_closed_form import RIDGE_LAMBDAS  # noqa: E402
from experiments.e3_sg12_spike_delay_rls import (  # noqa: E402
    audit_delay_data,
    build_action_alphabet,
)
from experiments.e3_sg13_suffix_spike_kernel import (  # noqa: E402
    KernelSpec,
    _fit_kernel_spec,
    build_game_folds,
    evaluate_cached_kernel_stream,
    evaluate_online_kernel_replication,
    extract_kernel_records,
    suffix_spike_kernel,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg13r_l5")
DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg14_phase_bound_kernel.json")
DEFAULT_FRESH_BASELINE = Path(
    "results/e3_scan/e3_sg10r_fresh_game_baselines.json"
)
DEFAULT_ADDITIVE_REFERENCE = Path("results/e3_scan/e3_sg13r_fresh_games.json")
DEFAULT_SG14_REFERENCE = Path("results/e3_scan/e3_sg14_phase_bound_kernel.json")
FRESH_BASELINE_SHA256 = (
    "1E5E4A49E2B0000D91D2E2EB71CFECEE270FC4913F8E35E93C213C08DD8927A6"
)
ADDITIVE_REFERENCE_SHA256 = (
    "A2EDBE97273FA1AEDCE8A34393C618B48CF613A41D6C55BA234A3D22061A8F96"
)
SG14_REFERENCE_SHA256 = (
    "15D345DE44B73BBA4E39BD2D3199E616AE88F70CCEF9B4D55757CD183B460B2B"
)
CONFIRMATION_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20261001, 20261009)),
    "test": tuple(range(20261009, 20261017)),
}
EXPECTED_COUNTS = {"train": 480, "valid": 120, "test": 120}
EXPECTED_GROUPS = {"train": 160, "valid": 40, "test": 40}


KERNEL_SPECS = (
    KernelSpec(
        "old_additive_phase",
        (1.0, 1.0, 1.0, 1.0),
        1.0,
    ),
    KernelSpec(
        "phase_product_only",
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        phase_suffix_weights=(1.0, 1.0, 1.0, 1.0),
    ),
    KernelSpec(
        "base_plus_phase_product",
        (1.0, 1.0, 1.0, 1.0),
        0.0,
        primary=True,
        phase_suffix_weights=(1.0, 1.0, 1.0, 1.0),
    ),
    KernelSpec(
        "candidate_phase_product",
        (1.0, 0.0, 0.0, 0.0),
        0.0,
        phase_suffix_weights=(1.0, 0.0, 0.0, 0.0),
    ),
    KernelSpec(
        "depth_weighted_product",
        (1.0, 1.0, 1.0, 1.0),
        0.0,
        phase_suffix_weights=(1.0, 2.0, 4.0, 8.0),
    ),
)
PRIMARY_SPEC = next(spec for spec in KERNEL_SPECS if spec.primary)


def _load_reference(
    path: Path, expected_sha: str, expected_experiment: str
) -> Tuple[Dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != expected_sha:
        raise ValueError(
            f"reference SHA mismatch for {path}: expected {expected_sha}, got {digest}"
        )
    value = json.loads(payload)
    if value["experiment"] != expected_experiment:
        raise ValueError(f"unexpected reference experiment in {path}")
    return value, digest


def _quality_gate(
    metrics: Mapping[str, Any], fresh_baseline: Mapping[str, Any]
) -> bool:
    return (
        metrics["exact_vector_accuracy"] >= 0.98
        and metrics["macro_channel_accuracy"] >= 0.98
        and all(value >= 0.95 for value in metrics["channel_accuracy"].values())
        and metrics["class_recall"]["reward"][sg10.REWARD_LABELS[1]] >= 0.90
        and metrics["class_recall"]["done"][sg10.DONE_LABELS[1]] >= 0.90
        and metrics["exact_vector_accuracy"]
        >= fresh_baseline["decision"]["best_ann_exact_vector_accuracy"] - 0.02
        and metrics["macro_channel_accuracy"]
        >= fresh_baseline["decision"]["best_ann_macro_channel_accuracy"] - 0.02
    )


def _error_audit(
    examples: Sequence[sg10.MultiChannelExample],
    records: Mapping[str, Any],
    runtime: Mapping[str, torch.Tensor],
    spec: KernelSpec,
) -> Dict[str, Any]:
    kernel = suffix_spike_kernel(
        records["keys"],
        runtime["prototype_keys"],
        records["phases"],
        runtime["prototype_phases"],
        spec,
    )
    predictions = sg10._prediction_matrix(kernel @ runtime["alpha"])
    channel_errors = Counter()
    exit_by_step = Counter()
    exit_by_source = Counter()
    records_out = []
    for index, example in enumerate(examples):
        predicted = tuple(int(value) for value in predictions[index].tolist())
        for channel_index, (name, _labels) in enumerate(sg10.CHANNEL_SPECS):
            if predicted[channel_index] != example.target_indices[channel_index]:
                channel_errors[name] += 1
        if predicted[3] != example.target_indices[3]:
            exit_by_step[example.step_index] += 1
            exit_by_source[example.source] += 1
            records_out.append(
                {
                    "example_id": example.example_id,
                    "step_index": example.step_index,
                    "source": example.source,
                    "context_actions": example.context_actions,
                    "candidate_action": example.candidate_action,
                    "target_exit": example.target_labels[3],
                    "predicted_exit": sg10.EXIT_LABELS[predicted[3]],
                }
            )
    return {
        "channel_error_counts": dict(channel_errors),
        "exit_errors_by_step": dict(sorted(exit_by_step.items())),
        "exit_errors_by_source": dict(sorted(exit_by_source.items())),
        "step2_step3_exit_error_count": exit_by_step[2] + exit_by_step[3],
        "records": records_out,
    }


def _decision(
    data_audit: Mapping[str, Any],
    kernel_results: Mapping[str, Mapping[str, Any]],
    error_audit: Mapping[str, Any],
    online_results: Sequence[Mapping[str, Any]],
    stream_results: Sequence[Mapping[str, Any]],
    fresh_baseline: Mapping[str, Any],
    additive_reference: Mapping[str, Any],
    *,
    quick: bool,
    fresh_confirmation: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "quality_gate": "SMOKE",
            "mechanism_gate": "SMOKE",
            "online_equivalence_gate": "SMOKE",
            "training_speed_gate": "SMOKE",
            "cached_stream_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": (
                "formal_sg14r_third_fresh_confirmation"
                if fresh_confirmation
                else "formal_sg14_phase_bound_kernel"
            ),
        }
    primary = kernel_results[PRIMARY_SPEC.name]
    metrics = primary["test"]
    quality = _quality_gate(metrics, fresh_baseline)
    additive_exact = additive_reference["decision"]["primary_test_metrics"][
        "exact_vector_accuracy"
    ]
    mechanism = (
        error_audit["step2_step3_exit_error_count"] <= 2
        and (
            metrics["exact_vector_accuracy"] >= additive_exact + 0.04
            or quality
        )
    )
    online_equivalent = all(
        result["prediction_equivalent_to_batch"]
        and result["max_score_abs_difference_from_batch"] <= 1e-6
        for result in online_results
    )
    transformer_walls = [
        seed["training"]["transformer"]["elapsed_seconds"]
        for seed in fresh_baseline["seeds"]
    ]
    training_speed = (
        online_equivalent
        and primary["timing"]["selection_plus_full_fit_wall_seconds"]
        <= sg0._mean(transformer_walls)
        and all(
            online["timing"]["elapsed_seconds"] <= transformer_wall
            for online, transformer_wall in zip(online_results, transformer_walls)
        )
    )
    transformer_parameter_bytes = max(
        seed["parameter_counts"]["transformer"]["total"] * 4
        for seed in fresh_baseline["seeds"]
    )
    stream_comparison = []
    for replication, (stream, reference_seed) in enumerate(
        zip(stream_results, fresh_baseline["seeds"])
    ):
        ann = reference_seed["cached_stream"]["transformer"]["generic"][
            "candidate_timing"
        ]
        passed = (
            abs(stream["exact_vector_accuracy"] - metrics["exact_vector_accuracy"])
            <= 1e-12
            and stream["max_full_score_abs_difference"] <= 1e-6
            and stream["candidate_timing"]["p50_ms"] <= ann["p50_ms"]
            and stream["candidate_timing"]["p95_ms"] <= ann["p95_ms"]
            and stream[
                "logical_model_storage_bytes_uint8_keys_float32_alpha"
            ]
            <= transformer_parameter_bytes
        )
        stream_comparison.append(
            {
                "replication": replication,
                "snn_p50_ms": stream["candidate_timing"]["p50_ms"],
                "snn_p95_ms": stream["candidate_timing"]["p95_ms"],
                "transformer_p50_ms": ann["p50_ms"],
                "transformer_p95_ms": ann["p95_ms"],
                "passed": passed,
            }
        )
    stream_pass = quality and all(record["passed"] for record in stream_comparison)
    overall = (
        data_audit["passed"]
        and quality
        and mechanism
        and online_equivalent
        and training_speed
        and stream_pass
    )
    return {
        "data_gate": "PASS" if data_audit["passed"] else "FAIL",
        "quality_gate": "PASS" if quality else "FAIL",
        "mechanism_gate": "PASS" if mechanism else "FAIL",
        "online_equivalence_gate": "PASS" if online_equivalent else "FAIL",
        "training_speed_gate": "PASS" if training_speed else "FAIL",
        "cached_stream_gate": "PASS" if stream_pass else "FAIL",
        "overall": "PASS" if overall else "FAIL",
        "primary_kernel": PRIMARY_SPEC.name,
        "primary_test_metrics": metrics,
        "primary_cross_validated_metrics": primary["cross_validated"],
        "old_additive_exact": additive_exact,
        "error_audit": error_audit,
        "selection_plus_fit_wall_seconds": primary["timing"][
            "selection_plus_full_fit_wall_seconds"
        ],
        "fresh_transformer_mean_training_wall_seconds": sg0._mean(
            transformer_walls
        ),
        "per_replication_stream_comparison": stream_comparison,
        "independent_confirmation_required": not fresh_confirmation,
        "fresh_confirmation": fresh_confirmation,
        "next_route": (
            (
                "sg15_closed_loop_candidate_planner"
                if fresh_confirmation
                else "sg14r_third_fresh_games_confirmation"
            )
            if overall
            else "sg15_reservoir_content_times_delay_phase_kernel"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    fresh_baseline, fresh_digest = _load_reference(
        args.fresh_baseline.expanduser().resolve(),
        FRESH_BASELINE_SHA256,
        "E3-SG10 multichannel TextWorld event delta",
    )
    additive_reference, additive_digest = _load_reference(
        args.additive_reference.expanduser().resolve(),
        ADDITIVE_REFERENCE_SHA256,
        "E3-SG13R fresh-game suffix spike kernel confirmation",
    )
    sg14_reference = None
    sg14_digest = None
    if args.fresh_confirmation:
        sg14_reference, sg14_digest = _load_reference(
            args.sg14_reference.expanduser().resolve(),
            SG14_REFERENCE_SHA256,
            "E3-SG14 phase-bound hierarchical spike kernel",
        )
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = sg10.build_multichannel_examples(corpus_root, corpus)
    base_audit = sg10.audit_multichannel_examples(
        examples,
        vocabulary,
        expected_counts=dict(zip(SPLITS, args.expected_counts)),
        expected_groups=dict(zip(SPLITS, args.expected_groups)),
    )
    alphabet = build_action_alphabet(examples)
    delay_audit = audit_delay_data(examples, alphabet)
    alphabet_index = {token: index for index, token in enumerate(alphabet)}
    records = {
        split: extract_kernel_records(
            examples[split], alphabet_index=alphabet_index, device=device
        )
        for split in SPLITS
    }
    folds = build_game_folds(
        records["train"]["game_seeds"], fold_count=args.cv_folds
    )
    data_audit = {
        "sg10_multichannel": base_audit,
        "spike_delay_line": delay_audit,
        "cv_fold_count": len(folds),
        "official_valid_or_test_used_for_selection": False,
        "test_already_observed_for_mechanism_design": True,
        "independent_confirmation_required": True,
        "passed": (
            base_audit["passed"]
            and delay_audit["passed"]
            and len(folds) == args.cv_folds
            and len(set(len(fold) for fold in folds)) == 1
        ),
    }
    if args.fresh_confirmation:
        observed_seeds = {
            split: tuple(sorted({example.game_seed for example in examples[split]}))
            for split in SPLITS
        }
        exact_seed_match = observed_seeds == CONFIRMATION_SEEDS
        pairwise_disjoint = all(
            not (set(observed_seeds[left]) & set(observed_seeds[right]))
            for left_index, left in enumerate(SPLITS)
            for right in SPLITS[left_index + 1 :]
        )
        data_audit["fresh_confirmation"] = {
            "observed_seeds": observed_seeds,
            "expected_seeds": CONFIRMATION_SEEDS,
            "exact_seed_match": exact_seed_match,
            "pairwise_seed_disjoint": pairwise_disjoint,
            "kernel_and_lambda_frozen_before_generation": True,
        }
        data_audit["passed"] = (
            data_audit["passed"] and exact_seed_match and pairwise_disjoint
        )
    if not data_audit["passed"]:
        raise AssertionError("SG14 data audit failed")
    kernel_results = {}
    runtimes = {}
    for spec in KERNEL_SPECS:
        kernel_results[spec.name], runtimes[spec.name] = _fit_kernel_spec(
            spec,
            records,
            lambdas=args.ridge_lambdas,
            fold_count=args.cv_folds,
            device=device,
        )
    primary_result = kernel_results[PRIMARY_SPEC.name]
    primary_runtime = runtimes[PRIMARY_SPEC.name]
    confirmation_reproduction = None
    if args.fresh_confirmation:
        if sg14_reference is None:
            raise AssertionError("SG14 confirmation reference was not loaded")
        frozen = sg14_reference["kernel_results"][PRIMARY_SPEC.name]
        confirmation_reproduction = {
            "original_primary_kernel": frozen["kernel"]["name"],
            "current_primary_kernel": primary_result["kernel"]["name"],
            "original_selected_lambda": frozen["selected_lambda"],
            "current_selected_lambda": primary_result["selected_lambda"],
            "cross_validated_exact_difference": abs(
                frozen["cross_validated"]["exact_vector_accuracy"]
                - primary_result["cross_validated"]["exact_vector_accuracy"]
            ),
            "train_exact_difference": abs(
                frozen["train"]["exact_vector_accuracy"]
                - primary_result["train"]["exact_vector_accuracy"]
            ),
        }
        confirmation_reproduction["passed"] = (
            confirmation_reproduction["original_primary_kernel"]
            == confirmation_reproduction["current_primary_kernel"]
            == PRIMARY_SPEC.name
            and confirmation_reproduction["original_selected_lambda"]
            == confirmation_reproduction["current_selected_lambda"]
            == 1e-6
            and confirmation_reproduction["cross_validated_exact_difference"]
            <= 1e-12
            and confirmation_reproduction["train_exact_difference"] <= 1e-12
        )
        if not confirmation_reproduction["passed"]:
            raise AssertionError("SG14R frozen train/kernel reproduction failed")
    error_audit = _error_audit(
        examples["test"], records["test"], primary_runtime, PRIMARY_SPEC
    )
    online_results = []
    stream_results = []
    for seed in args.seeds:
        schedule = sg10.build_length_stratified_schedule(
            examples["train"],
            epochs=1,
            batch_groups=args.batch_groups,
            seed=14_001_000 + seed,
        )
        online = evaluate_online_kernel_replication(
            records,
            PRIMARY_SPEC,
            primary_runtime,
            primary_result,
            schedule,
            device=device,
        )
        online["schedule_seed"] = seed
        online_results.append(online)
        stream = evaluate_cached_kernel_stream(
            examples["test"],
            alphabet_index=alphabet_index,
            spec=PRIMARY_SPEC,
            runtime=primary_runtime,
            device=device,
            timing_repeats=args.timing_repeats,
            timing_warmup_repeats=args.timing_warmup_repeats,
        )
        stream["timing_replication"] = seed
        stream_results.append(stream)
    decision = _decision(
        data_audit,
        kernel_results,
        error_audit,
        online_results,
        stream_results,
        fresh_baseline,
        additive_reference,
        quick=args.quick,
        fresh_confirmation=args.fresh_confirmation,
    )
    return {
        "schema_version": 1,
        "experiment": (
            "E3-SG14R third-fresh phase-bound spike kernel confirmation"
            if args.fresh_confirmation
            else "E3-SG14 phase-bound hierarchical spike kernel"
        ),
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": (
                "independent third procedural game confirmation"
                if args.fresh_confirmation
                else "mechanism repair on already observed fresh test"
            ),
            "statement": (
                "A PSD phase-by-suffix product binds world stage to action history "
                "and repairs the additive kernel's factual exit-count errors."
            ),
            "what_if": (
                "What if Transformer position-attention interaction can be replaced "
                "by an explicitly phase-gated spike associative kernel?"
            ),
        },
        "references": {
            "fresh_baseline": {
                "path": str(args.fresh_baseline.expanduser().resolve()),
                "sha256": fresh_digest,
            },
            "additive_kernel": {
                "path": str(args.additive_reference.expanduser().resolve()),
                "sha256": additive_digest,
            },
            "sg14_frozen_architecture": (
                {
                    "path": str(args.sg14_reference.expanduser().resolve()),
                    "sha256": sg14_digest,
                }
                if args.fresh_confirmation
                else None
            ),
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "primary_kernel": PRIMARY_SPEC.name,
            "kernel_specs": tuple(spec.name for spec in KERNEL_SPECS),
            "cv_folds": args.cv_folds,
            "ridge_lambdas": tuple(args.ridge_lambdas),
            "schedule_seeds": tuple(args.seeds),
            "threads": args.threads if device.type == "cpu" else None,
            "batch_groups": args.batch_groups,
            "timing_repeats_per_candidate": args.timing_repeats,
            "timing_warmup_repeats_per_candidate": args.timing_warmup_repeats,
            "fresh_confirmation": args.fresh_confirmation,
        },
        "dataset": {
            "synthetic": False,
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "action_alphabet": alphabet,
            "audit": data_audit,
        },
        "kernel_results": kernel_results,
        "confirmation_reproduction": confirmation_reproduction,
        "online_replications": online_results,
        "cached_stream_replications": stream_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--fresh-baseline", type=Path, default=DEFAULT_FRESH_BASELINE)
    parser.add_argument(
        "--additive-reference", type=Path, default=DEFAULT_ADDITIVE_REFERENCE
    )
    parser.add_argument("--sg14-reference", type=Path, default=DEFAULT_SG14_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=128)
    parser.add_argument("--timing-warmup-repeats", type=int, default=16)
    parser.add_argument(
        "--ridge-lambdas", nargs="+", type=float, default=RIDGE_LAMBDAS
    )
    parser.add_argument(
        "--expected-counts",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_COUNTS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-groups",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_GROUPS[split] for split in SPLITS),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh-confirmation", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.threads,
        args.cv_folds,
        args.batch_groups,
        args.timing_repeats,
        *args.ridge_lambdas,
        *args.expected_counts,
        *args.expected_groups,
    ) <= 0:
        parser.error("all numeric controls must be positive")
    if args.timing_warmup_repeats < 0:
        parser.error("timing-warmup-repeats must be nonnegative")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.timing_repeats = 1
        args.timing_warmup_repeats = 0
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
