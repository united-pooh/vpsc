"""SG15 strict phase-isolated suffix spike associative memory."""

from __future__ import annotations

import argparse
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
)
from experiments.e3_sg14_phase_bound_kernel import (  # noqa: E402
    DEFAULT_FRESH_BASELINE,
    FRESH_BASELINE_SHA256,
    _error_audit,
    _load_reference,
    _quality_gate,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS  # noqa: E402


DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg14r_l5")
DEFAULT_OUTPUT = Path("results/e3_scan/e3_sg15_phase_isolated_kernel.json")
DEFAULT_SG14R_REFERENCE = Path(
    "results/e3_scan/e3_sg14r_third_fresh_confirmation.json"
)
DEFAULT_SG15_REFERENCE = Path("results/e3_scan/e3_sg15_phase_isolated_kernel.json")
SG14R_REFERENCE_SHA256 = (
    "B765964EAF8845A65B71F98304A560EC935BE898188856E97EC9AB0C6A48013B"
)
EXPECTED_COUNTS = {"train": 480, "valid": 120, "test": 120}
EXPECTED_GROUPS = {"train": 160, "valid": 40, "test": 40}
MECHANISM_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20261001, 20261009)),
    "test": tuple(range(20261009, 20261017)),
}
CONFIRMATION_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20261101, 20261109)),
    "test": tuple(range(20261109, 20261117)),
}


PRIMARY_SPEC = KernelSpec(
    "strict_phase_suffix",
    (0.0, 0.0, 0.0, 0.0),
    0.0,
    primary=True,
    phase_suffix_weights=(1.0, 1.0, 1.0, 1.0),
)
KERNEL_SPECS = (
    PRIMARY_SPEC,
    KernelSpec(
        "base_plus_phase_product_control",
        (1.0, 1.0, 1.0, 1.0),
        0.0,
        phase_suffix_weights=(1.0, 1.0, 1.0, 1.0),
    ),
    KernelSpec(
        "old_additive_phase_control",
        (1.0, 1.0, 1.0, 1.0),
        1.0,
    ),
    KernelSpec(
        "strict_phase_suffix2_control",
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        phase_suffix_weights=(1.0, 1.0, 1.0, 0.0),
    ),
)


def _load_dynamic_reference(
    path: Path, expected_sha: str, expected_experiment: str
) -> Tuple[Dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != expected_sha.upper():
        raise ValueError(
            f"reference SHA mismatch for {path}: expected {expected_sha}, got {digest}"
        )
    result = json.loads(payload)
    if result["experiment"] != expected_experiment:
        raise ValueError("unexpected SG15 confirmation reference experiment")
    return result, digest


def _decision(
    data_audit: Mapping[str, Any],
    kernel_results: Mapping[str, Mapping[str, Any]],
    error_audit: Mapping[str, Any],
    online_results: Sequence[Mapping[str, Any]],
    stream_results: Sequence[Mapping[str, Any]],
    fresh_baseline: Mapping[str, Any],
    sg14r_reference: Mapping[str, Any],
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
                "formal_sg15r_fourth_fresh_confirmation"
                if fresh_confirmation
                else "formal_sg15_phase_isolated_kernel"
            ),
        }
    primary = kernel_results[PRIMARY_SPEC.name]
    metrics = primary["test"]
    quality = _quality_gate(metrics, fresh_baseline)
    base_exact = sg14r_reference["decision"]["primary_test_metrics"][
        "exact_vector_accuracy"
    ]
    mechanism = (
        error_audit["step2_step3_exit_error_count"] <= 1
        and (
            metrics["exact_vector_accuracy"] >= base_exact + 0.01
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
    comparisons = []
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
        comparisons.append(
            {
                "replication": replication,
                "snn_p50_ms": stream["candidate_timing"]["p50_ms"],
                "snn_p95_ms": stream["candidate_timing"]["p95_ms"],
                "transformer_p50_ms": ann["p50_ms"],
                "transformer_p95_ms": ann["p95_ms"],
                "passed": passed,
            }
        )
    stream_pass = quality and all(record["passed"] for record in comparisons)
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
        "fresh_confirmation": fresh_confirmation,
        "independent_confirmation_required": not fresh_confirmation,
        "primary_kernel": PRIMARY_SPEC.name,
        "primary_test_metrics": metrics,
        "primary_cross_validated_metrics": primary["cross_validated"],
        "base_plus_product_reference_exact": base_exact,
        "error_audit": error_audit,
        "selection_plus_fit_wall_seconds": primary["timing"][
            "selection_plus_full_fit_wall_seconds"
        ],
        "per_replication_stream_comparison": comparisons,
        "next_route": (
            (
                "sg16_closed_loop_candidate_planner"
                if fresh_confirmation
                else "sg15r_fourth_fresh_games_confirmation"
            )
            if overall
            else "sg16_observation_reservoir_times_strict_phase_kernel"
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
    sg14r_reference, sg14r_digest = _load_reference(
        args.sg14r_reference.expanduser().resolve(),
        SG14R_REFERENCE_SHA256,
        "E3-SG14R third-fresh phase-bound spike kernel confirmation",
    )
    sg15_reference = None
    sg15_digest = None
    if args.fresh_confirmation:
        if not args.sg15_reference_sha:
            raise ValueError("fresh confirmation requires --sg15-reference-sha")
        sg15_reference, sg15_digest = _load_dynamic_reference(
            args.sg15_reference.expanduser().resolve(),
            args.sg15_reference_sha,
            "E3-SG15 strict phase-isolated suffix spike kernel",
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
    expected_seed_map = CONFIRMATION_SEEDS if args.fresh_confirmation else MECHANISM_SEEDS
    observed_seed_map = {
        split: tuple(sorted({example.game_seed for example in examples[split]}))
        for split in SPLITS
    }
    seed_match = observed_seed_map == expected_seed_map
    pairwise_disjoint = all(
        not (set(observed_seed_map[left]) & set(observed_seed_map[right]))
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    )
    data_audit = {
        "sg10_multichannel": base_audit,
        "spike_delay_line": delay_audit,
        "cv_fold_count": len(folds),
        "observed_seeds": observed_seed_map,
        "expected_seeds": expected_seed_map,
        "exact_seed_match": seed_match,
        "pairwise_seed_disjoint": pairwise_disjoint,
        "official_valid_or_test_used_for_selection": False,
        "test_already_observed_for_design": not args.fresh_confirmation,
        "passed": (
            base_audit["passed"]
            and delay_audit["passed"]
            and len(folds) == args.cv_folds
            and len(set(len(fold) for fold in folds)) == 1
            and seed_match
            and pairwise_disjoint
        ),
    }
    if not data_audit["passed"]:
        raise AssertionError("SG15 data audit failed")
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
        if sg15_reference is None:
            raise AssertionError("SG15 reference was not loaded")
        frozen = sg15_reference["kernel_results"][PRIMARY_SPEC.name]
        confirmation_reproduction = {
            "original_kernel": frozen["kernel"]["name"],
            "current_kernel": primary_result["kernel"]["name"],
            "original_lambda": frozen["selected_lambda"],
            "current_lambda": primary_result["selected_lambda"],
            "cv_exact_difference": abs(
                frozen["cross_validated"]["exact_vector_accuracy"]
                - primary_result["cross_validated"]["exact_vector_accuracy"]
            ),
            "train_exact_difference": abs(
                frozen["train"]["exact_vector_accuracy"]
                - primary_result["train"]["exact_vector_accuracy"]
            ),
        }
        confirmation_reproduction["passed"] = (
            confirmation_reproduction["original_kernel"]
            == confirmation_reproduction["current_kernel"]
            == PRIMARY_SPEC.name
            and confirmation_reproduction["original_lambda"]
            == confirmation_reproduction["current_lambda"]
            == 1e-6
            and confirmation_reproduction["cv_exact_difference"] <= 1e-12
            and confirmation_reproduction["train_exact_difference"] <= 1e-12
        )
        if not confirmation_reproduction["passed"]:
            raise AssertionError("SG15R frozen train/kernel reproduction failed")
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
            seed=15_001_000 + seed,
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
        sg14r_reference,
        quick=args.quick,
        fresh_confirmation=args.fresh_confirmation,
    )
    return {
        "schema_version": 1,
        "experiment": (
            "E3-SG15R fourth-fresh strict phase suffix confirmation"
            if args.fresh_confirmation
            else "E3-SG15 strict phase-isolated suffix spike kernel"
        ),
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": (
                "independent fourth procedural game confirmation"
                if args.fresh_confirmation
                else "mechanism repair on observed third test"
            ),
            "statement": (
                "Strict phase isolation removes cross-stage negative transfer "
                "from sparse suffix associative memory."
            ),
            "what_if": (
                "What if event memories from different world phases must be "
                "orthogonal rather than merely down-weighted?"
            ),
        },
        "references": {
            "fresh_transformer_baseline": {
                "path": str(args.fresh_baseline.expanduser().resolve()),
                "sha256": fresh_digest,
            },
            "sg14r_failed_primary": {
                "path": str(args.sg14r_reference.expanduser().resolve()),
                "sha256": sg14r_digest,
            },
            "sg15_frozen_architecture": (
                {
                    "path": str(args.sg15_reference.expanduser().resolve()),
                    "sha256": sg15_digest,
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
    parser.add_argument("--sg14r-reference", type=Path, default=DEFAULT_SG14R_REFERENCE)
    parser.add_argument("--sg15-reference", type=Path, default=DEFAULT_SG15_REFERENCE)
    parser.add_argument("--sg15-reference-sha", type=str, default="")
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
