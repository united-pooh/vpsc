"""SG26A expanded SG22R raw-language CUDA Graph comparison.

This experiment moves the SG25F block-parallel SNN training backend from the
small SG24 corpus to the independently generated SG22R TextWorld corpus.  It
keeps every target token, uses four frozen padded buckets, and dispatches only
the three training examples longer than 128 steps to the exact SG25E serial
kernel.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_sg22r_seventh_fresh_confirmation as sg22r  # noqa: E402
from experiments import e3_sg25e_bucketed_batch_graph as sg25e  # noqa: E402
from experiments import e3_sg25f_parallel_affine_graph as sg25f  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402


MODES = ("snn_parallel", "lstm", "transformer")
BATCH_SIZE = 16
BUCKET_CAPACITIES = ((64, 27), (96, 55), (128, 71), (160, 71))
EXPECTED_BUCKET_COUNTS = (129, 116, 72, 3)
EXPECTED_COUNTS = {"train": 320, "valid": 80, "test": 80}
EXPECTED_PAIRS = {"train": 160, "valid": 40, "test": 40}
EXPECTED_UNIQUE_TARGETS = {"train": 145, "valid": 40, "test": 40}
EXPECTED_INPUT_MAX = {"train": 130, "valid": 153, "test": 133}
EXPECTED_TARGET_MAX = {"train": 71, "valid": 78, "test": 68}
EXPECTED_VOCAB_SIZE = 316
EXPECTED_SG22R_SHA256 = (
    "1a75839740a7913e555fbebd5eb462aa4c50d5324709b11f507a9fb607b7db92"
)
EXPECTED_SG25F_SHA256 = (
    "e30fc0e557399a96f0beebcf956fb3effca2a36ce97f008745da1fad40ad22bc"
)
EXPECTED_SG25G_SHA256 = (
    "79749885e5a11a0d1588c0ad169306a11c87f2e7085bcd15f637950e83c21e1e"
)
SG25F_PARALLEL_EXAMPLES_PER_SECOND = 20133.046722635696
TASK_MARGIN = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expanded_mode_logits(
    original: Any,
    mode: str,
    model: Any,
    batch: sg25e.DeviceBatch,
) -> torch.Tensor:
    if mode == "snn_parallel" and int(batch.input_ids.shape[1]) > 128:
        return sg25e._snn_batched_logits(
            model, batch.input_ids, batch.query_indices
        )
    return original(mode, model, batch)


@contextmanager
def _expanded_backend() -> Iterator[None]:
    """Install the frozen four-bucket and T>128 dispatch for one run."""

    original_buckets = sg25e.BUCKET_CAPACITIES
    original_logits = sg25f._mode_logits

    def dispatch(
        mode: str, model: Any, batch: sg25e.DeviceBatch
    ) -> torch.Tensor:
        return _expanded_mode_logits(original_logits, mode, model, batch)

    sg25e.BUCKET_CAPACITIES = BUCKET_CAPACITIES
    sg25f._mode_logits = dispatch
    try:
        yield
    finally:
        sg25f._mode_logits = original_logits
        sg25e.BUCKET_CAPACITIES = original_buckets


def expanded_data_audit(
    examples: Mapping[str, Sequence[Any]], vocabulary: Any
) -> Dict[str, Any]:
    legacy = sg0.audit_examples(examples, vocabulary)
    observed = {}
    for split, values in examples.items():
        observed[split] = {
            "example_count": len(values),
            "pair_count": len({value.pair_id for value in values}),
            "unique_target_count": len({value.target_text for value in values}),
            "input_max": max(value.input_length for value in values),
            "target_max_with_eos": max(len(value.target_ids) for value in values),
            "target_unknown_ratio": legacy["splits"][split][
                "target_unknown_ratio"
            ],
            "format_only_target_count": legacy["splits"][split][
                "format_only_target_count"
            ],
        }
    test_unique = observed["test"]["unique_target_count"]
    test_overlap_ratio = (
        legacy["target_overlap"]["test_in_train"] / test_unique
    )
    baseline = legacy["baselines"]["action_majority"]
    task_edit_threshold = baseline["edit_similarity"] + TASK_MARGIN
    exact_shape_pass = all(
        observed[split]["example_count"] == EXPECTED_COUNTS[split]
        and observed[split]["pair_count"] == EXPECTED_PAIRS[split]
        and observed[split]["unique_target_count"]
        == EXPECTED_UNIQUE_TARGETS[split]
        and observed[split]["input_max"] == EXPECTED_INPUT_MAX[split]
        and observed[split]["target_max_with_eos"]
        == EXPECTED_TARGET_MAX[split]
        for split in EXPECTED_COUNTS
    )
    content_pass = all(
        record["target_unknown_ratio"] < 0.10
        and record["format_only_target_count"] == 0
        for record in observed.values()
    )
    return {
        "legacy_sg24_audit_passed_expected_false": legacy["passed"],
        "legacy_audit": legacy,
        "observed": observed,
        "vocabulary_size": len(vocabulary),
        "test_target_overlap_with_train_ratio": test_overlap_ratio,
        "action_majority_edit_similarity": baseline["edit_similarity"],
        "action_majority_paired_action_sensitivity": baseline[
            "paired_action_sensitivity"
        ],
        "copy_observation_edit_similarity": legacy["baselines"][
            "copy_observation"
        ]["edit_similarity"],
        "task_edit_threshold": task_edit_threshold,
        "passed": exact_shape_pass
        and content_pass
        and len(vocabulary) == EXPECTED_VOCAB_SIZE
        and test_overlap_ratio <= 0.20
        and not legacy["passed"],
    }


def _bucket_index(example: Any) -> int:
    for index, (time_capacity, query_capacity) in enumerate(BUCKET_CAPACITIES):
        if (
            example.input_length <= time_capacity
            and len(example.target_ids) <= query_capacity
        ):
            return index
    raise AssertionError(
        f"example ({example.input_length}, {len(example.target_ids)}) "
        "exceeds SG26A frozen buckets"
    )


def expanded_bucket_audit(examples: Sequence[Any]) -> Dict[str, Any]:
    counts = [0 for _ in BUCKET_CAPACITIES]
    input_tokens = [0 for _ in BUCKET_CAPACITIES]
    target_tokens = [0 for _ in BUCKET_CAPACITIES]
    records = []
    for example in examples:
        index = _bucket_index(example)
        counts[index] += 1
        input_tokens[index] += example.input_length
        target_tokens[index] += len(example.target_ids)
    for index, (time_capacity, query_capacity) in enumerate(BUCKET_CAPACITIES):
        count = counts[index]
        records.append(
            {
                "time_capacity": time_capacity,
                "query_capacity": query_capacity,
                "example_count": count,
                "input_tokens": input_tokens[index],
                "target_tokens": target_tokens[index],
                "input_utilization": input_tokens[index]
                / (count * time_capacity),
                "target_utilization": target_tokens[index]
                / (count * query_capacity),
                "snn_backend": (
                    "sg25f_parallel" if time_capacity <= 128 else "sg25e_serial"
                ),
            }
        )
    fallback_count = counts[-1]
    return {
        "records": records,
        "counts": tuple(counts),
        "parallel_example_count": len(examples) - fallback_count,
        "serial_fallback_example_count": fallback_count,
        "serial_fallback_ratio": fallback_count / len(examples),
        "passed": tuple(counts) == EXPECTED_BUCKET_COUNTS,
    }


def _build_schedule(
    examples: Sequence[Any],
    vocabulary: Any,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> Tuple[sg25e.DeviceBatch, ...]:
    return sg25e.build_batch_schedule(
        examples,
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        seed=seed,
        device=device,
    )


def _coverage_prefix(
    batches: Sequence[sg25e.DeviceBatch], updates: int
) -> Tuple[sg25e.DeviceBatch, ...]:
    representatives: Dict[Tuple[int, int, int], sg25e.DeviceBatch] = {}
    for batch in batches:
        representatives.setdefault(batch.key, batch)
    ordered = [representatives[key] for key in sorted(representatives)]
    ordered.extend(batches)
    return tuple(ordered[:updates])


def fallback_dispatch_audit(
    batches: Sequence[sg25e.DeviceBatch],
    vocabulary: Any,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    batch = next(value for value in batches if value.input_ids.shape[1] > 128)
    model = sg25f._build_mode_model("snn_parallel", vocabulary, device=device)
    model.eval()
    with torch.no_grad():
        dispatched = sg25f._mode_logits("snn_parallel", model, batch)
        serial = sg25e._snn_batched_logits(
            model, batch.input_ids, batch.query_indices
        )
    maximum = float((dispatched - serial).abs().max().item())
    passed = bool(torch.equal(dispatched, serial))
    result = {
        "batch_key": batch.key,
        "real_example_count": batch.real_example_count,
        "dispatch_rule": "T>128 -> SG25E serial native fused scan",
        "maximum_logit_absolute_error": maximum,
        "exact_equal": passed,
        "passed": passed,
    }
    del model, dispatched, serial
    gc.collect()
    torch.cuda.empty_cache()
    return result


def precondition_graph_allocator(
    train_examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    batches = _build_schedule(
        train_examples,
        vocabulary,
        epochs=1,
        seed=26_090_000,
        device=device,
    )
    model = sg25f._build_mode_model("snn_parallel", vocabulary, device=device)
    trainer = sg25f.GraphTrainer(
        "snn_parallel", model, batches, device=device
    )
    audit = dict(trainer.capture_audit)
    del trainer, model, batches
    gc.collect()
    torch.cuda.empty_cache()
    return audit


def _per_bucket_benchmark(
    trainer: sg25f.GraphTrainer,
    batches: Sequence[sg25e.DeviceBatch],
) -> Dict[str, Any]:
    grouped: Dict[Tuple[int, int, int], list[sg25e.DeviceBatch]] = {}
    for batch in batches:
        grouped.setdefault(batch.key, []).append(batch)
    result = {}
    for key in sorted(grouped):
        record = sg25e.graph_benchmark(trainer, grouped[key])
        label = f"B{key[0]}_T{key[1]}_Q{key[2]}"
        result[label] = {
            **record,
            "snn_backend": (
                "sg25f_parallel" if key[1] <= 128 else "sg25e_serial"
            ),
        }
    return result


def _per_bucket_profile(
    trainer: sg25f.GraphTrainer,
    batches: Sequence[sg25e.DeviceBatch],
) -> Dict[str, Any]:
    representatives: Dict[Tuple[int, int, int], sg25e.DeviceBatch] = {}
    for batch in batches:
        representatives.setdefault(batch.key, batch)
    result = {}
    for key in sorted(representatives):
        label = f"B{key[0]}_T{key[1]}_Q{key[2]}"
        result[label] = {
            **sg25f.graph_profile(trainer, representatives[key]),
            "snn_backend": (
                "sg25f_parallel" if key[1] <= 128 else "sg25e_serial"
            ),
        }
    return result


def quality_audit(
    mode: str,
    trainer: sg25f.GraphTrainer,
    batches: Sequence[sg25e.DeviceBatch],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    trainer.reset()
    model = trainer.model
    pre = {
        split: sg0.evaluate_teacher(model, raw_examples[split], device=device)
        for split in ("valid", "test")
    }
    model.train(True)
    samples = []
    losses = []
    all_losses_finite = True
    started = time.perf_counter_ns()
    for batch in batches:
        record = trainer.replay(batch, timed=True, inspect=True)
        samples.append(record["wall_ms"])
        losses.append(record["loss"])
        all_losses_finite = all_losses_finite and math.isfinite(record["loss"])
    train_wall_seconds = (time.perf_counter_ns() - started) / 1e9
    post = {
        split: sg0.evaluate_teacher(model, raw_examples[split], device=device)
        for split in ("train", "valid", "test")
    }
    generation = sg0.generate_model(
        model,
        raw_examples["test"],
        vocabulary,
        max_tokens=sg0.MAX_GENERATION_TOKENS,
        device=device,
        include_records=True,
    )
    warmup = len(samples) // 5
    expected_updates = epochs * sum(
        math.ceil(count / BATCH_SIZE) for count in EXPECTED_BUCKET_COUNTS
    )
    return {
        "mode": mode,
        "epochs": epochs,
        "updates": len(batches),
        "expected_updates": expected_updates,
        "update_count_passed": len(batches) == expected_updates,
        "all_losses_finite": all_losses_finite,
        "train_wall_seconds": train_wall_seconds,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "timing": _sample_summary(samples[warmup:], 1),
        "pre_teacher": pre,
        "post_teacher": post,
        "generation": generation,
    }


def run_mode(
    mode: str,
    train_examples: Sequence[Any],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    benchmark_epochs: int,
    quality_epochs: int,
    equivalence_updates: int,
) -> Dict[str, Any]:
    benchmark_batches = _build_schedule(
        train_examples,
        vocabulary,
        epochs=benchmark_epochs,
        seed=26_100_000,
        device=device,
    )
    quality_batches = _build_schedule(
        train_examples,
        vocabulary,
        epochs=quality_epochs,
        seed=26_110_000,
        device=device,
    )
    equivalence_batches = _coverage_prefix(
        quality_batches, equivalence_updates
    )
    gc.collect()
    torch.cuda.empty_cache()
    clean_allocated = torch.cuda.memory_allocated(device)
    clean_reserved = torch.cuda.memory_reserved(device)
    model = sg25f._build_mode_model(mode, vocabulary, device=device)
    trainer = sg25f.GraphTrainer(mode, model, quality_batches, device=device)
    trainer.capture_audit.update(
        {
            "total_allocated_delta_from_clean_bytes": max(
                0, torch.cuda.memory_allocated(device) - clean_allocated
            ),
            "total_reserved_delta_from_clean_bytes": max(
                0, torch.cuda.memory_reserved(device) - clean_reserved
            ),
        }
    )
    equivalence = sg25f.graph_equivalence(
        mode,
        trainer,
        equivalence_batches,
        vocabulary,
        device=device,
        updates=len(equivalence_batches),
    )
    equivalence["coverage_prefix_keys"] = tuple(
        tuple(batch.key) for batch in equivalence_batches[: len(BUCKET_CAPACITIES)]
    )
    benchmark = sg25e.graph_benchmark(trainer, benchmark_batches)
    bucket_benchmark = _per_bucket_benchmark(trainer, benchmark_batches)
    profiler = sg25f.graph_profile(
        trainer, sg25e._fullest_batch(benchmark_batches)
    )
    bucket_profiler = _per_bucket_profile(trainer, benchmark_batches)
    quality = quality_audit(
        mode,
        trainer,
        quality_batches,
        raw_examples,
        vocabulary,
        device=device,
        epochs=quality_epochs,
    )
    result = {
        "capture": dict(trainer.capture_audit),
        "equivalence": equivalence,
        "benchmark": benchmark,
        "bucket_benchmark": bucket_benchmark,
        "profiler": profiler,
        "bucket_profiler": bucket_profiler,
        "quality": quality,
    }
    del (
        trainer,
        model,
        benchmark_batches,
        quality_batches,
        equivalence_batches,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _decision(
    data_audit: Mapping[str, Any],
    bucket_audit: Mapping[str, Any],
    fallback_audit: Mapping[str, Any],
    primary: Mapping[str, Mapping[str, Any]],
    canonical_parallel: Mapping[str, Any],
    *,
    quick: bool,
) -> Dict[str, Any]:
    capture_pass = all(
        record["capture"]["shape_count"] == len(BUCKET_CAPACITIES)
        for record in primary.values()
    )
    equivalence_pass = all(
        record["equivalence"]["passed"] for record in primary.values()
    )
    eps = {
        mode: record["benchmark"]["effective_examples_per_second"]
        for mode, record in primary.items()
    }
    target_tps = {
        mode: record["benchmark"]["effective_target_tokens_per_second"]
        for mode, record in primary.items()
    }
    p50 = {
        mode: record["benchmark"]["per_real_example_timing"]["p50_ms"]
        for mode, record in primary.items()
    }
    speed_pass = (
        eps["snn_parallel"] >= 0.75 * SG25F_PARALLEL_EXAMPLES_PER_SECOND
        and eps["snn_parallel"] >= eps["lstm"]
        and eps["snn_parallel"] >= eps["transformer"]
        and target_tps["snn_parallel"] >= target_tps["lstm"]
        and target_tps["snn_parallel"] >= target_tps["transformer"]
        and p50["snn_parallel"] <= p50["lstm"]
        and p50["snn_parallel"] <= p50["transformer"]
    )
    snn = primary["snn_parallel"]
    maximum_host_api = {
        mode: max(
            profile["host_launch_and_copy_api_count"]
            for profile in record["bucket_profiler"].values()
        )
        for mode, record in primary.items()
    }
    event_pass = all(
        maximum_host_api["snn_parallel"] <= maximum_host_api[mode]
        for mode in ("lstm", "transformer")
    )
    allocated_ratio = (
        snn["capture"]["allocated_delta_bytes"]
        / canonical_parallel["capture"]["allocated_delta_bytes"]
    )
    peak_ratio = (
        snn["capture"]["peak_additional_allocated_bytes"]
        / canonical_parallel["capture"]["peak_additional_allocated_bytes"]
    )
    memory_pass = allocated_ratio <= 1.50 and peak_ratio <= 1.50
    quality = {mode: record["quality"] for mode, record in primary.items()}
    all_finite = all(
        record["all_losses_finite"] and record["update_count_passed"]
        for record in quality.values()
    )
    ann_improvement_pass = all(
        quality[mode]["post_teacher"]["test"]["nll"]
        <= quality[mode]["pre_teacher"]["test"]["nll"] - 0.10
        for mode in ("lstm", "transformer")
    )
    best_ann_nll = min(
        quality[mode]["post_teacher"]["test"]["nll"]
        for mode in ("lstm", "transformer")
    )
    best_ann_edit = max(
        quality[mode]["generation"]["edit_similarity"]
        for mode in ("lstm", "transformer")
    )
    snn_quality = quality["snn_parallel"]
    snn_cross_pass = (
        snn_quality["post_teacher"]["test"]["nll"] <= best_ann_nll + 0.10
        and snn_quality["generation"]["edit_similarity"]
        >= best_ann_edit - 0.05
        and snn_quality["generation"]["paired_action_sensitivity"] >= 0.50
    )
    quality_pass = all_finite and ann_improvement_pass and snn_cross_pass
    task_threshold = data_audit["task_edit_threshold"]
    best_neural_edit = max(
        record["generation"]["edit_similarity"] for record in quality.values()
    )
    snn_edit = snn_quality["generation"]["edit_similarity"]
    task_pass = best_neural_edit >= task_threshold and snn_edit >= task_threshold
    infrastructure_gates = {
        "data_gate": bool(data_audit["passed"] and bucket_audit["passed"]),
        "fallback_gate": bool(fallback_audit["passed"]),
        "capture_gate": capture_pass,
        "equivalence_gate": equivalence_pass,
        "event_gate": event_pass,
        "memory_gate": memory_pass,
    }
    if quick:
        return {
            **{
                name: "PASS" if passed else "FAIL"
                for name, passed in infrastructure_gates.items()
            },
            "speed_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "task_validity_gate": "SMOKE",
            "overall": (
                "SMOKE" if all(infrastructure_gates.values()) else "FAIL"
            ),
            "effective_examples_per_second": eps,
            "effective_target_tokens_per_second": target_tps,
            "per_real_example_p50_ms": p50,
            "allocated_ratio_to_sg25f": allocated_ratio,
            "peak_ratio_to_sg25f": peak_ratio,
            "maximum_bucket_host_api": maximum_host_api,
            "diagnostic_speed_pass": speed_pass,
            "diagnostic_quality_pass": quality_pass,
            "diagnostic_task_validity_pass": task_pass,
        }
    gates = {
        **infrastructure_gates,
        "speed_gate": speed_pass,
        "quality_gate": quality_pass,
        "task_validity_gate": task_pass,
    }
    if not infrastructure_gates["data_gate"]:
        next_route = "repair_data_or_provenance"
    elif not task_pass and best_neural_edit < task_threshold:
        next_route = "increase_real_task_model_capacity"
    elif not task_pass:
        next_route = "snn_representation_learning"
    elif not quality_pass:
        next_route = "train_only_gradient_coherence_optimizer"
    elif not speed_pass:
        next_route = "sg26b_long_bucket_parallel_dispatch"
    else:
        next_route = "multimodal_rollout_closed_loop"
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "effective_examples_per_second": eps,
        "effective_target_tokens_per_second": target_tps,
        "per_real_example_p50_ms": p50,
        "snn_to_sg25f_speed_ratio": eps["snn_parallel"]
        / SG25F_PARALLEL_EXAMPLES_PER_SECOND,
        "allocated_ratio_to_sg25f": allocated_ratio,
        "peak_ratio_to_sg25f": peak_ratio,
        "maximum_bucket_host_api": maximum_host_api,
        "all_finite_and_update_count_pass": all_finite,
        "ann_nll_improvement_pass": ann_improvement_pass,
        "snn_cross_architecture_quality_pass": snn_cross_pass,
        "best_ann_test_nll": best_ann_nll,
        "best_ann_edit_similarity": best_ann_edit,
        "best_neural_edit_similarity": best_neural_edit,
        "snn_edit_similarity": snn_edit,
        "task_edit_threshold": task_threshold,
        "next_route": next_route,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG26A requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG26A requires the frozen V100 backend")

    references = {
        "sg22r": ROOT
        / "results/e3_scan/e3_sg22r_seventh_fresh_confirmation.json",
        "sg25f": ROOT / "results/e3_scan/e3_sg25f_parallel_affine_graph.json",
        "sg25g": ROOT / "results/e3_scan/e3_sg25g_time_dilated_adam.json",
    }
    reference_hashes = {name: _sha256(path) for name, path in references.items()}
    expected_hashes = {
        "sg22r": EXPECTED_SG22R_SHA256,
        "sg25f": EXPECTED_SG25F_SHA256,
        "sg25g": EXPECTED_SG25G_SHA256,
    }
    if reference_hashes != expected_hashes:
        raise AssertionError(
            f"SG26A reference hash mismatch: {reference_hashes}"
        )
    sg25f_payload = json.loads(references["sg25f"].read_text(encoding="utf-8"))
    canonical_parallel = sg25f_payload["speed_records"]["snn_parallel"]
    _, serial_extension = sg25f.load_serial_extension()
    _, parallel_extension = sg25f.load_parallel_extension()

    corpus_root = args.corpus_dir.expanduser().resolve()
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=sg22r.EXPECTED_SEEDS
    )
    corpus = sg0.load_event_corpus(corpus_root)
    raw_examples, vocabulary = sg0.build_counterfactual_examples(
        corpus_root, corpus
    )
    data_audit = expanded_data_audit(raw_examples, vocabulary)
    bucket_audit = expanded_bucket_audit(raw_examples["train"])
    if not data_audit["passed"] or not bucket_audit["passed"]:
        raise AssertionError("SG26A expanded data/bucket audit failed")

    with _expanded_backend():
        audit_batches = _build_schedule(
            raw_examples["train"],
            vocabulary,
            epochs=1,
            seed=26_080_000,
            device=device,
        )
        fallback_audit = fallback_dispatch_audit(
            audit_batches, vocabulary, device=device
        )
        del audit_batches
        allocator_preconditioning = precondition_graph_allocator(
            raw_examples["train"], vocabulary, device=device
        )
        primary = {
            mode: run_mode(
                mode,
                raw_examples["train"],
                raw_examples,
                vocabulary,
                device=device,
                benchmark_epochs=args.benchmark_epochs,
                quality_epochs=args.quality_epochs,
                equivalence_updates=args.equivalence_updates,
            )
            for mode in MODES
        }

    decision = _decision(
        data_audit,
        bucket_audit,
        fallback_audit,
        primary,
        canonical_parallel,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG26A expanded SG22R raw-language CUDA Graph comparison",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "modes": MODES,
            "batch_size": BATCH_SIZE,
            "bucket_capacities": BUCKET_CAPACITIES,
            "expected_bucket_counts": EXPECTED_BUCKET_COUNTS,
            "benchmark_epochs": args.benchmark_epochs,
            "quality_epochs": args.quality_epochs,
            "equivalence_updates": args.equivalence_updates,
            "optimizer": "AdamW(lr=1e-3,betas=.9/.999,wd=.01,fused,capturable)",
            "loss": "mean(valid-token CE per example), then mean(real examples)",
            "snn_dispatch": "T<=128 SG25F parallel; T>128 SG25E serial",
            "target_truncation": False,
            "copy_and_padding_included": True,
            "device": "cuda:0",
        },
        "provenance": {
            "references": {name: str(path) for name, path in references.items()},
            "reference_sha256": reference_hashes,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "manifest": manifest,
            "serial_extension": serial_extension,
            "parallel_extension": parallel_extension,
            "canonical_sg25f_parallel": canonical_parallel,
            "data_audit": data_audit,
            "bucket_audit": bucket_audit,
        },
        "fallback_dispatch_audit": fallback_audit,
        "allocator_preconditioning": allocator_preconditioning,
        "primary": primary,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir", type=Path, default=sg22r.DEFAULT_CORPUS
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg26a_expanded_raw_language.json"),
    )
    parser.add_argument("--benchmark-epochs", type=int, default=10)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--equivalence-updates", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(
        args.benchmark_epochs,
        args.quality_epochs,
        args.equivalence_updates,
    ) <= 0:
        parser.error("all counts must be positive")
    if args.quick:
        args.benchmark_epochs = 1
        args.quality_epochs = min(args.quality_epochs, 2)
        args.equivalence_updates = min(args.equivalence_updates, 8)
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
