"""SG26B train/valid selection of a length-mass-conserving LM loss."""

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
from typing import Any, Dict, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F


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
from experiments import e3_sg26a_expanded_raw_language as sg26a  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402


ALPHAS = (1.0, 0.5, 0.0)
MODES = sg26a.MODES
BATCH_SIZE = sg26a.BATCH_SIZE
VALID_GENERATION_TOKENS = 80
EXPECTED_SG26A_SHA256 = (
    "df6c0664603e876fc52e02c16932f86624a4ad9006cdbed73083b3b96036b98d"
)
SG25F_PARALLEL_EXAMPLES_PER_SECOND = 20133.046722635696


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alpha_label(alpha: float) -> str:
    return f"alpha_{str(alpha).replace('.', '_')}"


def length_mass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_mask: torch.Tensor,
    example_mask: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Interpolate exactly between example-mean and token-mean CE."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    lengths = token_mask.sum(dim=1).clamp_min(1.0)
    scaled_loss = (token_losses * token_mask).sum(dim=1) / lengths.pow(alpha)
    mass = lengths.pow(1.0 - alpha) * example_mask
    return (scaled_loss * example_mask).sum() / mass.sum().clamp_min(1.0)


@contextmanager
def _patched_loss(alpha: float) -> Iterator[None]:
    original = sg25e._masked_example_mean_loss

    def candidate(
        logits: torch.Tensor,
        targets: torch.Tensor,
        token_mask: torch.Tensor,
        example_mask: torch.Tensor,
    ) -> torch.Tensor:
        return length_mass_loss(
            logits,
            targets,
            token_mask,
            example_mask,
            alpha=alpha,
        )

    sg25e._masked_example_mean_loss = candidate
    try:
        yield
    finally:
        sg25e._masked_example_mean_loss = original


def formula_audit(*, device: torch.device) -> Dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(26_120_001)
    logits = torch.randn(3, 5, 7, generator=generator, device=device)
    targets = torch.randint(0, 7, (3, 5), generator=generator, device=device)
    token_mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
        dtype=torch.float32,
        device=device,
    )
    example_mask = torch.tensor([1, 1, 0], dtype=torch.float32, device=device)
    legacy = sg25e._masked_example_mean_loss(
        logits, targets, token_mask, example_mask
    )
    alpha_one = length_mass_loss(
        logits, targets, token_mask, example_mask, alpha=1.0
    )
    flat_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    token_reference = (flat_losses * token_mask).sum() / token_mask.sum()
    alpha_zero = length_mass_loss(
        logits, targets, token_mask, example_mask, alpha=0.0
    )
    one_gap = float((legacy - alpha_one).abs().item())
    zero_gap = float((token_reference - alpha_zero).abs().item())
    return {
        "alpha_one_legacy_absolute_gap": one_gap,
        "alpha_zero_token_mean_absolute_gap": zero_gap,
        "dummy_row_masked": True,
        # The alpha=0 reference associates the FP32 reductions differently;
        # tolerate two V100 ULPs while still recording the exact residual.
        "passed": one_gap <= 5e-7 and zero_gap <= 5e-7,
    }


def valid_action_majority(
    examples: Mapping[str, Sequence[Any]], vocabulary: Any
) -> Dict[str, Any]:
    majority = sg0._majority_targets(examples["train"])
    predictions = {
        value.example_id: majority[value.action_type]
        for value in examples["valid"]
    }
    metrics = sg0.evaluate_predictions(
        examples["valid"], predictions, vocabulary, include_records=False
    )
    return {
        **metrics,
        "task_edit_threshold": metrics["edit_similarity"] + sg26a.TASK_MARGIN,
        "source": "train action-conditional majority evaluated on valid",
    }


def valid_quality_audit(
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
    pre_valid = sg0.evaluate_teacher(
        model, raw_examples["valid"], device=device
    )
    model.train(True)
    samples = []
    losses = []
    all_finite = True
    started = time.perf_counter_ns()
    for batch in batches:
        record = trainer.replay(batch, timed=True, inspect=True)
        samples.append(record["wall_ms"])
        losses.append(record["loss"])
        all_finite = all_finite and math.isfinite(record["loss"])
    train_wall_seconds = (time.perf_counter_ns() - started) / 1e9
    post = {
        split: sg0.evaluate_teacher(model, raw_examples[split], device=device)
        for split in ("train", "valid")
    }
    generation = sg0.generate_model(
        model,
        raw_examples["valid"],
        vocabulary,
        max_tokens=VALID_GENERATION_TOKENS,
        device=device,
        include_records=True,
    )
    warmup = len(samples) // 5
    expected_updates = epochs * sum(
        math.ceil(count / BATCH_SIZE)
        for count in sg26a.EXPECTED_BUCKET_COUNTS
    )
    return {
        "mode": mode,
        "epochs": epochs,
        "updates": len(batches),
        "expected_updates": expected_updates,
        "update_count_passed": len(batches) == expected_updates,
        "all_losses_finite": all_finite,
        "train_wall_seconds": train_wall_seconds,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "timing": _sample_summary(samples[warmup:], 1),
        "pre_teacher": {"valid": pre_valid},
        "post_teacher": post,
        "generation": generation,
        "model_test_teacher_calls": 0,
        "model_test_generation_calls": 0,
    }


def run_candidate(
    mode: str,
    alpha: float,
    train_examples: Sequence[Any],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    benchmark_epochs: int,
    quality_epochs: int,
    equivalence_updates: int,
) -> Dict[str, Any]:
    benchmark_batches = sg26a._build_schedule(
        train_examples,
        vocabulary,
        epochs=benchmark_epochs,
        seed=26_130_000,
        device=device,
    )
    quality_batches = sg26a._build_schedule(
        train_examples,
        vocabulary,
        epochs=quality_epochs,
        seed=26_140_000,
        device=device,
    )
    equivalence_batches = sg26a._coverage_prefix(
        quality_batches, equivalence_updates
    )
    gc.collect()
    torch.cuda.empty_cache()
    with _patched_loss(alpha):
        model = sg25f._build_mode_model(mode, vocabulary, device=device)
        trainer = sg25f.GraphTrainer(
            mode, model, quality_batches, device=device
        )
        equivalence = sg25f.graph_equivalence(
            mode,
            trainer,
            equivalence_batches,
            vocabulary,
            device=device,
            updates=len(equivalence_batches),
        )
        benchmark = sg25e.graph_benchmark(trainer, benchmark_batches)
        quality = valid_quality_audit(
            mode,
            trainer,
            quality_batches,
            raw_examples,
            vocabulary,
            device=device,
            epochs=quality_epochs,
        )
    result = {
        "mode": mode,
        "alpha": alpha,
        "capture": dict(trainer.capture_audit),
        "equivalence": equivalence,
        "benchmark": benchmark,
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


def select_alpha(
    sweep: Mapping[str, Mapping[str, Any]],
    valid_baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    control = sweep[_alpha_label(1.0)]["quality"]
    control_nll = control["post_teacher"]["valid"]["nll"]
    eligible = {}
    for label, record in sweep.items():
        quality = record["quality"]
        generation = quality["generation"]
        eligible[label] = bool(
            quality["all_losses_finite"]
            and quality["update_count_passed"]
            and quality["post_teacher"]["valid"]["nll"]
            <= control_nll + 0.10
            and generation["paired_action_sensitivity"] >= 0.50
        )
    candidates = [label for label, passed in eligible.items() if passed]
    if not candidates:  # pragma: no cover - alpha=1 is expected eligible
        candidates = [_alpha_label(1.0)]
    selected = max(
        candidates,
        key=lambda label: (
            sweep[label]["quality"]["generation"]["edit_similarity"],
            sweep[label]["quality"]["generation"]["room_accuracy"] or -1.0,
            sweep[label]["alpha"],
        ),
    )
    selected_generation = sweep[selected]["quality"]["generation"]
    return {
        "eligible": eligible,
        "control_valid_nll": control_nll,
        "selected_label": selected,
        "selected_alpha": sweep[selected]["alpha"],
        "selected_valid_edit_similarity": selected_generation[
            "edit_similarity"
        ],
        "selected_valid_room_accuracy": selected_generation["room_accuracy"],
        "valid_task_edit_threshold": valid_baseline["task_edit_threshold"],
        "passed": selected_generation["edit_similarity"]
        >= valid_baseline["task_edit_threshold"],
    }


def _decision(
    formula: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    primary: Mapping[str, Mapping[str, Any]],
    *,
    quick: bool,
) -> Dict[str, Any]:
    capture_pass = all(
        record["capture"]["shape_count"] == len(sg26a.BUCKET_CAPACITIES)
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
    quality = {mode: record["quality"] for mode, record in primary.items()}
    all_finite = all(
        record["all_losses_finite"]
        and record["update_count_passed"]
        and record["model_test_teacher_calls"] == 0
        and record["model_test_generation_calls"] == 0
        for record in quality.values()
    )
    ann_improvement = all(
        quality[mode]["post_teacher"]["valid"]["nll"]
        <= quality[mode]["pre_teacher"]["valid"]["nll"] - 0.10
        for mode in ("lstm", "transformer")
    )
    best_ann_nll = min(
        quality[mode]["post_teacher"]["valid"]["nll"]
        for mode in ("lstm", "transformer")
    )
    best_ann_edit = max(
        quality[mode]["generation"]["edit_similarity"]
        for mode in ("lstm", "transformer")
    )
    snn = quality["snn_parallel"]
    cross_pass = (
        snn["post_teacher"]["valid"]["nll"] <= best_ann_nll + 0.10
        and snn["generation"]["edit_similarity"] >= best_ann_edit - 0.05
        and snn["generation"]["paired_action_sensitivity"] >= 0.50
    )
    quality_pass = all_finite and ann_improvement and cross_pass
    task_threshold = selection["valid_task_edit_threshold"]
    best_neural_edit = max(
        record["generation"]["edit_similarity"] for record in quality.values()
    )
    task_pass = (
        snn["generation"]["edit_similarity"] >= task_threshold
        and best_neural_edit >= task_threshold
    )
    infrastructure = {
        "formula_gate": bool(formula["passed"]),
        "data_gate": bool(data_audit["passed"]),
        "capture_gate": capture_pass,
        "equivalence_gate": equivalence_pass,
        "test_isolation_gate": all_finite,
    }
    diagnostics = {
        "effective_examples_per_second": eps,
        "effective_target_tokens_per_second": target_tps,
        "per_real_example_p50_ms": p50,
        "selected_alpha": selection["selected_alpha"],
        "selected_valid_edit_similarity": selection[
            "selected_valid_edit_similarity"
        ],
        "valid_task_edit_threshold": task_threshold,
        "best_ann_valid_nll": best_ann_nll,
        "best_ann_valid_edit_similarity": best_ann_edit,
        "best_neural_valid_edit_similarity": best_neural_edit,
        "snn_valid_nll": snn["post_teacher"]["valid"]["nll"],
        "snn_valid_edit_similarity": snn["generation"]["edit_similarity"],
    }
    if quick:
        return {
            **{
                name: "PASS" if passed else "FAIL"
                for name, passed in infrastructure.items()
            },
            "selection_gate": "SMOKE",
            "speed_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "task_validity_gate": "SMOKE",
            "overall": "SMOKE" if all(infrastructure.values()) else "FAIL",
            **diagnostics,
        }
    gates = {
        **infrastructure,
        "selection_gate": bool(selection["passed"]),
        "speed_gate": speed_pass,
        "quality_gate": quality_pass,
        "task_validity_gate": task_pass,
    }
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        **diagnostics,
        "ann_valid_nll_improvement_pass": ann_improvement,
        "snn_cross_architecture_quality_pass": cross_pass,
        "next_route": (
            "sg26c_fresh_corpus_confirmation"
            if all(gates.values())
            else "parallel_prefix_corruption_self_rollin"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG26B requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG26B requires the frozen V100 backend")
    reference = ROOT / "results/e3_scan/e3_sg26a_expanded_raw_language.json"
    reference_sha = _sha256(reference)
    if reference_sha != EXPECTED_SG26A_SHA256:
        raise AssertionError("SG26B SG26A reference hash mismatch")
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
    data_audit = sg26a.expanded_data_audit(raw_examples, vocabulary)
    bucket_audit = sg26a.expanded_bucket_audit(raw_examples["train"])
    valid_baseline = valid_action_majority(raw_examples, vocabulary)
    if not data_audit["passed"] or not bucket_audit["passed"]:
        raise AssertionError("SG26B data/bucket audit failed")
    formula = formula_audit(device=device)

    with sg26a._expanded_backend():
        snn_sweep = {
            _alpha_label(alpha): run_candidate(
                "snn_parallel",
                alpha,
                raw_examples["train"],
                raw_examples,
                vocabulary,
                device=device,
                benchmark_epochs=args.benchmark_epochs,
                quality_epochs=args.quality_epochs,
                equivalence_updates=args.equivalence_updates,
            )
            for alpha in ALPHAS
        }
        selection = select_alpha(snn_sweep, valid_baseline)
        selected_label = selection["selected_label"]
        selected_alpha = selection["selected_alpha"]
        primary = {
            "snn_parallel": snn_sweep[selected_label],
            **{
                mode: run_candidate(
                    mode,
                    selected_alpha,
                    raw_examples["train"],
                    raw_examples,
                    vocabulary,
                    device=device,
                    benchmark_epochs=args.benchmark_epochs,
                    quality_epochs=args.quality_epochs,
                    equivalence_updates=args.equivalence_updates,
                )
                for mode in ("lstm", "transformer")
            },
        }

    decision = _decision(
        formula,
        data_audit,
        selection,
        primary,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG26B length-mass-conserving objective",
        "formal": not args.quick,
        "development_selection_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "alphas": ALPHAS,
            "modes": MODES,
            "batch_size": BATCH_SIZE,
            "bucket_capacities": sg26a.BUCKET_CAPACITIES,
            "benchmark_epochs": args.benchmark_epochs,
            "quality_epochs": args.quality_epochs,
            "equivalence_updates": args.equivalence_updates,
            "valid_generation_tokens": VALID_GENERATION_TOKENS,
            "optimizer": "AdamW(lr=1e-3,betas=.9/.999,wd=.01,fused,capturable)",
            "loss_formula": "sum_i(sum_t CE_it / n_i^alpha) / sum_i(n_i^(1-alpha))",
            "model_test_teacher_calls": 0,
            "model_test_generation_calls": 0,
            "device": "cuda:0",
        },
        "provenance": {
            "sg26a_reference": str(reference),
            "sg26a_reference_sha256": reference_sha,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "manifest": manifest,
            "serial_extension": serial_extension,
            "parallel_extension": parallel_extension,
            "data_audit": data_audit,
            "bucket_audit": bucket_audit,
            "valid_action_majority": valid_baseline,
        },
        "formula_audit": formula,
        "snn_alpha_sweep": snn_sweep,
        "selection": selection,
        "primary_selected_alpha": primary,
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
        default=Path("results/e3_scan/e3_sg26b_length_mass_objective.json"),
    )
    parser.add_argument("--benchmark-epochs", type=int, default=10)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--equivalence-updates", type=int, default=8)
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
