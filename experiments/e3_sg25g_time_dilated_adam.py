"""SG25G time-dilated AdamW on SG25F B16 parallel training graphs."""

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
from typing import Any, Callable, Dict, Iterator, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments import e3_sg25e_bucketed_batch_graph as sg25e  # noqa: E402
from experiments import e3_sg25f_parallel_affine_graph as sg25f  # noqa: E402


PRIMARY_MODES = ("snn_parallel", "lstm", "transformer")
BATCH_SIZE = 16
BASE_LR = 1e-3
BASE_BETA1 = 0.9
BASE_BETA2 = 0.999
BASE_WEIGHT_DECAY = 0.01
EFFECTIVE_TIME = 40.0 / 3.0
DILATED_LR = BASE_LR * EFFECTIVE_TIME
DILATED_BETA1 = BASE_BETA1**EFFECTIVE_TIME
DILATED_BETA2 = BASE_BETA2**EFFECTIVE_TIME
DILATED_WEIGHT_DECAY = (
    1.0 - (1.0 - BASE_LR * BASE_WEIGHT_DECAY) ** EFFECTIVE_TIME
) / DILATED_LR
EXPECTED_SG25F_SHA256 = (
    "e30fc0e557399a96f0beebcf956fb3effca2a36ce97f008745da1fad40ad22bc"
)
SG25F_PARALLEL_EXAMPLES_PER_SECOND = 20133.046722635696
SG25C_SEED0_NLL = 2.6957537054423932
SG25C_SEED0_EDIT = 0.6513394872257832


OptimizerFactory = Callable[[Any], torch.optim.Optimizer]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trainable(model: Any) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def optimizer_factory(kind: str) -> OptimizerFactory:
    if kind == "time_dilated":
        learning_rate = DILATED_LR
        betas = (DILATED_BETA1, DILATED_BETA2)
        weight_decay = DILATED_WEIGHT_DECAY
    elif kind == "lr_only":
        learning_rate = DILATED_LR
        betas = (BASE_BETA1, BASE_BETA2)
        weight_decay = BASE_WEIGHT_DECAY
    elif kind == "default":
        learning_rate = BASE_LR
        betas = (BASE_BETA1, BASE_BETA2)
        weight_decay = BASE_WEIGHT_DECAY
    else:  # pragma: no cover
        raise ValueError(f"unknown optimizer kind: {kind}")

    def factory(model: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            _trainable(model),
            lr=learning_rate,
            betas=betas,
            eps=1e-8,
            weight_decay=weight_decay,
            fused=True,
            capturable=True,
        )

    return factory


@contextmanager
def _patched_optimizer(factory: OptimizerFactory) -> Iterator[None]:
    original = sg25f._optimizer
    sg25f._optimizer = factory
    try:
        yield
    finally:
        sg25f._optimizer = original


def _simulate_constant_gradient(
    *,
    steps: int,
    parameter: float,
    gradient: float,
    weight_decay: float,
) -> float:
    first_moment = 0.0
    second_moment = 0.0
    value = parameter
    for step in range(1, steps + 1):
        first_moment = (
            BASE_BETA1 * first_moment + (1.0 - BASE_BETA1) * gradient
        )
        second_moment = (
            BASE_BETA2 * second_moment
            + (1.0 - BASE_BETA2) * gradient * gradient
        )
        value *= 1.0 - BASE_LR * weight_decay
        corrected_first = first_moment / (1.0 - BASE_BETA1**step)
        corrected_second = second_moment / (1.0 - BASE_BETA2**step)
        value -= BASE_LR * corrected_first / (math.sqrt(corrected_second) + 1e-8)
    return value


def _compressed_constant_gradient(
    *,
    steps: int,
    parameter: float,
    gradient: float,
    weight_decay: float,
) -> float:
    learning_rate = BASE_LR * steps
    beta1 = BASE_BETA1**steps
    beta2 = BASE_BETA2**steps
    dilated_decay = (
        1.0 - (1.0 - BASE_LR * weight_decay) ** steps
    ) / learning_rate
    first_moment = (1.0 - beta1) * gradient
    second_moment = (1.0 - beta2) * gradient * gradient
    corrected_first = first_moment / (1.0 - beta1)
    corrected_second = second_moment / (1.0 - beta2)
    value = parameter * (1.0 - learning_rate * dilated_decay)
    value -= learning_rate * corrected_first / (
        math.sqrt(corrected_second) + 1e-8
    )
    return value


def formula_audit() -> Dict[str, Any]:
    steps = 16
    parameter = 1.25
    gradient = 0.3
    no_decay_sequential = _simulate_constant_gradient(
        steps=steps,
        parameter=parameter,
        gradient=gradient,
        weight_decay=0.0,
    )
    no_decay_compressed = _compressed_constant_gradient(
        steps=steps,
        parameter=parameter,
        gradient=gradient,
        weight_decay=0.0,
    )
    decay_sequential = _simulate_constant_gradient(
        steps=steps,
        parameter=parameter,
        gradient=gradient,
        weight_decay=BASE_WEIGHT_DECAY,
    )
    decay_compressed = _compressed_constant_gradient(
        steps=steps,
        parameter=parameter,
        gradient=gradient,
        weight_decay=BASE_WEIGHT_DECAY,
    )
    no_decay_error = abs(no_decay_sequential - no_decay_compressed)
    decay_error = abs(decay_sequential - decay_compressed)
    decay_relative_error = decay_error / abs(decay_sequential)
    return {
        "integer_control_steps": steps,
        "parameter_initial": parameter,
        "constant_gradient": gradient,
        "no_decay": {
            "sequential": no_decay_sequential,
            "compressed": no_decay_compressed,
            "absolute_error": no_decay_error,
        },
        "with_decay": {
            "sequential": decay_sequential,
            "compressed": decay_compressed,
            "absolute_error": decay_error,
            "relative_error": decay_relative_error,
        },
        "fractional_primary": {
            "effective_time": EFFECTIVE_TIME,
            "learning_rate": DILATED_LR,
            "beta1": DILATED_BETA1,
            "beta2": DILATED_BETA2,
            "weight_decay": DILATED_WEIGHT_DECAY,
        },
        "passed": no_decay_error <= 1e-7 and decay_relative_error <= 2e-4,
    }


def _quality_from_trainer(
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
    if mode.startswith("snn_"):
        basic_pass = (
            all_losses_finite
            and post["test"]["nll"] <= SG25C_SEED0_NLL + 0.10
            and generation["edit_similarity"] >= SG25C_SEED0_EDIT - 0.05
            and generation["paired_action_sensitivity"] >= 0.50
        )
    else:
        basic_pass = (
            all_losses_finite
            and post["test"]["nll"] <= pre["test"]["nll"] - 0.10
        )
    warmup = len(samples) // 5
    return {
        "mode": mode,
        "epochs": epochs,
        "updates": len(batches),
        "all_losses_finite": all_losses_finite,
        "train_wall_seconds": train_wall_seconds,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "timing": _sample_summary(samples[warmup:], 1),
        "pre_teacher": pre,
        "post_teacher": post,
        "generation": generation,
        "basic_passed": basic_pass,
    }


def run_primary_mode(
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
    benchmark_batches = sg25e.build_batch_schedule(
        train_examples,
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=benchmark_epochs,
        seed=25_950_000,
        device=device,
    )
    quality_batches = sg25e.build_batch_schedule(
        train_examples,
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=quality_epochs,
        seed=25_960_000,
        device=device,
    )
    factory = optimizer_factory("time_dilated")
    gc.collect()
    torch.cuda.empty_cache()
    clean_allocated = torch.cuda.memory_allocated(device)
    clean_reserved = torch.cuda.memory_reserved(device)
    with _patched_optimizer(factory):
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
            quality_batches,
            vocabulary,
            device=device,
            updates=equivalence_updates,
        )
        benchmark = sg25e.graph_benchmark(trainer, benchmark_batches)
        profiler = sg25f.graph_profile(
            trainer, sg25e._fullest_batch(benchmark_batches)
        )
        quality = _quality_from_trainer(
            mode,
            trainer,
            quality_batches,
            raw_examples,
            vocabulary,
            device=device,
            epochs=quality_epochs,
        )
    result = {
        "optimizer_kind": "time_dilated",
        "equivalence": equivalence,
        "benchmark": benchmark,
        "profiler": profiler,
        "capture": dict(trainer.capture_audit),
        "quality": quality,
    }
    del trainer, model, benchmark_batches, quality_batches
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_snn_control(
    kind: str,
    train_examples: Sequence[Any],
    raw_examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    batches = sg25e.build_batch_schedule(
        train_examples,
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        seed=25_960_000,
        device=device,
    )
    factory = optimizer_factory(kind)
    with _patched_optimizer(factory):
        model = sg25f._build_mode_model(
            "snn_parallel", vocabulary, device=device
        )
        trainer = sg25f.GraphTrainer(
            "snn_parallel", model, batches, device=device
        )
        quality = _quality_from_trainer(
            "snn_parallel",
            trainer,
            batches,
            raw_examples,
            vocabulary,
            device=device,
            epochs=epochs,
        )
    result = {
        "optimizer_kind": kind,
        "capture": dict(trainer.capture_audit),
        "quality": quality,
    }
    del trainer, model, batches
    gc.collect()
    torch.cuda.empty_cache()
    return result


def precondition_graph_allocator(
    train_examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    batches = sg25e.build_batch_schedule(
        train_examples,
        vocabulary,
        batch_size=BATCH_SIZE,
        epochs=2,
        seed=25_949_000,
        device=device,
    )
    factory = optimizer_factory("default")
    with _patched_optimizer(factory):
        model = sg25f._build_mode_model(
            "snn_parallel", vocabulary, device=device
        )
        trainer = sg25f.GraphTrainer(
            "snn_parallel", model, batches, device=device
        )
    audit = dict(trainer.capture_audit)
    del trainer, model, batches
    gc.collect()
    torch.cuda.empty_cache()
    return audit


def _decision(
    formula: Mapping[str, Any],
    primary: Mapping[str, Mapping[str, Any]],
    canonical_parallel: Mapping[str, Any],
    *,
    quick: bool,
) -> Dict[str, Any]:
    equivalence_pass = all(
        record["equivalence"]["passed"] for record in primary.values()
    )
    capture_pass = all(
        record["capture"]["shape_count"] == 3 for record in primary.values()
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
        eps["snn_parallel"] >= 0.90 * SG25F_PARALLEL_EXAMPLES_PER_SECOND
        and eps["snn_parallel"] >= eps["lstm"]
        and eps["snn_parallel"] >= eps["transformer"]
        and target_tps["snn_parallel"] >= target_tps["lstm"]
        and target_tps["snn_parallel"] >= target_tps["transformer"]
        and p50["snn_parallel"] <= p50["lstm"]
        and p50["snn_parallel"] <= p50["transformer"]
    )
    snn_record = primary["snn_parallel"]
    event_pass = (
        snn_record["profiler"]["host_launch_and_copy_api_count"]
        <= canonical_parallel["profiler"]["host_launch_and_copy_api_count"]
    )
    allocated_ratio = (
        snn_record["capture"]["allocated_delta_bytes"]
        / canonical_parallel["capture"]["allocated_delta_bytes"]
    )
    peak_ratio = (
        snn_record["capture"]["peak_additional_allocated_bytes"]
        / canonical_parallel["capture"]["peak_additional_allocated_bytes"]
    )
    memory_pass = allocated_ratio <= 1.10 and peak_ratio <= 1.10
    if quick:
        gates = {
            "formula_gate": bool(formula["passed"]),
            "capture_gate": capture_pass,
            "equivalence_gate": equivalence_pass,
            "event_gate": event_pass,
            "memory_gate": memory_pass,
        }
        return {
            **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
            "speed_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "overall": "SMOKE" if all(gates.values()) else "FAIL",
            "effective_examples_per_second": eps,
            "per_real_example_p50_ms": p50,
            "allocated_ratio": allocated_ratio,
            "peak_ratio": peak_ratio,
        }
    basic_quality_pass = all(
        record["quality"]["basic_passed"] for record in primary.values()
    )
    snn_quality = primary["snn_parallel"]["quality"]
    best_ann_nll = min(
        primary[name]["quality"]["post_teacher"]["test"]["nll"]
        for name in ("lstm", "transformer")
    )
    best_ann_edit = max(
        primary[name]["quality"]["generation"]["edit_similarity"]
        for name in ("lstm", "transformer")
    )
    cross_quality_pass = (
        snn_quality["post_teacher"]["test"]["nll"] <= best_ann_nll + 0.10
        and snn_quality["generation"]["edit_similarity"] >= best_ann_edit - 0.05
    )
    quality_pass = basic_quality_pass and cross_quality_pass
    gates = {
        "formula_gate": bool(formula["passed"]),
        "capture_gate": capture_pass,
        "equivalence_gate": equivalence_pass,
        "speed_gate": speed_pass,
        "event_gate": event_pass,
        "memory_gate": memory_pass,
        "quality_gate": quality_pass,
    }
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "effective_examples_per_second": eps,
        "effective_target_tokens_per_second": target_tps,
        "per_real_example_p50_ms": p50,
        "snn_to_sg25f_speed_ratio": eps["snn_parallel"]
        / SG25F_PARALLEL_EXAMPLES_PER_SECOND,
        "allocated_ratio": allocated_ratio,
        "peak_ratio": peak_ratio,
        "basic_quality_pass": basic_quality_pass,
        "cross_architecture_quality_pass": cross_quality_pass,
        "next_route": (
            "expanded_real_corpus" if all(gates.values()) else "virtual_per_example_adam"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG25G requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG25G requires the frozen V100 backend")
    reference = ROOT / "results/e3_scan/e3_sg25f_parallel_affine_graph.json"
    reference_sha = _sha256(reference)
    if reference_sha != EXPECTED_SG25F_SHA256:
        raise AssertionError("SG25G SG25F reference hash mismatch")
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    canonical_parallel = reference_payload["speed_records"]["snn_parallel"]
    _, extension = sg25f.load_parallel_extension()
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = sg0.load_event_corpus(corpus_root)
    raw_examples, vocabulary = sg0.build_counterfactual_examples(
        corpus_root, corpus
    )
    data_audit = sg0.audit_examples(raw_examples, vocabulary)
    bucket_audit = sg25e._bucket_audit(raw_examples["train"])
    if not data_audit["passed"] or not bucket_audit["passed"]:
        raise AssertionError("SG25G data/bucket audit failed")

    formula = formula_audit()
    allocator_preconditioning = precondition_graph_allocator(
        raw_examples["train"], vocabulary, device=device
    )
    primary = {
        mode: run_primary_mode(
            mode,
            raw_examples["train"],
            raw_examples,
            vocabulary,
            device=device,
            benchmark_epochs=args.benchmark_epochs,
            quality_epochs=args.quality_epochs,
            equivalence_updates=args.equivalence_updates,
        )
        for mode in PRIMARY_MODES
    }
    controls = None
    if not args.quick:
        controls = {
            "lr_only": run_snn_control(
                "lr_only",
                raw_examples["train"],
                raw_examples,
                vocabulary,
                device=device,
                epochs=args.quality_epochs,
            ),
            "step_matched": run_snn_control(
                "default",
                raw_examples["train"],
                raw_examples,
                vocabulary,
                device=device,
                epochs=args.step_matched_epochs,
            ),
        }
    decision = _decision(
        formula,
        primary,
        canonical_parallel,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG25G time-dilated AdamW",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "primary_modes": PRIMARY_MODES,
            "batch_size": BATCH_SIZE,
            "bucket_capacities": sg25e.BUCKET_CAPACITIES,
            "benchmark_epochs": args.benchmark_epochs,
            "quality_epochs": args.quality_epochs,
            "equivalence_updates": args.equivalence_updates,
            "step_matched_epochs": args.step_matched_epochs,
            "base_optimizer": {
                "lr": BASE_LR,
                "betas": (BASE_BETA1, BASE_BETA2),
                "weight_decay": BASE_WEIGHT_DECAY,
            },
            "time_dilated_optimizer": {
                "effective_time": EFFECTIVE_TIME,
                "lr": DILATED_LR,
                "betas": (DILATED_BETA1, DILATED_BETA2),
                "weight_decay": DILATED_WEIGHT_DECAY,
                "eps": 1e-8,
                "fused": True,
                "capturable": True,
            },
            "loss": "mean(valid-token CE per example), then mean(real examples)",
            "device": "cuda:0",
        },
        "provenance": {
            "sg25f_reference": str(reference),
            "sg25f_reference_sha256": reference_sha,
            "canonical_parallel": canonical_parallel,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "parallel_extension": extension,
            "data_audit": data_audit,
            "bucket_audit": bucket_audit,
        },
        "formula_audit": formula,
        "allocator_preconditioning": allocator_preconditioning,
        "primary": primary,
        "controls": controls,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg25g_time_dilated_adam.json"),
    )
    parser.add_argument("--benchmark-epochs", type=int, default=10)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--equivalence-updates", type=int, default=20)
    parser.add_argument("--step-matched-epochs", type=int, default=1334)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(
        args.benchmark_epochs,
        args.quality_epochs,
        args.equivalence_updates,
        args.step_matched_epochs,
    ) <= 0:
        parser.error("all counts must be positive")
    if args.quick:
        args.benchmark_epochs = 1
        args.quality_epochs = min(args.quality_epochs, 2)
        args.equivalence_updates = min(args.equivalence_updates, 5)
        args.step_matched_epochs = min(args.step_matched_epochs, 4)
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
