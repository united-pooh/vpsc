"""SG7 paired two-logit batched training for real TextWorld move deltas."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg1_history_generation import build_history_models  # noqa: E402
from experiments.e3_sg4_move_pair_ranking import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    EXPECTED_DATA_SEEDS,
)
from experiments.e3_sg6_move_delta import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_STEP_GROUPS,
    LABELS,
    MODEL_NAMES,
    MoveDeltaExample,
    audit_move_delta_examples,
    build_move_delta_examples,
    evaluate_move_delta,
)
from vpsc.world_model.cores import count_parameters  # noqa: E402
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


def build_paired_batch_schedule(
    examples: Sequence[MoveDeltaExample],
    *,
    epochs: int,
    batch_groups: int,
    seed: int,
) -> Tuple[Tuple[int, ...], ...]:
    groups: Dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.step_group_id].append(index)
    ordered_groups = []
    for group_id in sorted(groups):
        indices = tuple(
            sorted(groups[group_id], key=lambda index: examples[index].candidate_index)
        )
        if len(indices) != 2:
            raise ValueError(f"paired batch group {group_id} must contain two examples")
        if {examples[index].target_ids for index in indices} != {
            (examples[indices[0]].target_ids[0],),
            (examples[indices[1]].target_ids[0],),
        }:
            raise ValueError(f"paired batch group {group_id} has invalid targets")
        if examples[indices[0]].target_ids == examples[indices[1]].target_ids:
            raise ValueError(f"paired batch group {group_id} must contain both labels")
        ordered_groups.append(indices)
    if not ordered_groups:
        raise ValueError("paired batch schedule requires at least one group")
    if len(ordered_groups) % batch_groups:
        raise ValueError(
            "step group count must be divisible by batch_groups for fixed-size timing"
        )

    generator = random.Random(seed)
    schedule = []
    for _epoch in range(epochs):
        shuffled = list(range(len(ordered_groups)))
        generator.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_groups):
            group_indices = shuffled[start : start + batch_groups]
            schedule.append(
                tuple(
                    example_index
                    for group_index in group_indices
                    for example_index in ordered_groups[group_index]
                )
            )
    return tuple(schedule)


def _paired_batch_tensors(
    examples: Sequence[MoveDeltaExample],
    indices: Sequence[int],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = tuple(examples[index] for index in indices)
    lengths = {len(example.prompt_ids) for example in selected}
    if len(lengths) != 1:
        raise ValueError("paired binary batch requires equal prompt lengths")
    input_ids = torch.tensor(
        [example.prompt_ids for example in selected],
        dtype=torch.long,
        device=device,
    )
    query_indices = torch.tensor(
        [input_ids.shape[1] - 1], dtype=torch.long, device=device
    )
    label_to_index = {
        vocabulary.token_id(label): index for index, label in enumerate(LABELS)
    }
    target_indices = torch.tensor(
        [label_to_index[example.target_ids[0]] for example in selected],
        dtype=torch.long,
        device=device,
    )
    return input_ids, query_indices, target_indices


def evaluate_binary_teacher(
    model: Any,
    examples: Sequence[MoveDeltaExample],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    model.eval()
    label_ids = torch.tensor(
        [vocabulary.token_id(label) for label in LABELS],
        dtype=torch.long,
        device=device,
    )
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    relation_loss: Dict[str, float] = defaultdict(float)
    relation_correct: Counter[str] = Counter()
    relation_count: Counter[str] = Counter()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = tuple(range(start, min(start + batch_size, len(examples))))
            input_ids, query_indices, targets = _paired_batch_tensors(
                examples, indices, vocabulary, device=device
            )
            output = model(input_ids, None, detach_state=True)
            logits = output.logits.index_select(1, query_indices)[:, 0]
            binary_logits = logits.index_select(-1, label_ids)
            losses = F.cross_entropy(binary_logits, targets, reduction="none")
            predictions = binary_logits.argmax(dim=-1)
            correct = predictions == targets
            total_loss += float(losses.sum().item())
            total_correct += int(correct.sum().item())
            total_count += len(indices)
            for offset, example_index in enumerate(indices):
                relation = vocabulary.decode(
                    examples[example_index].target_ids
                )[0]
                relation_loss[relation] += float(losses[offset].item())
                relation_correct[relation] += int(correct[offset].item())
                relation_count[relation] += 1
    return {
        "binary_nll": total_loss / total_count,
        "binary_accuracy": total_correct / total_count,
        "example_count": total_count,
        "relations": {
            relation: {
                "binary_nll": relation_loss[relation] / relation_count[relation],
                "binary_accuracy": (
                    relation_correct[relation] / relation_count[relation]
                ),
                "example_count": relation_count[relation],
            }
            for relation in LABELS
        },
    }


def train_paired_binary(
    name: str,
    model: Any,
    examples: Sequence[MoveDeltaExample],
    vocabulary: Vocabulary,
    schedule: Sequence[Sequence[int]],
    *,
    epochs: int,
    batches_per_epoch: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.train(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-3, weight_decay=0.01, fused=True
    )
    label_ids = torch.tensor(
        [vocabulary.token_id(label) for label in LABELS],
        dtype=torch.long,
        device=device,
    )
    batch_timings = []
    example_equivalent_timings = []
    losses = []
    input_tokens = 0
    example_exposures = 0
    epoch_loss = 0.0
    epoch_examples = 0
    epoch_records = []
    started_all = time.perf_counter_ns()
    for update, indices in enumerate(schedule):
        input_ids, query_indices, targets = _paired_batch_tensors(
            examples, indices, vocabulary, device=device
        )
        _sync(device)
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits, _state = tw0._sparse_forward(
            model,
            input_ids,
            query_indices,
            None,
            use_eligibility=name in ("snn_at1", "snn_ra0"),
            detach_state=True,
        )
        binary_logits = logits[:, 0].index_select(-1, label_ids)
        loss = F.cross_entropy(binary_logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite SG7 loss for {name} at update {update + 1}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, foreach=True
        )
        optimizer.step()
        _sync(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        batch_size = len(indices)
        batch_timings.append(elapsed_ms)
        example_equivalent_timings.append(elapsed_ms / batch_size)
        value = float(loss.detach().item())
        losses.append(value)
        input_tokens += input_ids.numel()
        example_exposures += batch_size
        epoch_loss += value * batch_size
        epoch_examples += batch_size
        if (update + 1) % batches_per_epoch == 0:
            epoch_records.append(
                {
                    "epoch": len(epoch_records) + 1,
                    "binary_nll": epoch_loss / epoch_examples,
                    "example_exposures": epoch_examples,
                    "last_gradient_norm": float(gradient_norm),
                }
            )
            epoch_loss = 0.0
            epoch_examples = 0
    elapsed_seconds = (time.perf_counter_ns() - started_all) / 1e9
    warmup_updates = len(batch_timings) // 5
    steady_batch = batch_timings[warmup_updates:]
    steady_example = example_equivalent_timings[warmup_updates:]
    return {
        "epochs": epochs,
        "updates": len(schedule),
        "batches_per_epoch": batches_per_epoch,
        "batch_examples": len(schedule[0]),
        "example_exposures": example_exposures,
        "input_tokens": input_tokens,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_epoch": epoch_records[-1]["binary_nll"],
        "batch_timing": {
            **_sample_summary(steady_batch, 1),
            "warmup_updates_excluded": warmup_updates,
        },
        "example_equivalent_timing": {
            **_sample_summary(steady_example, 1),
            "warmup_updates_excluded": warmup_updates,
        },
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second_total": example_exposures / elapsed_seconds,
        "input_tokens_per_second_total": input_tokens / elapsed_seconds,
        "epoch_records": epoch_records,
    }


def _decision(
    data_audit: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    *,
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "task_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "speed_gate": "SMOKE",
            "response_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg7_paired_binary_batch",
        }

    mean_nll = {
        name: sg0._mean(
            seed["post_binary"][name]["test"]["binary_nll"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_accuracy = {
        name: sg0._mean(
            seed["move_delta"][name]["forced_accuracy"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_step = {
        name: sg0._mean(
            seed["move_delta"][name]["step_consistency"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    ann_improvement = all(
        seed["post_binary"][name]["test"]["binary_nll"]
        <= seed["pre_binary"][name]["test"]["binary_nll"] - 0.10
        for seed in seed_results
        for name in ("lstm", "transformer")
    )
    ra0_improvement = all(
        seed["post_binary"]["snn_ra0"]["test"]["binary_nll"]
        <= seed["pre_binary"]["snn_ra0"]["test"]["binary_nll"] - 0.10
        for seed in seed_results
    )
    best_ann_accuracy = max(mean_accuracy["lstm"], mean_accuracy["transformer"])
    best_ann_step = max(mean_step["lstm"], mean_step["transformer"])
    best_ann_nll = min(mean_nll["lstm"], mean_nll["transformer"])
    task_pass = (
        ann_improvement
        and best_ann_accuracy >= 0.98
        and best_ann_step >= 0.95
    )
    nll_gap_bptt = abs(mean_nll["snn_ra0"] - mean_nll["snn_bptt"])
    nll_gap_at1 = abs(mean_nll["snn_ra0"] - mean_nll["snn_at1"])
    accuracy_gap_bptt = abs(
        mean_accuracy["snn_ra0"] - mean_accuracy["snn_bptt"]
    )
    accuracy_gap_at1 = abs(
        mean_accuracy["snn_ra0"] - mean_accuracy["snn_at1"]
    )
    quality_pass = (
        mean_accuracy["snn_ra0"] >= best_ann_accuracy - 0.02
        and mean_accuracy["snn_ra0"] >= 0.98
        and mean_step["snn_ra0"] >= 0.95
        and mean_nll["snn_ra0"] <= best_ann_nll + 0.05
        and nll_gap_bptt <= 0.05
        and nll_gap_at1 <= 0.05
        and accuracy_gap_bptt <= 0.02
        and accuracy_gap_at1 <= 0.02
    )
    mean_example_p50 = {
        name: sg0._mean(
            seed["training"][name]["example_equivalent_timing"]["p50_ms"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_elapsed = {
        name: sg0._mean(
            seed["training"][name]["elapsed_seconds"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    at1_speedup = mean_example_p50["snn_at1"] / mean_example_p50["snn_ra0"]
    bptt_speedup = (
        mean_example_p50["snn_bptt"] / mean_example_p50["snn_ra0"]
    )
    speed_pass = (
        at1_speedup >= 1.25
        and bptt_speedup >= 1.25
        and mean_example_p50["snn_ra0"] <= mean_example_p50["lstm"]
        and mean_elapsed["snn_ra0"] <= mean_elapsed["lstm"]
    )
    response_pass = all(
        seed["move_delta"]["snn_ra0"]["timing"]["p50_ms"]
        <= seed["move_delta"]["lstm"]["timing"]["p50_ms"]
        and seed["move_delta"]["snn_ra0"]["timing"]["p95_ms"]
        <= seed["move_delta"]["lstm"]["timing"]["p95_ms"]
        for seed in seed_results
    )
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "task_gate": task_pass,
        "quality_gate": quality_pass,
        "speed_gate": speed_pass,
        "response_gate": response_pass,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    if overall == "PASS":
        next_route = "sg8_multichannel_delta_closed_loop"
    elif task_pass and quality_pass and (not speed_pass or not response_pass):
        next_route = "sg8_native_fused_batch_and_stream_scan"
    elif task_pass and not quality_pass:
        next_route = "sg8_bilinear_spike_binding_and_closed_form_readout"
    else:
        next_route = "sg8_bilinear_relation_task_control"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "mean_test_binary_nll": mean_nll,
        "mean_forced_accuracy": mean_accuracy,
        "mean_step_consistency": mean_step,
        "ann_every_seed_binary_nll_improvement_0_10": ann_improvement,
        "ra0_every_seed_binary_nll_improvement_0_10": ra0_improvement,
        "best_ann_nll": best_ann_nll,
        "best_ann_accuracy": best_ann_accuracy,
        "best_ann_step_consistency": best_ann_step,
        "ra0_vs_bptt_nll_gap": nll_gap_bptt,
        "ra0_vs_at1_nll_gap": nll_gap_at1,
        "ra0_vs_bptt_accuracy_gap": accuracy_gap_bptt,
        "ra0_vs_at1_accuracy_gap": accuracy_gap_at1,
        "mean_training_example_equivalent_p50_ms": mean_example_p50,
        "mean_training_elapsed_seconds": mean_elapsed,
        "ra0_vs_at1_training_speedup": at1_speedup,
        "ra0_vs_bptt_training_speedup": bptt_speedup,
        "next_route": next_route,
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
    corpus_root = args.corpus_dir.expanduser().resolve()
    expected_data_seeds = {
        "train": tuple(args.expected_train_seeds),
        "valid": tuple(args.expected_valid_seeds),
        "test": tuple(args.expected_test_seeds),
    }
    manifest = tw0._manifest_provenance(
        corpus_root, expected_seeds_by_split=expected_data_seeds
    )
    corpus = load_event_corpus(corpus_root)
    examples, vocabulary = build_move_delta_examples(corpus_root, corpus)
    expected_counts = dict(zip(SPLITS, args.expected_counts))
    expected_step_groups = dict(zip(SPLITS, args.expected_step_groups))
    data_audit = audit_move_delta_examples(
        examples,
        vocabulary,
        expected_counts=expected_counts,
        expected_step_groups=expected_step_groups,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG7 move-delta data audit failed; refusing experiment")
    if expected_step_groups["train"] % args.batch_groups:
        raise AssertionError("SG7 train step groups must divide batch_groups")
    batches_per_epoch = expected_step_groups["train"] // args.batch_groups

    seed_results = []
    for seed in args.seeds:
        models = build_history_models(
            9_900_000 + 100 * seed,
            vocabulary,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_heads=args.num_heads,
            device=device,
        )
        parameter_counts = {
            name: {
                "total": count_parameters(model),
                "core": count_parameters(model.core),
            }
            for name, model in models.items()
        }
        totals = tuple(record["total"] for record in parameter_counts.values())
        parameter_spread = (max(totals) - min(totals)) / sg0._mean(totals)
        if parameter_spread > 0.03:
            raise AssertionError(f"SG7 parameter spread failed: {parameter_counts}")
        pre_binary = {
            name: {
                split: evaluate_binary_teacher(
                    model, examples[split], vocabulary, device=device
                )
                for split in ("valid", "test")
            }
            for name, model in models.items()
        }
        schedule = build_paired_batch_schedule(
            examples["train"],
            epochs=args.epochs,
            batch_groups=args.batch_groups,
            seed=9_901_000 + seed,
        )
        training = {
            name: train_paired_binary(
                name,
                model,
                examples["train"],
                vocabulary,
                schedule,
                epochs=args.epochs,
                batches_per_epoch=batches_per_epoch,
                device=device,
            )
            for name, model in models.items()
        }
        post_binary = {
            name: {
                split: evaluate_binary_teacher(
                    model, examples[split], vocabulary, device=device
                )
                for split in ("train", "valid", "test")
            }
            for name, model in models.items()
        }
        move_delta = {
            name: evaluate_move_delta(
                model,
                examples["test"],
                vocabulary,
                device=device,
                include_records=True,
            )
            for name, model in models.items()
        }
        seed_results.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_relative_spread": parameter_spread,
                "pre_binary": pre_binary,
                "training": training,
                "post_binary": post_binary,
                "move_delta": move_delta,
            }
        )
    decision = _decision(data_audit, seed_results, quick=args.quick)
    return {
        "schema_version": 1,
        "experiment": "E3-SG7 paired binary batched TextWorld move-delta",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "established optimization direction",
            "statement": (
                "Paired two-logit batches reduce irrelevant vocabulary gradients "
                "and optimizer-step overhead for sparse world-state channels."
            ),
            "what_if": (
                "What if low-variance positive/negative event batches let SNN "
                "eligibility dynamics match ANN quality with fewer updates?"
            ),
        },
        "configuration": {
            "corpus_dir": str(corpus_root),
            "expected_data_seeds": expected_data_seeds,
            "expected_counts": expected_counts,
            "expected_step_groups": expected_step_groups,
            "epochs": args.epochs,
            "seeds": tuple(args.seeds),
            "threads": args.threads if device.type == "cpu" else None,
            "d_model": args.d_model,
            "state_dim": args.state_dim,
            "num_heads": args.num_heads,
            "batch_groups": args.batch_groups,
            "batch_examples": args.batch_groups * 2,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_updates_per_model": batches_per_epoch * args.epochs,
            "example_exposures_per_model": len(examples["train"]) * args.epochs,
            "loss": "cross_entropy_over_two_relation_logits",
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "optimizer_fused": True,
            "gradient_foreach": True,
            "max_prompt_tokens": args.max_prompt_tokens,
            "state_reset_per_batch": True,
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "corpus_provenance": tw0._corpus_provenance(corpus),
            "vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
                "source_split": "train",
                "source_fields": ("previous_move", "candidate_move", "label"),
                "tokenizer": corpus.tokenizer.metadata(),
            },
            "audit": data_audit,
        },
        "seeds": seed_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg7_paired_binary_batch.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=32)
    parser.add_argument(
        "--expected-counts",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_COUNTS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-step-groups",
        nargs=3,
        type=int,
        default=tuple(EXPECTED_STEP_GROUPS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-train-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["train"],
    )
    parser.add_argument(
        "--expected-valid-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["valid"],
    )
    parser.add_argument(
        "--expected-test-seeds",
        nargs="+",
        type=int,
        default=EXPECTED_DATA_SEEDS["test"],
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.epochs,
        args.threads,
        args.d_model,
        args.state_dim,
        args.num_heads,
        args.batch_groups,
        args.max_prompt_tokens,
        *args.expected_counts,
        *args.expected_step_groups,
        *args.expected_train_seeds,
        *args.expected_valid_seeds,
        *args.expected_test_seeds,
    ) <= 0:
        parser.error("all numeric experiment controls must be positive")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
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
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
