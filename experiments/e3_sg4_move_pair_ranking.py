"""SG4 hard move-vs-move TextWorld counterfactual outcome ranking."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _percentile,
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg1_history_generation import (  # noqa: E402
    build_history_models,
)
from vpsc.world_model.cores import count_parameters, state_nbytes  # noqa: E402
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


LABEL_A = "<option_a>"
LABEL_B = "<option_b>"
LABELS = (LABEL_A, LABEL_B)
MODEL_NAMES = sg0.MODEL_NAMES
DEFAULT_CORPUS_DIR = Path("results/e3_scan/textworld_sg2_l5")
EXPECTED_COUNTS = {"train": 384, "valid": 48, "test": 48}
EXPECTED_GROUPS = {"train": 192, "valid": 24, "test": 24}
EXPECTED_DATA_SEEDS = {
    "train": tuple(range(20260801, 20260833)),
    "valid": tuple(range(20260833, 20260837)),
    "test": tuple(range(20260837, 20260841)),
}


@dataclass(frozen=True)
class _RawRankingExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    candidate_index: int
    option_order: Tuple[int, int]
    group_id: str
    action: str
    option_a_text: str
    option_b_text: str
    prompt_tokens: Tuple[str, ...]
    target_token: str


@dataclass(frozen=True)
class RankingExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    candidate_index: int
    option_order: Tuple[int, int]
    group_id: str
    action: str
    action_type: str
    option_a_text: str
    option_b_text: str
    prompt_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    prompt_unknowns: int

    @property
    def example_id(self) -> str:
        return (
            f"{self.split}:{self.game_seed}:{self.step_index}:"
            f"{self.candidate_index}:{self.option_order[0]}"
        )

    @property
    def input_length(self) -> int:
        return len(self.prompt_ids)


def _ranking_prompt_tokens(
    corpus: Any,
    observations: Sequence[str],
    factual_actions: Sequence[str],
    step_index: int,
    candidate_action: str,
    option_a: str,
    option_b: str,
) -> Tuple[str, ...]:
    parts = [("<bos>",), corpus.tokenizer.tokenize("trajectory:")]
    for prior_index in range(step_index):
        parts.extend(
            (
                corpus.tokenizer.tokenize("observation:"),
                corpus.tokenizer.tokenize(observations[prior_index]),
                ("<eos>",),
                corpus.tokenizer.tokenize("action:"),
                corpus.tokenizer.tokenize(factual_actions[prior_index]),
                ("<eos>",),
            )
        )
    parts.extend(
        (
            corpus.tokenizer.tokenize("current observation:"),
            corpus.tokenizer.tokenize(observations[step_index]),
            ("<eos>",),
            corpus.tokenizer.tokenize("candidate action:"),
            corpus.tokenizer.tokenize(candidate_action),
            ("<eos>",),
            corpus.tokenizer.tokenize("option a:"),
            corpus.tokenizer.tokenize(option_a),
            ("<eos>",),
            corpus.tokenizer.tokenize("option b:"),
            corpus.tokenizer.tokenize(option_b),
            ("<eos>",),
            corpus.tokenizer.tokenize("answer:"),
        )
    )
    return tuple(token for part in parts for token in part)


def build_ranking_examples(
    corpus_root: Path,
    corpus: Any,
) -> Tuple[Dict[str, Tuple[RankingExample, ...]], Vocabulary]:
    raw_by_split: Dict[str, Tuple[_RawRankingExample, ...]] = {}
    for split in SPLITS:
        path = corpus_root / split / "episodes.jsonl"
        records = []
        for episode_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            episode = json.loads(line)
            if episode.get("split") != split:
                raise ValueError(f"ranking episode split mismatch: {path}")
            steps = episode["steps"]
            observations = tuple(
                sg0.normalize_textworld_observation(step["observation"])
                for step in steps
            )
            factual_actions = tuple(str(step["action"]) for step in steps)
            for step_index, step in enumerate(steps):
                factual_action = factual_actions[step_index]
                if sg0._action_type(factual_action) != "move":
                    continue
                move_counterfactuals = tuple(
                    counterfactual
                    for counterfactual in step["counterfactuals"]
                    if sg0._action_type(str(counterfactual["action"])) == "move"
                )
                if not move_counterfactuals:
                    continue
                if len(move_counterfactuals) != 1:
                    raise ValueError(
                        f"expected one hard move counterfactual in {split} "
                        f"seed {episode['seed']} step {step_index}"
                    )
                counterfactual = move_counterfactuals[0]
                actions = (factual_action, str(counterfactual["action"]))
                outcomes = (
                    sg0.normalize_textworld_observation(step["next_obs"]),
                    sg0.normalize_textworld_observation(counterfactual["next_obs"]),
                )
                if outcomes[0] == outcomes[1]:
                    raise ValueError("hard move options must have different outcomes")
                for candidate_index, candidate_action in enumerate(actions):
                    group_id = (
                        f"{split}:{episode['seed']}:{step_index}:"
                        f"{candidate_index}"
                    )
                    for swap in (0, 1):
                        order = (0, 1) if swap == 0 else (1, 0)
                        option_a = outcomes[order[0]]
                        option_b = outcomes[order[1]]
                        target = LABEL_A if order[0] == candidate_index else LABEL_B
                        records.append(
                            _RawRankingExample(
                                split=split,
                                episode_index=episode_index,
                                game_seed=int(episode["seed"]),
                                step_index=step_index,
                                candidate_index=candidate_index,
                                option_order=order,
                                group_id=group_id,
                                action=candidate_action,
                                option_a_text=option_a,
                                option_b_text=option_b,
                                prompt_tokens=_ranking_prompt_tokens(
                                    corpus,
                                    observations,
                                    factual_actions,
                                    step_index,
                                    candidate_action,
                                    option_a,
                                    option_b,
                                ),
                                target_token=target,
                            )
                        )
        raw_by_split[split] = tuple(records)

    vocabulary = Vocabulary.build(
        token
        for record in raw_by_split["train"]
        for token in record.prompt_tokens + (record.target_token,)
    )
    examples: Dict[str, Tuple[RankingExample, ...]] = {}
    for split in SPLITS:
        encoded = []
        for raw in raw_by_split[split]:
            prompt_ids = vocabulary.encode(raw.prompt_tokens)
            target_ids = vocabulary.encode((raw.target_token,))
            encoded.append(
                RankingExample(
                    split=raw.split,
                    episode_index=raw.episode_index,
                    game_seed=raw.game_seed,
                    step_index=raw.step_index,
                    candidate_index=raw.candidate_index,
                    option_order=raw.option_order,
                    group_id=raw.group_id,
                    action=raw.action,
                    action_type="move",
                    option_a_text=raw.option_a_text,
                    option_b_text=raw.option_b_text,
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    prompt_unknowns=sum(
                        value == vocabulary.unk_id for value in prompt_ids
                    ),
                )
            )
        examples[split] = tuple(encoded)
    return examples, vocabulary


def audit_ranking_examples(
    examples: Mapping[str, Sequence[RankingExample]],
    vocabulary: Vocabulary,
    *,
    expected_counts: Mapping[str, int],
    expected_groups: Mapping[str, int],
    max_prompt_tokens: int,
) -> Dict[str, Any]:
    split_records = {}
    all_valid = True
    for split, values in examples.items():
        groups: Dict[str, list[RankingExample]] = defaultdict(list)
        for example in values:
            groups[example.group_id].append(example)
        group_valid = all(
            len(group) == 2
            and {example.option_order for example in group} == {(0, 1), (1, 0)}
            and {example.target_ids for example in group}
            == {
                (vocabulary.token_id(LABEL_A),),
                (vocabulary.token_id(LABEL_B),),
            }
            for group in groups.values()
        )
        label_counts = Counter(
            vocabulary.decode(example.target_ids)[0] for example in values
        )
        prompt_tokens = sum(len(example.prompt_ids) for example in values)
        prompt_unknowns = sum(example.prompt_unknowns for example in values)
        duplicate_options = sum(
            example.option_a_text == example.option_b_text for example in values
        )
        record = {
            "example_count": len(values),
            "semantic_group_count": len(groups),
            "game_count": len({example.game_seed for example in values}),
            "label_counts": dict(sorted(label_counts.items())),
            "prompt_length": sg0._length_summary(
                [len(example.prompt_ids) for example in values]
            ),
            "prompt_token_count": prompt_tokens,
            "prompt_unknown_count": prompt_unknowns,
            "prompt_unknown_ratio": prompt_unknowns / prompt_tokens,
            "duplicate_option_count": duplicate_options,
            "swap_groups_valid": group_valid,
        }
        record_valid = (
            len(values) == expected_counts[split]
            and len(groups) == expected_groups[split]
            and label_counts[LABEL_A] == label_counts[LABEL_B] == len(values) // 2
            and max(len(example.prompt_ids) for example in values)
            <= max_prompt_tokens
            and (split == "train" or record["prompt_unknown_ratio"] < 0.10)
            and duplicate_options == 0
            and group_valid
            and all(example.action_type == "move" for example in values)
        )
        record["passed"] = record_valid
        split_records[split] = record
        all_valid = all_valid and record_valid
    return {
        "task": "hard factual-move versus counterfactual-move outcome ranking",
        "label_tokens": LABELS,
        "expected_counts": dict(expected_counts),
        "expected_groups": dict(expected_groups),
        "max_prompt_tokens": max_prompt_tokens,
        "position_baseline_accuracy": 0.50,
        "random_baseline_accuracy": 0.50,
        "inverse_history_oracle_accuracy": 1.0,
        "splits": split_records,
        "passed": all_valid,
    }


def evaluate_ranking(
    model: Any,
    examples: Sequence[RankingExample],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    include_records: bool,
) -> Dict[str, Any]:
    model.eval()
    label_ids = torch.tensor(
        [vocabulary.token_id(LABEL_A), vocabulary.token_id(LABEL_B)],
        dtype=torch.long,
        device=device,
    )
    forced_correct = 0
    open_correct = 0
    margins = []
    timings = []
    state_sizes = []
    semantic_predictions: Dict[str, list[int]] = defaultdict(list)
    records = []
    with torch.inference_mode():
        for example in examples:
            prompt = torch.tensor(
                example.prompt_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
            _sync(device)
            started = time.perf_counter_ns()
            output = model(prompt, None, detach_state=True)
            logits = output.logits[0, -1]
            logits.sum().item()
            _sync(device)
            timings.append((time.perf_counter_ns() - started) / 1e6)
            state_sizes.append(state_nbytes(output.state))
            pair_logits = logits.index_select(0, label_ids)
            forced_index = int(pair_logits.argmax().item())
            forced_id = int(label_ids[forced_index].item())
            target_id = example.target_ids[0]
            target_index = 0 if target_id == int(label_ids[0].item()) else 1
            forced_correct += int(forced_id == target_id)
            open_correct += int(int(logits.argmax().item()) == target_id)
            margins.append(
                float((pair_logits[target_index] - pair_logits[1 - target_index]).item())
            )
            semantic_prediction = example.option_order[forced_index]
            semantic_predictions[example.group_id].append(semantic_prediction)
            if include_records:
                records.append(
                    {
                        "example_id": example.example_id,
                        "group_id": example.group_id,
                        "candidate_index": example.candidate_index,
                        "action": example.action,
                        "option_order": example.option_order,
                        "target_label": vocabulary.decode(example.target_ids)[0],
                        "forced_label": LABELS[forced_index],
                        "semantic_prediction": semantic_prediction,
                        "correct": forced_id == target_id,
                        "margin": margins[-1],
                    }
                )
    swap_consistent = 0
    group_correct = 0
    for group_id, predictions in semantic_predictions.items():
        if len(predictions) != 2:
            raise AssertionError(f"ranking group lacks swapped pair: {group_id}")
        consistent = predictions[0] == predictions[1]
        swap_consistent += int(consistent)
        candidate_index = int(group_id.rsplit(":", 1)[1])
        group_correct += int(consistent and predictions[0] == candidate_index)
    group_count = len(semantic_predictions)
    return {
        "forced_accuracy": forced_correct / len(examples),
        "open_vocab_accuracy": open_correct / len(examples),
        "mean_target_margin": sg0._mean(margins),
        "swap_consistency": swap_consistent / group_count,
        "semantic_group_accuracy": group_correct / group_count,
        "example_count": len(examples),
        "group_count": group_count,
        "timing": {
            **_sample_summary(timings, 1),
            "p99_ms": _percentile(timings, 0.99),
            "state_bytes_max": max(state_sizes),
            "state_bytes_mean": sg0._mean(state_sizes),
        },
        "records": records if include_records else None,
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
            "next_route": "formal_sg4_move_pair_ranking",
        }
    mean_nll = {
        name: sg0._mean(
            seed["post_teacher"][name]["test"]["nll"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_accuracy = {
        name: sg0._mean(
            seed["ranking"][name]["forced_accuracy"] for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_swap = {
        name: sg0._mean(
            seed["ranking"][name]["swap_consistency"] for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    ann_improvement = all(
        seed["post_teacher"][name]["test"]["nll"]
        <= seed["pre_teacher"][name]["test"]["nll"] - 0.10
        for seed in seed_results
        for name in ("lstm", "transformer")
    )
    best_ann_accuracy = max(mean_accuracy["lstm"], mean_accuracy["transformer"])
    best_ann_swap = max(mean_swap["lstm"], mean_swap["transformer"])
    best_ann_nll = min(mean_nll["lstm"], mean_nll["transformer"])
    task_pass = (
        ann_improvement and best_ann_accuracy >= 0.90 and best_ann_swap >= 0.95
    )
    ra0_improvement = all(
        seed["post_teacher"]["snn_ra0"]["test"]["nll"]
        <= seed["pre_teacher"]["snn_ra0"]["test"]["nll"] - 0.10
        for seed in seed_results
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
        ra0_improvement
        and mean_nll["snn_ra0"] <= best_ann_nll + 0.10
        and nll_gap_bptt <= 0.10
        and nll_gap_at1 <= 0.10
        and mean_accuracy["snn_ra0"] >= best_ann_accuracy - 0.03
        and mean_accuracy["snn_ra0"] >= 0.90
        and accuracy_gap_bptt <= 0.03
        and accuracy_gap_at1 <= 0.03
        and mean_swap["snn_ra0"] >= 0.95
    )
    mean_training_p50 = {
        name: sg0._mean(
            seed["training"][name]["timing"]["p50_ms"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    at1_speedup = mean_training_p50["snn_at1"] / mean_training_p50["snn_ra0"]
    bptt_speedup = mean_training_p50["snn_bptt"] / mean_training_p50["snn_ra0"]
    speed_pass = (
        at1_speedup >= 1.25
        and bptt_speedup >= 1.25
        and mean_training_p50["snn_ra0"] <= mean_training_p50["lstm"]
    )
    response_pass = all(
        seed["ranking"]["snn_ra0"]["timing"]["p50_ms"]
        <= seed["ranking"]["lstm"]["timing"]["p50_ms"]
        and seed["ranking"]["snn_ra0"]["timing"]["p95_ms"]
        <= seed["ranking"]["lstm"]["timing"]["p95_ms"]
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
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "mean_test_teacher_nll": mean_nll,
        "mean_forced_accuracy": mean_accuracy,
        "mean_swap_consistency": mean_swap,
        "best_ann_nll": best_ann_nll,
        "best_ann_accuracy": best_ann_accuracy,
        "best_ann_swap_consistency": best_ann_swap,
        "ra0_vs_bptt_nll_gap": nll_gap_bptt,
        "ra0_vs_at1_nll_gap": nll_gap_at1,
        "ra0_vs_bptt_accuracy_gap": accuracy_gap_bptt,
        "ra0_vs_at1_accuracy_gap": accuracy_gap_at1,
        "mean_training_p50_ms": mean_training_p50,
        "ra0_vs_at1_training_speedup": at1_speedup,
        "ra0_vs_bptt_training_speedup": bptt_speedup,
        "next_route": (
            "ranking_guided_generation_or_closed_loop"
            if overall == "PASS"
            else "spiking_associative_memory_or_native_scan"
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
    examples, vocabulary = build_ranking_examples(corpus_root, corpus)
    expected_counts = dict(zip(SPLITS, args.expected_counts))
    expected_groups = dict(zip(SPLITS, args.expected_groups))
    data_audit = audit_ranking_examples(
        examples,
        vocabulary,
        expected_counts=expected_counts,
        expected_groups=expected_groups,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG4 ranking data audit failed; refusing model experiment")

    seed_results = []
    for seed in args.seeds:
        models = build_history_models(
            9_600_000 + 100 * seed,
            vocabulary,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_heads=args.num_heads,
            device=device,
        )
        parameter_counts = {
            name: {"total": count_parameters(model), "core": count_parameters(model.core)}
            for name, model in models.items()
        }
        totals = tuple(record["total"] for record in parameter_counts.values())
        parameter_spread = (max(totals) - min(totals)) / sg0._mean(totals)
        if parameter_spread > 0.02:
            raise AssertionError(f"SG4 parameter spread failed: {parameter_counts}")
        pre_teacher = {
            name: {
                split: sg0.evaluate_teacher(model, examples[split], device=device)
                for split in ("valid", "test")
            }
            for name, model in models.items()
        }
        schedule = sg0._training_schedule(
            len(examples["train"]), args.epochs, 9_601_000 + seed
        )
        training = {
            name: sg0.train_model(
                name,
                model,
                examples["train"],
                schedule,
                epochs=args.epochs,
                device=device,
            )
            for name, model in models.items()
        }
        post_teacher = {
            name: {
                split: sg0.evaluate_teacher(model, examples[split], device=device)
                for split in ("train", "valid", "test")
            }
            for name, model in models.items()
        }
        ranking = {
            name: evaluate_ranking(
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
                "pre_teacher": pre_teacher,
                "training": training,
                "post_teacher": post_teacher,
                "ranking": ranking,
            }
        )
    decision = _decision(data_audit, seed_results, quick=args.quick)
    return {
        "schema_version": 1,
        "experiment": "E3-SG4 hard move-pair counterfactual ranking",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "corpus_dir": str(corpus_root),
            "expected_data_seeds": expected_data_seeds,
            "expected_counts": expected_counts,
            "expected_groups": expected_groups,
            "epochs": args.epochs,
            "seeds": tuple(args.seeds),
            "threads": args.threads if device.type == "cpu" else None,
            "d_model": args.d_model,
            "state_dim": args.state_dim,
            "num_heads": args.num_heads,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "optimizer_fused": True,
            "gradient_foreach": True,
            "max_prompt_tokens": args.max_prompt_tokens,
            "state_reset_per_example": True,
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "corpus_provenance": tw0._corpus_provenance(corpus),
            "vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
                "source_split": "train",
                "source_fields": ("ranking_prompt", "label"),
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
        default=Path("results/e3_scan/e3_sg4_move_pair_ranking.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
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
        args.max_prompt_tokens,
        *args.expected_counts,
        *args.expected_groups,
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
