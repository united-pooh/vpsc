"""SG1 history-conditioned TextWorld counterfactual sequence generation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments.e2_textworld_lm import FROZEN_DATASET_SEEDS  # noqa: E402
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_sg1_history_identifiability import (  # noqa: E402
    build_identifiability_records,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3GatedTraceScanCore,
    StatefulLSTMCore,
    count_parameters,
)
from vpsc.world_model.event_corpus import (  # noqa: E402
    TextWorldEventCorpus,
    load_event_corpus,
)
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


MODEL_NAMES = sg0.MODEL_NAMES
EXPECTED_COUNTS = sg0.EXPECTED_COUNTS
EXPECTED_PAIRS = sg0.EXPECTED_PAIRS
MAX_CONTEXT_TOKENS = 384
MAX_TARGET_TOKENS = 70


def build_history_models(
    seed: int,
    vocabulary: Vocabulary,
    *,
    d_model: int,
    state_dim: int,
    num_heads: int,
    device: torch.device,
) -> Dict[str, Any]:
    generator = torch.Generator().manual_seed(seed + 10_000)
    shared = {
        "embedding": torch.randn(
            len(vocabulary), d_model, generator=generator
        )
        * (d_model**-0.5),
        "norm_weight": torch.ones(d_model),
        "norm_bias": torch.zeros(d_model),
        "head_bias": torch.zeros(len(vocabulary)),
    }
    shared["embedding"][vocabulary.pad_id].zero_()

    torch.manual_seed(seed)
    snn_bptt = tw0._common_model(
        E3GatedTraceScanCore(d_model, d_model, state_dim=state_dim),
        vocabulary=vocabulary,
    )
    snn_at1 = tw0._common_model(
        E3GatedTraceScanCore(d_model, d_model, state_dim=state_dim),
        vocabulary=vocabulary,
    )
    snn_at1.load_state_dict(snn_bptt.state_dict())
    snn_ra0 = tw0._common_model(
        E3GatedTraceScanCore(
            d_model,
            d_model,
            state_dim=state_dim,
            eligibility_backward_mode="reverse_adjoint",
        ),
        vocabulary=vocabulary,
    )
    snn_ra0.load_state_dict(snn_bptt.state_dict())
    torch.manual_seed(seed + 1)
    lstm = tw0._common_model(
        StatefulLSTMCore(d_model, d_model), vocabulary=vocabulary
    )
    torch.manual_seed(seed + 2)
    transformer = tw0._common_model(
        CausalTransformerCore(
            d_model,
            d_model,
            num_layers=1,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=tw0.SEQUENCE_LENGTH,
        ),
        vocabulary=vocabulary,
    )
    models = {
        "snn_bptt": snn_bptt,
        "snn_at1": snn_at1,
        "snn_ra0": snn_ra0,
        "lstm": lstm,
        "transformer": transformer,
    }
    for model in models.values():
        with torch.no_grad():
            model.embedding.weight.copy_(shared["embedding"])
            model.output_norm.weight.copy_(shared["norm_weight"])
            model.output_norm.bias.copy_(shared["norm_bias"])
            model.lm_head.bias.copy_(shared["head_bias"])
        model.to(device)
    return models


@dataclass(frozen=True)
class _RawHistoryExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    counterfactual_index: int
    pair_id: str
    action: str
    action_type: str
    observation_text: str
    target_text: str
    observation_tokens: Tuple[str, ...]
    previous_observation_tokens: Tuple[str, ...]
    prompt_tokens: Tuple[str, ...]
    target_tokens: Tuple[str, ...]


def _history_prompt_tokens(
    corpus: TextWorldEventCorpus,
    observations: Sequence[str],
    factual_actions: Sequence[str],
    step_index: int,
    candidate_action: str,
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
            corpus.tokenizer.tokenize("next observation:"),
        )
    )
    return tuple(token for part in parts for token in part)


def build_history_examples(
    root: Path,
    corpus: TextWorldEventCorpus,
) -> Tuple[
    Dict[str, Tuple[sg0.CounterfactualExample, ...]],
    Vocabulary,
    Dict[str, Tuple[int, ...]],
]:
    raw_by_split: Dict[str, Tuple[_RawHistoryExample, ...]] = {}
    for split in SPLITS:
        path = root / split / "episodes.jsonl"
        raw_records = []
        for episode_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            episode = json.loads(line)
            if episode.get("split") != split:
                raise ValueError(f"history episode split mismatch: {path}")
            steps = episode["steps"]
            observations = tuple(
                sg0.normalize_textworld_observation(step["observation"])
                for step in steps
            )
            factual_actions = tuple(str(step["action"]) for step in steps)
            for step_index, step in enumerate(steps):
                observation = observations[step_index]
                for counterfactual_index, counterfactual in enumerate(
                    step["counterfactuals"]
                ):
                    action = str(counterfactual["action"])
                    target = sg0.normalize_textworld_observation(
                        counterfactual["next_obs"]
                    )
                    prompt_tokens = _history_prompt_tokens(
                        corpus,
                        observations,
                        factual_actions,
                        step_index,
                        action,
                    )
                    raw_records.append(
                        _RawHistoryExample(
                            split=split,
                            episode_index=episode_index,
                            game_seed=int(episode["seed"]),
                            step_index=step_index,
                            counterfactual_index=counterfactual_index,
                            pair_id=f"{split}:{episode['seed']}:{step_index}",
                            action=action,
                            action_type=sg0._action_type(action),
                            observation_text=observation,
                            target_text=target,
                            observation_tokens=corpus.tokenizer.tokenize(observation),
                            previous_observation_tokens=(
                                corpus.tokenizer.tokenize(observations[step_index - 1])
                                if step_index > 0
                                else ()
                            ),
                            prompt_tokens=prompt_tokens,
                            target_tokens=(
                                corpus.tokenizer.tokenize(target) + ("<eos>",)
                            ),
                        )
                    )
        raw_by_split[split] = tuple(raw_records)

    vocabulary = Vocabulary.build(
        token
        for record in raw_by_split["train"]
        for token in record.prompt_tokens + record.target_tokens
    )
    examples: Dict[str, Tuple[sg0.CounterfactualExample, ...]] = {}
    previous_ids: Dict[str, Tuple[int, ...]] = {}
    for split in SPLITS:
        encoded = []
        for raw in raw_by_split[split]:
            prompt_ids = vocabulary.encode(raw.prompt_tokens)
            target_ids = vocabulary.encode(raw.target_tokens)
            observation_ids = vocabulary.encode(raw.observation_tokens)
            example = sg0.CounterfactualExample(
                split=raw.split,
                episode_index=raw.episode_index,
                game_seed=raw.game_seed,
                step_index=raw.step_index,
                counterfactual_index=raw.counterfactual_index,
                pair_id=raw.pair_id,
                action=raw.action,
                action_type=raw.action_type,
                observation_text=raw.observation_text,
                target_text=raw.target_text,
                observation_ids=observation_ids,
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                prompt_unknowns=sum(
                    value == vocabulary.unk_id for value in prompt_ids
                ),
                target_unknowns=sum(
                    value == vocabulary.unk_id for value in target_ids[:-1]
                ),
            )
            encoded.append(example)
            previous_ids[example.example_id] = vocabulary.encode(
                raw.previous_observation_tokens
            )
        examples[split] = tuple(encoded)
    return examples, vocabulary, previous_ids


def _history_rule_predictions(
    examples: Mapping[str, Sequence[sg0.CounterfactualExample]],
    previous_ids: Mapping[str, Tuple[int, ...]],
    vocabulary: Vocabulary,
) -> Dict[str, Tuple[int, ...]]:
    majority = sg0._majority_targets(examples["train"])
    predictions = {}
    for example in examples["test"]:
        if example.action_type == "move" and previous_ids[example.example_id]:
            prediction = previous_ids[example.example_id] + (vocabulary.eos_id,)
        elif example.action_type == "look":
            prediction = example.observation_ids + (vocabulary.eos_id,)
        else:
            prediction = majority[example.action_type]
        predictions[example.example_id] = prediction
    return predictions


def audit_history_examples(
    corpus_root: Path,
    examples: Mapping[str, Sequence[sg0.CounterfactualExample]],
    vocabulary: Vocabulary,
    previous_ids: Mapping[str, Tuple[int, ...]],
    *,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
    expected_pairs: Mapping[str, int] = EXPECTED_PAIRS,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_target_tokens: int = MAX_TARGET_TOKENS,
) -> Dict[str, Any]:
    targets = {
        split: {example.target_text for example in values}
        for split, values in examples.items()
    }
    split_records = {}
    for split, values in examples.items():
        target_count = sum(len(example.target_ids) - 1 for example in values)
        split_records[split] = {
            "example_count": len(values),
            "pair_count": len({example.pair_id for example in values}),
            "action_types": dict(
                sorted(Counter(example.action_type for example in values).items())
            ),
            "history_observation_count": sg0._length_summary(
                [example.step_index for example in values]
            ),
            "prompt_length": sg0._length_summary(
                [len(example.prompt_ids) for example in values]
            ),
            "target_length_with_eos": sg0._length_summary(
                [len(example.target_ids) for example in values]
            ),
            "input_length": sg0._length_summary(
                [example.input_length for example in values]
            ),
            "prompt_unknown_count": sum(
                example.prompt_unknowns for example in values
            ),
            "target_unknown_count": sum(
                example.target_unknowns for example in values
            ),
            "target_token_count_without_eos": target_count,
            "target_unknown_ratio": (
                sum(example.target_unknowns for example in values) / target_count
            ),
            "format_only_target_count": sum(
                not sg0._CONTENT_WORD.search(example.target_text)
                for example in values
            ),
        }
    overlap = {
        "valid_in_train": len(targets["valid"] & targets["train"]),
        "test_in_train": len(targets["test"] & targets["train"]),
        "valid_test": len(targets["valid"] & targets["test"]),
    }
    majority = sg0._majority_targets(examples["train"])
    test_predictions = {
        "copy_observation": {
            example.example_id: example.observation_ids + (vocabulary.eos_id,)
            for example in examples["test"]
        },
        "action_majority": {
            example.example_id: majority[example.action_type]
            for example in examples["test"]
        },
        "history_rule": _history_rule_predictions(
            examples, previous_ids, vocabulary
        ),
    }
    baselines = {
        name: sg0.evaluate_predictions(
            examples["test"], predictions, vocabulary, include_records=False
        )
        for name, predictions in test_predictions.items()
    }
    identifiability = build_identifiability_records(corpus_root)
    held_out_move = tuple(
        record
        for split in ("valid", "test")
        for record in identifiability[split]
        if record.action_type == "move"
    )
    known_surface = sum(
        record.target_surface_in_prior_history for record in held_out_move
    )
    data_pass = (
        all(len(examples[split]) == expected_counts[split] for split in SPLITS)
        and all(
            len({example.pair_id for example in examples[split]})
            == expected_pairs[split]
            for split in SPLITS
        )
        and max(
            len(example.prompt_ids)
            for values in examples.values()
            for example in values
        )
        <= max_context_tokens
        and max(
            len(example.target_ids)
            for values in examples.values()
            for example in values
        )
        <= max_target_tokens
        and split_records["valid"]["target_unknown_ratio"] < 0.10
        and split_records["test"]["target_unknown_ratio"] < 0.10
        and all(
            record["format_only_target_count"] == 0
            for record in split_records.values()
        )
        and overlap["test_in_train"] / len(targets["test"]) <= 0.20
        and len(held_out_move) >= 8
        and known_surface == len(held_out_move)
    )
    return {
        "history_mode": "full_factual_trajectory",
        "expected_counts": dict(expected_counts),
        "expected_pairs": dict(expected_pairs),
        "max_context_tokens": max_context_tokens,
        "max_target_tokens": max_target_tokens,
        "splits": split_records,
        "target_overlap": overlap,
        "held_out_move_target_surface_in_history": {
            "count": known_surface,
            "total": len(held_out_move),
            "ratio": known_surface / len(held_out_move),
        },
        "baselines": baselines,
        "passed": data_pass,
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
            "stream_gate": "SMOKE",
            "overall": "SMOKE",
            "next_route": "formal_sg1_history_generation",
        }
    mean_nll = {
        name: sg0._mean(
            seed["post_teacher"][name]["test"]["nll"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_edit = {
        name: sg0._mean(
            seed["generation"][name]["edit_similarity"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    mean_move_room = {
        name: sg0._mean(
            seed["generation"][name]["action_types"]["move"]["room_accuracy"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    teacher_improvement = all(
        seed["post_teacher"][name]["test"]["nll"]
        <= seed["pre_teacher"][name]["test"]["nll"] - 0.10
        for seed in seed_results
        for name in ("lstm", "transformer")
    )
    baseline_edit = data_audit["baselines"]["action_majority"][
        "edit_similarity"
    ]
    best_ann_nll = min(mean_nll["lstm"], mean_nll["transformer"])
    best_ann_edit = max(mean_edit["lstm"], mean_edit["transformer"])
    best_ann_move_room = max(
        mean_move_room["lstm"], mean_move_room["transformer"]
    )
    task_pass = (
        teacher_improvement
        and best_ann_edit >= baseline_edit + 0.05
        and best_ann_move_room >= 0.75
    )
    ra0_improvement = all(
        seed["post_teacher"]["snn_ra0"]["test"]["nll"]
        <= seed["pre_teacher"]["snn_ra0"]["test"]["nll"] - 0.10
        for seed in seed_results
    )
    nll_gap_bptt = abs(mean_nll["snn_ra0"] - mean_nll["snn_bptt"])
    nll_gap_at1 = abs(mean_nll["snn_ra0"] - mean_nll["snn_at1"])
    edit_gap_bptt = abs(mean_edit["snn_ra0"] - mean_edit["snn_bptt"])
    edit_gap_at1 = abs(mean_edit["snn_ra0"] - mean_edit["snn_at1"])
    paired_sensitivity = sg0._mean(
        seed["generation"]["snn_ra0"]["paired_action_sensitivity"]
        for seed in seed_results
    )
    quality_pass = (
        ra0_improvement
        and mean_nll["snn_ra0"] <= best_ann_nll + 0.25
        and nll_gap_bptt <= 0.10
        and nll_gap_at1 <= 0.10
        and mean_edit["snn_ra0"] >= best_ann_edit - 0.10
        and edit_gap_bptt <= 0.05
        and edit_gap_at1 <= 0.05
        and mean_edit["snn_ra0"] >= baseline_edit + 0.05
        and mean_move_room["snn_ra0"] >= best_ann_move_room - 0.25
        and mean_move_room["snn_ra0"] >= 0.50
        and paired_sensitivity >= 0.50
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
    stream_pass = all(
        seed["generation"]["snn_ra0"]["timing"]["generated_token"]["p50_ms"]
        <= seed["generation"]["lstm"]["timing"]["generated_token"]["p50_ms"]
        and seed["generation"]["snn_ra0"]["timing"]["generated_token"]["p95_ms"]
        <= seed["generation"]["lstm"]["timing"]["generated_token"]["p95_ms"]
        and seed["generation"]["snn_ra0"]["timing"]["prefill"]["p50_ms"]
        <= seed["generation"]["lstm"]["timing"]["prefill"]["p50_ms"]
        for seed in seed_results
    )
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "task_gate": task_pass,
        "quality_gate": quality_pass,
        "speed_gate": speed_pass,
        "stream_gate": stream_pass,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": overall,
        "mean_test_teacher_nll": mean_nll,
        "mean_test_greedy_edit_similarity": mean_edit,
        "mean_test_move_room_accuracy": mean_move_room,
        "best_ann_nll": best_ann_nll,
        "best_ann_edit_similarity": best_ann_edit,
        "best_ann_move_room_accuracy": best_ann_move_room,
        "action_majority_edit_similarity": baseline_edit,
        "ra0_vs_bptt_nll_gap": nll_gap_bptt,
        "ra0_vs_at1_nll_gap": nll_gap_at1,
        "ra0_vs_bptt_edit_gap": edit_gap_bptt,
        "ra0_vs_at1_edit_gap": edit_gap_at1,
        "ra0_paired_action_sensitivity": paired_sensitivity,
        "mean_training_p50_ms": mean_training_p50,
        "ra0_vs_at1_training_speedup": at1_speedup,
        "ra0_vs_bptt_training_speedup": bptt_speedup,
        "next_route": (
            "online_history_closed_loop"
            if overall == "PASS"
            else "paired_ranking_or_spiking_associative_memory_or_native_scan"
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
    examples, vocabulary, previous_ids = build_history_examples(
        corpus_root, corpus
    )
    expected_counts = dict(zip(SPLITS, args.expected_counts))
    expected_pairs = dict(zip(SPLITS, args.expected_pairs))
    data_audit = audit_history_examples(
        corpus_root,
        examples,
        vocabulary,
        previous_ids,
        expected_counts=expected_counts,
        expected_pairs=expected_pairs,
        max_context_tokens=args.max_context_tokens,
        max_target_tokens=args.max_target_tokens,
    )
    if not data_audit["passed"]:
        raise AssertionError("SG1 history data audit failed; refusing model experiment")

    seed_results = []
    for seed in args.seeds:
        models = build_history_models(
            9_500_000 + 100 * seed,
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
        if parameter_spread > 0.02:
            raise AssertionError(f"SG1 parameter spread failed: {parameter_counts}")
        pre_teacher = {
            name: {
                split: sg0.evaluate_teacher(model, examples[split], device=device)
                for split in ("valid", "test")
            }
            for name, model in models.items()
        }
        schedule = sg0._training_schedule(
            len(examples["train"]), args.epochs, 9_501_000 + seed
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
        generation = {
            name: sg0.generate_model(
                model,
                examples["test"],
                vocabulary,
                max_tokens=args.max_generation_tokens,
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
                "generation": generation,
            }
        )
    decision = _decision(data_audit, seed_results, quick=args.quick)
    return {
        "schema_version": 1,
        "experiment": args.experiment_label,
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "corpus_dir": str(corpus_root),
            "history_mode": "full_factual_trajectory",
            "seeds": tuple(args.seeds),
            "epochs": args.epochs,
            "threads": args.threads if device.type == "cpu" else None,
            "d_model": args.d_model,
            "state_dim": args.state_dim,
            "num_heads": args.num_heads,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "optimizer_fused": True,
            "gradient_foreach": True,
            "max_generation_tokens": args.max_generation_tokens,
            "state_reset_per_example": True,
            "target_truncation": False,
            "vocabulary_source": "history_normalized_train_prompt_and_target_only",
            "expected_counts": expected_counts,
            "expected_pairs": expected_pairs,
            "expected_data_seeds": expected_data_seeds,
            "max_context_tokens": args.max_context_tokens,
            "max_target_tokens": args.max_target_tokens,
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest,
            "corpus_provenance": tw0._corpus_provenance(corpus),
            "generation_vocabulary": {
                "size": len(vocabulary),
                "fingerprint": vocabulary.fingerprint,
                "tokenizer": corpus.tokenizer.metadata(),
                "source_split": "train",
                "source_fields": ("history_prompt", "normalized_target"),
            },
            "audit": data_audit,
        },
        "seeds": seed_results,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg1_history_generation.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--max-context-tokens", type=int, default=MAX_CONTEXT_TOKENS
    )
    parser.add_argument(
        "--max-target-tokens", type=int, default=MAX_TARGET_TOKENS
    )
    parser.add_argument(
        "--expected-counts",
        nargs=3,
        type=int,
        metavar=("TRAIN", "VALID", "TEST"),
        default=tuple(EXPECTED_COUNTS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--expected-pairs",
        nargs=3,
        type=int,
        metavar=("TRAIN", "VALID", "TEST"),
        default=tuple(EXPECTED_PAIRS[split] for split in SPLITS),
    )
    parser.add_argument(
        "--experiment-label",
        default="E3-SG1 history-conditioned counterfactual generation",
    )
    parser.add_argument(
        "--expected-train-seeds",
        nargs="+",
        type=int,
        default=FROZEN_DATASET_SEEDS["train"],
    )
    parser.add_argument(
        "--expected-valid-seeds",
        nargs="+",
        type=int,
        default=FROZEN_DATASET_SEEDS["valid"],
    )
    parser.add_argument(
        "--expected-test-seeds",
        nargs="+",
        type=int,
        default=FROZEN_DATASET_SEEDS["test"],
    )
    parser.add_argument(
        "--max-generation-tokens",
        type=int,
        default=sg0.MAX_GENERATION_TOKENS,
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
        args.max_generation_tokens,
        args.max_context_tokens,
        args.max_target_tokens,
        *args.expected_counts,
        *args.expected_pairs,
        *args.expected_train_seeds,
        *args.expected_valid_seeds,
        *args.expected_test_seeds,
    ) <= 0:
        parser.error("epochs, threads, limits, and expected counts must be positive")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.max_generation_tokens = min(args.max_generation_tokens, 32)
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
