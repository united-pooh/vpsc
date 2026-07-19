"""SG6 compact TextWorld move state-delta classification."""

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
from experiments.e3_sg1_history_generation import build_history_models  # noqa: E402
from experiments.e3_sg4_move_pair_ranking import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    EXPECTED_DATA_SEEDS,
)
from vpsc.world_model.cores import count_parameters, state_nbytes  # noqa: E402
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


PREVIOUS_ROOM = "<previous_room>"
NOVEL_ROOM = "<novel_room>"
LABELS = (PREVIOUS_ROOM, NOVEL_ROOM)
MODEL_NAMES = sg0.MODEL_NAMES
EXPECTED_COUNTS = {"train": 192, "valid": 24, "test": 24}
EXPECTED_STEP_GROUPS = {"train": 96, "valid": 12, "test": 12}


@dataclass(frozen=True)
class _RawMoveDeltaExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    candidate_index: int
    source: str
    step_group_id: str
    previous_action: str
    candidate_action: str
    outcome_text: str
    prior_match_lags: Tuple[int, ...]
    prompt_tokens: Tuple[str, ...]
    target_token: str


@dataclass(frozen=True)
class MoveDeltaExample:
    split: str
    episode_index: int
    game_seed: int
    step_index: int
    candidate_index: int
    source: str
    step_group_id: str
    previous_action: str
    candidate_action: str
    action_type: str
    outcome_text: str
    prior_match_lags: Tuple[int, ...]
    prompt_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    prompt_unknowns: int

    @property
    def example_id(self) -> str:
        return (
            f"{self.split}:{self.game_seed}:{self.step_index}:"
            f"{self.candidate_index}"
        )

    @property
    def input_length(self) -> int:
        return len(self.prompt_ids)


def _move_delta_prompt_tokens(
    corpus: Any,
    previous_action: str,
    candidate_action: str,
) -> Tuple[str, ...]:
    parts = (
        ("<bos>",),
        corpus.tokenizer.tokenize("previous move:"),
        corpus.tokenizer.tokenize(previous_action),
        ("<eos>",),
        corpus.tokenizer.tokenize("candidate move:"),
        corpus.tokenizer.tokenize(candidate_action),
        ("<eos>",),
        corpus.tokenizer.tokenize("next room relation:"),
    )
    return tuple(token for part in parts for token in part)


def _prior_match_lags(
    prior_observations: Sequence[str], outcome: str
) -> Tuple[int, ...]:
    step_index = len(prior_observations)
    return tuple(
        step_index - prior_index
        for prior_index, observation in enumerate(prior_observations)
        if observation == outcome
    )


def build_move_delta_examples(
    corpus_root: Path,
    corpus: Any,
) -> Tuple[Dict[str, Tuple[MoveDeltaExample, ...]], Vocabulary]:
    raw_by_split: Dict[str, Tuple[_RawMoveDeltaExample, ...]] = {}
    for split in SPLITS:
        path = corpus_root / split / "episodes.jsonl"
        records = []
        for episode_index, line in enumerate(
            path.read_text(encoding="utf-8").splitlines()
        ):
            episode = json.loads(line)
            if episode.get("split") != split:
                raise ValueError(f"move-delta episode split mismatch: {path}")
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
                if step_index == 0:
                    raise ValueError("hard move delta requires a previous factual move")

                counterfactual = move_counterfactuals[0]
                candidate_actions = (
                    factual_action,
                    str(counterfactual["action"]),
                )
                candidate_outcomes = (
                    sg0.normalize_textworld_observation(step["next_obs"]),
                    sg0.normalize_textworld_observation(
                        counterfactual["next_obs"]
                    ),
                )
                if candidate_actions[0] == candidate_actions[1]:
                    raise ValueError("hard move candidates must use different actions")
                if candidate_outcomes[0] == candidate_outcomes[1]:
                    raise ValueError("hard move candidates must have different outcomes")

                previous_action = factual_actions[step_index - 1]
                if sg0._action_type(previous_action) != "move":
                    raise ValueError("previous action for a hard move must be a move")
                prior_observations = observations[:step_index]
                step_group_id = f"{split}:{episode['seed']}:{step_index}"
                for candidate_index, (candidate_action, outcome) in enumerate(
                    zip(candidate_actions, candidate_outcomes)
                ):
                    lags = _prior_match_lags(prior_observations, outcome)
                    target = PREVIOUS_ROOM if lags else NOVEL_ROOM
                    records.append(
                        _RawMoveDeltaExample(
                            split=split,
                            episode_index=episode_index,
                            game_seed=int(episode["seed"]),
                            step_index=step_index,
                            candidate_index=candidate_index,
                            source=(
                                "factual"
                                if candidate_index == 0
                                else "counterfactual"
                            ),
                            step_group_id=step_group_id,
                            previous_action=previous_action,
                            candidate_action=candidate_action,
                            outcome_text=outcome,
                            prior_match_lags=lags,
                            prompt_tokens=_move_delta_prompt_tokens(
                                corpus, previous_action, candidate_action
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
    examples: Dict[str, Tuple[MoveDeltaExample, ...]] = {}
    for split in SPLITS:
        encoded = []
        for raw in raw_by_split[split]:
            prompt_ids = vocabulary.encode(raw.prompt_tokens)
            encoded.append(
                MoveDeltaExample(
                    split=raw.split,
                    episode_index=raw.episode_index,
                    game_seed=raw.game_seed,
                    step_index=raw.step_index,
                    candidate_index=raw.candidate_index,
                    source=raw.source,
                    step_group_id=raw.step_group_id,
                    previous_action=raw.previous_action,
                    candidate_action=raw.candidate_action,
                    action_type="move",
                    outcome_text=raw.outcome_text,
                    prior_match_lags=raw.prior_match_lags,
                    prompt_ids=prompt_ids,
                    target_ids=vocabulary.encode((raw.target_token,)),
                    prompt_unknowns=sum(
                        value == vocabulary.unk_id for value in prompt_ids
                    ),
                )
            )
        examples[split] = tuple(encoded)
    return examples, vocabulary


def audit_move_delta_examples(
    examples: Mapping[str, Sequence[MoveDeltaExample]],
    vocabulary: Vocabulary,
    *,
    expected_counts: Mapping[str, int],
    expected_step_groups: Mapping[str, int],
    max_prompt_tokens: int,
) -> Dict[str, Any]:
    split_records = {}
    all_valid = True
    previous_id = vocabulary.token_id(PREVIOUS_ROOM)
    novel_id = vocabulary.token_id(NOVEL_ROOM)
    for split, values in examples.items():
        groups: Dict[str, list[MoveDeltaExample]] = defaultdict(list)
        for example in values:
            groups[example.step_group_id].append(example)

        relationship_violations = []
        for example in values:
            target_id = example.target_ids[0]
            label_matches_membership = (
                target_id == previous_id
                if example.prior_match_lags
                else target_id == novel_id
            )
            source_relation_valid = (
                not example.prior_match_lags
                if example.source == "factual"
                else example.prior_match_lags == (1,)
            )
            if not label_matches_membership or not source_relation_valid:
                relationship_violations.append(example.example_id)

        group_valid = all(
            len(group) == 2
            and {example.candidate_index for example in group} == {0, 1}
            and {example.source for example in group}
            == {"factual", "counterfactual"}
            and {example.target_ids for example in group}
            == {(previous_id,), (novel_id,)}
            and len({example.candidate_action for example in group}) == 2
            and len({example.outcome_text for example in group}) == 2
            and next(
                example
                for example in group
                if example.source == "factual"
            ).prior_match_lags
            == ()
            and next(
                example
                for example in group
                if example.source == "counterfactual"
            ).prior_match_lags
            == (1,)
            for group in groups.values()
        )
        labels = Counter(
            vocabulary.decode(example.target_ids)[0] for example in values
        )
        source_labels = Counter(
            (
                example.source,
                vocabulary.decode(example.target_ids)[0],
            )
            for example in values
        )
        prompt_count = sum(len(example.prompt_ids) for example in values)
        unknowns = sum(example.prompt_unknowns for example in values)
        record = {
            "example_count": len(values),
            "step_group_count": len(groups),
            "game_count": len({example.game_seed for example in values}),
            "label_counts": dict(sorted(labels.items())),
            "source_label_counts": {
                f"{source}:{label}": count
                for (source, label), count in sorted(source_labels.items())
            },
            "prompt_length": sg0._length_summary(
                [len(example.prompt_ids) for example in values]
            ),
            "prompt_token_count": prompt_count,
            "prompt_unknown_count": unknowns,
            "prompt_unknown_ratio": unknowns / prompt_count,
            "factual_novel_count": sum(
                example.source == "factual"
                and example.prior_match_lags == ()
                for example in values
            ),
            "counterfactual_previous_lag1_count": sum(
                example.source == "counterfactual"
                and example.prior_match_lags == (1,)
                for example in values
            ),
            "relationship_violation_count": len(relationship_violations),
            "relationship_violation_examples": relationship_violations,
            "step_groups_valid": group_valid,
        }
        record_valid = (
            len(values) == expected_counts[split]
            and len(groups) == expected_step_groups[split]
            and labels[PREVIOUS_ROOM]
            == labels[NOVEL_ROOM]
            == len(values) // 2
            and record["prompt_length"]["max"] <= max_prompt_tokens
            and (split == "train" or record["prompt_unknown_ratio"] < 0.10)
            and record["factual_novel_count"] == len(groups)
            and record["counterfactual_previous_lag1_count"] == len(groups)
            and not relationship_violations
            and group_valid
            and all(example.action_type == "move" for example in values)
        )
        record["passed"] = record_valid
        split_records[split] = record
        all_valid = all_valid and record_valid

    return {
        "task": "compact real TextWorld move state-delta classification",
        "label_source": (
            "normalized candidate next_obs membership in prior factual "
            "observations"
        ),
        "label_tokens": LABELS,
        "expected_counts": dict(expected_counts),
        "expected_step_groups": dict(expected_step_groups),
        "max_prompt_tokens": max_prompt_tokens,
        "majority_accuracy": 0.50,
        "relationship_oracle_accuracy": 1.0,
        "splits": split_records,
        "passed": all_valid,
    }


def evaluate_move_delta(
    model: Any,
    examples: Sequence[MoveDeltaExample],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    include_records: bool,
) -> Dict[str, Any]:
    model.eval()
    label_ids = torch.tensor(
        [vocabulary.token_id(PREVIOUS_ROOM), vocabulary.token_id(NOVEL_ROOM)],
        dtype=torch.long,
        device=device,
    )
    forced_correct = 0
    open_correct = 0
    relation_correct: Counter[str] = Counter()
    relation_count: Counter[str] = Counter()
    margins = []
    timings = []
    state_sizes = []
    step_correct: Dict[str, list[bool]] = defaultdict(list)
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
            predicted_index = int(pair_logits.argmax().item())
            predicted_id = int(label_ids[predicted_index].item())
            target_id = example.target_ids[0]
            target_index = 0 if target_id == int(label_ids[0].item()) else 1
            correct = predicted_id == target_id
            forced_correct += int(correct)
            open_correct += int(int(logits.argmax().item()) == target_id)
            relation = vocabulary.decode(example.target_ids)[0]
            relation_count[relation] += 1
            relation_correct[relation] += int(correct)
            margins.append(
                float(
                    (
                        pair_logits[target_index]
                        - pair_logits[1 - target_index]
                    ).item()
                )
            )
            step_correct[example.step_group_id].append(correct)
            if include_records:
                records.append(
                    {
                        "example_id": example.example_id,
                        "step_group_id": example.step_group_id,
                        "source": example.source,
                        "candidate_index": example.candidate_index,
                        "previous_action": example.previous_action,
                        "candidate_action": example.candidate_action,
                        "prior_match_lags": example.prior_match_lags,
                        "target_label": relation,
                        "forced_label": LABELS[predicted_index],
                        "correct": correct,
                        "margin": margins[-1],
                    }
                )
    step_consistency = sum(all(values) for values in step_correct.values())
    return {
        "forced_accuracy": forced_correct / len(examples),
        "open_vocab_accuracy": open_correct / len(examples),
        "mean_target_margin": sg0._mean(margins),
        "step_consistency": step_consistency / len(step_correct),
        "relation_accuracy": {
            relation: relation_correct[relation] / relation_count[relation]
            for relation in LABELS
        },
        "example_count": len(examples),
        "step_group_count": len(step_correct),
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
            "next_route": "formal_sg6_move_delta",
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
        seed["post_teacher"][name]["test"]["nll"]
        <= seed["pre_teacher"][name]["test"]["nll"] - 0.10
        for seed in seed_results
        for name in ("lstm", "transformer")
    )
    ra0_improvement = all(
        seed["post_teacher"]["snn_ra0"]["test"]["nll"]
        <= seed["pre_teacher"]["snn_ra0"]["test"]["nll"] - 0.10
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
    mean_training_p50 = {
        name: sg0._mean(
            seed["training"][name]["timing"]["p50_ms"]
            for seed in seed_results
        )
        for name in MODEL_NAMES
    }
    at1_speedup = mean_training_p50["snn_at1"] / mean_training_p50["snn_ra0"]
    bptt_speedup = (
        mean_training_p50["snn_bptt"] / mean_training_p50["snn_ra0"]
    )
    speed_pass = (
        at1_speedup >= 1.25
        and bptt_speedup >= 1.25
        and mean_training_p50["snn_ra0"] <= mean_training_p50["lstm"]
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
        next_route = "sg7_multichannel_delta_closed_loop"
    elif task_pass and quality_pass and (not speed_pass or not response_pass):
        next_route = "sg7_native_fused_short_sequence_scan"
    elif task_pass and not quality_pass:
        next_route = "sg7_adaptive_multiscale_spike"
    else:
        next_route = "sg6_ann_task_encoding_audit"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "mean_test_teacher_nll": mean_nll,
        "mean_forced_accuracy": mean_accuracy,
        "mean_step_consistency": mean_step,
        "ann_every_seed_nll_improvement_0_10": ann_improvement,
        "ra0_every_seed_nll_improvement_0_10": ra0_improvement,
        "best_ann_nll": best_ann_nll,
        "best_ann_accuracy": best_ann_accuracy,
        "best_ann_step_consistency": best_ann_step,
        "ra0_vs_bptt_nll_gap": nll_gap_bptt,
        "ra0_vs_at1_nll_gap": nll_gap_at1,
        "ra0_vs_bptt_accuracy_gap": accuracy_gap_bptt,
        "ra0_vs_at1_accuracy_gap": accuracy_gap_at1,
        "mean_training_p50_ms": mean_training_p50,
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
        raise AssertionError("SG6 move-delta data audit failed; refusing model experiment")

    seed_results = []
    for seed in args.seeds:
        models = build_history_models(
            9_800_000 + 100 * seed,
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
            raise AssertionError(f"SG6 parameter spread failed: {parameter_counts}")
        pre_teacher = {
            name: {
                split: sg0.evaluate_teacher(model, examples[split], device=device)
                for split in ("valid", "test")
            }
            for name, model in models.items()
        }
        schedule = sg0._training_schedule(
            len(examples["train"]), args.epochs, 9_801_000 + seed
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
                "pre_teacher": pre_teacher,
                "training": training,
                "post_teacher": post_teacher,
                "move_delta": move_delta,
            }
        )
    decision = _decision(data_audit, seed_results, quick=args.quick)
    return {
        "schema_version": 1,
        "experiment": "E3-SG6 compact TextWorld move state-delta",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "research_hypothesis": {
            "epistemic_status": "speculative new idea",
            "statement": (
                "A compact event delta is a more learnable and composable SNN "
                "world-state basis than verbatim procedural room surface text."
            ),
            "what_if": (
                "What if real-time SNN world models should predict sparse state "
                "relations first and realize language only after dynamics?"
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
        default=Path("results/e3_scan/e3_sg6_move_delta.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=tw0.D_MODEL)
    parser.add_argument("--state-dim", type=int, default=tw0.STATE_DIM)
    parser.add_argument("--num-heads", type=int, default=4)
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
