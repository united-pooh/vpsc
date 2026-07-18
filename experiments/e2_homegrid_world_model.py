"""Run the preregistered E2-M0 HomeGrid dynamics pilot on CPU.

The runner consumes only the strict official HomeGrid corpus, builds the shared
LSTM/Transformer/E2 model suite per seed, and delegates training, one-step
evaluation, controlled rollout, and streaming timing to the common harness.
It never overwrites an existing result and never emits non-finite JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpsc.world_model.homegrid_corpus import HomeGridCorpus, load_homegrid_corpus
from vpsc.world_model.homegrid_model import (
    E2_POLICY,
    E2_POSITIVE_FACTOR,
    PARAMETER_TOLERANCE,
    TRANSFORMER_CACHE_TOKENS,
    assert_homegrid_parameter_budget,
    build_homegrid_model_suite,
)
from vpsc.world_model.homegrid_training import (
    HomeGridTrainingConfig,
    benchmark_homegrid_streaming,
    evaluate_homegrid_model,
    evaluate_homegrid_rollouts,
    seed_everything,
    train_homegrid_model,
)


SCHEMA_VERSION = 1
PILOT_SCOPE = "pilot_not_confirmatory"
SPLITS = ("train", "valid", "test")
EVALUATION_SPLITS = ("valid", "test")
ROLLOUT_HORIZONS = (1, 3, 5, 10)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = (
    REPOSITORY_ROOT / "results" / "e2_world_model" / "homegrid_dynamics"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "results"
    / "e2_world_model"
    / "homegrid_dynamics_pilot_s0_s1_s2.json"
)
TRAIN_ACTION_THRESHOLD = 2_000
TEST_CHANGED_PATCH_THRESHOLD = 1_000


def _refuse_existing_output(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing result: {destination}")
    return destination


def write_json_atomic_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish strict JSON and fail if another result exists."""

    destination = _refuse_existing_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                ensure_ascii=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing result: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _target_classes(corpus: HomeGridCorpus) -> tuple[tuple[int, ...], tuple[int, ...]]:
    reward = set()
    done = set()
    for episode in corpus.iter_episodes("train"):
        for transition in episode.transitions:
            reward.add(int(transition.reward_class))
            done.add(int(transition.done))
    reward_classes = tuple(sorted(reward))
    done_classes = tuple(sorted(done))
    if reward_classes != (0, 1, 2):
        raise ValueError(
            "frozen M0 reward-loss protocol requires train classes {0,1,2}, got "
            f"{set(reward_classes)}"
        )
    if done_classes != (0,):
        raise ValueError(
            "frozen M0 done-loss protocol requires the sole train class {0}, got "
            f"{set(done_classes)}"
        )
    return reward_classes, done_classes


def _finite_metric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FloatingPointError(f"required HomeGrid metric {label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"required HomeGrid metric {label} is non-finite: {value!r}")
    return result


def _nested(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise FloatingPointError(f"required HomeGrid metric is missing: {'.'.join(path)}")
        value = value[key]
    return value


def _assert_numeric_tree_finite(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_numeric_tree_finite(child, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _assert_numeric_tree_finite(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"HomeGrid metric {label} is non-finite: {value!r}")


def _assert_required_metrics_finite(model_record: Mapping[str, Any]) -> None:
    training_paths = (
        ("weighted_loss",),
        ("mean_gradient_norm",),
        ("elapsed_seconds",),
        ("transitions_per_second",),
        ("visual_tokens_per_second",),
        ("component_nll", "visual"),
        ("component_nll", "language"),
        ("component_nll", "read"),
        ("component_nll", "reward"),
        ("component_nll", "done"),
    )
    for path in training_paths:
        _finite_metric(_nested(model_record["training"], path), f"training.{'.'.join(path)}")

    for split in EVALUATION_SPLITS:
        evaluation = model_record["one_step"][split]
        for group in ("overall", "changed", "unchanged", "read_phase", "action_phase"):
            for metric in ("nll", "accuracy", "macro_f1_present_targets"):
                path = ("visual", group, metric)
                _finite_metric(
                    _nested(evaluation, path),
                    f"one_step.{split}.{'.'.join(path)}",
                )
        for group in ("next_language", "next_read", "reward"):
            for metric in ("nll", "accuracy", "brier"):
                path = (group, metric)
                _finite_metric(
                    _nested(evaluation, path),
                    f"one_step.{split}.{'.'.join(path)}",
                )
        for path in (
            ("baselines", "copy_current_frame", "overall_accuracy"),
            ("baselines", "copy_current_frame", "changed_accuracy"),
            ("baselines", "train_global_frequency", "overall_accuracy"),
        ):
            _finite_metric(
                _nested(evaluation, path),
                f"one_step.{split}.{'.'.join(path)}",
            )
        if evaluation["done"].get("enabled") is not False:
            raise RuntimeError("done metrics must remain disabled for train class {0}")

    for horizon in ROLLOUT_HORIZONS:
        for metric in (
            "anchors",
            "overall_accuracy",
            "changed_accuracy",
            "overall_patch_count",
            "changed_patch_count",
        ):
            path = ("horizons", str(horizon), metric)
            _finite_metric(
                _nested(model_record["rollout"], path),
                f"rollout.{'.'.join(path)}",
            )

    for metric in (
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "transitions_per_second",
        "state_nbytes",
    ):
        _finite_metric(
            _nested(model_record["streaming"], (metric,)),
            f"streaming.{metric}",
        )
    _assert_numeric_tree_finite(model_record, "model")


def _summary(values: Sequence[int | float]) -> Dict[str, float | int]:
    numeric = [float(value) for value in values]
    mean = math.fsum(numeric) / len(numeric)
    variance = math.fsum((value - mean) ** 2 for value in numeric) / len(numeric)
    return {
        "count": len(numeric),
        "mean": mean,
        "std_population": math.sqrt(variance),
        "min": min(numeric),
        "max": max(numeric),
    }


def _aggregate_numeric_tree(records: Sequence[Any]) -> Optional[Any]:
    if not records:
        return None
    if all(isinstance(record, Mapping) for record in records):
        first = records[0]
        result = {}
        for key in first:
            if all(key in record for record in records):
                child = _aggregate_numeric_tree([record[key] for record in records])
                if child is not None:
                    result[key] = child
        return result or None
    if all(
        isinstance(record, (int, float)) and not isinstance(record, bool)
        for record in records
    ):
        return _summary(records)
    return None


def _aggregate_results(seed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    models = {}
    for model_name in ("lstm", "transformer", "e2"):
        records = [seed["models"][model_name] for seed in seed_results]
        parameters = [record["parameters"] for record in records]
        if any(value != parameters[0] for value in parameters[1:]):
            raise RuntimeError(f"parameter statistics changed across seeds for {model_name}")
        models[model_name] = {
            "parameters": parameters[0],
            "metrics": _aggregate_numeric_tree(records),
        }
    return {
        "seed_count": len(seed_results),
        "seeds": [seed["seed"] for seed in seed_results],
        "statistics": "mean, population std, min, and max across model seeds",
        "models": models,
    }


def _counts(corpus: HomeGridCorpus) -> Dict[str, Dict[str, Any]]:
    fields = (
        "episode_count",
        "transition_count",
        "read_step_count",
        "action_step_count",
        "changed_patch_count",
        "transitions_with_changed_patches",
        "reward_sum",
        "done_count",
        "language_oov_current",
        "language_oov_next",
    )
    return {
        split: {
            field: corpus.split_metadata(split)[field]
            for field in fields
        }
        for split in SPLITS
    }


def _consumption_audit(
    model_results: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, Mapping[str, Any]],
    epochs: int,
) -> Dict[str, Any]:
    expected = {
        "training_transitions": int(counts["train"]["transition_count"]) * epochs,
        "valid_transitions": int(counts["valid"]["transition_count"]),
        "test_transitions": int(counts["test"]["transition_count"]),
    }
    observed = {
        model_name: {
            "training_transitions": record["training"]["transitions"],
            "valid_transitions": record["one_step"]["valid"]["transitions"],
            "test_transitions": record["one_step"]["test"]["transitions"],
        }
        for model_name, record in model_results.items()
    }
    signatures = {tuple(value.items()) for value in observed.values()}
    if len(signatures) != 1:
        raise RuntimeError("fairness invariant failed: models consumed different transitions")
    if next(iter(observed.values())) != expected:
        raise RuntimeError(
            "fairness invariant failed: observed transition counts do not match corpus/epochs"
        )
    return {
        "expected": expected,
        "observed_by_model": observed,
        "identical": True,
    }


def run_homegrid_pilot(args: argparse.Namespace) -> Dict[str, Any]:
    """Run all seeds/models and publish one complete M0 pilot record."""

    destination = _refuse_existing_output(args.output)
    if args.batch_size != 1:
        raise ValueError("the preregistered HomeGrid pilot requires batch_size=1")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("model seeds must be unique")

    corpus = load_homegrid_corpus(args.corpus_dir)
    reward_classes, done_classes = _target_classes(corpus)
    counts = _counts(corpus)
    test_episodes = tuple(corpus.iter_episodes("test"))
    if not test_episodes:
        raise ValueError("HomeGrid test split contains no episodes")
    first_test_episode = test_episodes[0]

    seed_results = []
    canonical_baselines: Optional[Dict[str, Any]] = None
    consumption_signatures = []
    for seed in args.seeds:
        seed_everything(seed)
        suite = build_homegrid_model_suite(
            language_vocab_size=len(corpus.vocabulary),
            d_model=args.d_model,
            num_heads=args.num_heads,
            transformer_cache_tokens=args.cache_window,
            parameter_tolerance=PARAMETER_TOLERANCE,
        )
        parameter_report = assert_homegrid_parameter_budget(
            suite,
            tolerance=PARAMETER_TOLERANCE,
        )
        model_results: Dict[str, Any] = {}
        for model_name, model in suite.models.items():
            training = train_homegrid_model(
                model,
                corpus.iter_chunks(
                    "train",
                    args.sequence_length,
                    epochs=args.epochs,
                ),
                HomeGridTrainingConfig(
                    seed=seed,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    reward_enabled=True,
                    done_enabled=False,
                    device="cpu",
                ),
            )
            one_step = {
                split: evaluate_homegrid_model(
                    model,
                    corpus.iter_chunks(split, args.sequence_length, epochs=1),
                    frequency_visual_token=corpus.most_frequent_visual_token,
                    reward_enabled=True,
                    done_enabled=False,
                    device="cpu",
                )
                for split in EVALUATION_SPLITS
            }
            rollout = evaluate_homegrid_rollouts(
                model,
                test_episodes,
                horizons=ROLLOUT_HORIZONS,
            )
            streaming = benchmark_homegrid_streaming(
                model,
                first_test_episode,
                warmup_steps=args.streaming_warmup_steps,
                measured_steps=args.streaming_steps,
            )
            record = {
                "parameters": model.parameter_stats().as_dict(),
                "training": training,
                "one_step": one_step,
                "rollout": rollout,
                "streaming": streaming,
            }
            _assert_required_metrics_finite(record)
            model_results[model_name] = record

            baselines = {
                split: one_step[split]["baselines"] for split in EVALUATION_SPLITS
            }
            if canonical_baselines is None:
                canonical_baselines = baselines
            elif baselines != canonical_baselines:
                raise RuntimeError("data-only HomeGrid baselines changed across model/seed")

        consumption = _consumption_audit(model_results, counts, args.epochs)
        signature = tuple(consumption["expected"].items())
        consumption_signatures.append(signature)
        seed_results.append(
            {
                "seed": seed,
                "parameter_budget": parameter_report.as_dict(),
                "e2_effective_gains": asdict(suite.e2.core.effective_gains()),
                "transition_consumption": consumption,
                "models": model_results,
            }
        )

    if len(set(consumption_signatures)) != 1:
        raise RuntimeError("fairness invariant failed: transition counts changed across seeds")
    if canonical_baselines is None:
        raise RuntimeError("no HomeGrid model results were produced")

    train_action_pass = counts["train"]["action_step_count"] >= TRAIN_ACTION_THRESHOLD
    test_changed_pass = (
        counts["test"]["changed_patch_count"] >= TEST_CHANGED_PATCH_THRESHOLD
    )
    metrics_complete = True
    pipeline_ready = train_action_pass and test_changed_pass and metrics_complete
    pipeline_status = "READY" if pipeline_ready else "PIPELINE_REVISE"
    loss_protocol = {
        "visual_weight": 1.0,
        "next_language_weight": 0.25,
        "next_read_weight": 0.10,
        "reward": {
            "weight": 0.10,
            "enabled": True,
            "train_target_classes": list(reward_classes),
            "reason": "all frozen reward classes 0/0.5/1 are present in train",
        },
        "done": {
            "weight": 0.0,
            "configured_weight_if_identifiable": 0.10,
            "enabled": False,
            "train_target_classes": list(done_classes),
            "reason": "train contains only done class 0",
        },
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "command": "homegrid-dynamics-m0-pilot",
        "scope": PILOT_SCOPE,
        "confirmatory": False,
        "automatic_decision": None,
        "automatic_model_decision": None,
        "pipeline_status": pipeline_status,
        "pipeline_gate": {
            "label": "M0_PIPELINE_READY" if pipeline_ready else "M0_PIPELINE_REVISE",
            "criteria": {
                "train_action_steps_at_least_2000": {
                    "observed": counts["train"]["action_step_count"],
                    "threshold": TRAIN_ACTION_THRESHOLD,
                    "passed": train_action_pass,
                },
                "test_changed_patches_at_least_1000": {
                    "observed": counts["test"]["changed_patch_count"],
                    "threshold": TEST_CHANGED_PATCH_THRESHOLD,
                    "passed": test_changed_pass,
                },
                "all_required_metrics_finite_and_complete": {
                    "passed": metrics_complete,
                },
            },
            "model_ranking_used": False,
        },
        "device": "cpu",
        "dataset": {
            "name": "official HomeGrid 0.1.1 homegrid-dynamics",
            "synthetic": False,
            "fallback_used": False,
            "corpus_root": str(corpus.root),
            "provenance": corpus.metadata(),
            "counts": counts,
            "baselines": canonical_baselines,
        },
        "loss_protocol": loss_protocol,
        "config": {
            "seeds": list(args.seeds),
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "transformer_kv_cache_tokens": args.cache_window,
            "e2_policy": E2_POLICY,
            "e2_positive_factor": E2_POSITIVE_FACTOR,
            "parameter_tolerance": PARAMETER_TOLERANCE,
            "one_step_evaluation": "full valid and test splits",
            "rollout_horizons": list(ROLLOUT_HORIZONS),
            "rollout_test_scope": "all test episodes",
            "streaming_episode": first_test_episode.episode_id,
            "streaming_warmup_steps": args.streaming_warmup_steps,
            "streaming_measured_steps": args.streaming_steps,
            "same_transition_order_and_count_for_all_models": True,
        },
        "results": seed_results,
        "aggregate": _aggregate_results(seed_results),
        "interpretation_boundary": (
            "Pilot pipeline readiness only. Model ranking does not establish a "
            "complete world model, autonomous planning, or real-time 2048-step evidence."
        ),
    }
    write_json_atomic_no_overwrite(destination, payload)
    return payload


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E2-M0 official HomeGrid dynamics CPU pilot (never confirmatory)",
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=_nonnegative_int,
        default=[0, 1, 2],
    )
    parser.add_argument("--d-model", type=_positive_int, default=32)
    parser.add_argument("--num-heads", type=_positive_int, default=4)
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=_positive_int, default=1)
    parser.add_argument(
        "--seq",
        "--sequence-length",
        dest="sequence_length",
        type=_positive_int,
        default=32,
    )
    parser.add_argument("--epochs", type=_positive_int, default=3)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument(
        "--cache-window",
        type=_positive_int,
        default=TRANSFORMER_CACHE_TOKENS,
    )
    parser.add_argument(
        "--streaming-warmup-steps",
        type=_nonnegative_int,
        default=32,
    )
    parser.add_argument("--streaming-steps", type=_positive_int, default=64)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_homegrid_pilot(args)
    print(f"wrote {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
