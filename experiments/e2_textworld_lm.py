"""Real TextWorld event-language-model comparison on CPU.

This runner consumes the already collected level-5 TextWorld
``token_events.txt`` corpus.  It never generates data and has no synthetic or
empty-data fallback.  LSTM, causal Transformer, and the frozen E2 hybrid are
trained and measured through the shared world-model factory and harness.

The output is explicitly a pilot record, not a confirmatory architecture
decision.  It does include the preregistered READY/REVISE pipeline gate, is
written atomically, and never overwrites an existing destination.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpsc.world_model.event_corpus import TextWorldEventCorpus, load_event_corpus
from vpsc.world_model.factory import (
    FairLMConfig,
    FrozenE2Config,
    assert_parameter_budget,
    build_model_suite,
)
from vpsc.world_model.training import (
    TrainingConfig,
    benchmark_streaming_step,
    evaluate_language_model,
    seed_everything,
    train_language_model,
)
from vpsc.world_model.wikitext import SPLITS, file_sha256


SCHEMA_VERSION = 2
PILOT_SCOPE = "pilot_not_confirmatory"
PARAMETER_TOLERANCE = 0.02
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPOSITORY_ROOT / "results" / "e2_world_model" / "textworld_l5"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "results"
    / "e2_world_model"
    / "textworld_event_lm_pilot_s0_s1_s2.json"
)
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.json"
EPISODES_FILENAME = "episodes.jsonl"
EVENTS_FILENAME = "token_events.txt"
EXPECTED_TEXTWORLD_VERSION = "1.7.0"
EXPECTED_DATASET_SCHEMA = "vpsc.textworld.coin_collector.v1"
EXPECTED_RUNNER_SCHEMA = "vpsc.e2_textworld_dataset.v1"
FROZEN_DATASET_SEEDS = {
    "train": (20260718, 20260719, 20260720, 20260721),
    "valid": (20260722,),
    "test": (20260723,),
}


def _strict_json_object(path: Path) -> Dict[str, Any]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"required TextWorld manifest is missing: {path}; no fallback was used"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"TextWorld manifest must be a JSON object: {path}")
    return value


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_int(value: object, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def _verified_artifact(path: Path, record: object, *, context: str) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{context} artifact record must be an object")
    if not path.is_file():
        raise FileNotFoundError(f"required TextWorld artifact is missing: {path}")
    expected_sha = record.get("sha256")
    expected_size = record.get("size_bytes")
    actual_sha = file_sha256(path)
    actual_size = path.stat().st_size
    if expected_sha != actual_sha:
        raise ValueError(
            f"{context} artifact SHA256 mismatch: expected {expected_sha!r}, "
            f"found {actual_sha!r}"
        )
    if expected_size != actual_size:
        raise ValueError(
            f"{context} artifact size mismatch: expected {expected_size!r}, "
            f"found {actual_size!r}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "size_bytes": actual_size,
    }


def _resolve_game_path(raw_path: object, *, root: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("manifest game_file must be a non-empty string")
    path = Path(raw_path).expanduser()
    if os.name == "nt" and raw_path.startswith("/mnt/") and len(raw_path) > 7:
        drive = raw_path[5]
        if raw_path[6] == "/" and drive.isalpha():
            path = Path(f"{drive.upper()}:/{raw_path[7:]}")
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _read_episode_records(path: Path) -> list[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read TextWorld episodes {path}: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"TextWorld episodes must be non-empty canonical JSONL: {path}")
    records: list[Dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant {constant}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"invalid episode JSON in {path} line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"episode record must be an object in {path} line {line_number}")
        records.append(value)
    return records


def _event_header_seeds(path: Path, split: str) -> tuple[int, ...]:
    seeds = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read TextWorld events {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t", 2)
        if len(fields) != 3 or fields[1] != "<|episode|>":
            continue
        try:
            payload = json.loads(fields[2])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid event episode header in {path} line {line_number}: {error}"
            ) from error
        if fields[0] != "000000" or not isinstance(payload, dict):
            raise ValueError(f"invalid event episode header in {path} line {line_number}")
        if payload.get("split") != split:
            raise ValueError(f"event episode split mismatch in {path} line {line_number}")
        seeds.append(
            _require_int(
                payload.get("seed"),
                context=f"event episode seed in {path} line {line_number}",
            )
        )
    if not seeds:
        raise ValueError(f"event artifact contains no episode headers: {path}")
    return tuple(seeds)


def _manifest_provenance(root: Path) -> Dict[str, Any]:
    """Verify the complete official-dataset proof chain for every split."""

    records: Dict[str, Any] = {}
    hash_inputs: Dict[str, Any] = {}
    seed_owner: Dict[int, str] = {}
    for split in SPLITS:
        split_root = root / split
        manifest_path = split_root / MANIFEST_FILENAME
        summary_path = split_root / SUMMARY_FILENAME
        episodes_path = split_root / EPISODES_FILENAME
        events_path = split_root / EVENTS_FILENAME
        manifest = _strict_json_object(manifest_path)
        summary = _strict_json_object(summary_path)

        for label, record in (("manifest", manifest), ("summary", summary)):
            if record.get("split") != split:
                raise ValueError(f"{label} split mismatch for {split_root}")
            if record.get("challenge") != "tw-coin_collector" or record.get("level") != 5:
                raise ValueError(f"{label} is not the required TextWorld level 5 task")
            if record.get("split_key") != "game_seed":
                raise ValueError(f"{label} split_key must be 'game_seed' for {split}")
        if manifest.get("schema_version") != EXPECTED_DATASET_SCHEMA:
            raise ValueError(f"unexpected manifest schema in {manifest_path}")
        if manifest.get("runner_schema_version") != EXPECTED_RUNNER_SCHEMA:
            raise ValueError(f"unexpected manifest runner schema in {manifest_path}")
        if summary.get("schema_version") != EXPECTED_RUNNER_SCHEMA:
            raise ValueError(f"unexpected summary schema in {summary_path}")
        if summary.get("counterfactual_limit") != 2:
            raise ValueError(f"counterfactual_limit must be 2 in {summary_path}")
        versions = summary.get("versions")
        if not isinstance(versions, dict) or versions.get("textworld") != EXPECTED_TEXTWORLD_VERSION:
            raise ValueError(
                f"TextWorld version must be {EXPECTED_TEXTWORLD_VERSION} in {summary_path}"
            )

        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"summary artifacts must be an object: {summary_path}")
        verified_artifacts = {
            "manifest": _verified_artifact(
                manifest_path, artifacts.get("manifest"), context=f"{split} manifest"
            ),
            "episodes_jsonl": _verified_artifact(
                episodes_path,
                artifacts.get("episodes_jsonl"),
                context=f"{split} episodes_jsonl",
            ),
            "token_events": _verified_artifact(
                events_path,
                artifacts.get("token_events"),
                context=f"{split} token_events",
            ),
        }

        games = manifest.get("games")
        summary_games = summary.get("games")
        if not isinstance(games, list) or not isinstance(summary_games, list) or not games:
            raise ValueError(f"manifest/summary contains no real TextWorld games: {split_root}")
        if len(games) != len(summary_games):
            raise ValueError(f"manifest/summary game count mismatch: {split_root}")
        manifest_seeds = []
        verified_games = []
        summary_by_seed: Dict[int, Mapping[str, Any]] = {}
        for game in summary_games:
            if not isinstance(game, dict):
                raise ValueError(f"summary game entry is not an object: {summary_path}")
            game_seed = _require_int(game.get("seed"), context="summary game seed")
            if game_seed in summary_by_seed:
                raise ValueError(f"duplicate summary game seed {game_seed} in {split}")
            summary_by_seed[game_seed] = game

        manifest_by_seed: Dict[int, Mapping[str, Any]] = {}
        for game in games:
            if not isinstance(game, dict):
                raise ValueError(f"manifest game entry is not an object: {manifest_path}")
            if game.get("split") != split or game.get("level") != 5:
                raise ValueError(f"manifest game split/level mismatch: {manifest_path}")
            seed = _require_int(game.get("seed"), context="manifest game seed")
            if seed in manifest_by_seed:
                raise ValueError(f"duplicate manifest game seed {seed} in {split}")
            manifest_by_seed[seed] = game
            manifest_seeds.append(seed)
            previous = seed_owner.get(seed)
            if previous is not None:
                raise ValueError(f"game seed {seed} leaks across {previous} and {split}")
            seed_owner[seed] = split

            summary_game = summary_by_seed.get(seed)
            if summary_game is None:
                raise ValueError(f"game seed {seed} missing from summary: {summary_path}")
            if summary_game.get("split") != split or summary_game.get("level") != 5:
                raise ValueError(f"summary game split/level mismatch for seed {seed}")
            if summary_game.get("path") != game.get("game_file"):
                raise ValueError(f"manifest/summary game path mismatch for seed {seed}")
            game_path = _resolve_game_path(game.get("game_file"), root=root)
            if not game_path.is_file() or game_path.suffix.lower() != ".z8":
                raise FileNotFoundError(f"required real TextWorld .z8 is missing: {game_path}")
            actual_size = game_path.stat().st_size
            actual_sha = file_sha256(game_path)
            if summary_game.get("size_bytes") != actual_size:
                raise ValueError(f"game size mismatch for seed {seed}: {game_path}")
            if summary_game.get("sha256") != actual_sha:
                raise ValueError(f"game SHA256 mismatch for seed {seed}: {game_path}")
            with game_path.open("rb") as handle:
                if handle.read(1) != b"\x08":
                    raise ValueError(f"game is not a Z-machine version-8 artifact: {game_path}")
            verified_games.append(
                {
                    "seed": seed,
                    "path": str(game_path),
                    "sha256": actual_sha,
                    "size_bytes": actual_size,
                }
            )

        expected_seeds = FROZEN_DATASET_SEEDS[split]
        seeds = tuple(manifest_seeds)
        summary_seed_values = summary.get("seeds")
        if not isinstance(summary_seed_values, list):
            raise ValueError(f"summary seeds must be a list in {summary_path}")
        summary_seeds = tuple(
            _require_int(value, context=f"summary {split} seed")
            for value in summary_seed_values
        )
        if seeds != expected_seeds or summary_seeds != expected_seeds:
            raise ValueError(
                f"{split} frozen seeds must be {list(expected_seeds)}, found {list(seeds)}"
            )
        if set(summary_by_seed) != set(expected_seeds):
            raise ValueError(f"summary seeds do not match frozen {split} seeds")

        episodes = _read_episode_records(episodes_path)
        episode_seeds = []
        for episode in episodes:
            seed = _require_int(episode.get("seed"), context="episode seed")
            episode_seeds.append(seed)
            game = manifest_by_seed.get(seed)
            summary_game = summary_by_seed.get(seed)
            if game is None or summary_game is None:
                raise ValueError(f"episode seed {seed} is absent from manifest/summary")
            if episode.get("split") != split or episode.get("level") != 5:
                raise ValueError(f"episode split/level mismatch for seed {seed}")
            if episode.get("challenge") != "tw-coin_collector":
                raise ValueError(f"episode challenge mismatch for seed {seed}")
            if episode.get("game_file") != game.get("game_file"):
                raise ValueError(f"episode game path mismatch for seed {seed}")
            if episode.get("game_sha256") != summary_game.get("sha256"):
                raise ValueError(f"episode game SHA256 mismatch for seed {seed}")
            episode_return = episode.get("return")
            if (
                episode.get("won") is not True
                or isinstance(episode_return, bool)
                or not isinstance(episode_return, (int, float))
                or float(episode_return) != 1.0
            ):
                raise ValueError(f"episode seed {seed} is not a verified 1.0-return win")
            if not isinstance(episode.get("steps"), list) or not episode["steps"]:
                raise ValueError(f"episode seed {seed} contains no real transitions")
        if tuple(episode_seeds) != expected_seeds:
            raise ValueError(f"episode records do not match frozen {split} seeds")
        event_seeds = _event_header_seeds(events_path, split)
        if event_seeds != tuple(episode_seeds):
            raise ValueError(f"event episode headers do not match episodes for {split}")

        manifest_sha = verified_artifacts["manifest"]["sha256"]
        summary_sha = file_sha256(summary_path)
        hash_inputs[split] = {
            "summary": summary_sha,
            "artifacts": {
                name: record["sha256"] for name, record in verified_artifacts.items()
            },
            "games": [game["sha256"] for game in verified_games],
        }
        records[split] = {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "size_bytes": manifest_path.stat().st_size,
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": summary_sha,
            "schema_version": manifest["schema_version"],
            "runner_schema_version": manifest["runner_schema_version"],
            "challenge": manifest["challenge"],
            "level": manifest["level"],
            "textworld_version": versions["textworld"],
            "counterfactual_limit": summary["counterfactual_limit"],
            "seeds": list(seeds),
            "game_count": len(games),
            "episode_count": len(episodes),
            "event_episode_count": len(event_seeds),
            "artifacts": verified_artifacts,
            "games": verified_games,
            "verified": True,
        }
    return {
        "fingerprint_sha256": _sha256_json(hash_inputs),
        "textworld_version": EXPECTED_TEXTWORLD_VERSION,
        "frozen_seeds": {split: list(seeds) for split, seeds in FROZEN_DATASET_SEEDS.items()},
        "cross_split_seed_disjoint": True,
        "splits": records,
        "verified": True,
    }


def _corpus_provenance(corpus: TextWorldEventCorpus) -> Dict[str, Any]:
    metadata = corpus.metadata()
    split_hashes = {
        split: corpus.split_metadata(split).source_sha256 for split in SPLITS
    }
    return {
        **metadata,
        "corpus_fingerprint_sha256": _sha256_json(
            {
                "event_source_sha256": split_hashes,
                "vocabulary_fingerprint_sha256": corpus.vocabulary.fingerprint,
            }
        ),
        "vocabulary": {
            "actual_size": len(corpus.vocabulary),
            "fingerprint_sha256": corpus.vocabulary.fingerprint,
            "built_from": "train_only",
        },
    }


def _episode_reset_audit(
    corpus: TextWorldEventCorpus,
    sequence_length: int,
) -> Dict[str, Any]:
    splits: Dict[str, Any] = {}
    for split in SPLITS:
        chunks = corpus.iter_chunks(split, sequence_length)
        reset_chunks = sum(bool(chunk.reset_state) for chunk in chunks)
        episode_count = corpus.episode_count(split)
        if reset_chunks != episode_count:
            raise RuntimeError(
                f"episode reset invariant failed for {split}: "
                f"{reset_chunks} reset chunks for {episode_count} episodes"
            )
        splits[split] = {
            "episode_count": episode_count,
            "reset_chunk_count": reset_chunks,
            "verified": True,
        }
    return {
        "policy": "reset state on the first chunk of every episode",
        "no_chunk_crosses_episode_boundary": True,
        "splits": splits,
    }


def _batch_stream(
    corpus: TextWorldEventCorpus,
    split: str,
    sequence_length: int,
) -> Iterable[Any]:
    return corpus.iter_chunks(
        split,
        sequence_length=sequence_length,
        drop_last=False,
        as_tensors=False,
    )


def _streaming_tokens(
    corpus: TextWorldEventCorpus,
    required_tokens: int,
) -> tuple[int, ...]:
    first_test_episode = next(corpus.iter_episode_token_ids("test"), None)
    if first_test_episode is None:
        raise ValueError("TextWorld test split contains no episodes")
    if len(first_test_episode) < required_tokens:
        raise ValueError(
            "first TextWorld test episode is too short for a boundary-safe "
            f"streaming benchmark: need {required_tokens}, have {len(first_test_episode)}"
        )
    return tuple(first_test_episode[:required_tokens])


def _summary(values: Sequence[float]) -> Dict[str, float | int]:
    if not values:
        raise ValueError("cannot aggregate an empty metric")
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


def _aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    exclude: Sequence[str] = (),
) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate no records")
    excluded = set(exclude)
    result: Dict[str, Any] = {}
    for key in records[0]:
        if key in excluded:
            continue
        values = [record[key] for record in records]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result[key] = _summary(values)
    return result


def _aggregate_seed_results(seed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    model_names = ("lstm", "transformer", "e2")
    models: Dict[str, Any] = {}
    for model_name in model_names:
        records = [result["models"][model_name] for result in seed_results]
        parameter_records = [record["parameters"] for record in records]
        if any(record != parameter_records[0] for record in parameter_records[1:]):
            raise RuntimeError(f"parameter statistics changed across seeds for {model_name}")
        models[model_name] = {
            "parameters": parameter_records[0],
            "training": _aggregate_records(
                [record["training"] for record in records],
                exclude=("seed",),
            ),
            "evaluation": {
                split: _aggregate_records(
                    [record["evaluation"][split] for record in records]
                )
                for split in SPLITS
            },
            "streaming": _aggregate_records(
                [record["streaming"] for record in records],
                exclude=("seed",),
            ),
        }
    return {
        "seed_count": len(seed_results),
        "seeds": [result["seed"] for result in seed_results],
        "statistics": "mean, population std, min, and max across seeds",
        "models": models,
    }


def _comparison_metrics_are_finite(seed_results: Sequence[Mapping[str, Any]]) -> bool:
    metric_groups = {
        "training": (
            "nll",
            "ppl",
            "mean_gradient_norm",
            "elapsed_seconds",
            "tokens_per_second",
        ),
        "evaluation": ("nll", "ppl"),
        "streaming": (
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "tokens_per_second",
        ),
    }
    for seed_result in seed_results:
        models = seed_result.get("models")
        if not isinstance(models, dict) or set(models) != {"lstm", "transformer", "e2"}:
            return False
        for record in models.values():
            training = record["training"]
            streaming = record["streaming"]
            if any(
                not math.isfinite(float(training[key]))
                for key in metric_groups["training"]
            ):
                return False
            if any(
                not math.isfinite(float(streaming[key]))
                for key in metric_groups["streaming"]
            ):
                return False
            for split in SPLITS:
                evaluation = record["evaluation"][split]
                if any(
                    not math.isfinite(float(evaluation[key]))
                    for key in metric_groups["evaluation"]
                ):
                    return False
    return True


def _refuse_existing_output(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing result: {destination}")
    return destination


def write_json_atomic_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish strict JSON, failing if the destination exists."""

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
            # A same-directory hard link atomically publishes the fully written
            # inode and, unlike os.replace, can never overwrite a racing writer.
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing result: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def run_textworld_lm(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the frozen real-event pilot and write one auditable JSON record."""

    destination = _refuse_existing_output(args.output)
    if args.batch_size != 1:
        raise ValueError("episode-aware TextWorld pilot requires batch_size=1")

    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = load_event_corpus(corpus_root)
    manifests = _manifest_provenance(corpus_root)
    corpus_record = _corpus_provenance(corpus)
    reset_audit = _episode_reset_audit(corpus, args.sequence_length)
    available_targets = {
        split: sum(
            len(episode) - 1 for episode in corpus.iter_episode_token_ids(split)
        )
        for split in SPLITS
    }
    streaming_tokens = _streaming_tokens(
        corpus,
        args.streaming_warmup_steps + args.streaming_steps,
    )

    factory_config = FairLMConfig(
        vocab_size=len(corpus.vocabulary),
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=0.0,
        padding_idx=corpus.vocabulary.pad_id,
        transformer_max_cache_tokens=args.cache_window,
        e2=FrozenE2Config(policy="hybrid", positive_factor=0.8),
        parameter_tolerance=PARAMETER_TOLERANCE,
        auto_match_parameters=True,
    )

    seed_results = []
    for seed in args.seeds:
        seed_everything(seed)
        suite = build_model_suite(factory_config)
        parameter_report = assert_parameter_budget(
            suite,
            tolerance=PARAMETER_TOLERANCE,
        )
        if parameter_report.relative_spread > PARAMETER_TOLERANCE + 1e-12:
            raise RuntimeError("strict parameter-spread invariant was not enforced")

        model_results: Dict[str, Any] = {}
        for model_name, model in suite.models.items():
            training = train_language_model(
                model,
                _batch_stream(corpus, "train", args.sequence_length),
                TrainingConfig(
                    seed=seed,
                    learning_rate=args.learning_rate,
                    max_steps=args.steps,
                    device="cpu",
                ),
            )
            evaluation = {
                split: asdict(
                    evaluate_language_model(
                        model,
                        _batch_stream(corpus, split, args.sequence_length),
                        max_steps=args.eval_steps,
                        device="cpu",
                    )
                )
                for split in SPLITS
            }
            streaming = benchmark_streaming_step(
                model,
                streaming_tokens,
                warmup_steps=args.streaming_warmup_steps,
                measured_steps=args.streaming_steps,
                seed=seed,
                device="cpu",
            )
            model_results[model_name] = {
                "parameters": model.parameter_stats().as_dict(),
                "training": asdict(training),
                "evaluation": evaluation,
                "streaming": asdict(streaming),
            }

        train_steps = {
            record["training"]["steps"] for record in model_results.values()
        }
        train_targets = {
            record["training"]["target_count"] for record in model_results.values()
        }
        evaluation_targets = {
            split: {
                record["evaluation"][split]["target_count"]
                for record in model_results.values()
            }
            for split in SPLITS
        }
        if len(train_steps) != 1 or len(train_targets) != 1:
            raise RuntimeError("fairness invariant failed for training data consumption")
        if any(len(counts) != 1 for counts in evaluation_targets.values()):
            raise RuntimeError("fairness invariant failed for evaluation data consumption")

        seed_results.append(
            {
                "seed": seed,
                "parameter_budget": parameter_report.as_dict(),
                "e2_effective_gains": asdict(suite.e2.core.effective_gains()),
                "models": model_results,
            }
        )

    train_consumed_values = {
        result["models"][model]["training"]["target_count"]
        for result in seed_results
        for model in ("lstm", "transformer", "e2")
    }
    evaluation_consumed_values = {
        split: {
            result["models"][model]["evaluation"][split]["target_count"]
            for result in seed_results
            for model in ("lstm", "transformer", "e2")
        }
        for split in SPLITS
    }
    if len(train_consumed_values) != 1 or any(
        len(values) != 1 for values in evaluation_consumed_values.values()
    ):
        raise RuntimeError("token accounting changed across model or experiment seed")
    train_consumed = next(iter(train_consumed_values))
    evaluation_consumed = {
        split: next(iter(values))
        for split, values in evaluation_consumed_values.items()
    }
    token_accounting = {
        "definition": "shifted causal-LM target tokens; BOS has no target",
        "available_target_tokens": available_targets,
        "consumed_training_target_tokens_per_model": train_consumed,
        "consumed_evaluation_target_tokens_per_model": evaluation_consumed,
        "configured_train_max_updates": args.steps,
        "configured_eval_max_chunks_per_split": args.eval_steps,
        "one_deterministic_pass_without_episode_repeat": True,
        "same_counts_across_models_and_seeds": True,
    }
    pipeline_checks = {
        "official_dataset_provenance_verified": bool(manifests["verified"]),
        "frozen_seed_splits_verified": bool(
            manifests["cross_split_seed_disjoint"]
        ),
        "episode_boundaries_verified": bool(
            reset_audit["no_chunk_crosses_episode_boundary"]
            and all(record["verified"] for record in reset_audit["splits"].values())
        ),
        "available_train_targets_at_least_10000": available_targets["train"]
        >= 10_000,
        "equal_data_consumption": True,
        "all_metrics_finite": _comparison_metrics_are_finite(seed_results),
        "held_out_event_evaluation_completed": all(
            evaluation_consumed[split] > 0 for split in ("valid", "test")
        ),
    }
    pipeline_status = "READY" if all(pipeline_checks.values()) else "REVISE"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "command": "textworld-l5-event-lm-pilot",
        "scope": PILOT_SCOPE,
        "confirmatory": False,
        "automatic_decision": None,
        "pipeline_status": pipeline_status,
        "pipeline_checks": pipeline_checks,
        "device": "cpu",
        "dataset": {
            "name": "TextWorld Coin Collector level 5 token events",
            "synthetic": False,
            "fallback_used": False,
            "corpus_root": str(corpus_root),
            "event_corpus": corpus_record,
            "manifests": manifests,
            "episode_reset_audit": reset_audit,
            "token_accounting": token_accounting,
        },
        "config": {
            "seeds": list(args.seeds),
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "dropout": 0.0,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "train_steps": args.steps,
            "eval_max_batches_per_split": args.eval_steps,
            "learning_rate": args.learning_rate,
            "transformer_kv_cache_tokens": args.cache_window,
            "streaming_warmup_steps": args.streaming_warmup_steps,
            "streaming_measured_steps": args.streaming_steps,
            "parameter_tolerance": PARAMETER_TOLERANCE,
            "e2_policy": "hybrid",
            "e2_positive_factor": 0.8,
            "same_data_order_for_all_models": True,
            "same_step_limits_for_all_models": True,
            "episode_state_reset": True,
        },
        "results": seed_results,
        "aggregate": _aggregate_seed_results(seed_results),
        "interpretation_boundary": (
            "pipeline_status only gates the event-LM data/training pipeline. "
            "It is not an H-WM pass/fail or confirmatory architecture claim."
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
        description=(
            "Real TextWorld L5 event-LM LSTM/Transformer/E2 CPU pilot; "
            "never confirmatory and never overwrites results"
        )
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
        default=64,
    )
    parser.add_argument("--steps", type=_positive_int, default=100)
    parser.add_argument(
        "--eval-max",
        "--eval-steps",
        dest="eval_steps",
        type=_positive_int,
        default=50,
    )
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--cache-window", type=_positive_int, default=128)
    parser.add_argument(
        "--streaming-warmup-steps",
        type=_nonnegative_int,
        default=10,
    )
    parser.add_argument("--streaming-steps", type=_positive_int, default=100)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_textworld_lm(args)
    print(f"wrote {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
