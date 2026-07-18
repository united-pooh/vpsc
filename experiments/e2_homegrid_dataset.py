"""Gate M official ``homegrid-dynamics`` dataset CLI.

The frozen preregistration default is 32/8/8 train/valid/test seeds, at most 96
transitions per episode, and an independent uniform action stream from
``random.Random(seed + 1_000_003).randrange(10)``. CLI arguments may override
the seeds or horizon; every output manifest and summary records both the frozen
default and the actual protocol.
"""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Dict, Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpsc.world_model.homegrid_dataset import (
    DEFAULT_MAX_STEPS,
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALID_SEEDS,
    HOMEGRID_DYNAMICS_ENV_ID,
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
    SPLITS,
    HomeGridDatasetEpisode,
    HomeGridDatasetError,
    HomeGridDatasetFile,
    HomeGridManifest,
    action_sampling_definition,
    aggregate_counts,
    build_homegrid_manifest,
    collect_homegrid_episode,
    frozen_default_protocol,
    quantization_definition,
    write_canonical_json_atomic,
    write_transition_jsonl_atomic,
)


RUNNER_SCHEMA_VERSION = "vpsc.e2_homegrid_gate_m.v1"
RUNNER_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "e2_homegrid_gate_m"


class HomeGridRunError(HomeGridDatasetError):
    """Raised when the CLI cannot produce a complete split artifact set."""


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


def _installed_version(distribution: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _versions() -> Dict[str, Any]:
    return {
        "runner": RUNNER_VERSION,
        "dataset_schema": DATASET_SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "homegrid": _installed_version("homegrid"),
        "gym": _installed_version("gym"),
        "numpy": _installed_version("numpy"),
        "platform": platform.platform(),
    }


def _artifact_record(artifact: HomeGridDatasetFile) -> Dict[str, Any]:
    return {
        "path": artifact.path,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def _split_paths(output_dir: Path, split: str) -> Dict[str, Path]:
    root = output_dir.expanduser().resolve() / split
    return {
        "transitions_jsonl": root / "transitions.jsonl",
        "manifest": root / "manifest.json",
        "summary": root / "summary.json",
    }


def _preflight_outputs(output_dir: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    conflicts = [
        path
        for split in SPLITS
        for path in _split_paths(output_dir, split).values()
        if path.exists()
    ]
    if conflicts:
        rendered = "\n  ".join(str(path) for path in conflicts)
        raise HomeGridRunError(
            "Gate M output artifacts already exist; refusing overwrite:\n  "
            f"{rendered}"
        )


def _actual_protocol(manifest: HomeGridManifest) -> Dict[str, Any]:
    return {
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "max_steps": manifest.max_steps,
        "train_seeds": [
            game.seed for game in manifest.games if game.split == "train"
        ],
        "valid_seeds": [
            game.seed for game in manifest.games if game.split == "valid"
        ],
        "test_seeds": [
            game.seed for game in manifest.games if game.split == "test"
        ],
        "action_sampling": action_sampling_definition(),
    }


def _protocol_overrides(manifest: HomeGridManifest) -> Dict[str, bool]:
    actual = _actual_protocol(manifest)
    frozen = frozen_default_protocol()
    return {
        "max_steps": actual["max_steps"] != frozen["max_steps"],
        "train_seeds": actual["train_seeds"] != frozen["train_seeds"],
        "valid_seeds": actual["valid_seeds"] != frozen["valid_seeds"],
        "test_seeds": actual["test_seeds"] != frozen["test_seeds"],
        "action_sampling": False,
    }


def _validate_episode(
    episode: HomeGridDatasetEpisode,
    expected_split: str,
    expected_seed: int,
    expected_max_steps: int,
) -> None:
    if episode.env_id != HOMEGRID_DYNAMICS_ENV_ID:
        raise HomeGridRunError("Collector returned a non-dynamics environment.")
    if episode.split != expected_split or episode.seed != expected_seed:
        raise HomeGridRunError(
            "Collector episode split/seed does not match the manifest spec."
        )
    if episode.max_steps != expected_max_steps:
        raise HomeGridRunError("Collector changed the fixed max_steps protocol.")
    if not episode.transitions or len(episode.transitions) > expected_max_steps:
        raise HomeGridRunError(
            "Collector returned zero transitions or exceeded max_steps."
        )


def _split_manifest(
    *,
    manifest: HomeGridManifest,
    split: str,
    episodes: Sequence[HomeGridDatasetEpisode],
    versions: Dict[str, Any],
    jsonl: HomeGridDatasetFile,
) -> Dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "split": split,
        "split_key": "environment_seed",
        "versions": versions,
        "uses_frozen_default_protocol": manifest.uses_frozen_default_protocol,
        "frozen_default_protocol": frozen_default_protocol(),
        "actual_protocol": _actual_protocol(manifest),
        "protocol_overrides": _protocol_overrides(manifest),
        "visual_quantization": quantization_definition(),
        "action_sampling": action_sampling_definition(),
        "seed_modes": sorted({episode.seed_mode for episode in episodes}),
        "episodes": [episode.manifest_record() for episode in episodes],
        "artifacts": {"transitions_jsonl": _artifact_record(jsonl)},
    }


def _write_split(
    *,
    output_dir: Path,
    manifest: HomeGridManifest,
    split: str,
    episodes: Sequence[HomeGridDatasetEpisode],
    versions: Dict[str, Any],
    overwrite: bool,
) -> HomeGridDatasetFile:
    paths = _split_paths(output_dir, split)
    jsonl = write_transition_jsonl_atomic(
        episodes,
        paths["transitions_jsonl"],
        overwrite=overwrite,
    )
    manifest_file = write_canonical_json_atomic(
        _split_manifest(
            manifest=manifest,
            split=split,
            episodes=episodes,
            versions=versions,
            jsonl=jsonl,
        ),
        paths["manifest"],
        overwrite=overwrite,
    )
    summary = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "split": split,
        "versions": versions,
        "uses_frozen_default_protocol": manifest.uses_frozen_default_protocol,
        "frozen_default_protocol": frozen_default_protocol(),
        "actual_protocol": _actual_protocol(manifest),
        "protocol_overrides": _protocol_overrides(manifest),
        "visual_quantization": quantization_definition(),
        "action_sampling": action_sampling_definition(),
        "seed_modes": sorted({episode.seed_mode for episode in episodes}),
        "seeds": [episode.seed for episode in episodes],
        "counts": aggregate_counts(episodes),
        "artifacts": {
            "transitions_jsonl": _artifact_record(jsonl),
            "manifest": _artifact_record(manifest_file),
        },
    }
    return write_canonical_json_atomic(
        summary,
        paths["summary"],
        overwrite=overwrite,
    )


def run_dataset(
    args: argparse.Namespace,
    *,
    collector: Callable[[Any], HomeGridDatasetEpisode] = collect_homegrid_episode,
) -> Dict[str, Any]:
    """Collect all real episodes before atomically writing any split artifact."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    _preflight_outputs(output_dir, overwrite=bool(args.overwrite))
    manifest = build_homegrid_manifest(
        train_seeds=args.train_seeds,
        valid_seeds=args.valid_seeds,
        test_seeds=args.test_seeds,
        max_steps=args.max_steps,
    )
    by_split: Dict[str, list[HomeGridDatasetEpisode]] = {
        split: [] for split in SPLITS
    }
    for spec in manifest.games:
        episode = collector(spec)
        _validate_episode(episode, spec.split, spec.seed, spec.max_steps)
        by_split[spec.split].append(episode)

    versions = _versions()
    summaries: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        episodes = tuple(sorted(by_split[split], key=lambda item: item.seed))
        artifact = _write_split(
            output_dir=output_dir,
            manifest=manifest,
            split=split,
            episodes=episodes,
            versions=versions,
            overwrite=bool(args.overwrite),
        )
        summaries[split] = _artifact_record(artifact)
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "uses_frozen_default_protocol": manifest.uses_frozen_default_protocol,
        "summary_artifacts": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Gate M from official homegrid-dynamics with deterministic "
            "seed-isolated uniform actions."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-steps", type=_positive_int, default=DEFAULT_MAX_STEPS
    )
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=_nonnegative_int,
        default=list(DEFAULT_TRAIN_SEEDS),
    )
    parser.add_argument(
        "--valid-seeds",
        nargs="+",
        type=_nonnegative_int,
        default=list(DEFAULT_VALID_SEEDS),
    )
    parser.add_argument(
        "--test-seeds",
        nargs="+",
        type=_nonnegative_int,
        default=list(DEFAULT_TEST_SEEDS),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing Gate M split artifacts",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_dataset(args)
    except (HomeGridDatasetError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
