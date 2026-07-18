"""Reproducible official TextWorld Coin Collector dataset runner.

This command is deliberately offline-by-default:

* without ``--generate`` every manifest ``.z8`` must already exist and pass
  a Z-machine header/size check;
* with ``--generate`` the only executable invoked is ``tw-make`` located next
  to the current Python executable, using argv with ``shell=False``;
* existing games and output artifacts are never replaced unless the caller
  explicitly supplies ``--overwrite``;
* collection delegates to the official TextWorld adapter, including the
  metadata walkthrough and ``Environment.copy`` counterfactuals.

Each split receives an atomic manifest, canonical JSONL episodes, deterministic
token-event text, and an atomic provenance summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpsc.world_model.textworld_dataset import (
    CHALLENGE,
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
    SPLITS,
    CoinCollectorEpisode,
    CoinCollectorGameSpec,
    DatasetFile,
    build_coin_collector_manifest,
    collect_coin_collector_game,
    episode_to_json_line,
    episode_to_token_event_text,
    file_sha256,
)


RUNNER_SCHEMA_VERSION = "vpsc.e2_textworld_dataset.v1"
RUNNER_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAMES_DIR = REPOSITORY_ROOT / "data" / "textworld" / "coin_collector"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "e2_textworld_dataset"
MINIMUM_Z8_BYTES = 64


class DatasetRunError(RuntimeError):
    """Raised when a run would be incomplete or irreproducible."""


@dataclass(frozen=True)
class ValidatedGame:
    """Validated immutable provenance for one compiled game file."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class GenerationStatus:
    """Whether and how a game was generated during this invocation."""

    generated: bool
    executed_command: Optional[Tuple[str, ...]]


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


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetRunError(f"Cannot serialize canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    overwrite: bool,
) -> DatasetFile:
    """Durably write bytes through a sibling temp file.

    The non-overwrite branch uses an atomic hard-link creation, so a file that
    appears after preflight still cannot be silently replaced.
    """

    destination = path.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise DatasetRunError(
            f"Refusing to overwrite existing artifact: {destination}. "
            "Pass --overwrite explicitly to replace dataset outputs."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise DatasetRunError(
                    f"Artifact appeared during the run; refusing overwrite: "
                    f"{destination}."
                ) from exc
            temporary.unlink()
    except OSError as exc:
        raise DatasetRunError(
            f"Atomic write failed for {destination}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)

    return DatasetFile(
        path=str(destination),
        sha256=file_sha256(destination),
        size_bytes=len(content),
    )


def validate_z8_file(path: Any) -> ValidatedGame:
    """Validate extension, regular file, size, version byte, and SHA256."""

    game = Path(path).expanduser().resolve()
    if game.suffix.lower() != ".z8":
        raise DatasetRunError(f"Expected an official .z8 game, got: {game}.")
    if not game.is_file():
        raise DatasetRunError(
            f"Required compiled game is missing: {game}. Re-run with "
            "--generate to invoke the environment-local official tw-make."
        )
    size = game.stat().st_size
    if size < MINIMUM_Z8_BYTES:
        raise DatasetRunError(
            f"Invalid .z8 file {game}: {size} bytes is smaller than the "
            f"{MINIMUM_Z8_BYTES}-byte Z-machine header."
        )
    with game.open("rb") as stream:
        version_byte = stream.read(1)
    if version_byte != b"\x08":
        raise DatasetRunError(
            f"Invalid .z8 file {game}: Z-machine version byte is "
            f"{version_byte.hex() or 'missing'}, expected 08."
        )
    return ValidatedGame(
        path=str(game),
        sha256=file_sha256(game),
        size_bytes=size,
    )


def resolve_sibling_tw_make(python_executable: Any = None) -> Path:
    """Find ``tw-make`` inside the selected/current Python environment.

    ``sys.executable`` is commonly a symlink in POSIX virtual environments.
    Resolving it can jump from ``<venv>/bin/python`` to ``/usr/bin/python`` and
    lose the environment-local console script. For the live runtime, search
    ``sys.prefix/bin`` or ``sys.prefix/Scripts`` first. An explicitly supplied
    executable (used by tests and embedding callers) is intentionally located
    by its *unresolved* parent.
    """

    names = ("tw-make.exe", "tw-make") if os.name == "nt" else (
        "tw-make",
        "tw-make.exe",
    )
    if python_executable is not None:
        invoked = Path(python_executable).expanduser().absolute()
        search_directories = (invoked.parent,)
        runtime_description = f"explicit Python executable {invoked}"
    else:
        environment_scripts = (
            Path(sys.prefix).expanduser().absolute()
            / ("Scripts" if os.name == "nt" else "bin")
        )
        invoked = Path(sys.executable).expanduser().absolute()
        # Keep the invoked parent as a constrained fallback for unusual Python
        # layouts, but never resolve it and never search PATH.
        search_directories = tuple(
            dict.fromkeys((environment_scripts, invoked.parent))
        )
        runtime_description = (
            f"Python environment prefix {Path(sys.prefix).expanduser().absolute()} "
            f"(invoked as {invoked})"
        )
    candidates = tuple(
        directory / name
        for directory in search_directories
        for name in names
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise DatasetRunError(
        "--generate was requested, but the official TextWorld CLI was not "
        f"found inside {runtime_description}. Checked: {rendered}. "
        "The PATH is intentionally not searched."
    )


def _split_paths(output_dir: Path, split: str) -> Dict[str, Path]:
    root = output_dir.expanduser().resolve() / split
    return {
        "manifest": root / "manifest.json",
        "episodes_jsonl": root / "episodes.jsonl",
        "token_events": root / "token_events.txt",
        "summary": root / "summary.json",
    }


def _all_output_paths(output_dir: Path) -> Tuple[Path, ...]:
    return tuple(
        path
        for split in SPLITS
        for path in _split_paths(output_dir, split).values()
    )


def _preflight_outputs(output_dir: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    conflicts = [path for path in _all_output_paths(output_dir) if path.exists()]
    if conflicts:
        rendered = "\n  ".join(str(path) for path in conflicts)
        raise DatasetRunError(
            "Dataset outputs already exist and --overwrite was not supplied:\n  "
            f"{rendered}"
        )


def _generation_command(
    spec: CoinCollectorGameSpec,
    tw_make: Path,
    *,
    overwrite: bool,
) -> Tuple[str, ...]:
    portable = spec.generation_command
    if not portable or portable[0] != "tw-make":
        raise DatasetRunError(
            "Manifest generation command is not the official tw-make entry point."
        )
    arguments = tuple(portable[1:])
    if not overwrite:
        # The reusable manifest advertises --force for exact regeneration, but
        # a default CLI run must remain non-overwriting even under a race where
        # the target appears after preflight and before subprocess execution.
        arguments = tuple(argument for argument in arguments if argument != "--force")
    return (str(tw_make),) + arguments


def _prepare_games(
    games: Iterable[CoinCollectorGameSpec],
    *,
    generate: bool,
    overwrite: bool,
    python_executable: Any,
    subprocess_run: Callable[..., Any],
) -> Tuple[
    Mapping[Tuple[str, int], ValidatedGame],
    Mapping[Tuple[str, int], GenerationStatus],
    Optional[str],
]:
    specs = tuple(games)
    need_generation = generate and (
        overwrite or any(not Path(spec.game_file).is_file() for spec in specs)
    )
    tw_make = (
        resolve_sibling_tw_make(python_executable) if need_generation else None
    )
    validated: Dict[Tuple[str, int], ValidatedGame] = {}
    statuses: Dict[Tuple[str, int], GenerationStatus] = {}

    for spec in specs:
        key = (spec.split, spec.seed)
        game_path = Path(spec.game_file)
        should_generate = bool(generate and (overwrite or not game_path.is_file()))
        executed: Optional[Tuple[str, ...]] = None
        if should_generate:
            if tw_make is None:
                raise DatasetRunError("Internal error: tw-make was not resolved.")
            game_path.parent.mkdir(parents=True, exist_ok=True)
            executed = _generation_command(spec, tw_make, overwrite=overwrite)
            try:
                subprocess_run(
                    list(executed),
                    check=True,
                    shell=False,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise DatasetRunError(
                    f"Official tw-make failed for split={spec.split} "
                    f"seed={spec.seed}: {exc}"
                ) from exc
        validated[key] = validate_z8_file(game_path)
        statuses[key] = GenerationStatus(
            generated=should_generate,
            executed_command=executed,
        )

    return validated, statuses, str(tw_make) if tw_make is not None else None


def _artifact_record(artifact: DatasetFile) -> Dict[str, Any]:
    return {
        "path": artifact.path,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def _split_manifest_record(
    level: int,
    split: str,
    specs: Sequence[CoinCollectorGameSpec],
) -> Dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "challenge": CHALLENGE,
        "level": level,
        "split": split,
        "split_key": "game_seed",
        "games": [spec.to_record() for spec in specs],
    }


def _runtime_versions() -> Dict[str, Any]:
    return {
        "runner": RUNNER_VERSION,
        "dataset_schema": DATASET_SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "textworld": _installed_version("textworld"),
        "vpsc": _installed_version("vpsc"),
        "platform": platform.platform(),
    }


def _python_runtime_record() -> Dict[str, Optional[str]]:
    def display(value: Any) -> Optional[str]:
        if value in {None, ""}:
            return None
        return str(Path(value).expanduser().absolute())

    return {
        "invoked_executable": display(sys.executable),
        "prefix": display(sys.prefix),
        "base_prefix": display(getattr(sys, "base_prefix", None)),
        "base_executable": display(getattr(sys, "_base_executable", None)),
    }


def _validate_collected_episode(
    episode: CoinCollectorEpisode,
    spec: CoinCollectorGameSpec,
    game: ValidatedGame,
) -> None:
    if (
        episode.split != spec.split
        or episode.seed != spec.seed
        or episode.level != spec.level
    ):
        raise DatasetRunError(
            "Collector returned episode provenance that does not match its manifest spec."
        )
    if episode.game_sha256 != game.sha256:
        raise DatasetRunError(
            f"Game changed between validation and collection for seed {spec.seed}: "
            f"{game.sha256} != {episode.game_sha256}."
        )
    if not episode.won:
        raise DatasetRunError(
            f"Collector returned a non-winning episode for seed {spec.seed}."
        )


def _write_split(
    *,
    output_dir: Path,
    level: int,
    split: str,
    specs: Sequence[CoinCollectorGameSpec],
    episodes: Sequence[CoinCollectorEpisode],
    games: Mapping[Tuple[str, int], ValidatedGame],
    statuses: Mapping[Tuple[str, int], GenerationStatus],
    tw_make: Optional[str],
    counterfactual_limit: int,
    generate_requested: bool,
    overwrite: bool,
    allow_unsupported_platform: bool,
) -> DatasetFile:
    paths = _split_paths(output_dir, split)
    manifest = _atomic_write_bytes(
        paths["manifest"],
        _canonical_json_bytes(_split_manifest_record(level, split, specs)),
        overwrite=overwrite,
    )
    jsonl_content = (
        "\n".join(episode_to_json_line(episode) for episode in episodes) + "\n"
    ).encode("utf-8")
    jsonl = _atomic_write_bytes(
        paths["episodes_jsonl"], jsonl_content, overwrite=overwrite
    )
    event_content = "".join(
        episode_to_token_event_text(episode) for episode in episodes
    ).encode("utf-8")
    token_events = _atomic_write_bytes(
        paths["token_events"], event_content, overwrite=overwrite
    )

    game_records = []
    for spec in specs:
        key = (split, spec.seed)
        game = games[key]
        status = statuses[key]
        game_records.append(
            {
                "seed": spec.seed,
                "level": spec.level,
                "split": spec.split,
                "path": game.path,
                "sha256": game.sha256,
                "size_bytes": game.size_bytes,
                "generation_command": list(spec.generation_command),
                "generation_command_text": spec.generation_command_text,
                "generated_this_run": status.generated,
                "executed_generation_command": (
                    list(status.executed_command)
                    if status.executed_command is not None
                    else None
                ),
            }
        )
    python_runtime = _python_runtime_record()
    summary_payload = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "challenge": CHALLENGE,
        "split": split,
        "split_key": "game_seed",
        "level": level,
        "seeds": [spec.seed for spec in specs],
        "counterfactual_limit": counterfactual_limit,
        "generate_requested": generate_requested,
        "overwrite_requested": overwrite,
        "allow_unsupported_platform": allow_unsupported_platform,
        "python_executable": python_runtime["invoked_executable"],
        "python_runtime": python_runtime,
        "tw_make_executable": tw_make,
        "versions": _runtime_versions(),
        "games": game_records,
        "artifacts": {
            "manifest": _artifact_record(manifest),
            "episodes_jsonl": _artifact_record(jsonl),
            "token_events": _artifact_record(token_events),
        },
    }
    return _atomic_write_bytes(
        paths["summary"],
        _canonical_json_bytes(summary_payload),
        overwrite=overwrite,
    )


def run_dataset(
    args: argparse.Namespace,
    *,
    subprocess_run: Callable[..., Any] = subprocess.run,
    collector: Callable[..., CoinCollectorEpisode] = collect_coin_collector_game,
    python_executable: Any = None,
) -> Dict[str, Any]:
    """Run generation/validation, official collection, and atomic writes."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    _preflight_outputs(output_dir, overwrite=bool(args.overwrite))
    manifest = build_coin_collector_manifest(
        args.games_dir,
        level=args.level,
        train_seeds=args.train_seeds,
        valid_seeds=args.valid_seeds,
        test_seeds=args.test_seeds,
    )
    games, statuses, tw_make = _prepare_games(
        manifest.games,
        generate=bool(args.generate),
        overwrite=bool(args.overwrite),
        python_executable=python_executable,
        subprocess_run=subprocess_run,
    )

    by_split: Dict[str, list[CoinCollectorEpisode]] = {
        split: [] for split in SPLITS
    }
    for spec in manifest.games:
        episode = collector(
            spec,
            counterfactual_limit=args.counterfactual_limit,
            allow_unsupported_platform=bool(args.allow_unsupported_platform),
        )
        _validate_collected_episode(
            episode, spec, games[(spec.split, spec.seed)]
        )
        by_split[spec.split].append(episode)

    summaries: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        split_specs = tuple(
            spec for spec in manifest.games if spec.split == split
        )
        split_episodes = tuple(
            sorted(by_split[split], key=lambda episode: episode.seed)
        )
        summary = _write_split(
            output_dir=output_dir,
            level=manifest.level,
            split=split,
            specs=split_specs,
            episodes=split_episodes,
            games=games,
            statuses=statuses,
            tw_make=tw_make,
            counterfactual_limit=args.counterfactual_limit,
            generate_requested=bool(args.generate),
            overwrite=bool(args.overwrite),
            allow_unsupported_platform=bool(args.allow_unsupported_platform),
        )
        summaries[split] = _artifact_record(summary)

    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "summary_artifacts": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build official TextWorld Coin Collector world-model datasets "
            "without synthetic environment transitions."
        )
    )
    parser.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--level", type=_positive_int, required=True)
    parser.add_argument(
        "--train-seeds", nargs="+", type=_nonnegative_int, required=True
    )
    parser.add_argument(
        "--valid-seeds", nargs="+", type=_nonnegative_int, required=True
    )
    parser.add_argument(
        "--test-seeds", nargs="+", type=_nonnegative_int, required=True
    )
    parser.add_argument("--counterfactual-limit", type=_positive_int, default=4)
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "generate missing games with the tw-make sibling of the current "
            "Python executable; never searches PATH"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing games (with --generate) and outputs",
    )
    parser.add_argument(
        "--allow-unsupported-platform",
        action="store_true",
        help="forward TextWorld's explicit native-platform override",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_dataset(args)
    except Exception as exc:
        if isinstance(exc, (DatasetRunError, RuntimeError, ValueError)):
            parser.error(str(exc))
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
