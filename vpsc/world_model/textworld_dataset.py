"""Deterministic datasets from official TextWorld Coin Collector games.

The production entry point in this module only accepts an existing ``.z8``
file and opens it through :func:`vpsc.world_model.textworld.open_textworld`.
It never predicts, replays, or otherwise simulates an environment transition.
Actual transitions come from ``Environment.step`` and counterfactual branches
come from the official ``Environment.copy`` implementation used by
``TextWorldAdapter.counterfactual_candidates``.

The generation command is based on TextWorld 1.7's registered challenge name
(``tw-coin_collector``), whose challenge-specific argument is ``--level``.
``--seed`` and ``--output`` are official ``tw-make`` general arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .textworld import TextWorldAdapter, WorldTransition, open_textworld


CHALLENGE = "tw-coin_collector"
SCHEMA_VERSION = "vpsc.textworld.coin_collector.v1"
SPLITS = ("train", "valid", "test")
_SPLIT_ORDER = {split: index for index, split in enumerate(SPLITS)}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CoinCollectorDatasetError(RuntimeError):
    """Base error for Coin Collector extraction and serialization."""


class ManifestValidationError(CoinCollectorDatasetError):
    """Raised when deterministic generation inputs are invalid."""


class SeedLeakageError(ManifestValidationError):
    """Raised when one game seed occurs in more than one data split."""


class WalkthroughValidationError(CoinCollectorDatasetError):
    """Raised when the official walkthrough does not produce a win."""


class DatasetSerializationError(CoinCollectorDatasetError):
    """Raised when a deterministic dataset artifact cannot be written."""


def _validate_level(level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int):
        raise ManifestValidationError("Coin Collector level must be an integer.")
    if level < 1 or level > 300:
        raise ManifestValidationError(
            "TextWorld Coin Collector level must be within [1, 300]."
        )

    # TextWorld 1.7 removed Glulx and rejects generated games over 100 rooms.
    quest_length = (level - 1) % 100 + 1
    room_multiplier = (level - 1) // 100 + 1
    if quest_length * room_multiplier > 100:
        raise ManifestValidationError(
            f"Coin Collector level {level} expands to "
            f"{quest_length * room_multiplier} rooms, but TextWorld 1.7's "
            "official Z-machine compiler supports at most 100."
        )


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestValidationError("Coin Collector seed must be an integer.")
    if seed < 0 or seed > 2**32 - 1:
        raise ManifestValidationError(
            "Coin Collector seed must be within NumPy RandomState's "
            "[0, 2**32 - 1] range."
        )


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ManifestValidationError(
            f"Unknown split {split!r}; expected one of {SPLITS}."
        )


@dataclass(frozen=True)
class CoinCollectorGameSpec:
    """One reproducible official game-generation unit."""

    seed: int
    level: int
    split: str
    game_file: str

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_level(self.level)
        _validate_split(self.split)
        if Path(self.game_file).suffix.lower() != ".z8":
            raise ManifestValidationError(
                "Coin Collector game_file must have the official .z8 extension."
            )

    @property
    def generation_command(self) -> Tuple[str, ...]:
        """Exact argv for the official TextWorld 1.7 ``tw-make`` CLI."""

        return (
            "tw-make",
            CHALLENGE,
            "--level",
            str(self.level),
            "--seed",
            str(self.seed),
            "--format",
            "z8",
            "--output",
            self.game_file,
            "--force",
            "--silent",
        )

    @property
    def generation_command_text(self) -> str:
        """Return a Windows-safe display form without executing the command."""

        return subprocess.list2cmdline(self.generation_command)

    def to_record(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "seed": self.seed,
            "level": self.level,
            "game_file": self.game_file,
            "generation_command": list(self.generation_command),
        }


@dataclass(frozen=True)
class CoinCollectorManifest:
    """Seed-disjoint train/valid/test generation manifest."""

    level: int
    games: Tuple[CoinCollectorGameSpec, ...]

    def __post_init__(self) -> None:
        _validate_level(self.level)
        if not self.games:
            raise ManifestValidationError("A manifest must contain at least one game.")
        for game in self.games:
            if game.level != self.level:
                raise ManifestValidationError(
                    "Every game in a fixed-level manifest must use manifest.level."
                )
        _assert_seed_disjoint(self.games)
        present = {game.split for game in self.games}
        missing = set(SPLITS) - present
        if missing:
            raise ManifestValidationError(
                "A comparison manifest needs non-empty train, valid, and test "
                f"splits; missing {sorted(missing)}."
            )

    def to_record(self) -> Dict[str, Any]:
        by_split = {
            split: [
                game.to_record()
                for game in self.games
                if game.split == split
            ]
            for split in SPLITS
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "challenge": CHALLENGE,
            "level": self.level,
            "split_key": "game_seed",
            "splits": by_split,
        }


def _seed_and_split(item: Any) -> Tuple[int, str]:
    try:
        seed = int(item.seed)
        split = str(item.split)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ManifestValidationError(
            "Split validation requires objects with integer seed and string split."
        ) from exc
    _validate_seed(seed)
    _validate_split(split)
    return seed, split


def _assert_seed_disjoint(items: Iterable[Any]) -> None:
    owner: Dict[int, str] = {}
    for item in items:
        seed, split = _seed_and_split(item)
        previous = owner.get(seed)
        if previous is not None and previous != split:
            raise SeedLeakageError(
                f"Game seed {seed} leaks across {previous!r} and {split!r}."
            )
        owner[seed] = split


def build_coin_collector_manifest(
    output_dir: Any,
    *,
    level: int,
    train_seeds: Iterable[int],
    valid_seeds: Iterable[int],
    test_seeds: Iterable[int],
) -> CoinCollectorManifest:
    """Build a deterministic manifest; no games are generated implicitly."""

    _validate_level(level)
    root = Path(output_dir).expanduser().resolve()
    seed_groups = {
        "train": tuple(train_seeds),
        "valid": tuple(valid_seeds),
        "test": tuple(test_seeds),
    }
    games: List[CoinCollectorGameSpec] = []
    for split in SPLITS:
        seeds = seed_groups[split]
        if not seeds:
            raise ManifestValidationError(f"{split} seeds must not be empty.")
        if len(set(seeds)) != len(seeds):
            raise ManifestValidationError(
                f"Duplicate game seed inside {split!r}: {seeds!r}."
            )
        for seed in sorted(seeds):
            _validate_seed(seed)
            filename = f"tw_coin_collector_l{level:03d}_s{seed:010d}.z8"
            games.append(
                CoinCollectorGameSpec(
                    seed=seed,
                    level=level,
                    split=split,
                    game_file=str(root / filename),
                )
            )
    return CoinCollectorManifest(level=level, games=tuple(games))


@dataclass(frozen=True)
class CounterfactualRecord:
    """One transition produced by a real ``Environment.copy`` branch."""

    action: str
    next_obs: str
    reward: float
    done: bool
    won: bool
    lost: bool
    admissible_actions_after: Tuple[str, ...]

    def to_record(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "next_obs": self.next_obs,
            "reward": self.reward,
            "done": self.done,
            "won": self.won,
            "lost": self.lost,
            "admissible_actions_after": list(self.admissible_actions_after),
        }


@dataclass(frozen=True)
class CoinCollectorStep:
    """One factual transition plus cloned counterfactual branches."""

    step: int
    observation: str
    admissible_actions: Tuple[str, ...]
    action: str
    next_obs: str
    reward: float
    done: bool
    counterfactuals: Tuple[CounterfactualRecord, ...]

    def to_record(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "observation": self.observation,
            "admissible_actions": list(self.admissible_actions),
            "action": self.action,
            "next_obs": self.next_obs,
            "reward": self.reward,
            "done": self.done,
            "counterfactuals": [
                counterfactual.to_record()
                for counterfactual in self.counterfactuals
            ],
        }


@dataclass(frozen=True)
class CoinCollectorEpisode:
    """Verified winning rollout from an official generated game."""

    split: str
    seed: int
    level: int
    game_file: str
    game_sha256: str
    game_uuid: str
    objective: Optional[str]
    initial_observation: str
    walkthrough: Tuple[str, ...]
    steps: Tuple[CoinCollectorStep, ...]
    won: bool
    generation_command: Tuple[str, ...]

    @property
    def return_(self) -> float:
        return float(sum(step.reward for step in self.steps))

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "challenge": CHALLENGE,
            "split": self.split,
            "seed": self.seed,
            "level": self.level,
            "game_file": self.game_file,
            "game_sha256": self.game_sha256,
            "game_uuid": self.game_uuid,
            "generation_command": list(self.generation_command),
            "objective": self.objective,
            "initial_observation": self.initial_observation,
            "walkthrough": list(self.walkthrough),
            "return": self.return_,
            "won": self.won,
            "steps": [step.to_record() for step in self.steps],
        }


@dataclass(frozen=True)
class DatasetFile:
    """Content digest returned after a deterministic file write."""

    path: str
    sha256: str
    size_bytes: int


def _state_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(key, default)
    getter = getattr(state, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return state[key]
    except (KeyError, TypeError, IndexError, AttributeError):
        return default


def walkthrough_from_state(state: Any) -> Tuple[str, ...]:
    """Read the requested official ``GameState['extra.walkthrough']`` value."""

    walkthrough = _state_value(state, "extra.walkthrough")
    if not isinstance(walkthrough, (list, tuple)) or not walkthrough:
        raise WalkthroughValidationError(
            "GameState has no non-empty 'extra.walkthrough'. Open the game "
            "with EnvInfos(extras=['walkthrough']); no plan was synthesized."
        )
    actions: List[str] = []
    for action in walkthrough:
        if not isinstance(action, str) or not action.strip():
            raise WalkthroughValidationError(
                "extra.walkthrough must contain only non-empty action strings."
            )
        actions.append(action)
    return tuple(actions)


def _normal_actions(actions: Iterable[str]) -> Tuple[str, ...]:
    normalized = {
        action
        for action in actions
        if isinstance(action, str) and action.strip()
    }
    return tuple(sorted(normalized))


def _candidate_actions(
    admissible: Tuple[str, ...],
    factual_action: str,
    limit: Optional[int],
) -> Tuple[str, ...]:
    if factual_action not in admissible:
        raise WalkthroughValidationError(
            f"Official walkthrough action {factual_action!r} is not in the "
            f"current admissible actions {admissible!r}."
        )
    if not admissible:
        raise WalkthroughValidationError(
            "No admissible actions are available for counterfactual branching."
        )
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ManifestValidationError(
                "counterfactual_limit must be None or a positive integer."
            )

    # Prefer truly alternative actions. If the state has no alternative, the
    # factual action is still evaluated on a clone to preserve branch evidence.
    ordered = tuple(
        action for action in admissible if action != factual_action
    ) + (factual_action,)
    return ordered if limit is None else ordered[:limit]


def _counterfactual_record(transition: WorldTransition) -> CounterfactualRecord:
    after = transition.info.get("admissible_commands", ()) or ()
    return CounterfactualRecord(
        action=str(transition.action),
        next_obs=str(transition.next_observation),
        reward=float(transition.reward),
        done=bool(transition.done),
        won=bool(transition.info.get("won", False)),
        lost=bool(transition.info.get("lost", False)),
        admissible_actions_after=_normal_actions(after),
    )


def collect_coin_collector_from_adapter(
    adapter: TextWorldAdapter,
    spec: CoinCollectorGameSpec,
    *,
    game_sha256: str,
    counterfactual_limit: Optional[int] = None,
) -> CoinCollectorEpisode:
    """Collect from an already-open core adapter using only official API calls.

    This seam exists for contract tests and for callers that already manage an
    official TextWorld interpreter. The production file entry point is
    :func:`collect_coin_collector_game` below.
    """

    if not _SHA256_PATTERN.fullmatch(game_sha256):
        raise ManifestValidationError(
            "game_sha256 must be a lowercase 64-character SHA256 digest."
        )

    initial_observation = adapter.reset()
    walkthrough = walkthrough_from_state(adapter.state)
    description = _state_value(adapter.state, "extra.desc")
    game_uuid = _state_value(adapter.state, "extra.uuid")
    if description != "Coin Collector" or not (
        isinstance(game_uuid, str) and game_uuid.startswith(f"{CHALLENGE}-")
    ):
        raise WalkthroughValidationError(
            "The opened game is not an official tw-coin_collector artifact: "
            "expected extra.desc='Coin Collector' and a tw-coin_collector UUID."
        )

    steps: List[CoinCollectorStep] = []
    for index, action in enumerate(walkthrough):
        admissible = _normal_actions(adapter.admissible_actions)
        candidates = _candidate_actions(
            admissible, action, counterfactual_limit
        )
        history_size = len(adapter.transitions)
        branches = adapter.counterfactual_candidates(candidates)
        if len(adapter.transitions) != history_size:
            raise WalkthroughValidationError(
                "Counterfactual evaluation mutated the factual transition history."
            )
        counterfactuals = tuple(
            _counterfactual_record(branch) for branch in branches
        )
        if not counterfactuals:
            raise WalkthroughValidationError(
                "At least one Environment.copy counterfactual is required per step."
            )

        transition = adapter.step(action)
        steps.append(
            CoinCollectorStep(
                step=index,
                observation=str(transition.observation),
                admissible_actions=admissible,
                action=action,
                next_obs=str(transition.next_observation),
                reward=float(transition.reward),
                done=bool(transition.done),
                counterfactuals=counterfactuals,
            )
        )
        final_action = index == len(walkthrough) - 1
        if transition.done and not final_action:
            raise WalkthroughValidationError(
                "The game terminated before the official walkthrough ended."
            )
        if final_action and not transition.done:
            raise WalkthroughValidationError(
                "The official walkthrough ended without a terminal transition."
            )

    won = bool(_state_value(adapter.state, "won", False))
    if not won:
        raise WalkthroughValidationError(
            "The official walkthrough completed but GameState.won is false."
        )

    return CoinCollectorEpisode(
        split=spec.split,
        seed=spec.seed,
        level=spec.level,
        game_file=spec.game_file,
        game_sha256=game_sha256,
        game_uuid=game_uuid,
        objective=adapter.objective,
        initial_observation=initial_observation,
        walkthrough=walkthrough,
        steps=tuple(steps),
        won=won,
        generation_command=spec.generation_command,
    )


def _require_real_z8(game_file: Any) -> Path:
    path = Path(game_file).expanduser()
    if path.suffix.lower() != ".z8":
        raise CoinCollectorDatasetError(
            f"Production collection requires a compiled .z8 file, got {path}."
        )
    if not path.is_file():
        raise CoinCollectorDatasetError(
            f"Compiled Coin Collector game does not exist: {path}. Run the "
            "manifest's official tw-make command first."
        )
    if path.stat().st_size == 0:
        raise CoinCollectorDatasetError(
            f"Compiled Coin Collector game is empty: {path}."
        )
    return path.resolve()


def collect_coin_collector_game(
    spec: CoinCollectorGameSpec,
    *,
    counterfactual_limit: Optional[int] = None,
    allow_unsupported_platform: bool = False,
) -> CoinCollectorEpisode:
    """Open a real ``.z8`` and execute its official winning walkthrough."""

    path = _require_real_z8(spec.game_file)
    resolved_spec = CoinCollectorGameSpec(
        seed=spec.seed,
        level=spec.level,
        split=spec.split,
        game_file=str(path),
    )
    adapter = open_textworld(
        path,
        allow_unsupported_platform=allow_unsupported_platform,
        extras=("walkthrough", "desc", "uuid"),
    )
    try:
        return collect_coin_collector_from_adapter(
            adapter,
            resolved_spec,
            game_sha256=file_sha256(path),
            counterfactual_limit=counterfactual_limit,
        )
    finally:
        adapter.close()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DatasetSerializationError(
            f"Dataset value is not canonical-JSON serializable: {exc}"
        ) from exc


def episode_to_json_line(episode: CoinCollectorEpisode) -> str:
    """Serialize exactly one episode as canonical JSONL content."""

    return _canonical_json(episode.to_record())


def _token_event_line(step: int, channel: str, value: Any) -> str:
    return f"{step:06d}\t<|{channel}|>\t{_canonical_json(value)}"


def episode_to_token_event_text(episode: CoinCollectorEpisode) -> str:
    """Serialize an episode as deterministic LLM-oriented event text."""

    header = {
        "schema_version": SCHEMA_VERSION,
        "challenge": CHALLENGE,
        "split": episode.split,
        "seed": episode.seed,
        "level": episode.level,
        "game_sha256": episode.game_sha256,
        "game_uuid": episode.game_uuid,
    }
    lines = [_token_event_line(0, "episode", header)]
    if episode.objective is not None:
        lines.append(_token_event_line(0, "goal", episode.objective))
    lines.append(
        _token_event_line(0, "observation", episode.initial_observation)
    )
    for step in episode.steps:
        lines.append(
            _token_event_line(
                step.step, "admissible_actions", list(step.admissible_actions)
            )
        )
        for counterfactual in step.counterfactuals:
            lines.append(
                _token_event_line(
                    step.step, "counterfactual", counterfactual.to_record()
                )
            )
        lines.extend(
            (
                _token_event_line(step.step, "action", step.action),
                _token_event_line(step.step + 1, "observation", step.next_obs),
                _token_event_line(step.step + 1, "reward", step.reward),
                _token_event_line(step.step + 1, "done", step.done),
            )
        )
    lines.append(_token_event_line(len(episode.steps), "won", episode.won))
    lines.append(
        _token_event_line(len(episode.steps), "end_episode", True)
    )
    return "\n".join(lines) + "\n"


def _episode_sort_key(episode: CoinCollectorEpisode) -> Tuple[Any, ...]:
    return (
        _SPLIT_ORDER[episode.split],
        episode.seed,
        episode.level,
        episode.game_sha256,
    )


def _ordered_episodes(
    episodes: Iterable[CoinCollectorEpisode],
) -> Tuple[CoinCollectorEpisode, ...]:
    result = tuple(episodes)
    if not result:
        raise DatasetSerializationError("Cannot write an empty dataset.")
    _assert_seed_disjoint(result)
    return tuple(sorted(result, key=_episode_sort_key))


def file_sha256(path: Any) -> str:
    """Hash file bytes in chunks and return a lowercase SHA256 digest."""

    source = Path(path)
    if not source.is_file():
        raise DatasetSerializationError(f"Cannot hash missing file: {source}.")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Any, content: bytes) -> DatasetFile:
    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    except OSError as exc:
        raise DatasetSerializationError(
            f"Unable to write deterministic dataset file {target}: {exc}"
        ) from exc
    return DatasetFile(
        path=str(target.resolve()),
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def write_manifest_json(manifest: CoinCollectorManifest, path: Any) -> DatasetFile:
    """Write a canonical UTF-8 generation manifest and return its digest."""

    content = (_canonical_json(manifest.to_record()) + "\n").encode("utf-8")
    return _write_bytes(path, content)


def write_episode_jsonl(
    episodes: Iterable[CoinCollectorEpisode], path: Any
) -> DatasetFile:
    """Write seed-disjoint episodes as deterministic canonical JSONL."""

    ordered = _ordered_episodes(episodes)
    content = (
        "\n".join(episode_to_json_line(episode) for episode in ordered) + "\n"
    ).encode("utf-8")
    return _write_bytes(path, content)


def write_token_event_text(
    episodes: Iterable[CoinCollectorEpisode], path: Any
) -> DatasetFile:
    """Write a deterministic token-event corpus and return its file digest."""

    ordered = _ordered_episodes(episodes)
    content = "".join(
        episode_to_token_event_text(episode) for episode in ordered
    ).encode("utf-8")
    return _write_bytes(path, content)


__all__ = [
    "CHALLENGE",
    "SCHEMA_VERSION",
    "SPLITS",
    "CoinCollectorDatasetError",
    "CoinCollectorEpisode",
    "CoinCollectorGameSpec",
    "CoinCollectorManifest",
    "CoinCollectorStep",
    "CounterfactualRecord",
    "DatasetFile",
    "DatasetSerializationError",
    "ManifestValidationError",
    "SeedLeakageError",
    "WalkthroughValidationError",
    "build_coin_collector_manifest",
    "collect_coin_collector_from_adapter",
    "collect_coin_collector_game",
    "episode_to_json_line",
    "episode_to_token_event_text",
    "file_sha256",
    "walkthrough_from_state",
    "write_episode_jsonl",
    "write_manifest_json",
    "write_token_event_text",
]
