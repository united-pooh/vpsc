"""Strict, episode-aware corpus for the official HomeGrid dynamics pilot.

The loader consumes only artifacts emitted by ``e2_homegrid_dataset.py``.  It
closes the provenance loop (summary -> manifest/transitions), rebuilds all
reported counts, enforces the frozen 32/8/8 seed protocol, and constructs a
language-token vocabulary from the training split only.  There is no generated,
synthetic, partially valid, or empty-data fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple, Union

from .homegrid_dataset import (
    ACTION_RNG_ALGORITHM,
    ACTION_SEED_OFFSET,
    DEFAULT_MAX_STEPS,
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALID_SEEDS,
    HOMEGRID_DYNAMICS_ENV_ID,
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
    SPLITS,
    VISUAL_TOKEN_COUNT,
    VISUAL_VOCAB_SIZE,
)
from .wikitext import file_sha256


PathLike = Union[str, Path]
SUMMARY_FILENAME = "summary.json"
MANIFEST_FILENAME = "manifest.json"
TRANSITIONS_FILENAME = "transitions.jsonl"
RUNNER_SCHEMA_VERSION = "vpsc.e2_homegrid_gate_m.v1"
LANGUAGE_SPECIAL_TOKENS = ("<pad>", "<unk>")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_SEEDS = {
    "train": DEFAULT_TRAIN_SEEDS,
    "valid": DEFAULT_VALID_SEEDS,
    "test": DEFAULT_TEST_SEEDS,
}
_REWARD_TO_CLASS = {0.0: 0, 0.5: 1, 1.0: 2}


class HomeGridCorpusError(RuntimeError):
    """Base class for fail-closed HomeGrid corpus errors."""


class HomeGridCorpusNotFoundError(FileNotFoundError, HomeGridCorpusError):
    """A mandatory source artifact is absent."""


class HomeGridCorpusFormatError(HomeGridCorpusError):
    """An artifact violates the frozen schema or provenance contract."""


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=reject_constant)
    except FileNotFoundError as error:
        raise HomeGridCorpusNotFoundError(
            f"required HomeGrid artifact is missing: {path}; no fallback was used"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise HomeGridCorpusFormatError(f"cannot parse strict JSON {path}: {error}") from error


def _require_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise HomeGridCorpusFormatError(f"{label} must be a JSON object")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HomeGridCorpusFormatError(f"{label} must be an integer")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise HomeGridCorpusFormatError(f"{label} must be boolean")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HomeGridCorpusFormatError(f"{label} must be a lowercase SHA-256")
    return value


def _verify_artifact_record(record: Any, actual: Path, label: str) -> None:
    item = _require_dict(record, label)
    recorded_path = item.get("path")
    if not isinstance(recorded_path, str) or Path(recorded_path).name != actual.name:
        raise HomeGridCorpusFormatError(f"{label} path does not name {actual.name}")
    recorded_sha = _require_sha(item.get("sha256"), f"{label}.sha256")
    recorded_size = _require_int(item.get("size_bytes"), f"{label}.size_bytes")
    try:
        actual_size = actual.stat().st_size
    except FileNotFoundError as error:
        raise HomeGridCorpusNotFoundError(
            f"required HomeGrid artifact is missing: {actual}; no fallback was used"
        ) from error
    if actual_size != recorded_size or file_sha256(actual) != recorded_sha:
        raise HomeGridCorpusFormatError(f"{label} SHA/size does not match {actual}")


@dataclass(frozen=True)
class HomeGridLanguageVocabulary:
    """Raw official language token IDs mapped to train-only local IDs."""

    raw_to_local: Mapping[int, int]
    local_to_raw: Tuple[int | None, ...]
    fingerprint: str

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.local_to_raw)

    def encode(self, raw_token: int) -> int:
        return self.raw_to_local.get(raw_token, self.unk_id)

    @classmethod
    def build(cls, raw_tokens: Sequence[int]) -> "HomeGridLanguageVocabulary":
        unique = tuple(sorted(set(raw_tokens)))
        if not unique:
            raise HomeGridCorpusFormatError("training split has no language tokens")
        mapping = {raw: index + len(LANGUAGE_SPECIAL_TOKENS) for index, raw in enumerate(unique)}
        fingerprint = hashlib.sha256(
            json.dumps(unique, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        return cls(
            raw_to_local=MappingProxyType(mapping),
            local_to_raw=(None, None, *unique),
            fingerprint=fingerprint,
        )


@dataclass(frozen=True)
class HomeGridTransition:
    step: int
    visual_tokens: Tuple[int, ...]
    next_visual_tokens: Tuple[int, ...]
    language_raw: int
    next_language_raw: int
    language_id: int
    next_language_id: int
    action: int
    is_read_step: bool
    next_is_read_step: bool
    reward: float
    reward_class: int
    done: bool
    changed_patch_count: int


@dataclass(frozen=True)
class HomeGridEpisode:
    split: str
    seed: int
    episode_id: str
    transitions: Tuple[HomeGridTransition, ...]


@dataclass(frozen=True)
class HomeGridChunk:
    """One batch-one temporal chunk; tuple fields have leading time axis."""

    episode_seed: int
    epoch: int
    offset: int
    reset_state: bool
    visual_tokens: Tuple[Tuple[int, ...], ...]
    next_visual_tokens: Tuple[Tuple[int, ...], ...]
    language_ids: Tuple[int, ...]
    next_language_ids: Tuple[int, ...]
    actions: Tuple[int, ...]
    read_flags: Tuple[bool, ...]
    next_read_flags: Tuple[bool, ...]
    reward_classes: Tuple[int, ...]
    done_targets: Tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class _RawTransition:
    split: str
    seed: int
    episode_id: str
    step: int
    visual_tokens: Tuple[int, ...]
    next_visual_tokens: Tuple[int, ...]
    language_raw: int
    next_language_raw: int
    action: int
    is_read_step: bool
    next_is_read_step: bool
    reward: float
    reward_class: int
    terminated: bool
    truncated: bool
    done: bool
    changed_patch_count: int


@dataclass(frozen=True)
class HomeGridCorpus:
    root: Path
    vocabulary: HomeGridLanguageVocabulary
    _episodes: Mapping[str, Tuple[HomeGridEpisode, ...]]
    _metadata: Mapping[str, Mapping[str, Any]]
    train_visual_frequency: Tuple[int, ...]

    def iter_episodes(self, split: str) -> Iterator[HomeGridEpisode]:
        _validate_split(split)
        yield from self._episodes[split]

    def iter_chunks(
        self,
        split: str,
        sequence_length: int = 32,
        *,
        epochs: int = 1,
    ) -> Iterator[HomeGridChunk]:
        _validate_split(split)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        for epoch in range(epochs):
            for episode in self._episodes[split]:
                for offset in range(0, len(episode.transitions), sequence_length):
                    rows = episode.transitions[offset : offset + sequence_length]
                    yield HomeGridChunk(
                        episode_seed=episode.seed,
                        epoch=epoch,
                        offset=offset,
                        reset_state=offset == 0,
                        visual_tokens=tuple(row.visual_tokens for row in rows),
                        next_visual_tokens=tuple(row.next_visual_tokens for row in rows),
                        language_ids=tuple(row.language_id for row in rows),
                        next_language_ids=tuple(row.next_language_id for row in rows),
                        actions=tuple(row.action for row in rows),
                        read_flags=tuple(row.is_read_step for row in rows),
                        next_read_flags=tuple(row.next_is_read_step for row in rows),
                        reward_classes=tuple(row.reward_class for row in rows),
                        done_targets=tuple(int(row.done) for row in rows),
                    )

    def split_metadata(self, split: str) -> Dict[str, Any]:
        _validate_split(split)
        return dict(self._metadata[split])

    @property
    def most_frequent_visual_token(self) -> int:
        return max(range(VISUAL_VOCAB_SIZE), key=self.train_visual_frequency.__getitem__)

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": "vpsc.homegrid_corpus.v1",
            "root": str(self.root),
            "language_vocabulary": {
                "size": len(self.vocabulary),
                "built_from": "train_current_and_next_only",
                "fingerprint_sha256": self.vocabulary.fingerprint,
                "special_tokens": list(LANGUAGE_SPECIAL_TOKENS),
            },
            "train_visual_frequency": list(self.train_visual_frequency),
            "most_frequent_visual_token": self.most_frequent_visual_token,
            "splits": {split: dict(self._metadata[split]) for split in SPLITS},
        }


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS!r}, got {split!r}")


def _visual_tokens(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != VISUAL_TOKEN_COUNT:
        raise HomeGridCorpusFormatError(
            f"{label} must contain exactly {VISUAL_TOKEN_COUNT} visual tokens"
        )
    result = tuple(_require_int(token, f"{label} token") for token in value)
    if any(token < 0 or token >= VISUAL_VOCAB_SIZE for token in result):
        raise HomeGridCorpusFormatError(f"{label} token is outside 0..63")
    return result


def _parse_transition(value: Any, split: str, line_number: int) -> _RawTransition:
    row = _require_dict(value, f"{split} transition line {line_number}")
    label = f"{split} transition line {line_number}"
    if row.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise HomeGridCorpusFormatError(f"{label} schema version mismatch")
    if row.get("split") != split or row.get("env_id") != HOMEGRID_DYNAMICS_ENV_ID:
        raise HomeGridCorpusFormatError(f"{label} split/env mismatch")
    seed = _require_int(row.get("episode_seed"), f"{label}.episode_seed")
    episode_id = row.get("episode_id")
    expected_id = f"{HOMEGRID_DYNAMICS_ENV_ID}:{split}:{seed}"
    if episode_id != expected_id:
        raise HomeGridCorpusFormatError(f"{label} episode_id mismatch")
    step = _require_int(row.get("step"), f"{label}.step")
    action = _require_int(row.get("action"), f"{label}.action")
    if action not in range(10):
        raise HomeGridCorpusFormatError(f"{label} action is outside 0..9")
    if row.get("action_rng_algorithm") != ACTION_RNG_ALGORITHM:
        raise HomeGridCorpusFormatError(f"{label} action RNG algorithm drifted")
    if row.get("action_seed") != seed + ACTION_SEED_OFFSET:
        raise HomeGridCorpusFormatError(f"{label} action seed mismatch")
    if row.get("max_steps") != DEFAULT_MAX_STEPS:
        raise HomeGridCorpusFormatError(f"{label} max_steps drifted")
    visual = _visual_tokens(row.get("visual_tokens"), f"{label}.visual_tokens")
    next_visual = _visual_tokens(
        row.get("next_visual_tokens"), f"{label}.next_visual_tokens"
    )
    changed = sum(left != right for left, right in zip(visual, next_visual))
    if row.get("changed_patch_count") != changed:
        raise HomeGridCorpusFormatError(f"{label} changed_patch_count mismatch")
    for field in ("raw_image_sha256", "next_raw_image_sha256"):
        _require_sha(row.get(field), f"{label}.{field}")
    language = _require_int(row.get("language_token"), f"{label}.language_token")
    next_language = _require_int(
        row.get("next_language_token"), f"{label}.next_language_token"
    )
    if language < 0 or next_language < 0:
        raise HomeGridCorpusFormatError(f"{label} language token cannot be negative")
    read = _require_bool(row.get("is_read_step"), f"{label}.is_read_step")
    next_read = _require_bool(
        row.get("next_is_read_step"), f"{label}.next_is_read_step"
    )
    if row.get("phase") != ("read" if read else "action"):
        raise HomeGridCorpusFormatError(f"{label} phase mismatch")
    if row.get("next_phase") != ("read" if next_read else "action"):
        raise HomeGridCorpusFormatError(f"{label} next_phase mismatch")
    reward = row.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise HomeGridCorpusFormatError(f"{label} reward must be numeric")
    reward = float(reward)
    if reward not in _REWARD_TO_CLASS:
        raise HomeGridCorpusFormatError(
            f"{label} reward {reward!r} is outside frozen classes 0/0.5/1"
        )
    terminated = _require_bool(row.get("terminated"), f"{label}.terminated")
    truncated = _require_bool(row.get("truncated"), f"{label}.truncated")
    done = _require_bool(row.get("done"), f"{label}.done")
    if done != (terminated or truncated):
        raise HomeGridCorpusFormatError(f"{label} done flag is inconsistent")
    return _RawTransition(
        split=split,
        seed=seed,
        episode_id=episode_id,
        step=step,
        visual_tokens=visual,
        next_visual_tokens=next_visual,
        language_raw=language,
        next_language_raw=next_language,
        action=action,
        is_read_step=read,
        next_is_read_step=next_read,
        reward=reward,
        reward_class=_REWARD_TO_CLASS[reward],
        terminated=terminated,
        truncated=truncated,
        done=done,
        changed_patch_count=changed,
    )


def _read_transition_lines(path: Path, split: str) -> Tuple[_RawTransition, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise HomeGridCorpusNotFoundError(
            f"required HomeGrid artifact is missing: {path}; no fallback was used"
        ) from error
    except (OSError, UnicodeError) as error:
        raise HomeGridCorpusFormatError(f"cannot read {path}: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        raise HomeGridCorpusFormatError(f"transition JSONL is empty or contains blanks: {path}")
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-standard JSON constant {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise HomeGridCorpusFormatError(
                f"invalid transition JSON in {path} line {line_number}: {error}"
            ) from error
        rows.append(_parse_transition(value, split, line_number))
    return tuple(rows)


def _counts(rows: Sequence[_RawTransition], episode_count: int) -> Dict[str, Any]:
    grouped: Dict[int, list[_RawTransition]] = {}
    for row in rows:
        grouped.setdefault(row.seed, []).append(row)
    cutoff_count = sum(
        len(episode_rows) == DEFAULT_MAX_STEPS and not episode_rows[-1].done
        for episode_rows in grouped.values()
    )
    return {
        "episode_count": episode_count,
        "transition_count": len(rows),
        "read_step_count": sum(row.is_read_step for row in rows),
        "action_step_count": sum(not row.is_read_step for row in rows),
        "changed_patch_count": sum(row.changed_patch_count for row in rows),
        "transitions_with_changed_patches": sum(row.changed_patch_count > 0 for row in rows),
        "nonzero_reward_count": sum(row.reward != 0.0 for row in rows),
        "positive_reward_count": sum(row.reward > 0.0 for row in rows),
        "negative_reward_count": sum(row.reward < 0.0 for row in rows),
        "reward_sum": sum(row.reward for row in rows),
        "terminated_count": sum(row.terminated for row in rows),
        "truncated_count": sum(row.truncated for row in rows),
        "done_count": sum(row.done for row in rows),
        "collector_cutoff_episode_count": cutoff_count,
    }


def _validate_protocol(value: Mapping[str, Any], label: str) -> None:
    if value.get("env_id") != HOMEGRID_DYNAMICS_ENV_ID:
        raise HomeGridCorpusFormatError(f"{label} env_id drifted")
    if value.get("max_steps") != DEFAULT_MAX_STEPS:
        raise HomeGridCorpusFormatError(f"{label} max_steps drifted")
    for split in SPLITS:
        if value.get(f"{split}_seeds") != list(_EXPECTED_SEEDS[split]):
            raise HomeGridCorpusFormatError(f"{label} {split} seeds drifted")
    action = _require_dict(value.get("action_sampling"), f"{label}.action_sampling")
    if action.get("algorithm") != ACTION_RNG_ALGORITHM:
        raise HomeGridCorpusFormatError(f"{label} action sampling drifted")


def _validate_split_sources(root: Path, split: str) -> Tuple[Tuple[_RawTransition, ...], Dict[str, Any]]:
    split_root = root / split
    summary_path = split_root / SUMMARY_FILENAME
    manifest_path = split_root / MANIFEST_FILENAME
    transitions_path = split_root / TRANSITIONS_FILENAME
    summary = _require_dict(_strict_json(summary_path), f"{split} summary")
    manifest = _require_dict(_strict_json(manifest_path), f"{split} manifest")

    for source, label in ((summary, "summary"), (manifest, "manifest")):
        if source.get("split") != split:
            raise HomeGridCorpusFormatError(f"{split} {label} split mismatch")
        if source.get("env_id") != HOMEGRID_DYNAMICS_ENV_ID:
            raise HomeGridCorpusFormatError(f"{split} {label} env mismatch")
        versions = _require_dict(source.get("versions"), f"{split} {label}.versions")
        if versions.get("homegrid") != "0.1.1":
            raise HomeGridCorpusFormatError(f"{split} requires official HomeGrid 0.1.1")
        if versions.get("dataset_schema") != DATASET_SCHEMA_VERSION:
            raise HomeGridCorpusFormatError(f"{split} dataset schema mismatch")
        if source.get("uses_frozen_default_protocol") is not True:
            raise HomeGridCorpusFormatError(f"{split} {label} is not the frozen protocol")
        _validate_protocol(
            _require_dict(source.get("actual_protocol"), f"{split} {label}.actual_protocol"),
            f"{split} {label}.actual_protocol",
        )

    if summary.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise HomeGridCorpusFormatError(f"{split} summary schema mismatch")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise HomeGridCorpusFormatError(f"{split} manifest schema mismatch")
    if manifest.get("runner_schema_version") != RUNNER_SCHEMA_VERSION:
        raise HomeGridCorpusFormatError(f"{split} runner schema mismatch")
    if summary.get("seeds") != list(_EXPECTED_SEEDS[split]):
        raise HomeGridCorpusFormatError(f"{split} summary seeds mismatch")
    artifacts = _require_dict(summary.get("artifacts"), f"{split} summary.artifacts")
    _verify_artifact_record(artifacts.get("manifest"), manifest_path, f"{split} manifest")
    _verify_artifact_record(
        artifacts.get("transitions_jsonl"), transitions_path, f"{split} transitions"
    )
    manifest_artifacts = _require_dict(
        manifest.get("artifacts"), f"{split} manifest.artifacts"
    )
    _verify_artifact_record(
        manifest_artifacts.get("transitions_jsonl"),
        transitions_path,
        f"{split} manifest transitions",
    )

    rows = _read_transition_lines(transitions_path, split)
    grouped: Dict[int, list[_RawTransition]] = {}
    for row in rows:
        grouped.setdefault(row.seed, []).append(row)
    if tuple(sorted(grouped)) != _EXPECTED_SEEDS[split]:
        raise HomeGridCorpusFormatError(f"{split} transition seeds mismatch")
    manifest_episodes = manifest.get("episodes")
    if not isinstance(manifest_episodes, list) or len(manifest_episodes) != len(grouped):
        raise HomeGridCorpusFormatError(f"{split} manifest episode count mismatch")
    by_seed = {}
    for item in manifest_episodes:
        episode = _require_dict(item, f"{split} manifest episode")
        seed = _require_int(episode.get("seed"), f"{split} manifest seed")
        if seed in by_seed:
            raise HomeGridCorpusFormatError(f"duplicate {split} manifest seed {seed}")
        by_seed[seed] = episode

    for seed in _EXPECTED_SEEDS[split]:
        episode_rows = grouped[seed]
        if not episode_rows or len(episode_rows) > DEFAULT_MAX_STEPS:
            raise HomeGridCorpusFormatError(
                f"{split} seed {seed} must contain 1..{DEFAULT_MAX_STEPS} transitions"
            )
        if [row.step for row in episode_rows] != list(range(len(episode_rows))):
            raise HomeGridCorpusFormatError(f"{split} seed {seed} step order mismatch")
        for left, right in zip(episode_rows, episode_rows[1:]):
            if left.done:
                raise HomeGridCorpusFormatError(f"{split} seed {seed} continues after done")
            if (
                left.next_visual_tokens != right.visual_tokens
                or left.next_language_raw != right.language_raw
                or left.next_is_read_step != right.is_read_step
            ):
                raise HomeGridCorpusFormatError(
                    f"{split} seed {seed} transition continuity mismatch"
                )
        if len(episode_rows) < DEFAULT_MAX_STEPS and not episode_rows[-1].done:
            raise HomeGridCorpusFormatError(
                f"{split} seed {seed} ended before max_steps without done"
            )
        record = by_seed.get(seed)
        if record is None:
            raise HomeGridCorpusFormatError(f"{split} manifest omits seed {seed}")
        if record.get("episode_id") != f"{HOMEGRID_DYNAMICS_ENV_ID}:{split}:{seed}":
            raise HomeGridCorpusFormatError(f"{split} manifest episode id mismatch")
        if record.get("action_seed") != seed + ACTION_SEED_OFFSET:
            raise HomeGridCorpusFormatError(f"{split} manifest action seed mismatch")
        if record.get("max_steps") != DEFAULT_MAX_STEPS:
            raise HomeGridCorpusFormatError(f"{split} manifest max_steps mismatch")
        expected_episode_counts = _counts(episode_rows, 1)
        if record.get("counts") != expected_episode_counts:
            raise HomeGridCorpusFormatError(f"{split} seed {seed} count mismatch")
        expected_cutoff = bool(expected_episode_counts["collector_cutoff_episode_count"])
        if record.get("collector_cutoff") is not expected_cutoff:
            raise HomeGridCorpusFormatError(f"{split} seed {seed} cutoff flag mismatch")

    observed_counts = _counts(rows, len(grouped))
    if summary.get("counts") != observed_counts:
        raise HomeGridCorpusFormatError(f"{split} summary counts do not match transitions")
    metadata = {
        "split": split,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "transitions_path": str(transitions_path.resolve()),
        "transitions_sha256": file_sha256(transitions_path),
        "episode_count": len(grouped),
        **observed_counts,
    }
    return rows, metadata


def load_homegrid_corpus(root: PathLike) -> HomeGridCorpus:
    """Load the exact preregistered official HomeGrid dynamics corpus."""

    resolved = Path(root).expanduser().resolve()
    raw_by_split: Dict[str, Tuple[_RawTransition, ...]] = {}
    metadata: Dict[str, Mapping[str, Any]] = {}
    all_seeds = set()
    for split in SPLITS:
        rows, split_metadata = _validate_split_sources(resolved, split)
        seeds = {row.seed for row in rows}
        if all_seeds.intersection(seeds):
            raise HomeGridCorpusFormatError("HomeGrid environment seed leaked across splits")
        all_seeds.update(seeds)
        raw_by_split[split] = rows
        metadata[split] = MappingProxyType(split_metadata)

    train_language = [
        token
        for row in raw_by_split["train"]
        for token in (row.language_raw, row.next_language_raw)
    ]
    vocabulary = HomeGridLanguageVocabulary.build(train_language)
    episodes_by_split: Dict[str, Tuple[HomeGridEpisode, ...]] = {}
    for split in SPLITS:
        grouped: Dict[int, list[_RawTransition]] = {}
        for row in raw_by_split[split]:
            grouped.setdefault(row.seed, []).append(row)
        episodes = []
        oov_current = 0
        oov_next = 0
        for seed in _EXPECTED_SEEDS[split]:
            transitions = []
            for row in grouped[seed]:
                language_id = vocabulary.encode(row.language_raw)
                next_language_id = vocabulary.encode(row.next_language_raw)
                oov_current += language_id == vocabulary.unk_id
                oov_next += next_language_id == vocabulary.unk_id
                transitions.append(
                    HomeGridTransition(
                        step=row.step,
                        visual_tokens=row.visual_tokens,
                        next_visual_tokens=row.next_visual_tokens,
                        language_raw=row.language_raw,
                        next_language_raw=row.next_language_raw,
                        language_id=language_id,
                        next_language_id=next_language_id,
                        action=row.action,
                        is_read_step=row.is_read_step,
                        next_is_read_step=row.next_is_read_step,
                        reward=row.reward,
                        reward_class=row.reward_class,
                        done=row.done,
                        changed_patch_count=row.changed_patch_count,
                    )
                )
            episodes.append(
                HomeGridEpisode(
                    split=split,
                    seed=seed,
                    episode_id=f"{HOMEGRID_DYNAMICS_ENV_ID}:{split}:{seed}",
                    transitions=tuple(transitions),
                )
            )
        split_metadata = dict(metadata[split])
        split_metadata["language_oov_current"] = oov_current
        split_metadata["language_oov_next"] = oov_next
        metadata[split] = MappingProxyType(split_metadata)
        episodes_by_split[split] = tuple(episodes)

    visual_frequency = Counter(
        token
        for episode in episodes_by_split["train"]
        for transition in episode.transitions
        for token in transition.next_visual_tokens
    )
    frequency_tuple = tuple(visual_frequency[index] for index in range(VISUAL_VOCAB_SIZE))
    if sum(frequency_tuple) != len(raw_by_split["train"]) * VISUAL_TOKEN_COUNT:
        raise HomeGridCorpusFormatError("training visual frequency count mismatch")
    return HomeGridCorpus(
        root=resolved,
        vocabulary=vocabulary,
        _episodes=MappingProxyType(episodes_by_split),
        _metadata=MappingProxyType(metadata),
        train_visual_frequency=frequency_tuple,
    )


__all__ = [
    "HomeGridChunk",
    "HomeGridCorpus",
    "HomeGridCorpusError",
    "HomeGridCorpusFormatError",
    "HomeGridCorpusNotFoundError",
    "HomeGridEpisode",
    "HomeGridLanguageVocabulary",
    "HomeGridTransition",
    "LANGUAGE_SPECIAL_TOKENS",
    "load_homegrid_corpus",
]
