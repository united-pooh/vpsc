"""Gate M datasets from the official ``homegrid-dynamics`` environment.

Production collection is intentionally narrow: it creates only the registered
official ``homegrid-dynamics`` environment, resets it through the existing
``HomeGridRecorder`` strict seed/compatibility path, and never substitutes a
fallback environment. Actions use an independent ``random.Random`` instance;
the environment's Python/NumPy RNG state is never used for policy sampling.

Raw RGB is never serialized. Each 96x96 uint8 RGB frame is represented by a
SHA256 over its C-order bytes and 144 visual tokens: 12x12 non-overlapping 8x8
patches, four mean-intensity bins per channel, combined as ``R*16 + G*4 + B``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .homegrid import (
    HOMEGRID_ACTIONS,
    HOMEGRID_SEED_MODE_COMPAT,
    HOMEGRID_SEED_MODE_GYM,
    HomeGridRecorder,
    make_homegrid_env,
)


SCHEMA_VERSION = "vpsc.homegrid_dynamics.gate_m.v1"
HOMEGRID_DYNAMICS_ENV_ID = "homegrid-dynamics"
SPLITS = ("train", "valid", "test")
_SPLIT_ORDER = {split: index for index, split in enumerate(SPLITS)}

FRAME_HEIGHT = 96
FRAME_WIDTH = 96
FRAME_CHANNELS = 3
PATCH_SIZE = 8
PATCH_ROWS = FRAME_HEIGHT // PATCH_SIZE
PATCH_COLUMNS = FRAME_WIDTH // PATCH_SIZE
VISUAL_TOKEN_COUNT = PATCH_ROWS * PATCH_COLUMNS
CHANNEL_LEVELS = 4
VISUAL_VOCAB_SIZE = CHANNEL_LEVELS**FRAME_CHANNELS
CHANNEL_BIN_WIDTH = 256 // CHANNEL_LEVELS

ACTION_RNG_ALGORITHM = "python_random.Random(seed + 1_000_003).randrange(10)"
ACTION_SEED_OFFSET = 1_000_003
DEFAULT_MAX_STEPS = 96
DEFAULT_TRAIN_SEEDS = tuple(range(2026071800, 2026071832))
DEFAULT_VALID_SEEDS = tuple(range(2026071900, 2026071908))
DEFAULT_TEST_SEEDS = tuple(range(2026072000, 2026072008))


class HomeGridDatasetError(RuntimeError):
    """Base error for official HomeGrid dataset extraction."""


class HomeGridManifestError(HomeGridDatasetError):
    """Raised when split seeds or fixed protocol fields are invalid."""


class HomeGridSeedLeakageError(HomeGridManifestError):
    """Raised when one environment seed occurs in multiple splits."""


class HomeGridFrameError(HomeGridDatasetError):
    """Raised when an official frame violates the frozen RGB contract."""


class HomeGridObservationError(HomeGridDatasetError):
    """Raised when language/read-phase observation fields are malformed."""


class HomeGridSerializationError(HomeGridDatasetError):
    """Raised when a canonical artifact cannot be written."""


def quantization_definition() -> Dict[str, Any]:
    """Return the complete frozen visual-token definition."""

    return {
        "input": {
            "shape": [FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS],
            "dtype": "uint8",
            "channel_order": "RGB",
            "raw_sha256_bytes": "C_order_RGB_bytes_only",
        },
        "patch": {
            "height": PATCH_SIZE,
            "width": PATCH_SIZE,
            "grid_rows": PATCH_ROWS,
            "grid_columns": PATCH_COLUMNS,
            "flatten_order": "row_major",
        },
        "channel_quantization": {
            "statistic": "exact_integer_sum_divided_by_64_pixel_count",
            "levels": CHANNEL_LEVELS,
            "mean_intervals": [[0, 64], [64, 128], [128, 192], [192, 256]],
            "interval_semantics": "left_closed_right_open_except_final",
        },
        "token_formula": "red_level * 16 + green_level * 4 + blue_level",
        "vocabulary_size": VISUAL_VOCAB_SIZE,
        "tokens_per_frame": VISUAL_TOKEN_COUNT,
    }


def action_sampling_definition() -> Dict[str, Any]:
    return {
        "algorithm": ACTION_RNG_ALGORITHM,
        "rng_isolation": "dedicated_random.Random_instance_per_episode",
        "episode_action_seed_formula": f"environment_seed + {ACTION_SEED_OFFSET}",
        "distribution": "discrete_uniform",
        "action_ids": list(sorted(HOMEGRID_ACTIONS)),
        "action_count": len(HOMEGRID_ACTIONS),
    }


def frozen_default_protocol() -> Dict[str, Any]:
    return {
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "max_steps": DEFAULT_MAX_STEPS,
        "train_seeds": list(DEFAULT_TRAIN_SEEDS),
        "valid_seeds": list(DEFAULT_VALID_SEEDS),
        "test_seeds": list(DEFAULT_TEST_SEEDS),
        "action_sampling": action_sampling_definition(),
    }


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise HomeGridManifestError("HomeGrid seeds must be integers.")
    if seed < 0 or seed > 2**32 - 1:
        raise HomeGridManifestError(
            "HomeGrid seeds must fit NumPy's audited compatibility range "
            "[0, 2**32 - 1]."
        )


def _validate_max_steps(max_steps: int) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise HomeGridManifestError("max_steps must be an integer.")
    if max_steps <= 0:
        raise HomeGridManifestError("max_steps must be positive.")


@dataclass(frozen=True)
class HomeGridEpisodeSpec:
    split: str
    seed: int
    max_steps: int = DEFAULT_MAX_STEPS
    env_id: str = HOMEGRID_DYNAMICS_ENV_ID

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise HomeGridManifestError(
                f"Unknown split {self.split!r}; expected one of {SPLITS}."
            )
        _validate_seed(self.seed)
        _validate_max_steps(self.max_steps)
        if self.env_id != HOMEGRID_DYNAMICS_ENV_ID:
            raise HomeGridManifestError(
                "Gate M accepts only the official 'homegrid-dynamics' environment."
            )

    @property
    def action_seed(self) -> int:
        return self.seed + ACTION_SEED_OFFSET

    @property
    def episode_id(self) -> str:
        return f"{self.env_id}:{self.split}:{self.seed}"

    def to_record(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "split": self.split,
            "seed": self.seed,
            "action_seed": self.action_seed,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class HomeGridManifest:
    games: Tuple[HomeGridEpisodeSpec, ...]
    max_steps: int

    def __post_init__(self) -> None:
        _validate_max_steps(self.max_steps)
        if not self.games:
            raise HomeGridManifestError("A HomeGrid manifest cannot be empty.")
        owners: Dict[int, str] = {}
        present = set()
        for game in self.games:
            if game.max_steps != self.max_steps:
                raise HomeGridManifestError(
                    "Every episode must share the manifest's fixed max_steps."
                )
            present.add(game.split)
            previous = owners.get(game.seed)
            if previous is not None and previous != game.split:
                raise HomeGridSeedLeakageError(
                    f"Environment seed {game.seed} leaks across {previous!r} "
                    f"and {game.split!r}."
                )
            owners[game.seed] = game.split
        missing = set(SPLITS) - present
        if missing:
            raise HomeGridManifestError(
                f"Train/valid/test must all be non-empty; missing {sorted(missing)}."
            )

    @property
    def uses_frozen_default_protocol(self) -> bool:
        actual = {
            split: tuple(game.seed for game in self.games if game.split == split)
            for split in SPLITS
        }
        return bool(
            self.max_steps == DEFAULT_MAX_STEPS
            and actual["train"] == DEFAULT_TRAIN_SEEDS
            and actual["valid"] == DEFAULT_VALID_SEEDS
            and actual["test"] == DEFAULT_TEST_SEEDS
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "env_id": HOMEGRID_DYNAMICS_ENV_ID,
            "split_key": "environment_seed",
            "max_steps": self.max_steps,
            "uses_frozen_default_protocol": self.uses_frozen_default_protocol,
            "frozen_default_protocol": frozen_default_protocol(),
            "action_sampling": action_sampling_definition(),
            "visual_quantization": quantization_definition(),
            "splits": {
                split: [
                    game.to_record()
                    for game in self.games
                    if game.split == split
                ]
                for split in SPLITS
            },
        }


def build_homegrid_manifest(
    *,
    train_seeds: Iterable[int] = DEFAULT_TRAIN_SEEDS,
    valid_seeds: Iterable[int] = DEFAULT_VALID_SEEDS,
    test_seeds: Iterable[int] = DEFAULT_TEST_SEEDS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> HomeGridManifest:
    """Build a seed-disjoint official-dynamics manifest without opening Gym."""

    _validate_max_steps(max_steps)
    seed_groups = {
        "train": tuple(train_seeds),
        "valid": tuple(valid_seeds),
        "test": tuple(test_seeds),
    }
    games: List[HomeGridEpisodeSpec] = []
    for split in SPLITS:
        seeds = seed_groups[split]
        if not seeds:
            raise HomeGridManifestError(f"{split} seeds must not be empty.")
        if len(set(seeds)) != len(seeds):
            raise HomeGridManifestError(
                f"Duplicate environment seed inside {split!r}: {seeds!r}."
            )
        for seed in sorted(seeds):
            _validate_seed(seed)
            games.append(
                HomeGridEpisodeSpec(
                    split=split,
                    seed=seed,
                    max_steps=max_steps,
                )
            )
    return HomeGridManifest(games=tuple(games), max_steps=max_steps)


@dataclass(frozen=True)
class EncodedFrame:
    visual_tokens: Tuple[int, ...]
    raw_image_sha256: str


@dataclass(frozen=True)
class EncodedObservation:
    frame: EncodedFrame
    language_token: int
    human_language: str
    is_read_step: bool

    @property
    def phase(self) -> str:
        return "read" if self.is_read_step else "action"


def quantize_rgb_frame(frame: Any) -> EncodedFrame:
    """Quantize one exact 96x96 uint8 RGB frame without floating point."""

    if not isinstance(frame, np.ndarray):
        raise HomeGridFrameError(
            f"HomeGrid image must be a NumPy array, got {type(frame).__name__}."
        )
    expected_shape = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
    if frame.shape != expected_shape:
        raise HomeGridFrameError(
            f"HomeGrid image shape must be {expected_shape}, got {frame.shape}."
        )
    if frame.dtype != np.uint8:
        raise HomeGridFrameError(
            f"HomeGrid image dtype must be uint8, got {frame.dtype}."
        )
    contiguous = np.ascontiguousarray(frame)
    raw_digest = hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()
    patches = contiguous.reshape(
        PATCH_ROWS,
        PATCH_SIZE,
        PATCH_COLUMNS,
        PATCH_SIZE,
        FRAME_CHANNELS,
    )
    channel_sums = patches.sum(axis=(1, 3), dtype=np.uint32)
    divisor = PATCH_SIZE * PATCH_SIZE * CHANNEL_BIN_WIDTH
    levels = np.minimum(channel_sums // divisor, CHANNEL_LEVELS - 1)
    tokens = (
        levels[:, :, 0] * (CHANNEL_LEVELS**2)
        + levels[:, :, 1] * CHANNEL_LEVELS
        + levels[:, :, 2]
    )
    flattened = tuple(int(token) for token in tokens.reshape(-1))
    if len(flattened) != VISUAL_TOKEN_COUNT:
        raise HomeGridFrameError(
            "Internal quantizer invariant failed: expected 144 visual tokens."
        )
    return EncodedFrame(
        visual_tokens=flattened,
        raw_image_sha256=raw_digest,
    )


def encode_homegrid_observation(
    observation: Mapping[str, Any],
) -> EncodedObservation:
    if not isinstance(observation, Mapping):
        raise HomeGridObservationError("HomeGrid observation must be a mapping.")
    missing = {
        key
        for key in ("image", "token", "log_language_info", "is_read_step")
        if key not in observation
    }
    if missing:
        raise HomeGridObservationError(
            f"HomeGrid observation is missing official fields: {sorted(missing)}."
        )
    token_array = np.asarray(observation["token"])
    if token_array.shape != ():
        raise HomeGridObservationError("HomeGrid language token must be scalar.")
    try:
        language_token = int(token_array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise HomeGridObservationError(
            "HomeGrid language token must be integer-convertible."
        ) from exc
    if language_token < 0:
        raise HomeGridObservationError("HomeGrid language token cannot be negative.")
    human_language = observation["log_language_info"]
    if not isinstance(human_language, str):
        raise HomeGridObservationError(
            "HomeGrid log_language_info must be a human-readable string."
        )
    read_value = observation["is_read_step"]
    if not isinstance(read_value, (bool, np.bool_)):
        raise HomeGridObservationError("HomeGrid is_read_step must be boolean.")
    return EncodedObservation(
        frame=quantize_rgb_frame(observation["image"]),
        language_token=language_token,
        human_language=human_language,
        is_read_step=bool(read_value),
    )


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        converted = [_safe_scalar(item) for item in value]
        return converted if all(item is not None for item in converted) else None
    return None


def summarize_homegrid_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep stable scalar/log fields and compact event descriptions only."""

    summary: Dict[str, Any] = {}
    for key in (
        "backend",
        "action_name",
        "success",
        "log_timesteps_with_task",
        "log_new_task",
        "log_dist_goal",
        "log_task_times",
        "seed",
        "seed_mode",
    ):
        if key in info:
            value = _safe_scalar(info[key])
            if value is not None:
                summary[key] = value
    events = info.get("events", ())
    if isinstance(events, (list, tuple)):
        compact_events = []
        for event in events:
            if not isinstance(event, Mapping):
                continue
            compact = {}
            for key in ("type", "description"):
                value = event.get(key)
                if isinstance(value, str):
                    compact[key] = value
            if compact:
                compact_events.append(compact)
        if compact_events:
            summary["events"] = compact_events
    return summary


@dataclass(frozen=True)
class HomeGridTransitionRecord:
    step: int
    visual_tokens: Tuple[int, ...]
    next_visual_tokens: Tuple[int, ...]
    raw_image_sha256: str
    next_raw_image_sha256: str
    language_token: int
    next_language_token: int
    human_language: str
    next_human_language: str
    is_read_step: bool
    next_is_read_step: bool
    action: int
    action_name: str
    reward: float
    terminated: bool
    truncated: bool
    changed_patch_count: int
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)

    def to_record(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "visual_tokens": list(self.visual_tokens),
            "next_visual_tokens": list(self.next_visual_tokens),
            "raw_image_sha256": self.raw_image_sha256,
            "next_raw_image_sha256": self.next_raw_image_sha256,
            "language_token": self.language_token,
            "next_language_token": self.next_language_token,
            "human_language": self.human_language,
            "next_human_language": self.next_human_language,
            "is_read_step": self.is_read_step,
            "next_is_read_step": self.next_is_read_step,
            "phase": "read" if self.is_read_step else "action",
            "next_phase": "read" if self.next_is_read_step else "action",
            "action": self.action,
            "action_name": self.action_name,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
            "changed_patch_count": self.changed_patch_count,
            "info": dict(self.info),
        }


@dataclass(frozen=True)
class HomeGridDatasetEpisode:
    split: str
    seed: int
    action_seed: int
    env_id: str
    max_steps: int
    seed_mode: str
    goal: Optional[str]
    reset_info: Mapping[str, Any]
    transitions: Tuple[HomeGridTransitionRecord, ...]

    @property
    def episode_id(self) -> str:
        return f"{self.env_id}:{self.split}:{self.seed}"

    @property
    def collector_cutoff(self) -> bool:
        return bool(
            len(self.transitions) == self.max_steps
            and self.transitions
            and not self.transitions[-1].done
        )

    def transition_records(self) -> Tuple[Dict[str, Any], ...]:
        common = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "split": self.split,
            "episode_seed": self.seed,
            "action_seed": self.action_seed,
            "action_rng_algorithm": ACTION_RNG_ALGORITHM,
            "seed_mode": self.seed_mode,
            "max_steps": self.max_steps,
            "goal": self.goal,
        }
        return tuple({**common, **transition.to_record()} for transition in self.transitions)

    def manifest_record(self) -> Dict[str, Any]:
        counts = episode_counts(self)
        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "seed": self.seed,
            "action_seed": self.action_seed,
            "seed_mode": self.seed_mode,
            "goal": self.goal,
            "max_steps": self.max_steps,
            "collector_cutoff": self.collector_cutoff,
            "reset_info": dict(self.reset_info),
            "counts": counts,
        }


def _action_space_size(recorder: HomeGridRecorder) -> int:
    action_space = getattr(recorder.env, "action_space", None)
    size = getattr(action_space, "n", None)
    try:
        parsed = int(size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HomeGridDatasetError(
            "Official homegrid-dynamics must expose Discrete(10) action_space."
        ) from exc
    if parsed != len(HOMEGRID_ACTIONS):
        raise HomeGridDatasetError(
            f"Expected official Discrete(10), received action_space.n={parsed}."
        )
    return parsed


def collect_homegrid_from_recorder(
    recorder: HomeGridRecorder,
    spec: HomeGridEpisodeSpec,
) -> HomeGridDatasetEpisode:
    """Collect one seeded rollout from an official-recorder contract object."""

    if recorder.env_id != HOMEGRID_DYNAMICS_ENV_ID or spec.env_id != recorder.env_id:
        raise HomeGridDatasetError(
            "Gate M collection accepts only a homegrid-dynamics recorder."
        )
    action_count = _action_space_size(recorder)
    recorder.reset(seed=spec.seed)
    if recorder.seed_mode not in {HOMEGRID_SEED_MODE_GYM, HOMEGRID_SEED_MODE_COMPAT}:
        raise HomeGridDatasetError(
            "Seeded HomeGrid reset did not report a strict seeded mode."
        )
    action_rng = random.Random(spec.action_seed)
    records: List[HomeGridTransitionRecord] = []
    for step in range(spec.max_steps):
        # Encode before env.step: the official preread wrapper mutates its
        # cached observation in place while streaming language tokens.
        current_mapping = recorder.observation
        if current_mapping is None:
            raise HomeGridObservationError("Recorder lost its current observation.")
        current = encode_homegrid_observation(current_mapping)
        action = action_rng.randrange(action_count)
        transition = recorder.step(action)
        next_observation = encode_homegrid_observation(
            transition.next_observation
        )
        changed = sum(
            left != right
            for left, right in zip(
                current.frame.visual_tokens,
                next_observation.frame.visual_tokens,
            )
        )
        if not math.isfinite(float(transition.reward)):
            raise HomeGridDatasetError("HomeGrid reward must be finite.")
        records.append(
            HomeGridTransitionRecord(
                step=step,
                visual_tokens=current.frame.visual_tokens,
                next_visual_tokens=next_observation.frame.visual_tokens,
                raw_image_sha256=current.frame.raw_image_sha256,
                next_raw_image_sha256=next_observation.frame.raw_image_sha256,
                language_token=current.language_token,
                next_language_token=next_observation.language_token,
                human_language=current.human_language,
                next_human_language=next_observation.human_language,
                is_read_step=current.is_read_step,
                next_is_read_step=next_observation.is_read_step,
                action=action,
                action_name=HOMEGRID_ACTIONS[action],
                reward=float(transition.reward),
                terminated=bool(transition.terminated),
                truncated=bool(transition.truncated),
                changed_patch_count=changed,
                info=summarize_homegrid_info(transition.info),
            )
        )
        if transition.done:
            break
    if not records:
        raise HomeGridDatasetError("HomeGrid rollout produced no transitions.")
    return HomeGridDatasetEpisode(
        split=spec.split,
        seed=spec.seed,
        action_seed=spec.action_seed,
        env_id=spec.env_id,
        max_steps=spec.max_steps,
        seed_mode=recorder.seed_mode,
        goal=recorder.goal,
        reset_info=summarize_homegrid_info(recorder.reset_info),
        transitions=tuple(records),
    )


def collect_homegrid_episode(spec: HomeGridEpisodeSpec) -> HomeGridDatasetEpisode:
    """Production entry: create and close the official environment only."""

    env = make_homegrid_env(HOMEGRID_DYNAMICS_ENV_ID)
    recorder = HomeGridRecorder(env, env_id=HOMEGRID_DYNAMICS_ENV_ID)
    try:
        return collect_homegrid_from_recorder(recorder, spec)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def episode_counts(episode: HomeGridDatasetEpisode) -> Dict[str, Any]:
    transitions = episode.transitions
    return {
        "episode_count": 1,
        "transition_count": len(transitions),
        "read_step_count": sum(item.is_read_step for item in transitions),
        "action_step_count": sum(not item.is_read_step for item in transitions),
        "changed_patch_count": sum(item.changed_patch_count for item in transitions),
        "transitions_with_changed_patches": sum(
            item.changed_patch_count > 0 for item in transitions
        ),
        "reward_sum": float(sum(item.reward for item in transitions)),
        "nonzero_reward_count": sum(item.reward != 0.0 for item in transitions),
        "positive_reward_count": sum(item.reward > 0.0 for item in transitions),
        "negative_reward_count": sum(item.reward < 0.0 for item in transitions),
        "terminated_count": sum(item.terminated for item in transitions),
        "truncated_count": sum(item.truncated for item in transitions),
        "done_count": sum(item.done for item in transitions),
        "collector_cutoff_episode_count": int(episode.collector_cutoff),
    }


def aggregate_counts(
    episodes: Sequence[HomeGridDatasetEpisode],
) -> Dict[str, Any]:
    keys = tuple(episode_counts(episodes[0])) if episodes else ()
    totals: Dict[str, Any] = {key: 0 for key in keys}
    for episode in episodes:
        for key, value in episode_counts(episode).items():
            totals[key] += value
    if "reward_sum" in totals:
        totals["reward_sum"] = float(totals["reward_sum"])
    return totals


@dataclass(frozen=True)
class HomeGridDatasetFile:
    path: str
    sha256: str
    size_bytes: int


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
        raise HomeGridSerializationError(
            f"Value is not canonical-JSON serializable: {exc}"
        ) from exc


def file_sha256(path: Any) -> str:
    source = Path(path)
    if not source.is_file():
        raise HomeGridSerializationError(f"Cannot hash missing file: {source}.")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(
    path: Any,
    content: bytes,
    *,
    overwrite: bool = False,
) -> HomeGridDatasetFile:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise HomeGridSerializationError(
            f"Refusing to overwrite existing artifact: {destination}."
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
                raise HomeGridSerializationError(
                    f"Artifact appeared during write: {destination}."
                ) from exc
            temporary.unlink()
    except OSError as exc:
        raise HomeGridSerializationError(
            f"Atomic write failed for {destination}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return HomeGridDatasetFile(
        path=str(destination),
        sha256=file_sha256(destination),
        size_bytes=len(content),
    )


def write_canonical_json_atomic(
    payload: Mapping[str, Any],
    path: Any,
    *,
    overwrite: bool = False,
) -> HomeGridDatasetFile:
    return atomic_write_bytes(
        path,
        (_canonical_json(dict(payload)) + "\n").encode("utf-8"),
        overwrite=overwrite,
    )


def write_transition_jsonl_atomic(
    episodes: Iterable[HomeGridDatasetEpisode],
    path: Any,
    *,
    overwrite: bool = False,
) -> HomeGridDatasetFile:
    values = tuple(episodes)
    if not values:
        raise HomeGridSerializationError("Cannot write an empty HomeGrid dataset.")
    owners: Dict[int, str] = {}
    for episode in values:
        previous = owners.get(episode.seed)
        if previous is not None and previous != episode.split:
            raise HomeGridSeedLeakageError(
                f"Environment seed {episode.seed} leaks across output splits."
            )
        owners[episode.seed] = episode.split
    ordered = sorted(
        values,
        key=lambda episode: (
            _SPLIT_ORDER[episode.split],
            episode.seed,
        ),
    )
    lines = [
        _canonical_json(record)
        for episode in ordered
        for record in episode.transition_records()
    ]
    return atomic_write_bytes(
        path,
        ("\n".join(lines) + "\n").encode("utf-8"),
        overwrite=overwrite,
    )


__all__ = [
    "ACTION_RNG_ALGORITHM",
    "ACTION_SEED_OFFSET",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_TEST_SEEDS",
    "DEFAULT_TRAIN_SEEDS",
    "DEFAULT_VALID_SEEDS",
    "HOMEGRID_DYNAMICS_ENV_ID",
    "SCHEMA_VERSION",
    "SPLITS",
    "EncodedFrame",
    "EncodedObservation",
    "HomeGridDatasetEpisode",
    "HomeGridDatasetError",
    "HomeGridDatasetFile",
    "HomeGridEpisodeSpec",
    "HomeGridFrameError",
    "HomeGridManifest",
    "HomeGridManifestError",
    "HomeGridObservationError",
    "HomeGridSeedLeakageError",
    "HomeGridSerializationError",
    "HomeGridTransitionRecord",
    "action_sampling_definition",
    "aggregate_counts",
    "atomic_write_bytes",
    "build_homegrid_manifest",
    "collect_homegrid_episode",
    "collect_homegrid_from_recorder",
    "encode_homegrid_observation",
    "episode_counts",
    "file_sha256",
    "frozen_default_protocol",
    "quantization_definition",
    "quantize_rgb_frame",
    "summarize_homegrid_info",
    "write_canonical_json_atomic",
    "write_transition_jsonl_atomic",
]
