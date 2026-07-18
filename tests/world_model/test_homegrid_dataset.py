import importlib.util
import json
from pathlib import Path
import random
import sys
import types

import numpy as np
import pytest


if importlib.util.find_spec("torch") is None:
    package = types.ModuleType("vpsc")
    package.__path__ = [str(Path(__file__).parents[2] / "vpsc")]
    sys.modules.setdefault("vpsc", package)
    world_model_package = types.ModuleType("vpsc.world_model")
    world_model_package.__path__ = [
        str(Path(__file__).parents[2] / "vpsc" / "world_model")
    ]
    sys.modules.setdefault("vpsc.world_model", world_model_package)

from vpsc.world_model.homegrid import (
    HOMEGRID_SEED_MODE_GYM,
    HomeGridRecorder,
    HomeGridSetupError,
    probe_homegrid,
)
from vpsc.world_model.homegrid_dataset import (
    ACTION_SEED_OFFSET,
    DEFAULT_MAX_STEPS,
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALID_SEEDS,
    HomeGridEpisodeSpec,
    HomeGridFrameError,
    HomeGridSeedLeakageError,
    HomeGridSerializationError,
    build_homegrid_manifest,
    collect_homegrid_episode,
    collect_homegrid_from_recorder,
    file_sha256,
    quantize_rgb_frame,
    write_transition_jsonl_atomic,
)


class _DiscreteTen:
    n = 10

    def seed(self, seed):
        self.seed_value = seed


def _image(value=0):
    return np.full((96, 96, 3), value, dtype=np.uint8)


def _observation(image, token, language, read):
    return {
        "image": image,
        "token": np.uint32(token),
        "log_language_info": language,
        "is_read_step": np.bool_(read),
        "token_embed": np.zeros(512, dtype=np.float32),
    }


class FakeDynamicsEnv:
    def __init__(self, terminate_at=3):
        self.action_space = _DiscreteTen()
        self.observation_space = _DiscreteTen()
        self.task = "open the cupboard"
        self.terminate_at = terminate_at
        self.actions = []
        self.step_index = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed_received = seed
        self.step_index = 0
        self.actions.clear()
        return (
            _observation(_image(0), 100, "cupboard is in kitchen", True),
            {
                "log_new_task": True,
                "log_dist_goal": 3,
                "events": [],
                "symbolic_state": {"must_not": "serialize"},
            },
        )

    def step(self, action):
        self.actions.append(action)
        self.step_index += 1
        # Deliberately disturb the global RNG. The dedicated action Random
        # instance must continue producing the preregistered sequence.
        random.seed(9000 + self.step_index)
        frame = _image(0)
        if self.step_index >= 2:
            frame[:8, :8] = np.array([255, 192, 128], dtype=np.uint8)
        terminated = self.step_index >= self.terminate_at
        read = self.step_index < 2
        return (
            _observation(
                frame,
                100 + self.step_index,
                "cupboard is in kitchen" if read else "open the cupboard",
                read,
            ),
            1.0 if terminated else 0.0,
            terminated,
            False,
            {
                "success": True if terminated else None,
                "log_new_task": False,
                "log_dist_goal": max(0, 3 - self.step_index),
                "events": [
                    {
                        "type": "dynamics",
                        "description": "the cupboard opened",
                        "obj": object(),
                    }
                ],
                "symbolic_state": {"must_not": "serialize"},
                "all_events": [[object()]],
            },
        )

    def close(self):
        self.closed = True


def _collect(seed=7, max_steps=5, terminate_at=3):
    env = FakeDynamicsEnv(terminate_at=terminate_at)
    recorder = HomeGridRecorder(env, env_id="homegrid-dynamics")
    episode = collect_homegrid_from_recorder(
        recorder,
        HomeGridEpisodeSpec(split="train", seed=seed, max_steps=max_steps),
    )
    return env, episode


def test_visual_quantizer_exact_thresholds_and_row_major_tokens():
    frame = _image(0)
    frame[:8, :8] = np.array([63, 64, 191], dtype=np.uint8)
    frame[:8, 8:16] = np.array([255, 192, 128], dtype=np.uint8)

    encoded = quantize_rgb_frame(frame)

    assert len(encoded.visual_tokens) == 144
    assert encoded.visual_tokens[:3] == (6, 62, 0)
    assert len(encoded.raw_image_sha256) == 64


def test_visual_quantizer_rejects_nonofficial_shape_and_dtype():
    with pytest.raises(HomeGridFrameError, match="shape"):
        quantize_rgb_frame(np.zeros((95, 96, 3), dtype=np.uint8))
    with pytest.raises(HomeGridFrameError, match="dtype"):
        quantize_rgb_frame(np.zeros((96, 96, 3), dtype=np.float32))


def test_manifest_freezes_preregistered_protocol_and_rejects_seed_leakage():
    manifest = build_homegrid_manifest()
    assert manifest.uses_frozen_default_protocol is True
    assert manifest.max_steps == DEFAULT_MAX_STEPS == 96
    assert tuple(
        game.seed for game in manifest.games if game.split == "train"
    ) == DEFAULT_TRAIN_SEEDS
    assert tuple(
        game.seed for game in manifest.games if game.split == "valid"
    ) == DEFAULT_VALID_SEEDS
    assert tuple(
        game.seed for game in manifest.games if game.split == "test"
    ) == DEFAULT_TEST_SEEDS
    assert max(DEFAULT_TEST_SEEDS) < 2**32

    with pytest.raises(HomeGridSeedLeakageError, match="leaks"):
        build_homegrid_manifest(
            train_seeds=[1], valid_seeds=[1], test_seeds=[2], max_steps=3
        )


def test_collection_preserves_phases_and_isolated_uniform_action_rng():
    env, episode = _collect(seed=7)
    expected_rng = random.Random(7 + ACTION_SEED_OFFSET)
    expected_actions = [expected_rng.randrange(10) for _ in range(3)]

    assert env.seed_received == 7
    assert episode.seed_mode == HOMEGRID_SEED_MODE_GYM
    assert env.actions == expected_actions
    assert [item.is_read_step for item in episode.transitions] == [True, True, False]
    assert [item.next_is_read_step for item in episode.transitions] == [True, False, False]
    assert [item.changed_patch_count for item in episode.transitions] == [0, 1, 0]
    assert episode.transitions[-1].reward == 1.0
    assert episode.transitions[-1].terminated is True
    assert "symbolic_state" not in episode.transitions[0].info
    assert episode.transitions[0].info["events"] == [
        {"type": "dynamics", "description": "the cupboard opened"}
    ]


def test_collection_stops_at_fixed_horizon_without_faking_truncation():
    env, episode = _collect(seed=9, max_steps=2, terminate_at=99)
    assert len(env.actions) == len(episode.transitions) == 2
    assert episode.collector_cutoff is True
    assert episode.transitions[-1].truncated is False


def test_canonical_jsonl_contains_tokens_and_hashes_but_no_raw_rgb(tmp_path):
    _, episode = _collect(seed=12)
    output = tmp_path / "transitions.jsonl"
    artifact = write_transition_jsonl_atomic([episode], output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert artifact.sha256 == file_sha256(output)
    assert len(records) == 3
    assert len(records[0]["visual_tokens"]) == 144
    assert len(records[0]["next_visual_tokens"]) == 144
    assert "raw_image_sha256" in records[0]
    assert "image" not in records[0]
    assert "token_embed" not in records[0]
    assert "symbolic_state" not in records[0]["info"]
    with pytest.raises(HomeGridSerializationError, match="overwrite"):
        write_transition_jsonl_atomic([episode], output)


def test_official_homegrid_dynamics_smoke_when_dependency_is_available():
    probe = probe_homegrid(verify_import=True)
    if (
        not probe.available
        or "homegrid-dynamics" not in probe.registered_env_ids
        or probe.data_assets_present is False
    ):
        pytest.skip(probe.detail)
    try:
        episode = collect_homegrid_episode(
            HomeGridEpisodeSpec(split="test", seed=2026072000, max_steps=1)
        )
    except HomeGridSetupError as exc:
        pytest.skip(str(exc))
    assert episode.env_id == "homegrid-dynamics"
    assert episode.seed_mode in {
        "gym_reset",
        "homegrid_0.1.1_compat_python_numpy_np_random_spaces",
    }
    assert len(episode.transitions) == 1
