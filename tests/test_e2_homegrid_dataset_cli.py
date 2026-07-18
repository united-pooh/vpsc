import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


if importlib.util.find_spec("torch") is None:
    package = types.ModuleType("vpsc")
    package.__path__ = [str(Path(__file__).parents[1] / "vpsc")]
    sys.modules.setdefault("vpsc", package)
    world_model_package = types.ModuleType("vpsc.world_model")
    world_model_package.__path__ = [
        str(Path(__file__).parents[1] / "vpsc" / "world_model")
    ]
    sys.modules.setdefault("vpsc.world_model", world_model_package)

from experiments import e2_homegrid_dataset as cli
from vpsc.world_model.homegrid_dataset import (
    ACTION_SEED_OFFSET,
    DEFAULT_MAX_STEPS,
    DEFAULT_TEST_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_VALID_SEEDS,
    HomeGridDatasetEpisode,
    HomeGridTransitionRecord,
    file_sha256,
)


def _args(tmp_path, *extra):
    return cli.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "gate-m"),
            "--max-steps",
            "2",
            "--train-seeds",
            "11",
            "--valid-seeds",
            "22",
            "--test-seeds",
            "33",
            *extra,
        ]
    )


def _transition(step, *, read, changed, done=False):
    tokens = tuple([0] * 144)
    next_tokens = list(tokens)
    for index in range(changed):
        next_tokens[index] = 63
    return HomeGridTransitionRecord(
        step=step,
        visual_tokens=tokens,
        next_visual_tokens=tuple(next_tokens),
        raw_image_sha256=f"{step + 1:064x}",
        next_raw_image_sha256=f"{step + 2:064x}",
        language_token=100 + step,
        next_language_token=101 + step,
        human_language="read text" if read else "act text",
        next_human_language="next text",
        is_read_step=read,
        next_is_read_step=False,
        action=step,
        action_name="left" if step == 0 else "right",
        reward=1.0 if done else 0.0,
        terminated=done,
        truncated=False,
        changed_patch_count=changed,
        info={"backend": "homegrid"},
    )


def _fake_episode(spec):
    return HomeGridDatasetEpisode(
        split=spec.split,
        seed=spec.seed,
        action_seed=spec.seed + ACTION_SEED_OFFSET,
        env_id=spec.env_id,
        max_steps=spec.max_steps,
        seed_mode="gym_reset",
        goal="open the cupboard",
        reset_info={"seed": spec.seed, "seed_mode": "gym_reset"},
        transitions=(
            _transition(0, read=True, changed=0),
            _transition(1, read=False, changed=2, done=True),
        ),
    )


def test_cli_defaults_are_the_frozen_preregistered_protocol():
    args = cli.build_parser().parse_args([])
    assert args.max_steps == DEFAULT_MAX_STEPS == 96
    assert args.train_seeds == list(DEFAULT_TRAIN_SEEDS)
    assert args.valid_seeds == list(DEFAULT_VALID_SEEDS)
    assert args.test_seeds == list(DEFAULT_TEST_SEEDS)
    assert args.overwrite is False


def test_cli_writes_atomic_split_manifest_jsonl_and_summary(tmp_path):
    args = _args(tmp_path)
    collected = []

    def collector(spec):
        collected.append(spec)
        return _fake_episode(spec)

    result = cli.run_dataset(args, collector=collector)

    assert len(collected) == 3
    assert result["uses_frozen_default_protocol"] is False
    for split, seed in (("train", 11), ("valid", 22), ("test", 33)):
        root = args.output_dir / split
        jsonl_path = root / "transitions.jsonl"
        manifest_path = root / "manifest.json"
        summary_path = root / "summary.json"
        assert jsonl_path.is_file()
        assert manifest_path.is_file()
        assert summary_path.is_file()

        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 2
        assert records[0]["episode_seed"] == seed
        assert len(records[0]["visual_tokens"]) == 144
        assert "image" not in records[0]

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert manifest["env_id"] == "homegrid-dynamics"
        assert manifest["seed_modes"] == ["gym_reset"]
        assert manifest["actual_protocol"][f"{split}_seeds"] == [seed]
        assert manifest["visual_quantization"]["vocabulary_size"] == 64
        assert "homegrid" in manifest["versions"]
        assert "gym" in manifest["versions"]
        assert summary["counts"] == {
            "action_step_count": 1,
            "changed_patch_count": 2,
            "collector_cutoff_episode_count": 0,
            "done_count": 1,
            "episode_count": 1,
            "negative_reward_count": 0,
            "nonzero_reward_count": 1,
            "positive_reward_count": 1,
            "read_step_count": 1,
            "reward_sum": 1.0,
            "terminated_count": 1,
            "transition_count": 2,
            "transitions_with_changed_patches": 1,
            "truncated_count": 0,
        }
        assert summary["artifacts"]["transitions_jsonl"][
            "sha256"
        ] == file_sha256(jsonl_path)
        assert summary["artifacts"]["manifest"]["sha256"] == file_sha256(
            manifest_path
        )
        assert result["summary_artifacts"][split]["sha256"] == file_sha256(
            summary_path
        )


def test_default_refuses_overwrite_before_collecting(tmp_path):
    args = _args(tmp_path)
    conflict = args.output_dir / "valid" / "manifest.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("old", encoding="utf-8")
    calls = []

    with pytest.raises(cli.HomeGridRunError, match="refusing overwrite"):
        cli.run_dataset(
            args,
            collector=lambda spec: calls.append(spec),
        )

    assert calls == []
    assert conflict.read_text(encoding="utf-8") == "old"
