import hashlib
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

from vpsc.world_model import homegrid_corpus as corpus_module
from vpsc.world_model.homegrid_corpus import (
    HomeGridCorpusFormatError,
    load_homegrid_corpus,
)
from vpsc.world_model.homegrid_dataset import (
    ACTION_RNG_ALGORITHM,
    ACTION_SEED_OFFSET,
    HOMEGRID_DYNAMICS_ENV_ID,
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
)


EXPECTED = {
    "train": (101,),
    "valid": (202,),
    "test": (303,),
}
MAX_STEPS = 2


def _canonical_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path):
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _tokens(first):
    return [first] + [0] * 143


def _transition(split, seed, step, language_base, *, early_done=False):
    current = _tokens(step)
    next_visual = _tokens(step + 1)
    done = bool(early_done)
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "episode_id": f"{HOMEGRID_DYNAMICS_ENV_ID}:{split}:{seed}",
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "split": split,
        "episode_seed": seed,
        "action_seed": seed + ACTION_SEED_OFFSET,
        "action_rng_algorithm": ACTION_RNG_ALGORITHM,
        "seed_mode": "gym_reset",
        "max_steps": MAX_STEPS,
        "goal": "open the cupboard",
        "step": step,
        "visual_tokens": current,
        "next_visual_tokens": next_visual,
        "raw_image_sha256": hashlib.sha256(
            f"{split}:{seed}:{step}:current".encode()
        ).hexdigest(),
        "next_raw_image_sha256": hashlib.sha256(
            f"{split}:{seed}:{step}:next".encode()
        ).hexdigest(),
        "language_token": language_base + step,
        "next_language_token": language_base + step + 1,
        "human_language": f"sentence {language_base + step}",
        "next_human_language": f"sentence {language_base + step + 1}",
        "is_read_step": step == 0,
        "next_is_read_step": False,
        "phase": "read" if step == 0 else "action",
        "next_phase": "action",
        "action": step,
        "action_name": "left" if step == 0 else "right",
        "reward": 1.0 if done else 0.0,
        "terminated": done,
        "truncated": False,
        "done": done,
        "changed_patch_count": 1,
        "info": {},
    }


def _counts(rows):
    cutoff = len(rows) == MAX_STEPS and not rows[-1]["done"]
    return {
        "episode_count": 1,
        "transition_count": len(rows),
        "read_step_count": sum(row["is_read_step"] for row in rows),
        "action_step_count": sum(not row["is_read_step"] for row in rows),
        "changed_patch_count": sum(row["changed_patch_count"] for row in rows),
        "transitions_with_changed_patches": sum(
            row["changed_patch_count"] > 0 for row in rows
        ),
        "nonzero_reward_count": sum(row["reward"] != 0 for row in rows),
        "positive_reward_count": sum(row["reward"] > 0 for row in rows),
        "negative_reward_count": sum(row["reward"] < 0 for row in rows),
        "reward_sum": sum(row["reward"] for row in rows),
        "terminated_count": sum(row["terminated"] for row in rows),
        "truncated_count": sum(row["truncated"] for row in rows),
        "done_count": sum(row["done"] for row in rows),
        "collector_cutoff_episode_count": int(cutoff),
    }


def _actual_protocol():
    return {
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "max_steps": MAX_STEPS,
        "train_seeds": list(EXPECTED["train"]),
        "valid_seeds": list(EXPECTED["valid"]),
        "test_seeds": list(EXPECTED["test"]),
        "action_sampling": {"algorithm": ACTION_RNG_ALGORITHM},
    }


def _write_split(root, split, *, early_done=False):
    split_root = root / split
    transitions_path = split_root / "transitions.jsonl"
    manifest_path = split_root / "manifest.json"
    summary_path = split_root / "summary.json"
    seed = EXPECTED[split][0]
    language_base = {"train": 10, "valid": 20, "test": 30}[split]
    row_count = 1 if early_done else MAX_STEPS
    rows = [
        _transition(
            split,
            seed,
            step,
            language_base,
            early_done=early_done and step == row_count - 1,
        )
        for step in range(row_count)
    ]
    transitions_path.parent.mkdir(parents=True, exist_ok=True)
    transitions_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    versions = {"homegrid": "0.1.1", "dataset_schema": DATASET_SCHEMA_VERSION}
    counts = _counts(rows)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "runner_schema_version": corpus_module.RUNNER_SCHEMA_VERSION,
        "split": split,
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "versions": versions,
        "uses_frozen_default_protocol": True,
        "actual_protocol": _actual_protocol(),
        "episodes": [
            {
                "episode_id": f"{HOMEGRID_DYNAMICS_ENV_ID}:{split}:{seed}",
                "seed": seed,
                "action_seed": seed + ACTION_SEED_OFFSET,
                "max_steps": MAX_STEPS,
                "collector_cutoff": bool(
                    counts["collector_cutoff_episode_count"]
                ),
                "counts": counts,
            }
        ],
        "artifacts": {"transitions_jsonl": _artifact(transitions_path)},
    }
    _canonical_write(manifest_path, manifest)
    summary = {
        "schema_version": corpus_module.RUNNER_SCHEMA_VERSION,
        "split": split,
        "env_id": HOMEGRID_DYNAMICS_ENV_ID,
        "versions": versions,
        "uses_frozen_default_protocol": True,
        "actual_protocol": _actual_protocol(),
        "seeds": [seed],
        "counts": counts,
        "artifacts": {
            "transitions_jsonl": _artifact(transitions_path),
            "manifest": _artifact(manifest_path),
        },
    }
    _canonical_write(summary_path, summary)


def _refresh_provenance(root, split):
    split_root = root / split
    transitions_path = split_root / "transitions.jsonl"
    manifest_path = split_root / "manifest.json"
    summary_path = split_root / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["transitions_jsonl"] = _artifact(transitions_path)
    _canonical_write(manifest_path, manifest)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["transitions_jsonl"] = _artifact(transitions_path)
    summary["artifacts"]["manifest"] = _artifact(manifest_path)
    _canonical_write(summary_path, summary)


@pytest.fixture
def corpus_root(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_module, "_EXPECTED_SEEDS", EXPECTED)
    monkeypatch.setattr(corpus_module, "DEFAULT_MAX_STEPS", MAX_STEPS)
    root = tmp_path / "corpus"
    for split in EXPECTED:
        _write_split(root, split)
    return root


def test_load_builds_train_only_vocabulary_and_oov_metadata(corpus_root):
    corpus = load_homegrid_corpus(corpus_root)

    assert len(corpus.vocabulary) == 5
    assert corpus.vocabulary.encode(10) == 2
    assert corpus.vocabulary.encode(11) == 3
    assert corpus.vocabulary.encode(12) == 4
    assert corpus.vocabulary.encode(20) == corpus.vocabulary.unk_id == 1
    valid = tuple(corpus.iter_episodes("valid"))[0]
    assert all(row.language_id == 1 for row in valid.transitions)
    assert all(row.next_language_id == 1 for row in valid.transitions)
    assert corpus.split_metadata("valid")["language_oov_current"] == 2
    assert corpus.split_metadata("valid")["language_oov_next"] == 2
    assert corpus.most_frequent_visual_token == 0
    assert sum(corpus.train_visual_frequency) == 2 * 144


def test_chunks_reset_each_epoch_and_never_cross_episode(corpus_root):
    corpus = load_homegrid_corpus(corpus_root)
    chunks = list(corpus.iter_chunks("train", sequence_length=1, epochs=2))

    assert [(chunk.epoch, chunk.offset) for chunk in chunks] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [chunk.reset_state for chunk in chunks] == [True, False, True, False]
    assert {chunk.episode_seed for chunk in chunks} == {EXPECTED["train"][0]}
    assert all(chunk.length == 1 for chunk in chunks)
    whole = list(corpus.iter_chunks("train", sequence_length=99, epochs=2))
    assert len(whole) == 2
    assert all(chunk.length == MAX_STEPS and chunk.reset_state for chunk in whole)


def test_early_done_episode_is_valid_when_manifest_and_counts_agree(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(corpus_module, "_EXPECTED_SEEDS", EXPECTED)
    monkeypatch.setattr(corpus_module, "DEFAULT_MAX_STEPS", MAX_STEPS)
    root = tmp_path / "early-done"
    for split in EXPECTED:
        _write_split(root, split, early_done=split == "valid")

    corpus = load_homegrid_corpus(root)
    valid = tuple(corpus.iter_episodes("valid"))[0]
    assert len(valid.transitions) == 1
    assert valid.transitions[-1].done is True
    metadata = corpus.split_metadata("valid")
    assert metadata["done_count"] == 1
    assert metadata["collector_cutoff_episode_count"] == 0


def test_tampered_transition_file_fails_closed_on_artifact_hash(corpus_root):
    path = corpus_root / "test" / "transitions.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(HomeGridCorpusFormatError, match="SHA/size"):
        load_homegrid_corpus(corpus_root)


def test_bad_reward_fails_after_hashes_are_refreshed(corpus_root):
    path = corpus_root / "train" / "transitions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["reward"] = -0.5
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    _refresh_provenance(corpus_root, "train")

    with pytest.raises(HomeGridCorpusFormatError, match="outside frozen classes"):
        load_homegrid_corpus(corpus_root)


def test_transition_continuity_mismatch_fails_with_valid_hashes(corpus_root):
    path = corpus_root / "valid" / "transitions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["language_token"] = 999
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    _refresh_provenance(corpus_root, "valid")

    with pytest.raises(HomeGridCorpusFormatError, match="continuity mismatch"):
        load_homegrid_corpus(corpus_root)
