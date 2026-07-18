from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

# Keep extraction-contract tests runnable without the training-only torch dep.
if importlib.util.find_spec("torch") is None:
    package = types.ModuleType("vpsc")
    package.__path__ = [str(Path(__file__).parents[2] / "vpsc")]
    sys.modules.setdefault("vpsc", package)
    world_model_package = types.ModuleType("vpsc.world_model")
    world_model_package.__path__ = [
        str(Path(__file__).parents[2] / "vpsc" / "world_model")
    ]
    sys.modules.setdefault("vpsc.world_model", world_model_package)

from vpsc.world_model.textworld import TextWorldAdapter
import vpsc.world_model.textworld_dataset as dataset_module
from vpsc.world_model.textworld_dataset import (
    CoinCollectorDatasetError,
    CoinCollectorGameSpec,
    SeedLeakageError,
    WalkthroughValidationError,
    build_coin_collector_manifest,
    collect_coin_collector_from_adapter,
    collect_coin_collector_game,
    episode_to_json_line,
    episode_to_token_event_text,
    file_sha256,
    write_episode_jsonl,
    write_manifest_json,
    write_token_event_text,
)


def _state(feedback, commands, *, won=False, include_identity=True):
    state = {
        "feedback": feedback,
        "description": feedback,
        "admissible_commands": tuple(commands),
        "objective": "Collect the coin.",
        "score": int(won),
        "won": won,
        "lost": False,
    }
    if include_identity:
        state.update(
            {
                "extra.walkthrough": ["go north", "take coin"],
                "extra.desc": "Coin Collector",
                "extra.uuid": "tw-coin_collector-contract-fixture",
            }
        )
    return state


class FakeCoinCollectorEnv:
    """Official core-API contract fake; never used by production entry."""

    def __init__(self, *, final_won=True):
        self.position = 0
        self.closed = False
        self.final_won = final_won

    def reset(self):
        self.position = 0
        return _state("You are in the first room.", ("go north", "look"))

    def step(self, command):
        if self.position == 0 and command == "go north":
            self.position = 1
            return _state(
                "You see a coin.", ("go south", "look", "take coin")
            ), 0, False
        if self.position == 1 and command == "take coin":
            self.position = 2
            return _state("Taken.", (), won=self.final_won), 1, True
        if command == "go south" and self.position == 1:
            self.position = 0
            return _state(
                "You are in the first room.", ("go north", "look")
            ), 0, False
        return _state(
            "Nothing changes.",
            ("go north", "look") if self.position == 0 else ("go south", "look", "take coin"),
        ), 0, False

    def copy(self):
        return deepcopy(self)

    def close(self):
        self.closed = True


def _spec(tmp_path, *, seed=11, split="train"):
    return CoinCollectorGameSpec(
        seed=seed,
        level=2,
        split=split,
        game_file=str(tmp_path / f"game-{seed}.z8"),
    )


def _episode(tmp_path, *, seed=11, split="train"):
    adapter = TextWorldAdapter(FakeCoinCollectorEnv(), source="contract-fake")
    return collect_coin_collector_from_adapter(
        adapter,
        _spec(tmp_path, seed=seed, split=split),
        game_sha256=(f"{seed:064x}"[-64:]),
    )


def test_manifest_uses_official_fixed_level_seed_commands(tmp_path):
    manifest = build_coin_collector_manifest(
        tmp_path / "games",
        level=10,
        train_seeds=[9, 3],
        valid_seeds=[101],
        test_seeds=[202],
    )

    assert [game.seed for game in manifest.games] == [3, 9, 101, 202]
    command = manifest.games[0].generation_command
    assert command[:6] == (
        "tw-make",
        "tw-coin_collector",
        "--level",
        "10",
        "--seed",
        "3",
    )
    assert command[-2:] == ("--force", "--silent")
    record = manifest.to_record()
    assert record["split_key"] == "game_seed"
    assert set(record["splits"]) == {"train", "valid", "test"}

    artifact = write_manifest_json(manifest, tmp_path / "manifest.json")
    assert artifact.sha256 == file_sha256(artifact.path)
    assert json.loads(Path(artifact.path).read_text(encoding="utf-8"))[
        "challenge"
    ] == "tw-coin_collector"


def test_manifest_rejects_seed_leakage(tmp_path):
    with pytest.raises(SeedLeakageError, match="seed 7 leaks"):
        build_coin_collector_manifest(
            tmp_path,
            level=2,
            train_seeds=[7],
            valid_seeds=[7],
            test_seeds=[9],
        )


def test_walkthrough_records_real_steps_and_copy_counterfactuals(tmp_path):
    adapter = TextWorldAdapter(FakeCoinCollectorEnv(), source="contract-fake")
    episode = collect_coin_collector_from_adapter(
        adapter,
        _spec(tmp_path),
        game_sha256="a" * 64,
        counterfactual_limit=1,
    )

    assert episode.won is True
    assert episode.walkthrough == ("go north", "take coin")
    assert [step.action for step in episode.steps] == ["go north", "take coin"]
    assert episode.steps[0].observation == "You are in the first room."
    assert episode.steps[0].next_obs == "You see a coin."
    assert episode.steps[-1].reward == 1.0
    assert episode.steps[-1].done is True
    assert len(episode.steps[0].counterfactuals) == 1
    assert episode.steps[0].counterfactuals[0].action == "look"
    assert len(adapter.transitions) == 2
    assert adapter.env.position == 2


def test_walkthrough_requires_terminal_won_state(tmp_path):
    adapter = TextWorldAdapter(
        FakeCoinCollectorEnv(final_won=False), source="contract-fake"
    )
    with pytest.raises(WalkthroughValidationError, match="won is false"):
        collect_coin_collector_from_adapter(
            adapter, _spec(tmp_path), game_sha256="b" * 64
        )


def test_production_entry_rejects_non_z8_before_opening_backend(tmp_path):
    # Construct a valid spec, then exercise the production file gate directly
    # through an extension-changing frozen dataclass replacement equivalent.
    bad = object.__new__(CoinCollectorGameSpec)
    object.__setattr__(bad, "seed", 1)
    object.__setattr__(bad, "level", 1)
    object.__setattr__(bad, "split", "test")
    object.__setattr__(bad, "game_file", str(tmp_path / "game.json"))

    with pytest.raises(CoinCollectorDatasetError, match="compiled .z8"):
        collect_coin_collector_game(bad)


def test_production_entry_requests_walkthrough_and_hashes_real_file(
    tmp_path, monkeypatch
):
    spec = _spec(tmp_path)
    game_path = Path(spec.game_file)
    game_path.write_bytes(b"\x08contract-z-machine-file")
    adapter = TextWorldAdapter(FakeCoinCollectorEnv(), source=str(game_path))
    opened = {}

    def fake_open(path, **kwargs):
        opened["path"] = Path(path)
        opened.update(kwargs)
        return adapter

    monkeypatch.setattr(dataset_module, "open_textworld", fake_open)
    episode = collect_coin_collector_game(spec)

    assert opened["path"] == game_path.resolve()
    assert opened["extras"] == ("walkthrough", "desc", "uuid")
    assert episode.game_sha256 == hashlib.sha256(game_path.read_bytes()).hexdigest()
    assert adapter.env.closed is True


def test_jsonl_and_token_event_serialization_are_deterministic(tmp_path):
    train = _episode(tmp_path, seed=11, split="train")
    valid = _episode(tmp_path, seed=22, split="valid")

    json_a = write_episode_jsonl(
        [valid, train], tmp_path / "a" / "episodes.jsonl"
    )
    json_b = write_episode_jsonl(
        [train, valid], tmp_path / "b" / "episodes.jsonl"
    )
    event_a = write_token_event_text(
        [valid, train], tmp_path / "a" / "events.txt"
    )
    event_b = write_token_event_text(
        [train, valid], tmp_path / "b" / "events.txt"
    )

    assert Path(json_a.path).read_bytes() == Path(json_b.path).read_bytes()
    assert Path(event_a.path).read_bytes() == Path(event_b.path).read_bytes()
    assert json_a.sha256 == json_b.sha256 == file_sha256(json_a.path)
    assert event_a.sha256 == event_b.sha256 == file_sha256(event_a.path)
    assert json_a.sha256 == hashlib.sha256(
        Path(json_a.path).read_bytes()
    ).hexdigest()

    first = json.loads(Path(json_a.path).read_text(encoding="utf-8").splitlines()[0])
    assert first["seed"] == 11
    assert first["steps"][0]["admissible_actions"] == ["go north", "look"]
    assert "<|counterfactual|>" in Path(event_a.path).read_text(encoding="utf-8")
    assert episode_to_json_line(train) + "\n" in Path(json_a.path).read_text(
        encoding="utf-8"
    )
    assert episode_to_token_event_text(train) in Path(event_a.path).read_text(
        encoding="utf-8"
    )
