import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest


# The dataset runner is intentionally usable without training-only torch.
if importlib.util.find_spec("torch") is None:
    package = types.ModuleType("vpsc")
    package.__path__ = [str(Path(__file__).parents[1] / "vpsc")]
    sys.modules.setdefault("vpsc", package)
    world_model_package = types.ModuleType("vpsc.world_model")
    world_model_package.__path__ = [
        str(Path(__file__).parents[1] / "vpsc" / "world_model")
    ]
    sys.modules.setdefault("vpsc.world_model", world_model_package)

from experiments import e2_textworld_dataset as cli
from vpsc.world_model.textworld_dataset import (
    CoinCollectorEpisode,
    CoinCollectorStep,
    CounterfactualRecord,
    file_sha256,
)


def _args(tmp_path, *extra):
    return cli.build_parser().parse_args(
        [
            "--games-dir",
            str(tmp_path / "games"),
            "--output-dir",
            str(tmp_path / "output"),
            "--level",
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


def _write_valid_z8(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x08" + b"z" * (cli.MINIMUM_Z8_BYTES - 1))


def _game_paths(args):
    manifest = cli.build_coin_collector_manifest(
        args.games_dir,
        level=args.level,
        train_seeds=args.train_seeds,
        valid_seeds=args.valid_seeds,
        test_seeds=args.test_seeds,
    )
    return manifest, [Path(spec.game_file) for spec in manifest.games]


def _fake_episode(spec):
    game_sha256 = file_sha256(spec.game_file)
    counterfactual = CounterfactualRecord(
        action="look",
        next_obs="Nothing changes.",
        reward=0.0,
        done=False,
        won=False,
        lost=False,
        admissible_actions_after=("look", "take coin"),
    )
    step = CoinCollectorStep(
        step=0,
        observation="You see a coin.",
        admissible_actions=("look", "take coin"),
        action="take coin",
        next_obs="Taken.",
        reward=1.0,
        done=True,
        counterfactuals=(counterfactual,),
    )
    return CoinCollectorEpisode(
        split=spec.split,
        seed=spec.seed,
        level=spec.level,
        game_file=spec.game_file,
        game_sha256=game_sha256,
        game_uuid=f"tw-coin_collector-test-{spec.seed}",
        objective="Collect the coin.",
        initial_observation="You see a coin.",
        walkthrough=("take coin",),
        steps=(step,),
        won=True,
        generation_command=spec.generation_command,
    )


def test_parser_is_offline_and_non_overwriting_by_default(tmp_path):
    args = _args(tmp_path)
    assert args.generate is False
    assert args.overwrite is False
    assert args.train_seeds == [11]
    assert args.valid_seeds == [22]
    assert args.test_seeds == [33]
    assert args.counterfactual_limit == 4


def test_missing_game_without_generate_fails_before_collection(tmp_path):
    args = _args(tmp_path)
    collected = []

    def collector(*collector_args, **collector_kwargs):
        collected.append((collector_args, collector_kwargs))
        raise AssertionError("collector must not run")

    with pytest.raises(cli.DatasetRunError, match="missing"):
        cli.run_dataset(args, collector=collector)

    assert collected == []
    assert not args.output_dir.exists()


def test_invalid_z8_hard_fails_before_collection(tmp_path):
    args = _args(tmp_path)
    _, paths = _game_paths(args)
    for path in paths:
        _write_valid_z8(path)
    paths[1].write_bytes(b"not-a-z8" + b"x" * 80)

    with pytest.raises(cli.DatasetRunError, match="version byte"):
        cli.run_dataset(
            args,
            collector=lambda *unused_args, **unused_kwargs: pytest.fail(
                "collector must not run"
            ),
        )
    assert not args.output_dir.exists()


def test_existing_output_refuses_before_generation(tmp_path):
    args = _args(tmp_path, "--generate")
    conflict = args.output_dir / "train" / "summary.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("old", encoding="utf-8")
    calls = []

    with pytest.raises(cli.DatasetRunError, match="already exist"):
        cli.run_dataset(
            args,
            subprocess_run=lambda *call_args, **call_kwargs: calls.append(
                (call_args, call_kwargs)
            ),
        )
    assert calls == []
    assert conflict.read_text(encoding="utf-8") == "old"


def test_generate_uses_only_sibling_cli_and_writes_hashed_split_artifacts(
    tmp_path,
):
    args = _args(tmp_path, "--generate", "--counterfactual-limit", "2")
    environment_bin = tmp_path / "venv" / "Scripts"
    python = environment_bin / "python.exe"
    tw_make = environment_bin / "tw-make.exe"
    environment_bin.mkdir(parents=True)
    python.write_bytes(b"python-placeholder")
    tw_make.write_bytes(b"official-entrypoint-placeholder")
    subprocess_calls = []
    collector_calls = []

    def fake_run(command, *, check, shell):
        subprocess_calls.append((tuple(command), check, shell))
        output = Path(command[command.index("--output") + 1])
        _write_valid_z8(output)

    def fake_collector(spec, **kwargs):
        collector_calls.append((spec, kwargs))
        return _fake_episode(spec)

    result = cli.run_dataset(
        args,
        subprocess_run=fake_run,
        collector=fake_collector,
        python_executable=python,
    )

    assert len(subprocess_calls) == 3
    assert all(call[0][0] == str(tw_make.resolve()) for call in subprocess_calls)
    assert all(call[1] is True and call[2] is False for call in subprocess_calls)
    assert all("--force" not in call[0] for call in subprocess_calls)
    assert all("tw-make" not in call[0][0] or call[0][0] == str(tw_make.resolve()) for call in subprocess_calls)
    assert len(collector_calls) == 3
    assert all(
        call[1]["counterfactual_limit"] == 2 for call in collector_calls
    )

    for split, seed in (("train", 11), ("valid", 22), ("test", 33)):
        split_root = args.output_dir / split
        manifest_path = split_root / "manifest.json"
        jsonl_path = split_root / "episodes.jsonl"
        events_path = split_root / "token_events.txt"
        summary_path = split_root / "summary.json"
        assert all(
            path.is_file()
            for path in (manifest_path, jsonl_path, events_path, summary_path)
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["seeds"] == [seed]
        assert summary["level"] == 2
        assert summary["counterfactual_limit"] == 2
        assert summary["tw_make_executable"] == str(tw_make.resolve())
        assert summary["games"][0]["generated_this_run"] is True
        assert summary["games"][0]["executed_generation_command"][0] == str(
            tw_make.resolve()
        )
        assert "python" in summary["versions"]
        assert "textworld" in summary["versions"]
        assert summary["artifacts"]["manifest"]["sha256"] == file_sha256(
            manifest_path
        )
        assert summary["artifacts"]["episodes_jsonl"][
            "sha256"
        ] == file_sha256(jsonl_path)
        assert summary["artifacts"]["token_events"]["sha256"] == file_sha256(
            events_path
        )
        assert summary["games"][0]["sha256"] == file_sha256(
            summary["games"][0]["path"]
        )
        assert result["summary_artifacts"][split]["sha256"] == hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest()


def test_existing_games_are_reused_with_generate_without_overwrite(tmp_path):
    args = _args(tmp_path, "--generate")
    _, paths = _game_paths(args)
    for path in paths:
        _write_valid_z8(path)
    subprocess_calls = []

    cli.run_dataset(
        args,
        subprocess_run=lambda *call_args, **call_kwargs: subprocess_calls.append(
            (call_args, call_kwargs)
        ),
        collector=lambda spec, **unused_kwargs: _fake_episode(spec),
        python_executable=tmp_path / "missing-python" / "python.exe",
    )

    assert subprocess_calls == []
    for split in cli.SPLITS:
        summary = json.loads(
            (args.output_dir / split / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["games"][0]["generated_this_run"] is False
        assert summary["games"][0]["executed_generation_command"] is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX virtualenv symlink regression")
def test_resolver_preserves_symlinked_venv_bin_and_prefers_sys_prefix(
    tmp_path, monkeypatch
):
    base_bin = tmp_path / "usr" / "bin"
    venv_bin = tmp_path / ".venv-wsl" / "bin"
    base_python = base_bin / "python3.12"
    invoked_python = venv_bin / "python"
    tw_make = venv_bin / "tw-make"
    base_bin.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    base_python.write_bytes(b"base-python")
    invoked_python.symlink_to(base_python)
    tw_make.write_bytes(b"official-tw-make")
    assert invoked_python.resolve() == base_python.resolve()

    monkeypatch.setattr(cli.sys, "prefix", str(venv_bin.parent))
    monkeypatch.setattr(cli.sys, "executable", str(invoked_python))

    assert cli.resolve_sibling_tw_make() == tw_make.absolute()
    assert cli.resolve_sibling_tw_make(invoked_python) == tw_make.absolute()
