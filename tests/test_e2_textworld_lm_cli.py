import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import e2_textworld_lm
from vpsc.world_model.textworld_dataset import (
    CoinCollectorEpisode,
    CoinCollectorGameSpec,
    CoinCollectorStep,
    CounterfactualRecord,
    episode_to_json_line,
    episode_to_token_event_text,
)
from vpsc.world_model.wikitext import file_sha256


FROZEN_SEEDS = {
    "train": (20260718, 20260719, 20260720, 20260721),
    "valid": (20260722,),
    "test": (20260723,),
}


def _write_json(path: Path, value: object) -> None:
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


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_z8(path: Path, seed: int) -> None:
    marker = f"official-textworld-fixture-{seed}".encode("ascii")
    content = b"\x08" + marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content + b"z" * (64 - len(content)))


def _episode(
    spec: CoinCollectorGameSpec,
    *,
    game_sha256: str,
    words: str,
) -> CoinCollectorEpisode:
    counterfactuals = (
        CounterfactualRecord(
            action="look",
            next_obs="The coin remains in the room.",
            reward=0.0,
            done=False,
            won=False,
            lost=False,
            admissible_actions_after=("look", "take coin"),
        ),
        CounterfactualRecord(
            action="inventory",
            next_obs="You are carrying nothing.",
            reward=0.0,
            done=False,
            won=False,
            lost=False,
            admissible_actions_after=("look", "take coin"),
        ),
    )
    step = CoinCollectorStep(
        step=0,
        observation=f"A room contains a coin and {words}.",
        admissible_actions=("look", "inventory", "take coin"),
        action="take coin",
        next_obs="You collect the coin and win.",
        reward=1.0,
        done=True,
        counterfactuals=counterfactuals,
    )
    return CoinCollectorEpisode(
        split=spec.split,
        seed=spec.seed,
        level=spec.level,
        game_file=spec.game_file,
        game_sha256=game_sha256,
        game_uuid=f"tw-coin_collector-fixture-{spec.seed}",
        objective="Collect the coin.",
        initial_observation=step.observation,
        walkthrough=("take coin",),
        steps=(step,),
        won=True,
        generation_command=spec.generation_command,
    )


def _write_fixture(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    games_root = root / "games"
    train_words = " ".join(f"trainword{index}" for index in range(72))
    split_words = {
        "train": train_words,
        "valid": "validation violet token",
        "test": "testing amber token",
    }

    for split, seeds in FROZEN_SEEDS.items():
        split_root = root / split
        split_root.mkdir(parents=True)
        specs = []
        episodes = []
        summary_games = []
        for seed in seeds:
            game_path = (
                games_root / f"tw_coin_collector_l005_s{seed:010d}.z8"
            ).resolve()
            _write_z8(game_path, seed)
            spec = CoinCollectorGameSpec(
                seed=seed,
                level=5,
                split=split,
                game_file=str(game_path),
            )
            game_sha256 = file_sha256(game_path)
            specs.append(spec)
            episodes.append(
                _episode(
                    spec,
                    game_sha256=game_sha256,
                    words=split_words[split],
                )
            )
            summary_games.append(
                {
                    "seed": seed,
                    "level": 5,
                    "split": split,
                    "path": str(game_path),
                    "sha256": game_sha256,
                    "size_bytes": game_path.stat().st_size,
                    "generation_command": list(spec.generation_command),
                    "generation_command_text": spec.generation_command_text,
                    "generated_this_run": True,
                    "executed_generation_command": list(spec.generation_command),
                }
            )
            paths[f"game_{seed}"] = game_path

        manifest_path = split_root / "manifest.json"
        episodes_path = split_root / "episodes.jsonl"
        event_path = split_root / "token_events.txt"
        summary_path = split_root / "summary.json"
        manifest = {
            "schema_version": "vpsc.textworld.coin_collector.v1",
            "runner_schema_version": "vpsc.e2_textworld_dataset.v1",
            "challenge": "tw-coin_collector",
            "level": 5,
            "split": split,
            "split_key": "game_seed",
            "games": [spec.to_record() for spec in specs],
        }
        _write_json(manifest_path, manifest)
        episodes_path.write_text(
            "\n".join(episode_to_json_line(episode) for episode in episodes)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        event_path.write_text(
            "".join(episode_to_token_event_text(episode) for episode in episodes),
            encoding="utf-8",
            newline="\n",
        )
        summary = {
            "schema_version": "vpsc.e2_textworld_dataset.v1",
            "challenge": "tw-coin_collector",
            "split": split,
            "split_key": "game_seed",
            "level": 5,
            "seeds": list(seeds),
            "counterfactual_limit": 2,
            "generate_requested": True,
            "overwrite_requested": False,
            "allow_unsupported_platform": False,
            "python_executable": "fixture-python",
            "python_runtime": {
                "invoked_executable": "fixture-python",
                "prefix": "fixture-prefix",
                "base_prefix": "fixture-prefix",
                "base_executable": "fixture-python",
            },
            "tw_make_executable": "fixture-tw-make",
            "versions": {
                "runner": "1",
                "dataset_schema": "vpsc.textworld.coin_collector.v1",
                "python": "fixture",
                "python_implementation": "CPython",
                "textworld": "1.7.0",
                "vpsc": "fixture",
                "platform": "fixture",
            },
            "games": summary_games,
            "artifacts": {
                "manifest": _artifact_record(manifest_path),
                "episodes_jsonl": _artifact_record(episodes_path),
                "token_events": _artifact_record(event_path),
            },
        }
        _write_json(summary_path, summary)
        paths[f"{split}_events"] = event_path
        paths[f"{split}_manifest"] = manifest_path
        paths[f"{split}_episodes"] = episodes_path
        paths[f"{split}_summary"] = summary_path
    return paths


def _refresh_manifest_artifact(summary_path: Path, manifest_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["manifest"] = _artifact_record(manifest_path)
    _write_json(summary_path, summary)


class E2TextWorldLMCLITests(unittest.TestCase):
    def test_preregistered_defaults(self) -> None:
        args = e2_textworld_lm.build_parser().parse_args([])

        self.assertEqual(args.seeds, [0, 1, 2])
        self.assertEqual(args.d_model, 32)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.sequence_length, 64)
        self.assertEqual(args.steps, 100)
        self.assertEqual(args.eval_steps, 50)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.cache_window, 128)

    def test_real_event_cli_runs_shared_suite_and_writes_auditable_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "textworld_l5"
            paths = _write_fixture(corpus_root)
            output = root / "nested" / "pilot.json"
            argv = [
                "--corpus-dir",
                str(corpus_root),
                "--output",
                str(output),
                "--seeds",
                "7",
                "--d-model",
                "4",
                "--num-heads",
                "1",
                "--seq",
                "4",
                "--steps",
                "1",
                "--eval-max",
                "1",
                "--cache-window",
                "4",
                "--streaming-warmup-steps",
                "0",
                "--streaming-steps",
                "1",
            ]

            result = e2_textworld_lm.main(argv)

            self.assertEqual(result, 0)
            with output.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["scope"], "pilot_not_confirmatory")
            self.assertFalse(payload["confirmatory"])
            self.assertIsNone(payload["automatic_decision"])
            self.assertFalse(payload["dataset"]["synthetic"])
            self.assertFalse(payload["dataset"]["fallback_used"])

            corpus = payload["dataset"]["event_corpus"]
            manifests = payload["dataset"]["manifests"]
            self.assertEqual(
                corpus["splits"]["train"]["source_sha256"],
                file_sha256(paths["train_events"]),
            )
            self.assertEqual(
                manifests["splits"]["test"]["sha256"],
                file_sha256(paths["test_manifest"]),
            )
            self.assertEqual(len(corpus["corpus_fingerprint_sha256"]), 64)
            self.assertEqual(len(manifests["fingerprint_sha256"]), 64)
            self.assertTrue(manifests["verified"])
            self.assertTrue(manifests["cross_split_seed_disjoint"])
            self.assertEqual(manifests["textworld_version"], "1.7.0")
            self.assertEqual(
                manifests["frozen_seeds"],
                {split: list(seeds) for split, seeds in FROZEN_SEEDS.items()},
            )
            for split, seeds in FROZEN_SEEDS.items():
                provenance = manifests["splits"][split]
                self.assertTrue(provenance["verified"])
                self.assertEqual(provenance["seeds"], list(seeds))
                self.assertEqual(provenance["game_count"], len(seeds))
                self.assertEqual(provenance["episode_count"], len(seeds))
                self.assertEqual(provenance["event_episode_count"], len(seeds))
                self.assertEqual(provenance["counterfactual_limit"], 2)
                self.assertEqual(provenance["textworld_version"], "1.7.0")
                self.assertEqual(
                    set(provenance["artifacts"]),
                    {"manifest", "episodes_jsonl", "token_events"},
                )
                self.assertEqual(len(provenance["games"]), len(seeds))

            reset = payload["dataset"]["episode_reset_audit"]
            self.assertTrue(reset["no_chunk_crosses_episode_boundary"])
            self.assertEqual(reset["splits"]["train"]["episode_count"], 4)
            self.assertEqual(reset["splits"]["train"]["reset_chunk_count"], 4)
            self.assertTrue(reset["splits"]["train"]["verified"])

            accounting = payload["dataset"]["token_accounting"]
            self.assertEqual(
                accounting["available_target_tokens"]["train"],
                corpus["splits"]["train"]["token_count"] - 4,
            )
            self.assertEqual(
                accounting["consumed_training_target_tokens_per_model"], 4
            )
            self.assertTrue(
                accounting["one_deterministic_pass_without_episode_repeat"]
            )
            self.assertTrue(accounting["same_counts_across_models_and_seeds"])
            expected_pipeline_status = (
                "READY" if all(payload["pipeline_checks"].values()) else "REVISE"
            )
            self.assertEqual(payload["pipeline_status"], expected_pipeline_status)
            self.assertTrue(payload["pipeline_checks"]["all_metrics_finite"])
            self.assertTrue(payload["pipeline_checks"]["equal_data_consumption"])

            self.assertEqual(payload["config"]["e2_policy"], "hybrid")
            self.assertEqual(payload["config"]["e2_positive_factor"], 0.8)
            self.assertEqual(payload["config"]["parameter_tolerance"], 0.02)
            seed_result = payload["results"][0]
            self.assertTrue(seed_result["parameter_budget"]["within_tolerance"])
            self.assertLessEqual(
                seed_result["parameter_budget"]["relative_spread"],
                0.02,
            )
            self.assertEqual(set(seed_result["models"]), {"lstm", "transformer", "e2"})
            for record in seed_result["models"].values():
                self.assertEqual(record["training"]["steps"], 1)
                self.assertGreater(record["training"]["tokens_per_second"], 0.0)
                for split in ("train", "valid", "test"):
                    self.assertEqual(record["evaluation"][split]["batches"], 1)
                    self.assertGreater(record["evaluation"][split]["nll"], 0.0)
                    self.assertGreater(record["evaluation"][split]["ppl"], 1.0)
                streaming = record["streaming"]
                self.assertIn("latency_p50_ms", streaming)
                self.assertIn("latency_p95_ms", streaming)
                self.assertIn("latency_p99_ms", streaming)
                self.assertIn("state_nbytes", streaming)

            aggregate = payload["aggregate"]
            self.assertEqual(aggregate["seed_count"], 1)
            self.assertEqual(aggregate["seeds"], [7])
            self.assertEqual(
                aggregate["models"]["lstm"]["evaluation"]["test"]["nll"]["count"],
                1,
            )
            self.assertFalse(list(output.parent.glob(f".{output.name}.*.tmp")))

    def test_existing_output_is_refused_before_loading_or_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pilot.json"
            output.write_text("sentinel\n", encoding="utf-8")
            args = e2_textworld_lm.build_parser().parse_args(
                ["--corpus-dir", str(root / "missing"), "--output", str(output)]
            )

            with mock.patch.object(e2_textworld_lm, "load_event_corpus") as loader:
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    e2_textworld_lm.run_textworld_lm(args)

            loader.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

    def test_missing_manifest_is_a_hard_failure_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "textworld_l5"
            _write_fixture(corpus_root)
            (corpus_root / "valid" / "manifest.json").unlink()
            output = root / "pilot.json"
            args = e2_textworld_lm.build_parser().parse_args(
                ["--corpus-dir", str(corpus_root), "--output", str(output)]
            )

            with self.assertRaisesRegex(FileNotFoundError, "no fallback was used"):
                e2_textworld_lm.run_textworld_lm(args)

            self.assertFalse(output.exists())

    def test_tampered_hashed_artifact_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "textworld_l5"
            paths = _write_fixture(corpus_root)
            manifest = json.loads(paths["valid_manifest"].read_text(encoding="utf-8"))
            manifest["tampered_after_summary"] = True
            _write_json(paths["valid_manifest"], manifest)
            output = root / "pilot.json"
            args = e2_textworld_lm.build_parser().parse_args(
                ["--corpus-dir", str(corpus_root), "--output", str(output)]
            )

            with self.assertRaisesRegex(ValueError, "artifact SHA256 mismatch"):
                e2_textworld_lm.run_textworld_lm(args)

            self.assertFalse(output.exists())

    def test_wrong_textworld_version_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "textworld_l5"
            paths = _write_fixture(corpus_root)
            summary = json.loads(paths["train_summary"].read_text(encoding="utf-8"))
            summary["versions"]["textworld"] = "1.6.0"
            _write_json(paths["train_summary"], summary)
            output = root / "pilot.json"
            args = e2_textworld_lm.build_parser().parse_args(
                ["--corpus-dir", str(corpus_root), "--output", str(output)]
            )

            with self.assertRaisesRegex(ValueError, "TextWorld version must be 1.7.0"):
                e2_textworld_lm.run_textworld_lm(args)

            self.assertFalse(output.exists())

    def test_cross_split_seed_leakage_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "textworld_l5"
            paths = _write_fixture(corpus_root)
            manifest = json.loads(paths["test_manifest"].read_text(encoding="utf-8"))
            manifest["games"][0]["seed"] = FROZEN_SEEDS["valid"][0]
            _write_json(paths["test_manifest"], manifest)
            _refresh_manifest_artifact(
                paths["test_summary"], paths["test_manifest"]
            )
            output = root / "pilot.json"
            args = e2_textworld_lm.build_parser().parse_args(
                ["--corpus-dir", str(corpus_root), "--output", str(output)]
            )

            with self.assertRaisesRegex(ValueError, "leaks across"):
                e2_textworld_lm.run_textworld_lm(args)

            self.assertFalse(output.exists())

    def test_wrong_but_disjoint_seed_is_not_accepted_as_frozen_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "textworld_l5"
            paths = _write_fixture(corpus_root)
            replacement_seed = 20260724
            manifest = json.loads(paths["test_manifest"].read_text(encoding="utf-8"))
            manifest["games"][0]["seed"] = replacement_seed
            _write_json(paths["test_manifest"], manifest)
            summary = json.loads(paths["test_summary"].read_text(encoding="utf-8"))
            summary["seeds"] = [replacement_seed]
            summary["games"][0]["seed"] = replacement_seed
            summary["artifacts"]["manifest"] = _artifact_record(
                paths["test_manifest"]
            )
            _write_json(paths["test_summary"], summary)
            output = root / "pilot.json"
            args = e2_textworld_lm.build_parser().parse_args(
                ["--corpus-dir", str(corpus_root), "--output", str(output)]
            )

            with self.assertRaisesRegex(ValueError, "frozen seeds must be"):
                e2_textworld_lm.run_textworld_lm(args)

            self.assertFalse(output.exists())

    def test_missing_real_z8_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus_root = root / "textworld_l5"
            paths = _write_fixture(corpus_root)
            paths[f"game_{FROZEN_SEEDS['train'][0]}"].unlink()
            output = root / "pilot.json"
            args = e2_textworld_lm.build_parser().parse_args(
                ["--corpus-dir", str(corpus_root), "--output", str(output)]
            )

            with self.assertRaisesRegex(
                FileNotFoundError, "required real TextWorld .z8 is missing"
            ):
                e2_textworld_lm.run_textworld_lm(args)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
