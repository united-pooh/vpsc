from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import e2_homegrid_world_model


@dataclass(frozen=True)
class _Transition:
    reward_class: int
    done: bool = False


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    transitions: tuple[_Transition, ...]


@dataclass(frozen=True)
class _Chunks:
    split: str
    epochs: int


class _Vocabulary:
    def __len__(self) -> int:
        return 9


class _Corpus:
    def __init__(
        self,
        root: Path,
        *,
        train_action_steps: int = 2_500,
        test_changed_patches: int = 1_500,
    ) -> None:
        self.root = root.resolve()
        self.vocabulary = _Vocabulary()
        self.most_frequent_visual_token = 7
        self._counts = {
            "train": self._split_counts(
                episodes=32,
                transitions=100,
                read_steps=12,
                action_steps=train_action_steps,
                changed_patches=5_000,
            ),
            "valid": self._split_counts(
                episodes=8,
                transitions=20,
                read_steps=3,
                action_steps=17,
                changed_patches=500,
            ),
            "test": self._split_counts(
                episodes=8,
                transitions=24,
                read_steps=4,
                action_steps=20,
                changed_patches=test_changed_patches,
            ),
        }
        self._train_episode = _Episode(
            "train-fixture",
            (_Transition(0), _Transition(1), _Transition(2)),
        )
        self._test_episode = _Episode(
            "test-fixture",
            tuple(_Transition(0) for _ in range(96)),
        )

    @staticmethod
    def _split_counts(
        *,
        episodes: int,
        transitions: int,
        read_steps: int,
        action_steps: int,
        changed_patches: int,
    ) -> dict[str, object]:
        return {
            "episode_count": episodes,
            "transition_count": transitions,
            "read_step_count": read_steps,
            "action_step_count": action_steps,
            "changed_patch_count": changed_patches,
            "transitions_with_changed_patches": max(1, transitions // 3),
            "reward_sum": 1.5,
            "done_count": 0,
            "language_oov_current": 0,
            "language_oov_next": 0,
        }

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "fixture.homegrid_corpus.v1",
            "root": str(self.root),
            "language_vocabulary": {
                "size": len(self.vocabulary),
                "built_from": "train_current_and_next_only",
                "fingerprint_sha256": "a" * 64,
            },
            "splits": {
                split: {
                    **counts,
                    "summary_sha256": "b" * 64,
                    "manifest_sha256": "c" * 64,
                    "transitions_sha256": "d" * 64,
                }
                for split, counts in self._counts.items()
            },
        }

    def split_metadata(self, split: str) -> dict[str, object]:
        return dict(self._counts[split])

    def iter_episodes(self, split: str):
        if split == "train":
            return iter((self._train_episode,))
        if split == "test":
            return iter((self._test_episode,))
        return iter((_Episode("valid-fixture", (_Transition(0),)),))

    def iter_chunks(self, split: str, sequence_length: int, *, epochs: int = 1):
        del sequence_length
        return _Chunks(split, epochs)


def _training(corpus: _Corpus, chunks: _Chunks, config) -> dict[str, object]:
    transitions = int(corpus.split_metadata("train")["transition_count"]) * config.epochs
    return {
        "seed": config.seed,
        "epochs": config.epochs,
        "chunks": 12,
        "transitions": transitions,
        "visual_targets": transitions * 144,
        "weighted_loss": 2.0,
        "component_nll": {
            "visual": 1.0,
            "language": 1.1,
            "read": 0.4,
            "reward": 0.6,
            "done": 0.2,
        },
        "mean_gradient_norm": 0.5,
        "elapsed_seconds": 1.0,
        "transitions_per_second": 300.0,
        "visual_tokens_per_second": 43_200.0,
        "loss_weights": {
            "visual": 1.0,
            "language": 0.25,
            "read": 0.1,
            "reward": 0.1,
            "done": 0.0,
        },
        "reward_loss_enabled": True,
        "done_loss_enabled": False,
    }


def _visual_metric(count: int) -> dict[str, object]:
    return {
        "count": count,
        "nll": 1.2,
        "accuracy": 0.6,
        "macro_f1_present_targets": 0.5,
        "present_target_classes": [0, 1],
    }


def _evaluation(corpus: _Corpus, chunks: _Chunks) -> dict[str, object]:
    transitions = int(corpus.split_metadata(chunks.split)["transition_count"])
    return {
        "chunks": 2,
        "transitions": transitions,
        "visual": {
            name: _visual_metric(transitions * 144)
            for name in ("overall", "changed", "unchanged", "read_phase", "action_phase")
        },
        "next_language": {
            "enabled": True,
            "count": transitions,
            "nll": 0.7,
            "accuracy": 0.8,
            "brier": 0.3,
            "target_classes": [0, 1],
        },
        "next_read": {
            "enabled": True,
            "count": transitions,
            "nll": 0.2,
            "accuracy": 0.9,
            "brier": 0.1,
            "target_classes": [0, 1],
        },
        "reward": {
            "enabled": True,
            "count": transitions,
            "nll": 0.4,
            "accuracy": 0.7,
            "brier": 0.2,
            "target_classes": [0, 1, 2],
        },
        "done": {
            "enabled": False,
            "reason": "training split did not contain multiple target classes",
            "target_classes": [0],
        },
        "baselines": {
            "copy_current_frame": {
                "overall_accuracy": 0.75,
                "changed_accuracy": 0.0,
            },
            "train_global_frequency": {
                "token": 7,
                "overall_accuracy": 0.2,
            },
        },
        "change_mask_definition": "fixture",
    }


def _rollout() -> dict[str, object]:
    return {
        "horizons": {
            str(horizon): {
                "anchors": 10,
                "overall_accuracy": 0.5,
                "changed_accuracy": 0.25,
                "overall_patch_count": 1_440,
                "changed_patch_count": 120,
            }
            for horizon in (1, 3, 5, 10)
        },
        "anchor_phase": "action_only",
        "conditioning": "fixture",
    }


def _streaming() -> dict[str, object]:
    return {
        "warmup_steps": 32,
        "measured_steps": 64,
        "history_steps": 96,
        "latency_mean_ms": 0.5,
        "latency_p50_ms": 0.4,
        "latency_p95_ms": 0.8,
        "latency_p99_ms": 1.0,
        "transitions_per_second": 2_000.0,
        "state_nbytes": 256,
        "timed_scope": "fixture",
    }


class E2HomeGridWorldModelCLITests(unittest.TestCase):
    def _patches(self, corpus: _Corpus, training=None):
        train = training or (lambda model, chunks, config: _training(corpus, chunks, config))
        return (
            mock.patch.object(
                e2_homegrid_world_model,
                "load_homegrid_corpus",
                return_value=corpus,
            ),
            mock.patch.object(e2_homegrid_world_model, "train_homegrid_model", side_effect=train),
            mock.patch.object(
                e2_homegrid_world_model,
                "evaluate_homegrid_model",
                side_effect=lambda model, chunks, **kwargs: _evaluation(corpus, chunks),
            ),
            mock.patch.object(
                e2_homegrid_world_model,
                "evaluate_homegrid_rollouts",
                side_effect=lambda *args, **kwargs: _rollout(),
            ),
            mock.patch.object(
                e2_homegrid_world_model,
                "benchmark_homegrid_streaming",
                side_effect=lambda *args, **kwargs: _streaming(),
            ),
        )

    @staticmethod
    def _argv(corpus: _Corpus, output: Path) -> list[str]:
        return [
            "--corpus-dir",
            str(corpus.root),
            "--output",
            str(output),
            "--seeds",
            "7",
            "--d-model",
            "4",
            "--num-heads",
            "1",
        ]

    def test_preregistered_defaults(self) -> None:
        args = e2_homegrid_world_model.build_parser().parse_args([])
        self.assertEqual(args.corpus_dir, e2_homegrid_world_model.DEFAULT_CORPUS_DIR)
        self.assertEqual(args.output, e2_homegrid_world_model.DEFAULT_OUTPUT)
        self.assertEqual(args.seeds, [0, 1, 2])
        self.assertEqual(args.d_model, 32)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.sequence_length, 32)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.learning_rate, 1e-3)
        self.assertEqual(args.cache_window, 128)
        self.assertEqual(args.streaming_warmup_steps, 32)
        self.assertEqual(args.streaming_steps, 64)

    def test_mocked_cli_writes_ready_auditable_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = _Corpus(root / "corpus")
            output = root / "nested" / "pilot.json"
            patches = self._patches(corpus)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = e2_homegrid_world_model.main(self._argv(corpus, output))

            self.assertEqual(result, 0)
            with output.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["scope"], "pilot_not_confirmatory")
            self.assertFalse(payload["confirmatory"])
            self.assertIsNone(payload["automatic_decision"])
            self.assertEqual(payload["pipeline_status"], "READY")
            self.assertFalse(payload["pipeline_gate"]["model_ranking_used"])
            self.assertTrue(
                payload["pipeline_gate"]["criteria"]
                ["all_required_metrics_finite_and_complete"]["passed"]
            )
            self.assertEqual(
                payload["dataset"]["provenance"]["splits"]["test"]
                ["transitions_sha256"],
                "d" * 64,
            )
            self.assertEqual(payload["dataset"]["counts"]["train"]["action_step_count"], 2_500)
            self.assertEqual(payload["dataset"]["baselines"]["test"]
                             ["copy_current_frame"]["changed_accuracy"], 0.0)
            self.assertTrue(payload["loss_protocol"]["reward"]["enabled"])
            self.assertEqual(payload["loss_protocol"]["reward"]["train_target_classes"], [0, 1, 2])
            self.assertFalse(payload["loss_protocol"]["done"]["enabled"])
            self.assertEqual(payload["loss_protocol"]["done"]["train_target_classes"], [0])
            seed = payload["results"][0]
            self.assertTrue(seed["parameter_budget"]["within_tolerance"])
            self.assertTrue(seed["transition_consumption"]["identical"])
            self.assertEqual(set(seed["models"]), {"lstm", "transformer", "e2"})
            self.assertEqual(payload["aggregate"]["seed_count"], 1)
            self.assertFalse(list(output.parent.glob(f".{output.name}.*.tmp")))

    def test_below_data_threshold_reports_pipeline_revise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = _Corpus(root / "corpus", train_action_steps=1_999)
            output = root / "pilot.json"
            patches = self._patches(corpus)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                e2_homegrid_world_model.main(self._argv(corpus, output))
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["pipeline_status"], "PIPELINE_REVISE")
            criterion = payload["pipeline_gate"]["criteria"]
            self.assertFalse(criterion["train_action_steps_at_least_2000"]["passed"])
            self.assertTrue(criterion["test_changed_patches_at_least_1000"]["passed"])

    def test_existing_output_is_refused_before_corpus_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pilot.json"
            output.write_text("sentinel\n", encoding="utf-8")
            args = e2_homegrid_world_model.build_parser().parse_args(
                ["--corpus-dir", str(root / "missing"), "--output", str(output)]
            )
            with mock.patch.object(e2_homegrid_world_model, "load_homegrid_corpus") as loader:
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    e2_homegrid_world_model.run_homegrid_pilot(args)
            loader.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

    def test_transition_fairness_mismatch_is_fatal_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = _Corpus(root / "corpus")
            output = root / "pilot.json"
            calls = 0

            def unfair_training(model, chunks, config):
                nonlocal calls
                calls += 1
                record = _training(corpus, chunks, config)
                if calls == 2:
                    record["transitions"] = int(record["transitions"]) - 1
                return record

            patches = self._patches(corpus, unfair_training)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(RuntimeError, "consumed different transitions"):
                    e2_homegrid_world_model.main(self._argv(corpus, output))
            self.assertFalse(output.exists())

    def test_nonfinite_required_metric_is_fatal_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = _Corpus(root / "corpus")
            output = root / "pilot.json"

            def nonfinite_training(model, chunks, config):
                record = _training(corpus, chunks, config)
                record["weighted_loss"] = float("nan")
                return record

            patches = self._patches(corpus, nonfinite_training)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaisesRegex(FloatingPointError, "non-finite"):
                    e2_homegrid_world_model.main(self._argv(corpus, output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
