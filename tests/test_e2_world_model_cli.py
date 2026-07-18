import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from experiments import e2_world_model
from vpsc.world_model.wikitext import (
    WikiText2Source,
    WikiTextNotFoundError,
    file_sha256,
)


class E2WorldModelCLITests(unittest.TestCase):
    def _tiny_source(self, root: Path) -> WikiText2Source:
        archive = root / "tiny-cli-source.zip"
        train_tokens = " ".join(f"token{index}" for index in range(48))
        contents = {
            "tiny/wiki.train.raw": train_tokens + "\n" + train_tokens + "\n",
            "tiny/wiki.valid.raw": "token1 token2 token3 token4 token5 token6\n",
            "tiny/wiki.test.raw": "token7 token8 token9 token10 token11 token12\n",
        }
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for member, text in contents.items():
                handle.writestr(member, text.encode("utf-8"))
        return WikiText2Source(
            name="tiny-wikitext-cli-v1",
            version="fixture-1",
            url=archive.as_uri(),
            sha256=file_sha256(archive),
            archive_filename="tiny-wikitext-cli-v1.zip",
            archive_bytes=archive.stat().st_size,
            split_members=(
                ("train", "tiny/wiki.train.raw"),
                ("valid", "tiny/wiki.valid.raw"),
                ("test", "tiny/wiki.test.raw"),
            ),
            homepage="local offline fixture",
            license="test only",
        )

    def test_probe_writes_atomic_environment_and_official_adapter_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "probe.json"
            result = e2_world_model.main(["probe", "--output", str(output)])

            self.assertEqual(result, 0)
            with output.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["command"], "probe")
            self.assertEqual(payload["scope"], "pilot_not_confirmatory")
            self.assertFalse(payload["confirmatory"])
            self.assertIsNone(payload["automatic_decision"])
            self.assertEqual(payload["environment"]["pilot_device"], "cpu")
            self.assertIn("python_version", payload["environment"])
            self.assertIn("torch_version", payload["environment"])
            self.assertEqual(
                set(payload["official_adapter_probes"]),
                {"textworld", "homegrid", "messenger"},
            )
            self.assertFalse(payload["verify_imports_requested"])
            self.assertFalse(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                "atomic writer left a temporary file behind",
            )

    def test_wikitext_pilot_uses_verified_fixture_and_shared_harness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._tiny_source(root)
            cache = root / "cache"
            output = root / "results" / "pilot.json"
            argv = [
                "wikitext-pilot",
                "--cache-dir",
                str(cache),
                "--output",
                str(output),
                "--download",
                "--seeds",
                "7",
                "--d-model",
                "4",
                "--num-heads",
                "1",
                "--vocab",
                "64",
                "--batch",
                "1",
                "--seq",
                "2",
                "--steps",
                "1",
                "--eval-steps",
                "1",
                "--cache-window",
                "4",
                "--e2-policy",
                "hybrid",
                "--positive-factor",
                "0.8",
                "--streaming-warmup-steps",
                "0",
                "--streaming-steps",
                "1",
            ]

            with mock.patch.object(
                e2_world_model,
                "WIKITEXT2_RAW_SOURCE",
                source,
            ):
                result = e2_world_model.main(argv)

            self.assertEqual(result, 0)
            with output.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["command"], "wikitext-pilot")
            self.assertEqual(payload["scope"], "pilot_not_confirmatory")
            self.assertFalse(payload["confirmatory"])
            self.assertIsNone(payload["automatic_decision"])
            self.assertEqual(payload["device"], "cpu")
            self.assertNotIn("pass", payload)

            dataset = payload["dataset"]
            self.assertTrue(dataset["verified_archive"])
            self.assertFalse(dataset["synthetic"])
            self.assertTrue(dataset["download_requested"])
            self.assertEqual(dataset["source"]["sha256"], source.sha256)
            self.assertEqual(dataset["manifest"]["source"], source.metadata())
            self.assertEqual(dataset["vocabulary"]["built_from"], "train_only")
            self.assertLessEqual(dataset["vocabulary"]["actual_size"], 64)

            self.assertEqual(payload["config"]["seeds"], [7])
            self.assertTrue(payload["config"]["same_data_order_for_all_models"])
            self.assertTrue(payload["config"]["same_step_limits_for_all_models"])
            self.assertEqual(payload["config"]["e2_policy"], "hybrid")
            self.assertEqual(payload["config"]["e2_positive_factor"], 0.8)

            self.assertEqual(len(payload["results"]), 1)
            seed_result = payload["results"][0]
            self.assertEqual(seed_result["seed"], 7)
            self.assertTrue(seed_result["parameter_budget"]["within_tolerance"])
            self.assertEqual(seed_result["e2_effective_gains"]["i_to_e"], 0.0)
            self.assertEqual(
                set(seed_result["models"]),
                {"lstm", "transformer", "e2"},
            )
            for record in seed_result["models"].values():
                self.assertEqual(record["train"]["steps"], 1)
                self.assertEqual(record["valid"]["batches"], 1)
                self.assertEqual(record["test"]["batches"], 1)
                self.assertEqual(record["streaming"]["measured_steps"], 1)
                self.assertGreater(record["parameters"]["model_total"], 0)
                self.assertEqual(record["parameters"]["output_norm_total"], 8)

            self.assertFalse(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                "atomic writer left a temporary file behind",
            )

    def test_wikitext_default_does_not_authorize_download(self) -> None:
        args = e2_world_model.build_parser().parse_args(["wikitext-pilot"])
        self.assertFalse(args.download)
        with mock.patch.object(
            e2_world_model,
            "load_wikitext2",
            side_effect=WikiTextNotFoundError("fixture cache missing"),
        ) as loader:
            with self.assertRaises(WikiTextNotFoundError):
                e2_world_model.run_wikitext_pilot(args)
        self.assertFalse(loader.call_args.kwargs["download"])


if __name__ == "__main__":
    unittest.main()
