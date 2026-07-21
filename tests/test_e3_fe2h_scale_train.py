from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.e3_fe2h_scale_train import (
    ScaleConfig,
    _apply_shared_initialization,
    _build_scale_model,
    _full_epoch_batches,
    _route_specialization,
    _save_checkpoint,
)


def _config(root: Path, **overrides: object) -> ScaleConfig:
    values = {
        "out": root / "artifact.json",
        "cache_dir": root,
        "checkpoint": None,
        "label": "test",
        "d_model": 128,
        "state_dim": 128,
        "tile_size": 32,
        "active_tiles": 2,
        "block_size": 32,
        "rank": 16,
        "batch_size": 4,
        "seq_len": 32,
        "max_steps": 2,
        "train_mode": "random",
        "epochs": 1,
        "warmup_steps": 1,
        "valid_steps": 1,
        "full_validation": False,
        "log_every": 1,
        "finite_check_every": 1,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "route_supervision_weight": 0.01,
        "homeostasis_weight": 0.01,
        "seed": 0,
        "amp": False,
        "amp_init_scale": 256.0,
        "amp_growth_interval": 100_000,
        "low_rank_output": True,
        "sample_interval_ms": 200,
        "estimator_factor": 1.5,
        "estimator_overhead_gib": 0.75,
        "save_optimizer": False,
        "checkpoint_every_steps": 0,
        "tqdm_progress": False,
        "shared_init": None,
        "create_shared_init": False,
    }
    values.update(overrides)
    return ScaleConfig(**values)


class FullEpochBatchTests(unittest.TestCase):
    def test_each_epoch_covers_every_sequence_once_with_partial_tail(self) -> None:
        batches = list(
            _full_epoch_batches(
                10,
                4,
                2,
                generator=torch.Generator().manual_seed(7),
            )
        )
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2, 4, 4, 2])
        for epoch in range(2):
            observed = torch.cat(batches[epoch * 3 : (epoch + 1) * 3]).sort().values
            self.assertTrue(torch.equal(observed, torch.arange(10)))

    def test_seed_reproduces_identical_order(self) -> None:
        left = list(
            _full_epoch_batches(
                17,
                5,
                1,
                generator=torch.Generator().manual_seed(11),
            )
        )
        right = list(
            _full_epoch_batches(
                17,
                5,
                1,
                generator=torch.Generator().manual_seed(11),
            )
        )
        self.assertEqual(len(left), len(right))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))


class SharedInitializationTests(unittest.TestCase):
    def test_common_parameters_match_across_tile_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            init_path = root / "shared.pt"
            torch.manual_seed(0)
            coarse, _ = _build_scale_model(
                64,
                _config(
                    root,
                    shared_init=init_path,
                    create_shared_init=True,
                ),
            )
            coarse_record = _apply_shared_initialization(
                coarse,
                _config(
                    root,
                    shared_init=init_path,
                    create_shared_init=True,
                ),
            )
            self.assertEqual(coarse_record["matched_parameter_ratio"], 1.0)

            torch.manual_seed(0)
            micro_cfg = _config(
                root,
                tile_size=16,
                active_tiles=4,
                shared_init=init_path,
            )
            micro, _ = _build_scale_model(64, micro_cfg)
            micro_record = _apply_shared_initialization(micro, micro_cfg)
            self.assertGreater(micro_record["matched_parameter_ratio"], 0.99)
            self.assertEqual(
                {item["name"] for item in micro_record["unmatched_keys"]},
                {
                    "core.router_projection.net.2.weight",
                    "core.router_projection.net.2.bias",
                },
            )
            self.assertTrue(
                torch.equal(
                    coarse.embedding.weight,
                    micro.embedding.weight,
                )
            )
            self.assertTrue(
                torch.equal(
                    coarse.core.input_event_projection.weight,
                    micro.core.input_event_projection.weight,
                )
            )


class RouteSpecializationTests(unittest.TestCase):
    def test_perfect_token_route_relation_exceeds_shuffle_baseline(self) -> None:
        result = _route_specialization(
            [torch.tensor([1] * 100 + [2] * 100)],
            [torch.tensor([3] * 100 + [12] * 100)],
            hard_tile_sum=torch.tensor([100.0, 100.0, 100.0, 100.0]),
            hard_block_count=200,
            tile_count=4,
            active_tiles=2,
        )
        self.assertEqual(result["unique_route_count"], 2)
        self.assertAlmostEqual(result["top_route_share"], 0.5)
        self.assertGreater(result["excess_token_route_mutual_information_nats"], 0.5)
        self.assertEqual(result["hard_dead_tile_ratio"], 0.0)


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_contains_step_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rolling.pt"
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            cfg = _config(
                root,
                checkpoint=path,
                save_optimizer=True,
                checkpoint_every_steps=1000,
            )
            _save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                completed_steps=1000,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["completed_steps"], 1000)
            self.assertIn("optimizer", payload)
            self.assertEqual(payload["configuration"]["checkpoint_every_steps"], 1000)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
