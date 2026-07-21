from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.e3_fe2h_full_matched_summary import (
    EXPECTED_TRAIN_TOKENS,
    EXPECTED_TRAIN_STEPS,
    EXPECTED_VALID_TOKENS,
    EXPECTED_VALID_STEPS,
    VARIANTS,
    build_summary,
)


def _artifact(tile_count: int, active_tiles: int, validation_bpc: float) -> dict:
    tile_size = 8192 // tile_count
    specialization_pass = tile_count > 4
    return {
        "status": "COMPLETED",
        "failure": None,
        "configuration": {
            "d_model": 8192,
            "state_dim": 8192,
            "tile_size": tile_size,
            "active_tiles": active_tiles,
            "rank": 512,
            "batch_size": 112,
            "seq_len": 128,
            "block_size": 32,
            "seed": 0,
            "epochs": 1,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "route_supervision_weight": 0.01,
            "homeostasis_weight": 1.0,
            "amp": True,
            "amp_init_scale": 256.0,
            "amp_growth_interval": 100_000,
            "checkpoint_every_steps": 1_000,
            "tqdm_progress": True,
            "train_mode": "full_epoch",
            "full_validation": True,
        },
        "model": {
            "parameter_stats": {"model_total": 1000 + tile_count},
            "shared_initialization": {
                "enabled": True,
                "matched_parameter_ratio": 0.9999,
            },
        },
        "training": {
            "tokens": EXPECTED_TRAIN_TOKENS,
            "coverage_fraction": 1.0,
            "completed_steps": EXPECTED_TRAIN_STEPS,
            "elapsed_s": 100.0,
            "tokens_per_s": 1000.0 / tile_count,
            "mean_bpc": validation_bpc + 0.2,
            "windows": [{"bpc": 10.0}, {"bpc": validation_bpc + 0.1}],
        },
        "validation": {
            "tokens": EXPECTED_VALID_TOKENS,
            "coverage_fraction": 1.0,
            "finite": True,
            "steps": EXPECTED_VALID_STEPS,
            "elapsed_s": 10.0,
            "tokens_per_s": 5000.0,
            "bpc": validation_bpc,
            "homeostasis": {
                "activation_rate": [active_tiles / tile_count] * tile_count,
                "hotspot_share": active_tiles / tile_count,
                "dead_tile_ratio": 0.0,
            },
            "route_specialization": {
                "supported": True,
                "hard_activation_rate": [active_tiles / tile_count] * tile_count,
                "hard_hotspot_share": active_tiles / tile_count,
                "hard_dead_tile_ratio": 0.0,
                "unique_route_count": 2,
                "theoretical_route_count": 6,
                "route_entropy_nats": 0.6,
                "top_route_share": 0.6,
                "token_route_mutual_information_nats": 0.1,
                "shuffled_token_route_mutual_information_nats": 0.03,
                "excess_token_route_mutual_information_nats": 0.07,
                "excess_token_route_mi_over_route_entropy": 0.116,
            },
        },
        "matched_gates": {
            "training_complete": True,
            "validation_complete": True,
            "soft_homeostasis_pass": True,
            "hard_homeostasis_pass": True,
            "specialization_pass": specialization_pass,
        },
        "gpu_utilization": {
            "post_warmup": {
                "gpu_util_percent": {"mean": 50.0, "p50": 50.0, "p90": 60.0},
                "power_w": {"mean": 40.0, "p90": 50.0},
            }
        },
        "memory": {
            "torch_peak_allocated_gib": 1.0,
            "torch_peak_reserved_gib": 1.2,
            "device_samples_full_training": {"memory_used_mib": {"max": 1400.0}},
        },
        "checkpoint": {
            "path": "checkpoint.pt",
            "bytes": 1234,
            "includes_optimizer": True,
            "periodic_interval_steps": 1_000,
            "periodic_steps": [1_000],
            "last_saved_step": EXPECTED_TRAIN_STEPS,
            "atomic_replace": True,
        },
    }


class FullMatchedSummaryTests(unittest.TestCase):
    def test_complete_summary_applies_quality_and_sparsity_gates(self) -> None:
        bpcs = {
            "coarse_k2": 6.0,
            "micro_k16": 5.9,
            "micro_k8": 5.95,
            "micro_k4": 6.1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, tile_count, active_tiles in VARIANTS:
                path = root / f"e3_fe2h_full_{name}.json"
                path.write_text(
                    json.dumps(_artifact(tile_count, active_tiles, bpcs[name])),
                    encoding="utf-8",
                )
            summary = build_summary(root)
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(summary["decision"]["quality_winner"], "micro_k16")
        self.assertEqual(
            summary["decision"]["lowest_active_fraction_within_1pct_and_no_hard_dead_tiles"],
            "micro_k8",
        )
        self.assertTrue(summary["hypothesis"]["h1_micro50_improves"])
        self.assertTrue(summary["hypothesis"]["h3_any_micro_specialization_pass"])

    def test_missing_artifacts_are_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = build_summary(Path(temporary))
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(len(summary["audit_errors"]), 4)

    def test_missing_required_telemetry_and_checkpoint_are_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, tile_count, active_tiles in VARIANTS:
                artifact = _artifact(tile_count, active_tiles, 6.0)
                if name == "micro_k8":
                    artifact["gpu_utilization"]["post_warmup"]["power_w"]["mean"] = None
                    artifact["checkpoint"] = None
                (root / f"e3_fe2h_full_{name}.json").write_text(
                    json.dumps(artifact), encoding="utf-8"
                )
            summary = build_summary(root)
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertIn("micro_k8: power_mean_w telemetry missing or non-finite", summary["audit_errors"])
        self.assertIn("micro_k8: checkpoint path missing", summary["audit_errors"])


if __name__ == "__main__":
    unittest.main()
