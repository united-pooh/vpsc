import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import e3_fe2h_neuron_tile as experiment


def _honest_variant_paths():
    return {
        "base_e3": {
            "path_label": "base_e3",
            "supported": True,
            "hardware_executed_sparsity": False,
        },
        "fe2h_dense_mask": {
            "path_label": "dense_mask",
            "supported": True,
            "hardware_executed_sparsity": False,
        },
        "fe2h_tile_sparse": {
            "path_label": "tile_sparse",
            "supported": True,
            "hardware_executed_sparsity": True,
            "unsupported_reason": None,
        },
        "fe2h_low_rank_tile_sparse": {
            "path_label": "low_rank_tile_sparse",
            "supported": True,
            "hardware_executed_sparsity": True,
            "dense_input_projection_retained": True,
            "unsupported_reason": None,
        },
        "all_lowrank_dense_mask": {
            "path_label": "all_lowrank_dense_mask",
            "supported": False,
            "hardware_executed_sparsity": False,
            "unsupported_reason": "router dims are too small for rank 16/32",
        },
    }


def _minimal_artifact():
    variant_paths = _honest_variant_paths()
    decision = experiment.make_gate_decision(
        mechanism={"status": "PASS"},
        numerics={"status": "PASS"},
        memory={"status": "PASS"},
        speed={"status": "PASS"},
        quality={"status": "PASS"},
        variant_paths=variant_paths,
    )
    return {
        "schema_version": experiment.SCHEMA_VERSION,
        "formal": False,
        "environment": {"resolved_device": "cpu"},
        "configuration": {"mode": "smoke"},
        "provenance": {"variant_paths": variant_paths},
        "mechanism": {"status": "PASS"},
        "numerics": {"status": "PASS"},
        "memory": {"status": "PASS"},
        "speed": {"status": "PASS"},
        "quality": {"status": "PASS"},
        "decision": decision,
    }


class _FakeAuxModel(torch.nn.Module):
    def __init__(self, *, router_weight: float = 1.0) -> None:
        super().__init__()
        self.router_weight = torch.nn.Parameter(torch.tensor(router_weight))
        self.ce_anchor = torch.nn.Parameter(torch.tensor(0.0))


class FE2HNeuronTileArtifactTests(unittest.TestCase):
    def test_schema_validation_requires_top_level_fields(self) -> None:
        artifact = _minimal_artifact()
        artifact.pop("decision")
        artifact.pop("quality")
        errors = experiment.validate_artifact_schema(artifact)
        self.assertIn("missing required field: decision", errors)
        self.assertIn("missing required field: quality", errors)

    def test_predicted_memory_gate_pauses_above_16_gib(self) -> None:
        gate = experiment.predicted_memory_gate(16.01)
        self.assertEqual(gate["status"], "PAUSE")
        self.assertFalse(gate["can_launch"])

    def test_predicted_memory_gate_refuses_above_32_gib(self) -> None:
        gate = experiment.predicted_memory_gate(32.01)
        self.assertEqual(gate["status"], "REFUSE")
        self.assertFalse(gate["can_launch"])
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = experiment.ExperimentConfig(
                mode="smoke",
                device_request="cpu",
                out=Path(tmpdir) / "artifact.json",
                cache_dir=Path(tmpdir) / "cache",
                d_model=32,
                state_dim=64,
                tile_size=16,
                active_tiles=2,
                block_size=4,
                rank=16,
                batch_size=2,
                seq_len=8,
                epochs=1,
                warmup_steps=1,
                benchmark_steps=1,
                smoke_vocab_size=32,
                smoke_train_sequences=4,
                smoke_valid_sequences=2,
            )
            fake_fe2h_memory = {
                "core_total_gib": 40.0,
                "core_forward_only_gib": 20.0,
                "model_total_gib": 40.0,
                "projection_bytes": 1,
                "core": {},
            }
            with mock.patch.object(
                experiment,
                "_estimate_generic_model_memory_gib",
                return_value=40.0,
            ), mock.patch.object(
                experiment,
                "_estimate_fe2h_memory",
                return_value=fake_fe2h_memory,
            ), mock.patch.object(
                experiment.CausalLanguageModel,
                "to",
                side_effect=AssertionError("device transfer should not happen before memory preflight"),
            ), mock.patch.object(
                experiment,
                "_mechanism_equivalence_record",
                side_effect=AssertionError("mechanism should not run after refuse preflight"),
            ), mock.patch.object(
                experiment,
                "_numerics_records",
                side_effect=AssertionError("numerics should not run after refuse preflight"),
            ):
                artifact = experiment._run_experiment(cfg)
            self.assertEqual(artifact["memory"]["status"], "REFUSE")
            self.assertEqual(artifact["mechanism"]["status"], "NOT_RUN")
            self.assertEqual(artifact["numerics"]["status"], "NOT_RUN")
            self.assertEqual(artifact["speed"]["status"], "NOT_RUN")
            self.assertEqual(artifact["quality"]["status"], "NOT_RUN")
            self.assertEqual(artifact["decision"]["memory_gate"], "REFUSE")
            self.assertEqual(artifact["decision"]["mechanism_gate"], "NOT_RUN")
            self.assertEqual(artifact["decision"]["first_blocker"], "memory")
            self.assertEqual(artifact["decision"]["overall"], "REFUSE")
            self.assertEqual(experiment.validate_artifact_schema(artifact), [])

    def test_unsupported_backend_fail_closes_later_gates(self) -> None:
        decision = experiment.make_gate_decision(
            mechanism={"status": "UNSUPPORTED", "reason": "requested CUDA but unavailable"},
            numerics={"status": "PASS"},
            memory={"status": "PASS"},
            speed={"status": "PASS"},
            quality={"status": "PASS"},
            variant_paths=_honest_variant_paths(),
        )
        self.assertEqual(decision["overall"], "FAIL")
        self.assertEqual(decision["mechanism_gate"], "UNSUPPORTED")
        self.assertEqual(decision["numerics_gate"], "NOT_RUN")
        self.assertEqual(decision["quality_gate"], "NOT_RUN")

    def test_dense_fallback_is_rejected_and_path_labels_must_stay_honest(self) -> None:
        dishonest = _honest_variant_paths()
        dishonest["fe2h_dense_mask"]["hardware_executed_sparsity"] = True
        dishonest["fe2h_tile_sparse"]["hardware_executed_sparsity"] = False
        errors = experiment.validate_variant_paths(dishonest)
        self.assertTrue(
            any("dense-mask path cannot claim hardware_executed_sparsity=true" in error for error in errors)
        )
        self.assertTrue(
            any("supported tile-sparse path must report real hardware_executed_sparsity" in error for error in errors)
        )

    def test_no_speedup_negative_result_is_retained_and_quality_is_gated(self) -> None:
        decision = experiment.make_gate_decision(
            mechanism={"status": "PASS"},
            numerics={"status": "PASS"},
            memory={"status": "PASS"},
            speed={
                "status": "FAIL",
                "best_sparse_speedup_over_base_e3": 0.84,
                "retained_negative_result": True,
            },
            quality={"status": "PASS"},
            variant_paths=_honest_variant_paths(),
        )
        self.assertEqual(decision["overall"], "NEGATIVE")
        self.assertEqual(decision["speed_gate"], "FAIL")
        self.assertEqual(decision["quality_gate"], "NOT_RUN")
        self.assertTrue(decision["retained_negative_result"])

    def test_homeostasis_gate_passes_healthy_metrics_and_blocks_collapsed_mechanism(self) -> None:
        class FakeHomeostasis:
            def __init__(
                self,
                activation_rate,
                entropy=1.10,
                gini=0.08,
                p99_tile_load=0.62,
                block_fill=0.50,
                hotspot_share=0.62,
                dead_tile_ratio=0.0,
                target_activation_rate=0.50,
            ) -> None:
                self.activation_rate = activation_rate
                self.entropy = entropy
                self.gini = gini
                self.p99_tile_load = p99_tile_load
                self.block_fill = block_fill
                self.hotspot_share = hotspot_share
                self.dead_tile_ratio = dead_tile_ratio
                self.target_activation_rate = target_activation_rate

        healthy = FakeHomeostasis(
            activation_rate=experiment.torch.tensor([0.58, 0.48, 0.45, 0.49])
        )
        healthy_gate = experiment._homeostasis_gate_result(healthy, batch_windows=8)
        self.assertEqual(healthy_gate["status"], "PASS")
        self.assertEqual(healthy_gate["failures"], [])
        self.assertIn("activation_rate", healthy_gate)
        self.assertIn("hotspot_share", healthy_gate)
        self.assertEqual(healthy_gate["batch_windows"], 8)

        collapsed = FakeHomeostasis(
            activation_rate=experiment.torch.tensor([1.0, 0.7, 0.3, 0.0]),
            hotspot_share=0.71,
            dead_tile_ratio=0.25,
        )
        collapsed_gate = experiment._homeostasis_gate_result(collapsed, batch_windows=8)
        self.assertEqual(collapsed_gate["status"], "FAIL")
        self.assertIn("hotspot_share_exceeds_0.70", collapsed_gate["failures"])
        self.assertIn("dead_tile_ratio_nonzero", collapsed_gate["failures"])

        decision = experiment.make_gate_decision(
            mechanism={"status": "FAIL", "homeostasis": collapsed_gate},
            numerics={"status": "PASS"},
            memory={"status": "PASS"},
            speed={"status": "PASS"},
            quality={"status": "PASS"},
            variant_paths=_honest_variant_paths(),
        )
        self.assertEqual(decision["mechanism_gate"], "FAIL")
        self.assertEqual(decision["numerics_gate"], "NOT_RUN")
        self.assertEqual(decision["speed_gate"], "NOT_RUN")
        self.assertEqual(decision["quality_gate"], "NOT_RUN")

    def test_training_aux_terms_drive_finite_nonzero_router_grad(self) -> None:
        model = _FakeAuxModel()

        def fake_forward(_model, _inputs, *, targets=None, route_override=None, sparse_inference=False):
            self.assertIs(_model, model)
            self.assertIsNotNone(targets)
            self.assertIsNone(route_override)
            self.assertFalse(sparse_inference)
            return {
                "loss": model.ce_anchor * 0.0 + 1.25,
                "diagnostics": SimpleNamespace(
                    route_supervision_loss=model.router_weight * 2.0,
                    homeostasis=SimpleNamespace(loss=model.router_weight.square() * 3.0),
                ),
            }

        with mock.patch.object(experiment, "_forward_model", side_effect=fake_forward):
            record = experiment._run_epoch(
                model,
                torch.zeros((1, 2), dtype=torch.long),
                torch.zeros((1, 2), dtype=torch.long),
                batch_size=1,
                device=torch.device("cpu"),
                optimizer=experiment._optimizer(model),
                seed=0,
            )

        self.assertIsNone(record["first_failure"])
        self.assertTrue(record["loss_breakdown"]["aux_applied"])
        self.assertAlmostEqual(record["loss_breakdown"]["route_supervision"], 2.0, places=6)
        self.assertAlmostEqual(record["loss_breakdown"]["homeostasis"], 3.0, places=6)
        grad = model.router_weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool(torch.isfinite(grad).all().item()))
        self.assertGreater(float(grad.abs().sum().item()), 0.0)

    def test_training_bpc_uses_cross_entropy_only_even_with_aux_terms(self) -> None:
        model = _FakeAuxModel()

        def fake_forward(_model, _inputs, *, targets=None, route_override=None, sparse_inference=False):
            self.assertIs(_model, model)
            self.assertIsNotNone(targets)
            self.assertIsNone(route_override)
            self.assertFalse(sparse_inference)
            return {
                "loss": model.ce_anchor * 0.0 + 2.0,
                "diagnostics": SimpleNamespace(
                    route_supervision_loss=model.router_weight * 20.0,
                    homeostasis=SimpleNamespace(loss=model.router_weight * 30.0),
                ),
            }

        with mock.patch.object(experiment, "_forward_model", side_effect=fake_forward):
            record = experiment._run_epoch(
                model,
                torch.zeros((1, 3), dtype=torch.long),
                torch.zeros((1, 3), dtype=torch.long),
                batch_size=1,
                device=torch.device("cpu"),
                optimizer=experiment._optimizer(model),
                seed=0,
            )

        self.assertIsNone(record["first_failure"])
        self.assertAlmostEqual(record["ce"], 2.0, places=6)
        self.assertAlmostEqual(record["bpc"], 2.0 / math.log(2.0), places=6)
        self.assertAlmostEqual(record["loss_breakdown"]["route_supervision"], 20.0, places=6)
        self.assertAlmostEqual(record["loss_breakdown"]["homeostasis"], 30.0, places=6)
        self.assertAlmostEqual(record["loss_breakdown"]["total"], 2.5, places=6)

    def test_nonfinite_aux_loss_fails_closed_before_optimizer_step(self) -> None:
        model = _FakeAuxModel(router_weight=1.5)
        router_before = float(model.router_weight.detach().item())

        def fake_forward(_model, _inputs, *, targets=None, route_override=None, sparse_inference=False):
            self.assertIs(_model, model)
            self.assertIsNotNone(targets)
            self.assertIsNone(route_override)
            self.assertFalse(sparse_inference)
            return {
                "loss": model.ce_anchor * 0.0 + 1.0,
                "diagnostics": SimpleNamespace(
                    route_supervision_loss=model.router_weight * torch.tensor(float("nan")),
                    homeostasis=SimpleNamespace(loss=model.router_weight.square()),
                ),
            }

        with mock.patch.object(experiment, "_forward_model", side_effect=fake_forward):
            record = experiment._run_epoch(
                model,
                torch.zeros((1, 1), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
                batch_size=1,
                device=torch.device("cpu"),
                optimizer=experiment._optimizer(model),
                seed=0,
            )

        self.assertIsNotNone(record["first_failure"])
        self.assertEqual(record["first_failure"]["type"], "FE2HFiniteGuardError")
        self.assertIn("loss:route_supervision", record["first_failure"]["message"])
        self.assertIsNone(model.router_weight.grad)
        self.assertEqual(float(model.router_weight.detach().item()), router_before)
        self.assertEqual(record["ce"], 0.0)
        self.assertEqual(record["loss_breakdown"]["total"], 0.0)

    def test_smoke_cli_runs_on_cpu_with_tiny_temp_output_only(self) -> None:
        canonical_output = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "e3_scan"
            / "e3_fe2h_neuron_tile.json"
        )
        existed_before = canonical_output.exists()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "artifact.json"
            cache_dir = Path(tmpdir) / "cache"
            result = experiment.main(
                [
                    "--mode",
                    "smoke",
                    "--device",
                    "cpu",
                    "--out",
                    str(out),
                    "--cache-dir",
                    str(cache_dir),
                    "--d-model",
                    "32",
                    "--state-dim",
                    "64",
                    "--tile-size",
                    "16",
                    "--active-tiles",
                    "2",
                    "--block-size",
                    "4",
                    "--rank",
                    "16",
                    "--batch-size",
                    "2",
                    "--seq-len",
                    "8",
                    "--epochs",
                    "1",
                    "--seed",
                    "3",
                    "--warmup-steps",
                    "1",
                    "--benchmark-steps",
                    "1",
                    "--smoke-vocab-size",
                    "32",
                    "--smoke-train-sequences",
                    "4",
                    "--smoke-valid-sequences",
                    "2",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(out.exists())
            artifact = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(artifact["formal"])
            self.assertEqual(artifact["configuration"]["out"], str(out.resolve()))
            self.assertEqual(artifact["mechanism"]["status"], "PASS")
            training_loss = artifact["provenance"]["training_loss"]
            self.assertEqual(training_loss["weights"]["ce"], 1.0)
            self.assertEqual(
                training_loss["weights"]["route_supervision"],
                experiment.ROUTE_SUPERVISION_LOSS_WEIGHT,
            )
            self.assertEqual(
                training_loss["weights"]["homeostasis"],
                experiment.HOMEOSTASIS_LOSS_WEIGHT,
            )
            self.assertIn("homeostasis", artifact["mechanism"])
            homeostasis = artifact["mechanism"]["homeostasis"]
            for key in (
                "activation_rate",
                "entropy",
                "gini",
                "p99_tile_load",
                "block_fill",
                "hotspot_share",
                "dead_tile_ratio",
                "target_activation_rate",
                "activation_rate_max_abs_tolerance",
            ):
                self.assertIn(key, homeostasis)
            train_history = artifact["numerics"]["variants"]["fe2h_dense_mask"]["train_history"]
            self.assertTrue(train_history)
            self.assertIn("loss_breakdown", train_history[0])
            self.assertIn("loss_provenance", train_history[0])
            self.assertTrue(train_history[0]["loss_breakdown"]["aux_applied"])
            self.assertEqual(
                train_history[0]["loss_provenance"]["route_supervision"],
                "diagnostics.route_supervision_loss",
            )
            self.assertEqual(
                train_history[0]["loss_provenance"]["homeostasis"],
                "diagnostics.homeostasis.loss",
            )
            self.assertEqual(
                experiment.validate_artifact_schema(artifact),
                [],
            )
        self.assertEqual(canonical_output.exists(), existed_before)


if __name__ == "__main__":
    unittest.main()
