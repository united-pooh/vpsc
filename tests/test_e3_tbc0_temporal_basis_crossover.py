import unittest

import torch

from experiments.e3_tbc0_temporal_basis_crossover import (
    IGNORE_INDEX,
    IntervenableTemporalMoECore,
    ModelConfig,
    TaskConfig,
    analyse_phase_a_grid,
    build_model,
    count_parameters,
    find_parameter_matched_base_state_dim,
    make_event_recall_dataset,
)


class TemporalBasisCrossoverTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260722)
        self.task = TaskConfig(
            train_sequences=8,
            valid_sequences=4,
            seq_len=32,
            horizons=(2, 6, 12),
            gap_min=1,
            gap_max=2,
        )
        self.model_cfg = ModelConfig(d_model=8, state_dim=8, n_experts=3)

    def test_event_recall_queries_do_not_reveal_payload(self) -> None:
        dataset = make_event_recall_dataset(8, self.task, seed=17)

        self.assertGreater(dataset.query_count, 0)
        self.assertTrue(torch.all(dataset.inputs[dataset.query_mask] == self.task.query_token))
        self.assertTrue(torch.all(dataset.targets[dataset.query_mask] < self.task.payload_values))
        self.assertTrue(torch.all(dataset.targets[~dataset.query_mask] == IGNORE_INDEX))
        self.assertFalse(bool((dataset.event_mask & dataset.query_mask).any()))

    def test_factorised_event_probability_controls_realised_density(self) -> None:
        low_task = TaskConfig(
            train_sequences=256,
            valid_sequences=8,
            seq_len=64,
            horizons=(8,),
            event_probability=0.05,
        )
        high_task = TaskConfig(
            train_sequences=256,
            valid_sequences=8,
            seq_len=64,
            horizons=(8,),
            event_probability=0.30,
        )
        low = make_event_recall_dataset(256, low_task, seed=31)
        high = make_event_recall_dataset(256, high_task, seed=31)

        low_density = float(low.event_mask.float().mean())
        high_density = float(high.event_mask.float().mean())
        self.assertGreater(high_density, low_density * 2.0)
        self.assertTrue(torch.all(high.inputs[high.query_mask] == high_task.query_token))

    def test_temporal_and_homogeneous_have_identical_capacity(self) -> None:
        temporal = build_model("temporal", self.task, self.model_cfg)
        homogeneous = build_model("homogeneous", self.task, self.model_cfg)

        self.assertEqual(count_parameters(temporal), count_parameters(homogeneous))
        temporal_bounds = [
            (expert.min_decay, expert.max_decay) for expert in temporal.core.experts
        ]
        homogeneous_bounds = [
            (expert.min_decay, expert.max_decay) for expert in homogeneous.core.experts
        ]
        self.assertGreater(len(set(temporal_bounds)), 1)
        self.assertEqual(len(set(homogeneous_bounds)), 1)

    def test_parameter_matched_base_is_within_five_percent(self) -> None:
        temporal = build_model("temporal", self.task, self.model_cfg)
        target = count_parameters(temporal)
        state_dim, count, gap = find_parameter_matched_base_state_dim(
            self.task, self.model_cfg, target
        )

        matched = build_model(
            "base_param_matched",
            self.task,
            self.model_cfg,
            matched_state_dim=state_dim,
        )
        self.assertEqual(count, count_parameters(matched))
        self.assertLessEqual(gap, 0.05)

    def test_router_interventions_are_deterministic(self) -> None:
        model = build_model("temporal", self.task, self.model_cfg)
        self.assertIsInstance(model.core, IntervenableTemporalMoECore)
        embedded = torch.randn(2, 9, self.model_cfg.d_model)
        raw = model.core.raw_gate_logits(embedded)

        model.core.set_router_intervention("uniform")
        torch.testing.assert_close(model.core._gate_logits(embedded), torch.zeros_like(raw))
        model.core.set_router_intervention("reverse_time")
        torch.testing.assert_close(model.core._gate_logits(embedded), raw.flip(dims=(1,)))
        model.core.set_router_intervention("none")
        torch.testing.assert_close(model.core._gate_logits(embedded), raw)

        uniform = build_model("uniform", self.task, self.model_cfg)
        self.assertEqual(uniform.core.router_intervention, "uniform")
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in uniform.core.change_router.parameters()
            )
        )
        torch.testing.assert_close(
            uniform.core._gate_logits(embedded),
            torch.zeros_like(uniform.core.raw_gate_logits(embedded)),
        )

    def test_all_variants_have_finite_forward_and_backward(self) -> None:
        dataset = make_event_recall_dataset(4, self.task, seed=29)
        temporal = build_model("temporal", self.task, self.model_cfg)
        target = count_parameters(temporal)
        matched_state_dim, _, _ = find_parameter_matched_base_state_dim(
            self.task, self.model_cfg, target
        )

        for variant in (
            "base_same_width",
            "base_param_matched",
            "temporal",
            "homogeneous",
            "uniform",
        ):
            with self.subTest(variant=variant):
                model = build_model(
                    variant,
                    self.task,
                    self.model_cfg,
                    matched_state_dim=matched_state_dim,
                )
                output = model(dataset.inputs, targets=dataset.targets)
                self.assertTrue(torch.isfinite(output.loss))
                self.assertEqual(int(output.target_count), dataset.query_count)
                output.loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

    def test_phase_a_analysis_applies_frozen_gates(self) -> None:
        cells = []
        for horizon in (2, 24):
            for probability in (0.05, 0.30):
                rows = []
                short_usage = 0.20 + (0.05 if probability == 0.30 else 0.0)
                long_usage = 0.20 + (0.05 if horizon == 24 else 0.0)
                middle_usage = 1.0 - short_usage - long_usage
                for seed in (0, 1, 2):
                    rows.extend(
                        [
                            {
                                "variant": "temporal",
                                "seed": seed,
                                "parameter_gap_to_temporal": 0.0,
                                "normal": {"query_accuracy": 0.70, "query_nll": 0.50},
                                "uniform_intervention": {"query_accuracy": 0.60},
                                "reverse_time_intervention": {"query_accuracy": 0.65},
                                "routing": {
                                    "mean_usage": [short_usage, middle_usage, long_usage]
                                },
                            },
                            {
                                "variant": "homogeneous",
                                "seed": seed,
                                "normal": {"query_accuracy": 0.60, "query_nll": 0.70},
                            },
                            {
                                "variant": "base_param_matched",
                                "seed": seed,
                                "parameter_gap_to_temporal": 0.01,
                                "normal": {"query_accuracy": 0.55, "query_nll": 0.75},
                            },
                        ]
                    )
                cells.append(
                    {
                        "grid_horizon": horizon,
                        "grid_event_probability": probability,
                        "dataset": {"valid_event_density": probability * 0.8},
                        "rows": rows,
                    }
                )

        analysis = analyse_phase_a_grid(cells)

        self.assertEqual(analysis["verdict"], "GO")
        self.assertTrue(all(analysis["gates"].values()))


if __name__ == "__main__":
    unittest.main()
