import unittest

import torch

from vpsc.world_model.cores import (
    CausalTransformerCore,
    E2SignedCore,
    StatefulLSTMCore,
    state_nbytes,
)
from vpsc.world_model.factory import (
    FairLMConfig,
    FrozenE2Config,
    ParameterBudgetError,
    assert_parameter_budget,
    build_model_suite,
)


class FairModelFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31415)

    def test_count_only_matching_reaches_two_percent_budget(self) -> None:
        config = FairLMConfig(
            vocab_size=17,
            d_model=8,
            num_heads=2,
            dropout=0.15,
            parameter_tolerance=0.02,
            auto_match_parameters=True,
        )
        suite = build_model_suite(config)
        report = assert_parameter_budget(suite)

        self.assertLessEqual(report.relative_spread, 0.02)
        self.assertGreater(report.base_relative_spread, 0.02)
        self.assertTrue(report.auto_matched)
        self.assertEqual(set(report.totals), {"lstm", "transformer", "e2"})
        self.assertTrue(all(count > 0 for count in report.cores.values()))
        self.assertEqual(report.selected_e2_state_dim, suite.e2.core.state_dim)

        expected_hidden = round(
            config.d_model * report.selected_transformer_mlp_ratio
        )
        actual_hidden = suite.transformer.core.layers[0].mlp[0].out_features
        self.assertEqual(actual_hidden, expected_hidden)

        for model in suite.models.values():
            self.assertEqual(model.vocab_size, config.vocab_size)
            self.assertEqual(model.embedding_dim, config.d_model)
            self.assertEqual(model.input_dropout.p, config.dropout)
            self.assertEqual(model.output_dropout.p, config.dropout)
            self.assertTrue(model.tie_weights)
            self.assertIs(model.embedding.weight, model.lm_head.weight)

        self.assertIsInstance(suite.lstm.core, StatefulLSTMCore)
        self.assertEqual(suite.lstm.core.num_layers, 1)
        self.assertIsInstance(suite.transformer.core, CausalTransformerCore)
        self.assertEqual(suite.transformer.core.num_layers, 1)
        self.assertIsInstance(suite.e2.core, E2SignedCore)

    def test_all_three_models_run_the_same_lm_and_streaming_interface(self) -> None:
        suite = build_model_suite(
            FairLMConfig(
                vocab_size=19,
                d_model=8,
                num_heads=2,
                dropout=0.0,
                transformer_max_cache_tokens=8,
            )
        )
        assert_parameter_budget(suite)
        token_ids = torch.tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
            ],
            dtype=torch.long,
        )

        for name, model in suite.models.items():
            with self.subTest(model=name):
                model.eval()
                with torch.no_grad():
                    teacher_forced = model.teacher_forced(token_ids)
                    first = model.step(token_ids[:, 0])
                    second = model.step(token_ids[:, 1], first.state)

                self.assertEqual(teacher_forced.logits.shape, (2, 4, 19))
                self.assertEqual(teacher_forced.loss_mode, "causal_shift")
                self.assertTrue(torch.isfinite(teacher_forced.loss))
                self.assertEqual(first.logits.shape, (2, 1, 19))
                self.assertEqual(second.logits.shape, (2, 1, 19))
                self.assertGreater(state_nbytes(second.state), 0)

        transformer_state = suite.transformer.step(token_ids[:, 0]).state
        self.assertEqual(transformer_state.position, 1)
        self.assertTrue(
            all(cache.key.shape[2] == 1 for cache in transformer_state.layers)
        )

    def test_e2_policy_and_ablations_survive_parameter_matching(self) -> None:
        frozen = FrozenE2Config(
            policy="hybrid",
            no_positive=True,
            state_reset=True,
            margin_scale=0.91,
            positive_factor=0.8,
            negative_factor=1.2,
            hybrid_cutoff=0.9,
            g_ee=7.1,
            g_ei=6.2,
            g_ie=9.3,
            g_ii=5.4,
            theta_e=2.1,
            theta_i=5.2,
            tau_e=1.0,
            tau_i=4.8,
            dt=0.5,
            micro_steps=2,
        )
        suite = build_model_suite(
            FairLMConfig(
                vocab_size=17,
                d_model=8,
                num_heads=2,
                e2=frozen,
                auto_match_parameters=True,
            )
        )
        core = suite.e2.core

        self.assertEqual(core.policy, "hybrid")
        self.assertTrue(core.no_positive)
        self.assertTrue(core.state_reset)
        self.assertEqual(core.margin_scale, frozen.margin_scale)
        self.assertEqual(core.positive_factor, frozen.positive_factor)
        self.assertEqual(core.negative_factor, frozen.negative_factor)
        self.assertEqual(core.hybrid_cutoff, frozen.hybrid_cutoff)
        self.assertEqual(core.g_ee, frozen.g_ee)
        self.assertEqual(core.g_ei, frozen.g_ei)
        self.assertEqual(core.g_ie, frozen.g_ie)
        self.assertEqual(core.g_ii, frozen.g_ii)
        self.assertEqual(core.theta_e, frozen.theta_e)
        self.assertEqual(core.theta_i, frozen.theta_i)
        self.assertEqual(core.alpha_e, frozen.dt / frozen.tau_e)
        self.assertEqual(core.alpha_i, frozen.dt / frozen.tau_i)
        self.assertEqual(core.micro_steps, frozen.micro_steps)
        self.assertEqual(core.effective_gains().e_to_e, 0.0)
        self.assertEqual(core.effective_gains().i_to_e, 0.0)

    def test_budget_assertion_exposes_unmatched_frozen_defaults(self) -> None:
        suite = build_model_suite(
            FairLMConfig(
                vocab_size=17,
                d_model=8,
                num_heads=2,
                auto_match_parameters=False,
            )
        )
        self.assertFalse(suite.parameter_report.auto_matched)
        self.assertEqual(
            suite.parameter_report.selected_transformer_mlp_ratio,
            2.0,
        )
        self.assertEqual(suite.parameter_report.selected_e2_state_dim, 8)
        with self.assertRaises(ParameterBudgetError):
            assert_parameter_budget(suite)


if __name__ == "__main__":
    unittest.main()
