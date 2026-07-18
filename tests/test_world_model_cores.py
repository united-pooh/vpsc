import unittest

import torch

from vpsc.world_model.cores import (
    CausalTransformerCore,
    E2CoreState,
    E2SignedCore,
    StatefulLSTMCore,
    state_nbytes,
)


class TemporalCoreContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1729)

    def test_all_cores_shape_and_chunk_state_continuity(self) -> None:
        factories = {
            "lstm": lambda: StatefulLSTMCore(4, 6),
            "transformer": lambda: CausalTransformerCore(
                4,
                6,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
                max_cache_tokens=8,
            ),
            "e2": lambda: E2SignedCore(4, 6),
        }
        first_chunk = torch.linspace(-1.0, 1.0, 2 * 3 * 4).reshape(2, 3, 4)
        second_chunk = torch.linspace(0.75, -0.5, 2 * 2 * 4).reshape(2, 2, 4)
        whole_sequence = torch.cat((first_chunk, second_chunk), dim=1)

        for name, factory in factories.items():
            with self.subTest(core=name):
                torch.manual_seed(1729)
                core = factory().eval()
                with torch.no_grad():
                    whole = core(whole_sequence)
                    first = core(first_chunk)
                    continued = core(second_chunk, first.state)
                    fresh = core(second_chunk)

                self.assertEqual(whole.sequence.shape, (2, 5, 6))
                self.assertEqual(first.sequence.shape, (2, 3, 6))
                self.assertEqual(continued.sequence.shape, (2, 2, 6))
                self.assertEqual(whole.sequence.device.type, "cpu")
                self.assertGreater(state_nbytes(first.state), 0)
                torch.testing.assert_close(
                    continued.sequence,
                    whole.sequence[:, 3:],
                    atol=1e-5,
                    rtol=1e-5,
                )
                self.assertFalse(
                    torch.allclose(continued.sequence, fresh.sequence, atol=1e-7, rtol=1e-7),
                    f"{name} ignored the supplied prior state",
                )

    def test_lstm_full_sequence_matches_streaming_steps(self) -> None:
        core = StatefulLSTMCore(5, 7, num_layers=2, dropout=0.0).eval()
        sequence = torch.randn(3, 6, 5)

        with torch.no_grad():
            full = core(sequence)
            state = None
            pieces = []
            for index in range(sequence.shape[1]):
                streamed = core.step(sequence[:, index], state)
                self.assertEqual(streamed.sequence.shape, (3, 1, 7))
                pieces.append(streamed.sequence)
                state = streamed.state

        torch.testing.assert_close(
            full.sequence,
            torch.cat(pieces, dim=1),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(full.state.hidden, state.hidden)
        torch.testing.assert_close(full.state.cell, state.cell)

    def test_transformer_full_sequence_matches_streaming_steps(self) -> None:
        core = CausalTransformerCore(
            5,
            8,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            max_cache_tokens=None,
        ).eval()
        sequence = torch.randn(2, 7, 5)

        with torch.no_grad():
            full = core(sequence)
            state = None
            pieces = []
            for index in range(sequence.shape[1]):
                streamed = core.step(sequence[:, index], state)
                self.assertEqual(streamed.sequence.shape, (2, 1, 8))
                pieces.append(streamed.sequence)
                state = streamed.state

        torch.testing.assert_close(
            full.sequence,
            torch.cat(pieces, dim=1),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(full.state.position, 7)
        self.assertEqual(state.position, 7)
        for full_cache, streamed_cache in zip(full.state.layers, state.layers):
            torch.testing.assert_close(full_cache.key, streamed_cache.key)
            torch.testing.assert_close(full_cache.value, streamed_cache.value)

    def test_transformer_cache_window_and_exact_logical_bytes(self) -> None:
        batch_size = 2
        model_dim = 8
        num_layers = 2
        max_cache_tokens = 3
        core = CausalTransformerCore(
            input_dim=4,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=2,
            dropout=0.0,
            max_cache_tokens=max_cache_tokens,
        ).eval()

        self.assertEqual(state_nbytes(core.initial_state(batch_size)), 0)
        with torch.no_grad():
            result = core(torch.randn(batch_size, 7, 4))

        element_size = result.sequence.element_size()
        expected_bytes = (
            2
            * num_layers
            * batch_size
            * max_cache_tokens
            * model_dim
            * element_size
        )
        self.assertEqual(state_nbytes(result.state), expected_bytes)
        self.assertEqual(result.state.position, 7)
        for cache in result.state.layers:
            self.assertEqual(cache.key.shape, (batch_size, 2, max_cache_tokens, 4))
            self.assertEqual(cache.value.shape, cache.key.shape)

        with torch.no_grad():
            advanced = core.step(torch.randn(batch_size, 4), result.state)
        self.assertEqual(advanced.state.position, 8)
        self.assertEqual(state_nbytes(advanced.state), expected_bytes)
        self.assertTrue(
            all(cache.key.shape[2] == max_cache_tokens for cache in advanced.state.layers)
        )


class E2SignedCoreTests(unittest.TestCase):
    def _controlled_core(
        self,
        *,
        g_ee: float = 0.0,
        g_ei: float = 0.0,
        g_ie: float = 0.0,
        g_ii: float = 0.0,
        no_positive: bool = False,
    ) -> E2SignedCore:
        core = E2SignedCore(
            3,
            3,
            state_dim=3,
            policy="exact",
            no_positive=no_positive,
            g_ee=g_ee,
            g_ei=g_ei,
            g_ie=g_ie,
            g_ii=g_ii,
            theta_e=0.0,
            theta_i=0.0,
            tau_e=1.0,
            tau_i=1.0,
            dt=1.0,
        ).eval()
        with torch.no_grad():
            core.input_to_e.weight.zero_()
            core.input_to_e.bias.zero_()
            core.input_to_i.weight.zero_()
            core.input_to_i.bias.zero_()
        return core

    def test_all_four_recurrent_channels_have_effective_fixed_signs(self) -> None:
        state = E2CoreState(
            excitatory=torch.tensor([[0.20, 0.35, 0.50]]),
            inhibitory=torch.tensor([[0.15, 0.30, 0.45]]),
        )
        token = torch.zeros(1, 3)
        baseline_core = self._controlled_core()

        with torch.no_grad():
            baseline = baseline_core.step(token, state).state
            positive_e = self._controlled_core(g_ee=1.5).step(token, state).state
            negative_e = self._controlled_core(g_ei=1.5).step(token, state).state
            positive_i = self._controlled_core(g_ie=1.5).step(token, state).state
            negative_i = self._controlled_core(g_ii=1.5).step(token, state).state

        self.assertTrue(torch.all(positive_e.excitatory > baseline.excitatory))
        torch.testing.assert_close(positive_e.inhibitory, baseline.inhibitory)

        self.assertTrue(torch.all(negative_e.excitatory < baseline.excitatory))
        torch.testing.assert_close(negative_e.inhibitory, baseline.inhibitory)

        self.assertTrue(torch.all(positive_i.inhibitory > baseline.inhibitory))
        torch.testing.assert_close(positive_i.excitatory, baseline.excitatory)

        self.assertTrue(torch.all(negative_i.inhibitory < baseline.inhibitory))
        torch.testing.assert_close(negative_i.excitatory, baseline.excitatory)

        for logits in (
            baseline_core.e_to_e_logits,
            baseline_core.i_to_e_logits,
            baseline_core.e_to_i_logits,
            baseline_core.i_to_i_logits,
        ):
            magnitudes = torch.softmax(logits, dim=-1)
            self.assertTrue(torch.all(magnitudes > 0.0))
            torch.testing.assert_close(magnitudes.sum(dim=-1), torch.ones(3))

    def test_feedback_policies_preserve_frozen_e1_semantics(self) -> None:
        exact = E2SignedCore(
            3,
            3,
            policy="exact",
            positive_factor=0.8,
            negative_factor=1.1,
        ).effective_gains()
        margin = E2SignedCore(
            3,
            3,
            policy="margin",
            positive_factor=0.8,
            negative_factor=1.1,
        ).effective_gains()
        hybrid_low = E2SignedCore(
            3,
            3,
            policy="hybrid",
            positive_factor=0.8,
            hybrid_cutoff=1.0,
        ).effective_gains()
        hybrid_at_cutoff = E2SignedCore(
            3,
            3,
            policy="hybrid",
            positive_factor=1.0,
            hybrid_cutoff=1.0,
        ).effective_gains()

        self.assertAlmostEqual(exact.e_to_e, 7.75 * 0.8)
        self.assertAlmostEqual(exact.i_to_e, 6.70 * 1.1)
        self.assertAlmostEqual(margin.e_to_e, exact.e_to_e * 0.95)
        self.assertAlmostEqual(margin.i_to_e, exact.i_to_e * 0.95)
        self.assertAlmostEqual(margin.e_to_i, 10.0 * 0.95)
        self.assertAlmostEqual(margin.i_to_i, 6.30 * 0.95)
        self.assertEqual(hybrid_low.i_to_e, 0.0)
        self.assertGreater(hybrid_low.e_to_e, 0.0)
        self.assertAlmostEqual(hybrid_at_cutoff.i_to_e, 6.70 * 0.95)

    def test_no_positive_removes_e_to_e_dynamics_only(self) -> None:
        state = E2CoreState(
            excitatory=torch.tensor([[0.20, 0.35, 0.50]]),
            inhibitory=torch.tensor([[0.15, 0.30, 0.45]]),
        )
        token = torch.zeros(1, 3)
        full = self._controlled_core(g_ee=2.0)
        ablated = self._controlled_core(g_ee=2.0, no_positive=True)
        zero_gain = self._controlled_core(g_ee=0.0)

        with torch.no_grad():
            full_state = full.step(token, state).state
            ablated_state = ablated.step(token, state).state
            zero_state = zero_gain.step(token, state).state

        self.assertGreater(full.effective_gains().e_to_e, 0.0)
        self.assertEqual(ablated.effective_gains().e_to_e, 0.0)
        self.assertTrue(torch.all(full_state.excitatory > ablated_state.excitatory))
        torch.testing.assert_close(ablated_state.excitatory, zero_state.excitatory)
        torch.testing.assert_close(ablated_state.inhibitory, zero_state.inhibitory)

    def test_state_reset_removes_cross_token_memory(self) -> None:
        torch.manual_seed(99)
        normal = E2SignedCore(4, 4, state_reset=False).eval()
        reset = E2SignedCore(4, 4, state_reset=True).eval()
        reset.load_state_dict(normal.state_dict())
        first_token = torch.tensor([[4.0, -3.0, 2.0, -1.0]])
        second_token = torch.zeros(1, 4)

        with torch.no_grad():
            reset_first = reset.step(first_token)
            reset_continued = reset.step(second_token, reset_first.state)
            reset_fresh = reset.step(second_token)
            reset_pair = reset(torch.stack((first_token, second_token), dim=1))

            normal_first = normal.step(first_token)
            normal_continued = normal.step(second_token, normal_first.state)
            normal_fresh = normal.step(second_token)

        torch.testing.assert_close(reset_continued.sequence, reset_fresh.sequence)
        torch.testing.assert_close(
            reset_continued.state.excitatory, reset_fresh.state.excitatory
        )
        torch.testing.assert_close(
            reset_continued.state.inhibitory, reset_fresh.state.inhibitory
        )
        torch.testing.assert_close(reset_pair.sequence[:, 1:], reset_fresh.sequence)
        self.assertFalse(
            torch.allclose(
                normal_continued.sequence,
                normal_fresh.sequence,
                atol=1e-7,
                rtol=1e-7,
            )
        )


if __name__ == "__main__":
    unittest.main()
