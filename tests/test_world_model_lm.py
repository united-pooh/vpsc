import math
import unittest

import torch
import torch.nn.functional as F

from vpsc.world_model.cores import (
    CausalTransformerCore,
    E2SignedCore,
    StatefulLSTMCore,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.lm import CausalLanguageModel
from vpsc.world_model.factory import (
    FairLMConfig,
    FrozenE2Config,
    assert_parameter_budget,
    build_model_suite,
)


class CausalLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2027)

    def test_teacher_forced_loss_shifts_once_and_ignores_padding(self) -> None:
        model = CausalLanguageModel(
            vocab_size=7,
            core=StatefulLSTMCore(5, 5),
            dropout=0.0,
            padding_idx=0,
        ).eval()
        token_ids = torch.tensor(
            [
                [1, 2, 3, 0],
                [1, 4, 0, 0],
            ],
            dtype=torch.long,
        )

        with torch.no_grad():
            shifted = model.teacher_forced(token_ids)
            aligned = model(
                token_ids[:, :-1],
                targets=token_ids[:, 1:],
            )

        expected = F.cross_entropy(
            shifted.logits[:, :-1].reshape(-1, 7),
            token_ids[:, 1:].reshape(-1),
            ignore_index=0,
        )
        self.assertEqual(shifted.logits.shape, (2, 4, 7))
        self.assertEqual(shifted.loss_mode, "causal_shift")
        self.assertEqual(aligned.loss_mode, "aligned")
        self.assertEqual(int(shifted.target_count), 3)
        self.assertEqual(int(aligned.target_count), 3)
        torch.testing.assert_close(shifted.loss, expected)
        torch.testing.assert_close(aligned.loss, expected)
        torch.testing.assert_close(aligned.logits, shifted.logits[:, :-1])

        all_padding = torch.zeros(2, 3, dtype=torch.long)
        with torch.no_grad():
            ignored = model.teacher_forced(all_padding)
        self.assertEqual(int(ignored.target_count), 0)
        self.assertTrue(torch.isfinite(ignored.loss))
        self.assertEqual(float(ignored.loss), 0.0)

    def test_full_sequence_matches_explicit_steps_for_all_cores(self) -> None:
        factories = {
            "lstm": lambda: StatefulLSTMCore(8, 8, num_layers=2, dropout=0.0),
            "transformer": lambda: CausalTransformerCore(
                8,
                8,
                num_layers=2,
                num_heads=2,
                dropout=0.0,
                max_cache_tokens=None,
            ),
            "e2": lambda: E2SignedCore(8, 8, state_dim=8),
        }
        token_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ],
            dtype=torch.long,
        )

        for name, factory in factories.items():
            with self.subTest(core=name):
                torch.manual_seed(2027)
                model = CausalLanguageModel(
                    vocab_size=12,
                    core=factory(),
                    dropout=0.0,
                    padding_idx=0,
                ).eval()
                with torch.no_grad():
                    full = model(token_ids)
                    state = None
                    streamed_logits = []
                    for index in range(token_ids.shape[1]):
                        step = model.step(token_ids[:, index], state)
                        self.assertEqual(step.logits.shape, (2, 1, 12))
                        self.assertEqual(step.hidden_states.shape, (2, 1, 8))
                        self.assertIsNone(step.loss)
                        streamed_logits.append(step.logits)
                        state = step.state

                self.assertEqual(full.logits.shape, (2, 5, 12))
                self.assertEqual(full.hidden_states.shape, (2, 5, 8))
                self.assertGreater(state_nbytes(state), 0)
                torch.testing.assert_close(
                    full.logits,
                    torch.cat(streamed_logits, dim=1),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_embedding_uses_width_scaled_normal_and_keeps_padding_zero(self) -> None:
        vocab_size = 4096
        embedding_dim = 32
        model = CausalLanguageModel(
            vocab_size=vocab_size,
            core=StatefulLSTMCore(embedding_dim, embedding_dim),
            padding_idx=0,
            tie_weights=True,
        )
        expected_std = embedding_dim**-0.5
        non_padding = model.embedding.weight.detach()[1:]

        self.assertLess(abs(float(non_padding.mean())), expected_std * 0.03)
        self.assertLess(
            abs(float(non_padding.std(unbiased=False)) - expected_std),
            expected_std * 0.03,
        )
        torch.testing.assert_close(
            model.embedding.weight.detach()[0],
            torch.zeros(embedding_dim),
            rtol=0.0,
            atol=0.0,
        )
        self.assertIs(model.embedding.weight, model.lm_head.weight)

        # A tied output head would normally add a non-zero gradient to every
        # vocabulary row, including padding.  E2' freezes that row at zero.
        output = model.teacher_forced(torch.tensor([[1, 2, 3]], dtype=torch.long))
        output.loss.backward()
        torch.testing.assert_close(
            model.embedding.weight.grad[0],
            torch.zeros(embedding_dim),
            rtol=0.0,
            atol=0.0,
        )
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        torch.testing.assert_close(
            model.embedding.weight.detach()[0],
            torch.zeros(embedding_dim),
            rtol=0.0,
            atol=0.0,
        )

    def test_preregistered_first_batch_scale_acceptance_for_all_cores(self) -> None:
        vocab_size = 4096
        d_model = 32
        torch.manual_seed(20260718)
        suite = build_model_suite(
            FairLMConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=4,
                dropout=0.0,
                transformer_max_cache_tokens=128,
                e2=FrozenE2Config(policy="hybrid", positive_factor=0.8),
            )
        )
        assert_parameter_budget(suite)
        input_ids = (
            torch.arange(4 * 64, dtype=torch.long).reshape(4, 64)
            % (vocab_size - 4)
        ) + 4
        targets = ((input_ids - 4 + 1) % (vocab_size - 4)) + 4
        expected_nll = math.log(vocab_size)
        logits_stds = {}
        initial_nlls = {}

        for name, model in suite.models.items():
            model.eval()
            with torch.no_grad():
                output = model(input_ids, targets=targets)
            logits_std = float(output.logits.std(unbiased=False))
            initial_nll = float(output.loss)
            logits_stds[name] = logits_std
            initial_nlls[name] = initial_nll

            self.assertGreaterEqual(logits_std, 0.5, name)
            self.assertLessEqual(logits_std, 1.5, name)
            self.assertLessEqual(abs(initial_nll - expected_nll), 1.0, name)
            torch.testing.assert_close(
                output.hidden_states.mean(dim=-1),
                torch.zeros_like(output.hidden_states[..., 0]),
                atol=1e-5,
                rtol=0.0,
            )

        self.assertLessEqual(max(initial_nlls.values()) - min(initial_nlls.values()), 0.5)
        self.assertLessEqual(max(logits_stds.values()) / min(logits_stds.values()), 2.0)

    def test_weight_tying_and_parameter_breakdown_are_unambiguous(self) -> None:
        vocab_size = 11
        embedding_dim = 6
        core = StatefulLSTMCore(embedding_dim, embedding_dim)
        model = CausalLanguageModel(
            vocab_size=vocab_size,
            core=core,
            padding_idx=0,
            tie_weights=True,
            head_bias=True,
        )

        self.assertIs(model.embedding.weight, model.lm_head.weight)
        self.assertEqual(
            model.embedding.weight.data_ptr(), model.lm_head.weight.data_ptr()
        )
        stats = model.parameter_stats()
        shared = vocab_size * embedding_dim
        self.assertTrue(stats.tied_weights)
        self.assertEqual(stats.shared_embedding_head, shared)
        self.assertEqual(stats.embedding.total, shared)
        self.assertEqual(stats.output_norm.total, 2 * embedding_dim)
        self.assertEqual(stats.lm_head.total, shared + vocab_size)
        self.assertEqual(
            stats.wrapper_unique.total,
            shared + vocab_size + 2 * embedding_dim,
        )
        self.assertEqual(
            stats.model.total,
            count_parameters(model, trainable_only=False),
        )
        self.assertEqual(stats.model.total, stats.core.total + stats.wrapper_unique.total)
        self.assertEqual(stats.as_dict()["core_type"], "StatefulLSTMCore")
        self.assertEqual(stats.as_dict()["output_norm_total"], 2 * embedding_dim)

        with self.assertRaisesRegex(ValueError, "weight tying"):
            CausalLanguageModel(
                vocab_size=9,
                core=StatefulLSTMCore(4, 5),
                tie_weights=True,
            )


if __name__ == "__main__":
    unittest.main()
