import unittest

import torch

from vpsc.world_model.cores import TransformerCoreState, state_nbytes
from vpsc.world_model.homegrid_model import (
    ACTION_COUNT,
    DONE_CLASS_COUNT,
    E2_POLICY,
    E2_POSITIVE_FACTOR,
    READ_CLASS_COUNT,
    REWARD_CLASS_COUNT,
    TRANSFORMER_CACHE_TOKENS,
    VISUAL_PATCHES,
    VISUAL_VOCAB_SIZE,
    HomeGridWorldModelConfig,
    assert_homegrid_parameter_budget,
    build_homegrid_model_suite,
)


def _inputs(
    language_vocab_size: int,
    *,
    batch: int = 2,
    time: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7301)
    visual = torch.randint(
        VISUAL_VOCAB_SIZE,
        (batch, time, VISUAL_PATCHES),
        generator=generator,
    )
    language = torch.randint(
        language_vocab_size,
        (batch, time),
        generator=generator,
    )
    actions = torch.randint(
        ACTION_COUNT,
        (batch, time),
        generator=generator,
    )
    read = torch.randint(2, (batch, time), generator=generator, dtype=torch.bool)
    return visual, language, actions, read


class HomeGridWorldModelTests(unittest.TestCase):
    def test_shared_shapes_frozen_e2_and_parameter_fairness(self) -> None:
        language_vocab_size = 23
        torch.manual_seed(17)
        suite = build_homegrid_model_suite(language_vocab_size, d_model=32)
        visual, language, actions, read = _inputs(language_vocab_size)

        report = assert_homegrid_parameter_budget(suite)
        self.assertTrue(report.within_tolerance)
        self.assertLessEqual(report.relative_spread, 0.02)
        self.assertEqual(set(report.totals), {"lstm", "transformer", "e2"})

        wrapper_counts = set()
        for model in suite.models.values():
            output = model(visual, language, actions, read, detach_state=True)
            self.assertEqual(
                output.next_visual_logits.shape,
                (2, 4, VISUAL_PATCHES, VISUAL_VOCAB_SIZE),
            )
            self.assertEqual(
                output.next_language_logits.shape,
                (2, 4, language_vocab_size),
            )
            self.assertEqual(output.next_read_logits.shape, (2, 4, READ_CLASS_COUNT))
            self.assertEqual(output.reward_logits.shape, (2, 4, REWARD_CLASS_COUNT))
            self.assertEqual(output.done_logits.shape, (2, 4, DONE_CLASS_COUNT))
            self.assertEqual(output.hidden_states.shape, (2, 4, 32))
            self.assertEqual(
                output.last_next_visual_logits.shape,
                (2, VISUAL_PATCHES, VISUAL_VOCAB_SIZE),
            )
            self.assertEqual(
                output.last_next_language_logits.shape,
                (2, language_vocab_size),
            )
            self.assertGreater(state_nbytes(output.state), 0)
            stats = model.parameter_stats()
            self.assertEqual(stats.model.total, report.totals[next(
                name for name, candidate in suite.models.items() if candidate is model
            )])
            self.assertEqual(stats.output_norm.total, 64)
            wrapper_counts.add(stats.shared_wrapper.total)

        self.assertEqual(len(wrapper_counts), 1)
        e2_core = suite.e2.core
        self.assertEqual(e2_core.policy, E2_POLICY)
        self.assertEqual(e2_core.positive_factor, E2_POSITIVE_FACTOR)
        self.assertGreater(e2_core.effective_gains().e_to_e, 0.0)
        self.assertEqual(e2_core.effective_gains().i_to_e, 0.0)

    def test_step_matches_forward_for_every_head_and_core(self) -> None:
        language_vocab_size = 19
        torch.manual_seed(29)
        suite = build_homegrid_model_suite(language_vocab_size, d_model=8)
        visual, language, actions, read = _inputs(language_vocab_size, time=5)

        for name, model in suite.models.items():
            model.eval()
            with torch.no_grad():
                full = model(visual, language, actions, read)
                state = None
                streamed = []
                for index in range(visual.shape[1]):
                    result = model.step(
                        visual[:, index],
                        language[:, index],
                        actions[:, index],
                        read[:, index],
                        state,
                    )
                    streamed.append(result)
                    state = result.state

            for field in (
                "next_visual_logits",
                "next_language_logits",
                "next_read_logits",
                "reward_logits",
                "done_logits",
                "hidden_states",
            ):
                with self.subTest(model=name, field=field):
                    torch.testing.assert_close(
                        getattr(full, field),
                        torch.cat([getattr(result, field) for result in streamed], dim=1),
                        atol=2e-5,
                        rtol=2e-5,
                    )
            self.assertGreater(state_nbytes(state), 0)

    def test_transformer_uses_real_growing_kv_cache_with_128_window(self) -> None:
        language_vocab_size = 11
        torch.manual_seed(41)
        model = build_homegrid_model_suite(language_vocab_size, d_model=8).transformer
        visual, language, actions, read = _inputs(
            language_vocab_size,
            batch=1,
            time=3,
        )
        self.assertEqual(model.core.max_cache_tokens, TRANSFORMER_CACHE_TOKENS)

        state = None
        observed_lengths = []
        observed_bytes = []
        with torch.no_grad():
            for index in range(3):
                output = model.step(
                    visual[:, index],
                    language[:, index],
                    actions[:, index],
                    read[:, index],
                    state,
                )
                state = output.state
                self.assertIsInstance(state, TransformerCoreState)
                observed_lengths.append(state.layers[0].key.shape[2])
                observed_bytes.append(state_nbytes(state))

        self.assertEqual(observed_lengths, [1, 2, 3])
        self.assertEqual(state.position, 3)
        self.assertEqual(state.layers[0].key.shape, (1, 4, 3, 2))
        self.assertEqual(state.layers[0].value.shape, state.layers[0].key.shape)
        self.assertLess(observed_bytes[0], observed_bytes[1])
        self.assertLess(observed_bytes[1], observed_bytes[2])

    def test_config_and_model_reject_bad_shapes_dtypes_and_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "language_vocab_size"):
            HomeGridWorldModelConfig(language_vocab_size=1)
        with self.assertRaisesRegex(ValueError, "divide d_model"):
            HomeGridWorldModelConfig(language_vocab_size=8, d_model=10)

        language_vocab_size = 13
        model = build_homegrid_model_suite(language_vocab_size, d_model=8).lstm
        visual, language, actions, read = _inputs(language_vocab_size)

        bad_calls = []
        bad_calls.append(
            (
                "visual shape",
                (visual[:, :, :-1], language, actions, read),
                ValueError,
            )
        )
        bad_calls.append(
            (
                "language shape",
                (visual, language[:, :-1], actions, read),
                ValueError,
            )
        )
        bad_calls.append(
            (
                "visual dtype",
                (visual.float(), language, actions, read),
                TypeError,
            )
        )
        invalid_visual = visual.clone()
        invalid_visual[0, 0, 0] = VISUAL_VOCAB_SIZE
        bad_calls.append(
            (
                "visual range",
                (invalid_visual, language, actions, read),
                ValueError,
            )
        )
        invalid_language = language.clone()
        invalid_language[0, 0] = language_vocab_size
        bad_calls.append(
            (
                "language range",
                (visual, invalid_language, actions, read),
                ValueError,
            )
        )
        invalid_actions = actions.clone()
        invalid_actions[0, 0] = ACTION_COUNT
        bad_calls.append(
            (
                "action range",
                (visual, language, invalid_actions, read),
                ValueError,
            )
        )
        invalid_read = read.to(dtype=torch.int64)
        invalid_read[0, 0] = 2
        bad_calls.append(
            (
                "read range",
                (visual, language, actions, invalid_read),
                ValueError,
            )
        )
        for name, inputs, error in bad_calls:
            with self.subTest(case=name):
                with self.assertRaises(error):
                    model(*inputs)

        with self.assertRaisesRegex(ValueError, r"\[batch, 144\]"):
            model.step(visual[:, 0, :-1], language[:, 0], actions[:, 0], read[:, 0])
        with self.assertRaisesRegex(ValueError, r"\[batch\]"):
            model.step(
                visual[:, 0],
                language[:, :1],
                actions[:, 0],
                read[:, 0],
            )


if __name__ == "__main__":
    unittest.main()
