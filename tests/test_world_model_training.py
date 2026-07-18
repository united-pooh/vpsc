from dataclasses import dataclass
import json
import math
import unittest

import torch
import torch.nn as nn

from vpsc.world_model.cores import StatefulLSTMCore
from vpsc.world_model.lm import CausalLanguageModel
from vpsc.world_model.training import (
    LanguageModelOutput,
    TrainingConfig,
    benchmark_streaming_step,
    evaluate_language_model,
    seed_everything,
    train_language_model,
)


@dataclass(frozen=True)
class _Batch:
    input_ids: object
    target_ids: object
    reset_state: bool


class _FixedLogitModel(nn.Module):
    """Input 0 is uniform; input 1 gives target-zero probability 0.9."""

    def forward(
        self,
        input_ids: torch.Tensor,
        state: object = None,
        *,
        detach_state: bool = False,
    ) -> LanguageModelOutput:
        del detach_state
        target_zero_logit = (input_ids == 1).to(torch.float32) * math.log(9.0)
        logits = torch.stack((target_zero_logit, torch.zeros_like(target_zero_logit)), dim=-1)
        if state is None:
            state = torch.zeros(input_ids.shape[0], 1)
        return LanguageModelOutput(logits=logits, state=state)

    def step(
        self,
        input_id: torch.Tensor,
        state: object = None,
        *,
        detach_state: bool = False,
    ) -> LanguageModelOutput:
        return self.forward(input_id.unsqueeze(1), state, detach_state=detach_state)


class _StateRecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(0.2))
        self.state_was_none = []
        self.incoming_state_requires_grad = []
        self.detach_flags = []

    def forward(
        self,
        input_ids: torch.Tensor,
        state: object = None,
        *,
        detach_state: bool = False,
    ) -> LanguageModelOutput:
        self.state_was_none.append(state is None)
        self.incoming_state_requires_grad.append(
            None if state is None else bool(state.requires_grad)
        )
        self.detach_flags.append(detach_state)
        if state is None:
            state = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

        positive = self.logit_scale.expand(*input_ids.shape)
        logits = torch.stack((positive, -positive), dim=-1)
        next_state = state + self.logit_scale
        if detach_state:
            next_state = next_state.detach()
        return LanguageModelOutput(logits=logits, state=next_state)

    def step(
        self,
        input_id: torch.Tensor,
        state: object = None,
        *,
        detach_state: bool = False,
    ) -> LanguageModelOutput:
        return self.forward(input_id.unsqueeze(1), state, detach_state=detach_state)


class WorldModelTrainingTests(unittest.TestCase):
    def test_evaluation_nll_is_weighted_by_target_count(self) -> None:
        batches = [
            _Batch(
                input_ids=torch.tensor([[0]]),
                target_ids=torch.tensor([[0]]),
                reset_state=True,
            ),
            _Batch(
                input_ids=torch.tensor([[1, 1, 1]]),
                target_ids=torch.tensor([[0, 0, 0]]),
                reset_state=False,
            ),
        ]

        metrics = evaluate_language_model(_FixedLogitModel(), batches)
        expected_nll = (math.log(2.0) + 3.0 * math.log(10.0 / 9.0)) / 4.0

        self.assertEqual(metrics.batches, 2)
        self.assertEqual(metrics.target_count, 4)
        self.assertAlmostEqual(metrics.nll, expected_nll, places=6)
        self.assertAlmostEqual(metrics.ppl, math.exp(expected_nll), places=6)
        self.assertEqual(json.loads(metrics.to_json())["target_count"], 4)

    def test_training_carries_detached_state_and_honours_reset(self) -> None:
        model = _StateRecordingModel()
        batches = [
            _Batch(((0, 1),), ((1, 0),), True),
            _Batch(((1, 0),), ((0, 1),), False),
            _Batch(((0, 0),), ((1, 1),), True),
        ]
        metrics = train_language_model(
            model,
            batches,
            TrainingConfig(
                seed=4,
                learning_rate=1e-2,
                weight_decay=0.0,
                gradient_clip_norm=0.5,
                max_steps=3,
            ),
        )

        self.assertEqual(model.state_was_none, [True, False, True])
        self.assertEqual(model.incoming_state_requires_grad, [None, False, None])
        self.assertEqual(model.detach_flags, [True, True, True])
        self.assertEqual(metrics.steps, 3)
        self.assertEqual(metrics.target_count, 6)
        self.assertTrue(math.isfinite(metrics.nll))
        self.assertTrue(math.isfinite(metrics.ppl))
        self.assertGreaterEqual(metrics.mean_gradient_norm, 0.0)
        self.assertGreater(metrics.elapsed_seconds, 0.0)
        self.assertGreater(metrics.tokens_per_second, 0.0)
        payload = json.loads(metrics.to_json())
        self.assertGreater(payload["elapsed_seconds"], 0.0)
        self.assertGreater(payload["tokens_per_second"], 0.0)

    def test_token_budget_is_a_hard_bound_with_final_time_slice(self) -> None:
        model = _StateRecordingModel()
        batches = [
            _Batch(((0, 1),), ((1, 0),), True),
            _Batch(((1, 0),), ((0, 1),), False),
            _Batch(((0, 0),), ((1, 1),), False),
        ]
        metrics = train_language_model(
            model,
            batches,
            TrainingConfig(
                seed=8,
                learning_rate=1e-2,
                weight_decay=0.0,
                gradient_clip_norm=1.0,
                max_steps=10,
                token_budget=3,
            ),
        )

        self.assertEqual(metrics.steps, 2)
        self.assertEqual(metrics.target_count, 3)
        self.assertEqual(model.state_was_none, [True, False])

    def test_training_is_deterministic_for_equal_initialisation_and_seed(self) -> None:
        def make_model() -> CausalLanguageModel:
            seed_everything(123)
            return CausalLanguageModel(
                vocab_size=7,
                core=StatefulLSTMCore(4, 5),
            )

        batches = [
            _Batch(
                torch.tensor([[0, 1, 2], [2, 3, 4]]),
                torch.tensor([[1, 2, 3], [3, 4, 5]]),
                True,
            ),
            _Batch(
                torch.tensor([[3, 4], [5, 6]]),
                torch.tensor([[4, 5], [6, 0]]),
                False,
            ),
        ]
        config = TrainingConfig(
            seed=77,
            learning_rate=5e-3,
            weight_decay=1e-3,
            gradient_clip_norm=0.75,
            max_steps=2,
        )
        first_model = make_model()
        second_model = make_model()

        first_metrics = train_language_model(first_model, batches, config)
        second_metrics = train_language_model(second_model, batches, config)

        deterministic_fields = (
            "seed",
            "steps",
            "target_count",
            "nll",
            "ppl",
            "mean_gradient_norm",
        )
        for field in deterministic_fields:
            self.assertEqual(getattr(first_metrics, field), getattr(second_metrics, field))
        self.assertGreater(first_metrics.elapsed_seconds, 0.0)
        self.assertGreater(second_metrics.elapsed_seconds, 0.0)
        self.assertGreater(first_metrics.tokens_per_second, 0.0)
        self.assertGreater(second_metrics.tokens_per_second, 0.0)
        for name, first_parameter in first_model.state_dict().items():
            torch.testing.assert_close(
                first_parameter,
                second_model.state_dict()[name],
                atol=0.0,
                rtol=0.0,
            )
        json.dumps(first_metrics.to_dict(), allow_nan=False)

    def test_streaming_benchmark_reports_latency_throughput_and_state_bytes(self) -> None:
        seed_everything(31)
        model = CausalLanguageModel(
            vocab_size=6,
            core=StatefulLSTMCore(3, 4),
        )
        model.train(True)

        metrics = benchmark_streaming_step(
            model,
            [0, 1, 2, 3],
            warmup_steps=2,
            measured_steps=6,
            seed=31,
        )

        self.assertTrue(model.training)
        self.assertEqual(metrics.seed, 31)
        self.assertEqual(metrics.warmup_steps, 2)
        self.assertEqual(metrics.measured_steps, 6)
        self.assertGreater(metrics.latency_mean_ms, 0.0)
        self.assertLessEqual(metrics.latency_p50_ms, metrics.latency_p95_ms)
        self.assertLessEqual(metrics.latency_p95_ms, metrics.latency_p99_ms)
        self.assertGreater(metrics.tokens_per_second, 0.0)
        # One-layer LSTM: hidden + cell, each [1, batch=1, hidden=4] float32.
        self.assertEqual(metrics.state_nbytes, 2 * 1 * 1 * 4 * 4)
        payload = json.loads(metrics.to_json())
        self.assertEqual(
            set(payload),
            {
                "seed",
                "warmup_steps",
                "measured_steps",
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "tokens_per_second",
                "state_nbytes",
            },
        )


if __name__ == "__main__":
    unittest.main()
