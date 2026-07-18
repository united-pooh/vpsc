# ruff: noqa: E402

import math

import pytest


torch = pytest.importorskip("torch")

from vpsc.world_model.cores import StatefulLSTMCore
from vpsc.world_model.homegrid_corpus import (
    HomeGridChunk,
    HomeGridEpisode,
    HomeGridTransition,
)
from vpsc.world_model.homegrid_model import (
    HomeGridWorldModel,
    HomeGridWorldModelOutput,
)
from vpsc.world_model.homegrid_training import (
    HomeGridTrainingConfig,
    benchmark_homegrid_streaming,
    evaluate_homegrid_model,
    evaluate_homegrid_rollouts,
    train_homegrid_model,
)


@pytest.fixture(autouse=True)
def _single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _visual(first):
    return (first,) + (0,) * 143


def _chunk():
    return HomeGridChunk(
        episode_seed=7,
        epoch=0,
        offset=0,
        reset_state=True,
        visual_tokens=(_visual(0), _visual(1)),
        next_visual_tokens=(_visual(1), _visual(2)),
        language_ids=(2, 3),
        next_language_ids=(3, 4),
        actions=(0, 1),
        read_flags=(True, False),
        next_read_flags=(False, False),
        reward_classes=(0, 2),
        done_targets=(0, 1),
    )


def _transition(step, *, read=False):
    return HomeGridTransition(
        step=step,
        visual_tokens=_visual(step),
        next_visual_tokens=_visual(step + 1),
        language_raw=10 + step,
        next_language_raw=11 + step,
        language_id=2 + step,
        next_language_id=3 + step,
        action=step % 10,
        is_read_step=read,
        next_is_read_step=False,
        reward=0.0,
        reward_class=0,
        done=False,
        changed_patch_count=1,
    )


def _episode():
    return HomeGridEpisode(
        split="test",
        seed=7,
        episode_id="homegrid-dynamics:test:7",
        transitions=(
            _transition(0, read=True),
            _transition(1),
            _transition(2),
        ),
    )


class CopyPredictor(torch.nn.Module):
    """Deterministic fake: predicts current visual token at every patch."""

    def __init__(self, language_vocab_size=8):
        super().__init__()
        self.language_vocab_size = language_vocab_size
        self.step_state_inputs = []

    def _output(self, visual, language, read, state):
        batch, time, patches = visual.shape
        visual_logits = torch.full(
            (batch, time, patches, 64),
            -20.0,
            device=visual.device,
        )
        visual_logits.scatter_(-1, visual.unsqueeze(-1), 20.0)
        language_logits = torch.full(
            (batch, time, self.language_vocab_size),
            -20.0,
            device=visual.device,
        )
        language_logits.scatter_(-1, language.unsqueeze(-1), 20.0)
        read_logits = torch.full((batch, time, 2), -20.0, device=visual.device)
        read_logits.scatter_(-1, read.long().unsqueeze(-1), 20.0)
        reward_logits = torch.zeros((batch, time, 3), device=visual.device)
        done_logits = torch.zeros((batch, time, 2), device=visual.device)
        next_state = (
            torch.ones(1, device=visual.device)
            if state is None
            else state + 1
        )
        return HomeGridWorldModelOutput(
            next_visual_logits=visual_logits,
            next_language_logits=language_logits,
            next_read_logits=read_logits,
            reward_logits=reward_logits,
            done_logits=done_logits,
            state=next_state,
            hidden_states=torch.zeros(batch, time, 1, device=visual.device),
        )

    def forward(
        self,
        visual,
        language,
        actions,
        read,
        state=None,
        *,
        detach_state=False,
    ):
        del actions, detach_state
        return self._output(visual, language, read, state)

    def step(
        self,
        visual,
        language,
        actions,
        read,
        state=None,
        *,
        detach_state=False,
    ):
        del actions, detach_state
        self.step_state_inputs.append(
            None if state is None else state.detach().clone()
        )
        return self._output(
            visual.unsqueeze(1),
            language.unsqueeze(1),
            read.unsqueeze(1),
            state,
        )


def test_tiny_real_model_training_is_finite_and_updates_parameters():
    model = HomeGridWorldModel(
        language_vocab_size=8,
        d_model=8,
        core=StatefulLSTMCore(8, 8, num_layers=1, dropout=0.0),
    )
    before = model.heads["next_visual"].weight.detach().clone()

    metrics = train_homegrid_model(
        model,
        [_chunk()],
        HomeGridTrainingConfig(
            seed=3,
            learning_rate=1e-3,
            weight_decay=0.0,
            epochs=1,
            reward_enabled=True,
            done_enabled=True,
        ),
    )

    assert metrics["chunks"] == 1
    assert metrics["transitions"] == 2
    assert metrics["visual_targets"] == 288
    assert math.isfinite(metrics["weighted_loss"])
    assert math.isfinite(metrics["mean_gradient_norm"])
    assert all(math.isfinite(value) for value in metrics["component_nll"].values())
    assert not torch.equal(before, model.heads["next_visual"].weight)


def test_evaluation_reports_change_masks_and_copy_frequency_baselines():
    metrics = evaluate_homegrid_model(
        CopyPredictor(),
        [_chunk()],
        frequency_visual_token=0,
        reward_enabled=False,
        done_enabled=False,
    )

    assert metrics["transitions"] == 2
    assert metrics["visual"]["overall"]["count"] == 288
    assert metrics["visual"]["changed"]["count"] == 2
    assert metrics["visual"]["changed"]["accuracy"] == 0.0
    assert metrics["visual"]["unchanged"]["count"] == 286
    assert metrics["visual"]["unchanged"]["accuracy"] == 1.0
    expected = 286 / 288
    assert metrics["baselines"]["copy_current_frame"][
        "overall_accuracy"
    ] == pytest.approx(expected)
    assert metrics["baselines"]["train_global_frequency"][
        "overall_accuracy"
    ] == pytest.approx(expected)
    assert metrics["baselines"]["train_global_frequency"]["token"] == 0
    assert metrics["reward"]["enabled"] is False
    assert metrics["done"]["enabled"] is False


def test_rollouts_report_requested_horizons_without_read_phase_anchors():
    metrics = evaluate_homegrid_rollouts(
        CopyPredictor(),
        [_episode()],
        horizons=(1, 2, 4, 2),
    )

    assert list(metrics["horizons"]) == ["1", "2", "4"]
    assert metrics["horizons"]["1"]["anchors"] == 2
    assert metrics["horizons"]["2"]["anchors"] == 1
    assert metrics["horizons"]["4"]["anchors"] == 0
    assert metrics["horizons"]["1"]["overall_patch_count"] == 288
    assert metrics["horizons"]["2"]["overall_patch_count"] == 144
    assert metrics["horizons"]["4"]["overall_accuracy"] is None
    assert metrics["anchor_phase"] == "action_only"


def test_streaming_benchmark_threads_state_across_warmup_and_measured_steps():
    model = CopyPredictor()
    metrics = benchmark_homegrid_streaming(
        model,
        _episode(),
        warmup_steps=1,
        measured_steps=2,
    )

    assert metrics["warmup_steps"] == 1
    assert metrics["measured_steps"] == 2
    assert metrics["history_steps"] == 3
    assert metrics["state_nbytes"] == 4
    assert metrics["latency_mean_ms"] >= 0.0
    assert metrics["transitions_per_second"] > 0.0
    assert model.step_state_inputs[0] is None
    assert torch.equal(model.step_state_inputs[1], torch.tensor([1.0]))
    assert torch.equal(model.step_state_inputs[2], torch.tensor([2.0]))
