"""Training, held-out evaluation, rollout, and latency metrics for HomeGrid M0."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .cores import state_nbytes
from .homegrid_corpus import HomeGridChunk, HomeGridEpisode
from .homegrid_model import HomeGridWorldModel, VISUAL_PATCHES, VISUAL_VOCAB_SIZE


Tensor = torch.Tensor


@dataclass(frozen=True)
class HomeGridTrainingConfig:
    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    epochs: int = 3
    language_weight: float = 0.25
    read_weight: float = 0.10
    reward_weight: float = 0.10
    done_weight: float = 0.10
    reward_enabled: bool = True
    done_enabled: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.weight_decay < 0.0 or self.epochs <= 0:
            raise ValueError("weight decay cannot be negative and epochs must be positive")
        for name in ("language_weight", "read_weight", "reward_weight", "done_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if torch.device(self.device).type != "cpu":
            raise ValueError("the reproducible HomeGrid harness is CPU-only")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _chunk_tensors(chunk: HomeGridChunk, device: torch.device) -> Dict[str, Tensor]:
    return {
        "visual": torch.tensor(chunk.visual_tokens, dtype=torch.long, device=device).unsqueeze(0),
        "next_visual": torch.tensor(
            chunk.next_visual_tokens, dtype=torch.long, device=device
        ).unsqueeze(0),
        "language": torch.tensor(chunk.language_ids, dtype=torch.long, device=device).unsqueeze(0),
        "next_language": torch.tensor(
            chunk.next_language_ids, dtype=torch.long, device=device
        ).unsqueeze(0),
        "actions": torch.tensor(chunk.actions, dtype=torch.long, device=device).unsqueeze(0),
        "read": torch.tensor(chunk.read_flags, dtype=torch.long, device=device).unsqueeze(0),
        "next_read": torch.tensor(
            chunk.next_read_flags, dtype=torch.long, device=device
        ).unsqueeze(0),
        "reward": torch.tensor(
            chunk.reward_classes, dtype=torch.long, device=device
        ).unsqueeze(0),
        "done": torch.tensor(chunk.done_targets, dtype=torch.long, device=device).unsqueeze(0),
    }


def _losses(output: Any, tensors: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    return {
        "visual": F.cross_entropy(
            output.next_visual_logits.reshape(-1, VISUAL_VOCAB_SIZE),
            tensors["next_visual"].reshape(-1),
        ),
        "language": F.cross_entropy(
            output.next_language_logits.reshape(-1, output.next_language_logits.shape[-1]),
            tensors["next_language"].reshape(-1),
        ),
        "read": F.cross_entropy(
            output.next_read_logits.reshape(-1, 2), tensors["next_read"].reshape(-1)
        ),
        "reward": F.cross_entropy(
            output.reward_logits.reshape(-1, 3), tensors["reward"].reshape(-1)
        ),
        "done": F.cross_entropy(
            output.done_logits.reshape(-1, 2), tensors["done"].reshape(-1)
        ),
    }


def _weighted_loss(losses: Mapping[str, Tensor], config: HomeGridTrainingConfig) -> Tensor:
    total = losses["visual"]
    total = total + config.language_weight * losses["language"]
    total = total + config.read_weight * losses["read"]
    if config.reward_enabled:
        total = total + config.reward_weight * losses["reward"]
    if config.done_enabled:
        total = total + config.done_weight * losses["done"]
    return total


def train_homegrid_model(
    model: HomeGridWorldModel[Any],
    chunks: Iterable[HomeGridChunk],
    config: HomeGridTrainingConfig,
) -> Dict[str, Any]:
    """Train a deterministic ordered stream with stateful truncated BPTT."""

    device = torch.device(config.device)
    seed_everything(config.seed)
    model.to(device)
    model.train(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    state: Any = None
    chunk_count = 0
    transition_count = 0
    visual_target_count = 0
    total_weighted_loss = 0.0
    component_sums = {name: 0.0 for name in ("visual", "language", "read", "reward", "done")}
    gradient_sum = 0.0
    started = time.perf_counter_ns()
    for chunk in chunks:
        if chunk.reset_state:
            state = None
        tensors = _chunk_tensors(chunk, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            tensors["visual"],
            tensors["language"],
            tensors["actions"],
            tensors["read"],
            state,
            detach_state=True,
        )
        state = output.state
        losses = _losses(output, tensors)
        loss = _weighted_loss(losses, config)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("HomeGrid training produced a non-finite loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=config.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient).item()):
            raise FloatingPointError("HomeGrid training produced a non-finite gradient")
        optimizer.step()
        count = chunk.length
        chunk_count += 1
        transition_count += count
        visual_target_count += count * VISUAL_PATCHES
        total_weighted_loss += float(loss.detach().item()) * count
        for name, component in losses.items():
            component_sums[name] += float(component.detach().item()) * count
        gradient_sum += float(gradient.detach().item())
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
    if transition_count == 0:
        raise ValueError("HomeGrid training consumed no transitions")
    return {
        "seed": config.seed,
        "epochs": config.epochs,
        "chunks": chunk_count,
        "transitions": transition_count,
        "visual_targets": visual_target_count,
        "weighted_loss": total_weighted_loss / transition_count,
        "component_nll": {
            name: value / transition_count for name, value in component_sums.items()
        },
        "mean_gradient_norm": gradient_sum / chunk_count,
        "elapsed_seconds": elapsed,
        "transitions_per_second": transition_count / elapsed,
        "visual_tokens_per_second": visual_target_count / elapsed,
        "loss_weights": {
            "visual": 1.0,
            "language": config.language_weight,
            "read": config.read_weight,
            "reward": config.reward_weight if config.reward_enabled else 0.0,
            "done": config.done_weight if config.done_enabled else 0.0,
        },
        "reward_loss_enabled": config.reward_enabled,
        "done_loss_enabled": config.done_enabled,
    }


class _VisualAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.nll_sum = 0.0
        self.correct = 0
        self.confusion = torch.zeros(
            VISUAL_VOCAB_SIZE, VISUAL_VOCAB_SIZE, dtype=torch.int64
        )

    def add(self, nll: Tensor, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        selected = mask.reshape(-1)
        count = int(selected.sum().item())
        if count == 0:
            return
        flat_prediction = prediction.reshape(-1)[selected].to(device="cpu")
        flat_target = target.reshape(-1)[selected].to(device="cpu")
        self.count += count
        self.nll_sum += float(nll.reshape(-1)[selected].sum().item())
        self.correct += int((flat_prediction == flat_target).sum().item())
        indices = flat_target * VISUAL_VOCAB_SIZE + flat_prediction
        bins = torch.bincount(indices, minlength=VISUAL_VOCAB_SIZE**2)
        self.confusion += bins.reshape(VISUAL_VOCAB_SIZE, VISUAL_VOCAB_SIZE)

    def result(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "nll": None,
                "accuracy": None,
                "macro_f1_present_targets": None,
                "present_target_classes": [],
            }
        true_positive = self.confusion.diag().to(dtype=torch.float64)
        false_positive = self.confusion.sum(dim=0) - self.confusion.diag()
        false_negative = self.confusion.sum(dim=1) - self.confusion.diag()
        denominator = 2 * true_positive + false_positive + false_negative
        target_support = self.confusion.sum(dim=1)
        present = target_support > 0
        f1 = torch.where(
            denominator > 0,
            2 * true_positive / denominator.to(dtype=torch.float64),
            torch.zeros_like(true_positive),
        )
        return {
            "count": self.count,
            "nll": self.nll_sum / self.count,
            "accuracy": self.correct / self.count,
            "macro_f1_present_targets": float(f1[present].mean().item()),
            "present_target_classes": present.nonzero().reshape(-1).tolist(),
        }


class _ClassAccumulator:
    def __init__(self, classes: int, enabled: bool = True) -> None:
        self.classes = classes
        self.enabled = enabled
        self.count = 0
        self.nll_sum = 0.0
        self.correct = 0
        self.brier_sum = 0.0
        self.target_classes = set()

    def add(self, logits: Tensor, target: Tensor) -> None:
        flat_logits = logits.reshape(-1, self.classes)
        flat_target = target.reshape(-1)
        self.target_classes.update(int(value) for value in flat_target.unique().tolist())
        if not self.enabled:
            return
        probabilities = flat_logits.softmax(dim=-1)
        one_hot = F.one_hot(flat_target, num_classes=self.classes).to(probabilities.dtype)
        self.count += int(flat_target.numel())
        self.nll_sum += float(
            F.cross_entropy(flat_logits, flat_target, reduction="sum").item()
        )
        self.correct += int((flat_logits.argmax(dim=-1) == flat_target).sum().item())
        self.brier_sum += float(((probabilities - one_hot) ** 2).sum(dim=-1).sum().item())

    def result(self) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "reason": "training split did not contain multiple target classes",
                "target_classes": sorted(self.target_classes),
            }
        return {
            "enabled": True,
            "count": self.count,
            "nll": self.nll_sum / self.count,
            "accuracy": self.correct / self.count,
            "brier": self.brier_sum / self.count,
            "target_classes": sorted(self.target_classes),
        }


def evaluate_homegrid_model(
    model: HomeGridWorldModel[Any],
    chunks: Iterable[HomeGridChunk],
    *,
    frequency_visual_token: int,
    reward_enabled: bool,
    done_enabled: bool,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate one-step action-conditioned prediction with change-aware metrics."""

    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise ValueError("the reproducible HomeGrid harness is CPU-only")
    model.to(resolved)
    was_training = model.training
    model.eval()
    state: Any = None
    visual = {
        name: _VisualAccumulator()
        for name in ("overall", "changed", "unchanged", "read_phase", "action_phase")
    }
    language = _ClassAccumulator(model.language_vocab_size)
    read = _ClassAccumulator(2)
    reward = _ClassAccumulator(3, enabled=reward_enabled)
    done = _ClassAccumulator(2, enabled=done_enabled)
    copy_correct = 0
    frequency_correct = 0
    total_visual = 0
    transition_count = 0
    chunk_count = 0
    try:
        with torch.inference_mode():
            for chunk in chunks:
                if chunk.reset_state:
                    state = None
                tensors = _chunk_tensors(chunk, resolved)
                output = model(
                    tensors["visual"],
                    tensors["language"],
                    tensors["actions"],
                    tensors["read"],
                    state,
                    detach_state=True,
                )
                state = output.state
                flat_logits = output.next_visual_logits.reshape(-1, VISUAL_VOCAB_SIZE)
                visual_nll = F.cross_entropy(
                    flat_logits,
                    tensors["next_visual"].reshape(-1),
                    reduction="none",
                ).reshape_as(tensors["next_visual"])
                prediction = output.next_visual_logits.argmax(dim=-1)
                changed = tensors["visual"] != tensors["next_visual"]
                read_mask = tensors["read"].to(dtype=torch.bool).unsqueeze(-1).expand_as(changed)
                all_mask = torch.ones_like(changed, dtype=torch.bool)
                visual["overall"].add(
                    visual_nll, prediction, tensors["next_visual"], all_mask
                )
                visual["changed"].add(
                    visual_nll, prediction, tensors["next_visual"], changed
                )
                visual["unchanged"].add(
                    visual_nll, prediction, tensors["next_visual"], ~changed
                )
                visual["read_phase"].add(
                    visual_nll, prediction, tensors["next_visual"], read_mask
                )
                visual["action_phase"].add(
                    visual_nll, prediction, tensors["next_visual"], ~read_mask
                )
                language.add(output.next_language_logits, tensors["next_language"])
                read.add(output.next_read_logits, tensors["next_read"])
                reward.add(output.reward_logits, tensors["reward"])
                done.add(output.done_logits, tensors["done"])
                copy_correct += int(
                    (tensors["visual"] == tensors["next_visual"]).sum().item()
                )
                frequency_correct += int(
                    (tensors["next_visual"] == frequency_visual_token).sum().item()
                )
                total_visual += int(tensors["next_visual"].numel())
                transition_count += chunk.length
                chunk_count += 1
    finally:
        model.train(was_training)
    if transition_count == 0:
        raise ValueError("HomeGrid evaluation consumed no transitions")
    return {
        "chunks": chunk_count,
        "transitions": transition_count,
        "visual": {name: accumulator.result() for name, accumulator in visual.items()},
        "next_language": language.result(),
        "next_read": read.result(),
        "reward": reward.result(),
        "done": done.result(),
        "baselines": {
            "copy_current_frame": {
                "overall_accuracy": copy_correct / total_visual,
                "changed_accuracy": 0.0,
            },
            "train_global_frequency": {
                "token": frequency_visual_token,
                "overall_accuracy": frequency_correct / total_visual,
            },
        },
        "change_mask_definition": "next_visual_token != current_visual_token",
    }


def _step_inputs(row: Any, visual: Optional[Sequence[int]] = None) -> Tuple[Tensor, ...]:
    source_visual = row.visual_tokens if visual is None else visual
    return (
        torch.tensor(source_visual, dtype=torch.long).unsqueeze(0),
        torch.tensor([row.language_id], dtype=torch.long),
        torch.tensor([row.action], dtype=torch.long),
        torch.tensor([int(row.is_read_step)], dtype=torch.long),
    )


def evaluate_homegrid_rollouts(
    model: HomeGridWorldModel[Any],
    episodes: Iterable[HomeGridEpisode],
    *,
    horizons: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """Controlled visual rollout with future actions/language fixed to real data."""

    ordered_horizons = tuple(sorted(set(int(value) for value in horizons)))
    if not ordered_horizons or ordered_horizons[0] <= 0:
        raise ValueError("rollout horizons must be positive")
    model.to(torch.device("cpu"))
    was_training = model.training
    model.eval()
    accumulators = {
        horizon: {"overall": _VisualAccumulator(), "changed": _VisualAccumulator()}
        for horizon in ordered_horizons
    }
    anchor_counts = {horizon: 0 for horizon in ordered_horizons}
    try:
        with torch.inference_mode():
            for episode in episodes:
                main_state: Any = None
                rows = episode.transitions
                for index, row in enumerate(rows):
                    inputs = _step_inputs(row)
                    factual = model.step(*inputs, main_state, detach_state=True)
                    main_state = factual.state
                    if row.is_read_step:
                        continue
                    predicted_visual = factual.last_next_visual_logits.argmax(dim=-1)
                    branch_state = factual.state
                    for step_ahead in range(1, max(ordered_horizons) + 1):
                        target_index = index + step_ahead - 1
                        if target_index >= len(rows):
                            break
                        target = rows[target_index].next_visual_tokens
                        if step_ahead in accumulators:
                            target_tensor = torch.tensor(target, dtype=torch.long).unsqueeze(0)
                            anchor_tensor = torch.tensor(
                                row.visual_tokens, dtype=torch.long
                            ).unsqueeze(0)
                            nll_dummy = torch.zeros_like(target_tensor, dtype=torch.float32)
                            overall_mask = torch.ones_like(target_tensor, dtype=torch.bool)
                            changed_mask = target_tensor != anchor_tensor
                            prediction = predicted_visual
                            accumulators[step_ahead]["overall"].add(
                                nll_dummy, prediction, target_tensor, overall_mask
                            )
                            accumulators[step_ahead]["changed"].add(
                                nll_dummy, prediction, target_tensor, changed_mask
                            )
                            anchor_counts[step_ahead] += 1
                        if step_ahead >= max(ordered_horizons):
                            break
                        next_index = index + step_ahead
                        if next_index >= len(rows):
                            break
                        next_row = rows[next_index]
                        branch = model.step(
                            *_step_inputs(next_row, predicted_visual[0].tolist()),
                            branch_state,
                            detach_state=True,
                        )
                        branch_state = branch.state
                        predicted_visual = branch.last_next_visual_logits.argmax(dim=-1)
    finally:
        model.train(was_training)
    return {
        "horizons": {
            str(horizon): {
                "anchors": anchor_counts[horizon],
                "overall_accuracy": accumulators[horizon]["overall"].result()["accuracy"],
                "changed_accuracy": accumulators[horizon]["changed"].result()["accuracy"],
                "overall_patch_count": accumulators[horizon]["overall"].count,
                "changed_patch_count": accumulators[horizon]["changed"].count,
            }
            for horizon in ordered_horizons
        },
        "anchor_phase": "action_only",
        "conditioning": "real future actions/language/read flags; predicted visual recursively fed back",
        "changed_mask_definition": "horizon target token != anchor current token",
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def benchmark_homegrid_streaming(
    model: HomeGridWorldModel[Any],
    episode: HomeGridEpisode,
    *,
    warmup_steps: int = 32,
    measured_steps: int = 64,
) -> Dict[str, Any]:
    """Measure the complete batch-one transition update after tensor preparation."""

    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("invalid streaming benchmark lengths")
    if warmup_steps + measured_steps > len(episode.transitions):
        raise ValueError("episode is too short for the streaming benchmark")
    prepared = [_step_inputs(row) for row in episode.transitions[: warmup_steps + measured_steps]]
    model.to(torch.device("cpu"))
    was_training = model.training
    model.eval()
    state: Any = None
    durations = []
    try:
        with torch.inference_mode():
            for index, inputs in enumerate(prepared):
                if index < warmup_steps:
                    output = model.step(*inputs, state, detach_state=True)
                else:
                    started = time.perf_counter_ns()
                    output = model.step(*inputs, state, detach_state=True)
                    ended = time.perf_counter_ns()
                    durations.append((ended - started) / 1_000_000.0)
                state = output.state
    finally:
        model.train(was_training)
    total_seconds = sum(durations) / 1_000.0
    return {
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "history_steps": warmup_steps + measured_steps,
        "latency_mean_ms": sum(durations) / len(durations),
        "latency_p50_ms": _percentile(durations, 50.0),
        "latency_p95_ms": _percentile(durations, 95.0),
        "latency_p99_ms": _percentile(durations, 99.0),
        "transitions_per_second": measured_steps / total_seconds,
        "state_nbytes": state_nbytes(state),
        "timed_scope": "model.step including multimodal encoder, temporal core, and all heads",
    }


__all__ = [
    "HomeGridTrainingConfig",
    "benchmark_homegrid_streaming",
    "evaluate_homegrid_model",
    "evaluate_homegrid_rollouts",
    "seed_everything",
    "train_homegrid_model",
]
