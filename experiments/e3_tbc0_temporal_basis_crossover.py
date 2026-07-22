#!/usr/bin/env python3
"""TBC-0: deconfound capacity, decay bands, and routing in temporal MoE-SNNs.

The first stage uses a deterministic synthetic event-recall task so the
mechanism can be exercised without downloading a corpus.  It is a smoke and
instrumentation study, not evidence that d4 beats an ANN language model.

Variants:

* ``base_same_width``: one gated-trace core at the nominal state width.
* ``base_param_matched``: one core widened to match temporal-MoE parameters.
* ``temporal``: distinct short/medium/long decay bands and change routing.
* ``homogeneous``: the same MoE capacity and router, but identical decay bands.
* ``uniform``: distinct decay bands with a frozen uniform router.

The temporal variants also support evaluation-only uniform and time-reversed
router interventions.  These interventions test whether a trained model uses
the routing signal rather than merely benefiting from extra parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vpsc.world_model.cores import E3GatedTraceScanCore  # noqa: E402
from vpsc.world_model.devices import choose_device, device_label, synchronize  # noqa: E402
from vpsc.world_model.lm import CausalLanguageModel  # noqa: E402
from vpsc.world_model.scaling_variants import TemporalMoEGatedTraceCore  # noqa: E402


IGNORE_INDEX = -100
RouterIntervention = Literal["none", "uniform", "reverse_time"]
VariantName = Literal[
    "base_same_width",
    "base_param_matched",
    "temporal",
    "homogeneous",
    "uniform",
]


@dataclass(frozen=True)
class TaskConfig:
    train_sequences: int = 128
    valid_sequences: int = 64
    seq_len: int = 64
    payload_values: int = 8
    horizons: Tuple[int, ...] = (2, 8, 24)
    gap_min: int = 1
    gap_max: int = 4
    event_probability: Optional[float] = None

    @property
    def fill_token(self) -> int:
        return self.payload_values

    @property
    def query_token(self) -> int:
        return self.payload_values + 1

    @property
    def vocab_size(self) -> int:
        return self.payload_values + 2


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 16
    state_dim: int = 16
    n_experts: int = 3


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-3
    grad_clip: float = 1.0


@dataclass(frozen=True)
class EventRecallDataset:
    inputs: Tensor
    targets: Tensor
    event_mask: Tensor
    query_mask: Tensor

    @property
    def query_count(self) -> int:
        return int(self.query_mask.sum())


class IntervenableTemporalMoECore(TemporalMoEGatedTraceCore):
    """Temporal MoE with deterministic evaluation-only router interventions."""

    def __init__(self, *args, router_intervention: RouterIntervention = "none", **kwargs):
        super().__init__(*args, **kwargs)
        self.router_intervention: RouterIntervention = "none"
        self.set_router_intervention(router_intervention)

    def set_router_intervention(self, mode: RouterIntervention) -> None:
        if mode not in ("none", "uniform", "reverse_time"):
            raise ValueError(f"unknown router intervention: {mode}")
        self.router_intervention = mode

    def raw_gate_logits(self, x: Tensor) -> Tensor:
        return super()._gate_logits(x)

    def _gate_logits(self, x: Tensor) -> Tensor:
        logits = self.raw_gate_logits(x)
        if self.router_intervention == "uniform":
            return torch.zeros_like(logits)
        if self.router_intervention == "reverse_time":
            return logits.flip(dims=(1,))
        return logits


def temporal_decay_bands(n_experts: int) -> List[Tuple[float, float]]:
    """Return the frozen d4 short-to-long bands without constructing a model."""

    if n_experts < 2:
        raise ValueError("n_experts must be at least two")
    lo = torch.linspace(0.50, 0.80, n_experts).tolist()
    hi = torch.linspace(0.80, 0.99, n_experts).tolist()
    return [(float(lo[index]), float(hi[index])) for index in range(n_experts)]


def homogeneous_decay_bands(n_experts: int) -> List[Tuple[float, float]]:
    """Give every expert the same broad decay support while preserving capacity."""

    if n_experts < 2:
        raise ValueError("n_experts must be at least two")
    return [(0.50, 0.99) for _ in range(n_experts)]


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def build_model(
    variant: VariantName,
    task: TaskConfig,
    model_cfg: ModelConfig,
    *,
    matched_state_dim: Optional[int] = None,
) -> CausalLanguageModel:
    state_dim = model_cfg.state_dim
    if variant == "base_param_matched":
        if matched_state_dim is None:
            raise ValueError("base_param_matched requires matched_state_dim")
        state_dim = int(matched_state_dim)

    if variant in ("base_same_width", "base_param_matched"):
        core = E3GatedTraceScanCore(
            model_cfg.d_model,
            model_cfg.d_model,
            state_dim=state_dim,
        )
    else:
        bands = (
            homogeneous_decay_bands(model_cfg.n_experts)
            if variant == "homogeneous"
            else temporal_decay_bands(model_cfg.n_experts)
        )
        intervention: RouterIntervention = "uniform" if variant == "uniform" else "none"
        core = IntervenableTemporalMoECore(
            model_cfg.d_model,
            model_cfg.d_model,
            state_dim=model_cfg.state_dim,
            n_experts=model_cfg.n_experts,
            decay_bands=bands,
            router_intervention=intervention,
        )
        if variant == "uniform":
            core.change_router.requires_grad_(False)

    return CausalLanguageModel(
        task.vocab_size,
        core,
        padding_idx=None,
        ignore_index=IGNORE_INDEX,
    )


def find_parameter_matched_base_state_dim(
    task: TaskConfig,
    model_cfg: ModelConfig,
    target_parameters: int,
    *,
    maximum_state_dim: int = 8192,
) -> Tuple[int, int, float]:
    """Find the single-core state width closest to a temporal MoE parameter count."""

    best_state = 1
    best_count = count_parameters(
        build_model(
            "base_param_matched",
            task,
            model_cfg,
            matched_state_dim=best_state,
        )
    )
    best_gap = abs(best_count - target_parameters)
    low, high = 1, maximum_state_dim
    while low <= high:
        state_dim = (low + high) // 2
        count = count_parameters(
            build_model(
                "base_param_matched",
                task,
                model_cfg,
                matched_state_dim=state_dim,
            )
        )
        gap = abs(count - target_parameters)
        if gap < best_gap:
            best_state, best_count, best_gap = state_dim, count, gap
        if count < target_parameters:
            low = state_dim + 1
        elif count > target_parameters:
            high = state_dim - 1
        else:
            break
    relative_gap = best_gap / max(1, target_parameters)
    return best_state, best_count, relative_gap


def make_event_recall_dataset(
    sequences: int,
    task: TaskConfig,
    *,
    seed: int,
) -> EventRecallDataset:
    """Create sparse delayed queries whose input token does not reveal the answer."""

    if sequences <= 0 or task.seq_len <= 2:
        raise ValueError("sequences must be positive and seq_len must exceed two")
    if task.payload_values <= 1:
        raise ValueError("payload_values must exceed one")
    if not task.horizons or min(task.horizons) <= 0 or max(task.horizons) >= task.seq_len:
        raise ValueError("horizons must lie inside (0, seq_len)")
    if task.gap_min < 0 or task.gap_max < task.gap_min:
        raise ValueError("invalid gap range")
    if task.event_probability is not None and not 0.0 < task.event_probability < 1.0:
        raise ValueError("event_probability must lie inside (0, 1)")

    rng = random.Random(seed)
    inputs = torch.full(
        (sequences, task.seq_len), task.fill_token, dtype=torch.long
    )
    targets = torch.full(
        (sequences, task.seq_len), IGNORE_INDEX, dtype=torch.long
    )
    event_mask = torch.zeros((sequences, task.seq_len), dtype=torch.bool)
    query_mask = torch.zeros_like(event_mask)

    for row in range(sequences):
        if task.event_probability is None:
            event_time = rng.randrange(0, min(4, task.seq_len - 1))
            while event_time < task.seq_len:
                horizon = rng.choice(task.horizons)
                query_time = event_time + horizon
                if query_time >= task.seq_len:
                    break
                payload = rng.randrange(task.payload_values)
                inputs[row, event_time] = payload
                event_mask[row, event_time] = True
                inputs[row, query_time] = task.query_token
                targets[row, query_time] = payload
                query_mask[row, query_time] = True
                event_time = query_time + 1 + rng.randint(task.gap_min, task.gap_max)
            continue

        # Factorised grid mode: event proposals are Bernoulli and can overlap
        # in flight.  A due query owns its timestep, so its input is always the
        # answer-independent QUERY token.  This makes event probability an
        # independent control rather than forcing the next event to wait for
        # the previous query, as the original gap-based smoke generator did.
        scheduled_queries: Dict[int, int] = {}
        for time_index in range(task.seq_len):
            if time_index in scheduled_queries:
                inputs[row, time_index] = task.query_token
                targets[row, time_index] = scheduled_queries[time_index]
                query_mask[row, time_index] = True
                continue
            if rng.random() >= task.event_probability:
                continue
            horizon = rng.choice(task.horizons)
            query_time = time_index + horizon
            if query_time >= task.seq_len or query_time in scheduled_queries:
                continue
            payload = rng.randrange(task.payload_values)
            inputs[row, time_index] = payload
            event_mask[row, time_index] = True
            scheduled_queries[query_time] = payload

    if not bool(query_mask.any()):
        raise RuntimeError("event-recall dataset contains no queries")
    return EventRecallDataset(inputs, targets, event_mask, query_mask)


def _batches(size: int, batch_size: int, order: Tensor) -> Iterable[Tensor]:
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


def evaluate_model(
    model: CausalLanguageModel,
    dataset: EventRecallDataset,
    *,
    batch_size: int,
    device: torch.device,
    intervention: RouterIntervention = "none",
) -> Dict[str, float]:
    core = model.core
    previous: Optional[RouterIntervention] = None
    if isinstance(core, IntervenableTemporalMoECore):
        previous = core.router_intervention
        core.set_router_intervention(intervention)
    elif intervention != "none":
        raise ValueError("router intervention requires a temporal MoE core")

    model.eval()
    loss_sum = 0.0
    correct = 0
    queries = 0
    synchronize(device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            order = torch.arange(dataset.inputs.shape[0])
            for indices in _batches(len(order), batch_size, order):
                inputs = dataset.inputs[indices].to(device)
                targets = dataset.targets[indices].to(device)
                output = model(inputs, targets=targets)
                count = int(output.target_count)
                loss_sum += float(output.loss) * count
                mask = targets.ne(IGNORE_INDEX)
                correct += int(output.logits.argmax(dim=-1)[mask].eq(targets[mask]).sum())
                queries += count
    finally:
        if isinstance(core, IntervenableTemporalMoECore) and previous is not None:
            core.set_router_intervention(previous)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "query_nll": loss_sum / max(1, queries),
        "query_accuracy": correct / max(1, queries),
        "query_count": float(queries),
        "elapsed_s": elapsed,
        "tokens_per_s": dataset.inputs.numel() / max(elapsed, 1e-9),
    }


def _pearson(x: Tensor, y: Tensor) -> float:
    x = x.detach().double().flatten()
    y = y.detach().double().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denominator) == 0.0:
        return 0.0
    return float((x * y).sum() / denominator)


def routing_diagnostics(
    model: CausalLanguageModel,
    dataset: EventRecallDataset,
    *,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    core = model.core
    if not isinstance(core, IntervenableTemporalMoECore):
        return None
    model.eval()
    with torch.no_grad():
        inputs = dataset.inputs.to(device)
        embedded = model.embedding(inputs)
        raw_logits = core.raw_gate_logits(embedded)
        weights = torch.softmax(raw_logits, dim=-1)
        difference = torch.zeros_like(embedded)
        difference[:, 1:] = embedded[:, 1:] - embedded[:, :-1]
        change = difference.norm(dim=-1)
        short_weight = weights[..., 0]
        events = dataset.event_mask.to(device)
        fillers = ~(dataset.event_mask | dataset.query_mask).to(device)
        usage = weights.mean(dim=(0, 1))
        entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        entropy /= math.log(core.n_experts)
        event_mean = float(short_weight[events].mean()) if bool(events.any()) else 0.0
        filler_mean = float(short_weight[fillers].mean()) if bool(fillers.any()) else 0.0
    return {
        "change_short_weight_correlation": _pearson(change, short_weight),
        "short_weight_event_minus_filler": event_mean - filler_mean,
        "normalised_usage_entropy": float(entropy),
        "mean_usage": [float(value) for value in usage],
    }


def train_model(
    model: CausalLanguageModel,
    train_data: EventRecallDataset,
    *,
    train_cfg: TrainConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=train_cfg.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    synchronize(device)
    started = time.perf_counter()
    updates = 0
    for _epoch in range(train_cfg.epochs):
        order = torch.randperm(train_data.inputs.shape[0], generator=generator)
        for indices in _batches(len(order), train_cfg.batch_size, order):
            inputs = train_data.inputs[indices].to(device)
            targets = train_data.targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs, targets=targets)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, train_cfg.grad_clip)
            optimizer.step()
            updates += 1
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "train_elapsed_s": elapsed,
        "train_tokens_per_s": (
            train_data.inputs.numel() * train_cfg.epochs / max(elapsed, 1e-9)
        ),
        "updates": float(updates),
    }


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    variants = sorted({str(row["variant"]) for row in rows})
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            "seeds": float(len(subset)),
            "params_total": float(subset[0]["params_total"]),
            "normal_query_nll_mean": float(
                np.mean([row["normal"]["query_nll"] for row in subset])
            ),
            "normal_query_accuracy_mean": float(
                np.mean([row["normal"]["query_accuracy"] for row in subset])
            ),
            "train_tokens_per_s_mean": float(
                np.mean([row["training"]["train_tokens_per_s"] for row in subset])
            ),
        }
        if subset[0]["uniform_intervention"] is not None:
            summary[variant]["uniform_accuracy_delta_mean"] = float(
                np.mean(
                    [
                        row["uniform_intervention"]["query_accuracy"]
                        - row["normal"]["query_accuracy"]
                        for row in subset
                    ]
                )
            )
            summary[variant]["reverse_time_accuracy_delta_mean"] = float(
                np.mean(
                    [
                        row["reverse_time_intervention"]["query_accuracy"]
                        - row["normal"]["query_accuracy"]
                        for row in subset
                    ]
                )
            )
    return summary


def run_study(
    task: TaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    variants: Sequence[VariantName],
    seeds: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    reference = build_model("temporal", task, model_cfg)
    target_parameters = count_parameters(reference)
    matched_state_dim, matched_count, matched_gap = find_parameter_matched_base_state_dim(
        task, model_cfg, target_parameters
    )

    train_data = make_event_recall_dataset(task.train_sequences, task, seed=701)
    valid_data = make_event_recall_dataset(task.valid_sequences, task, seed=1701)
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            torch.manual_seed(seed)
            model = build_model(
                variant,
                task,
                model_cfg,
                matched_state_dim=matched_state_dim,
            ).to(device)
            training = train_model(
                model,
                train_data,
                train_cfg=train_cfg,
                device=device,
                seed=seed,
            )
            normal = evaluate_model(
                model,
                valid_data,
                batch_size=train_cfg.batch_size,
                device=device,
                intervention="uniform" if variant == "uniform" else "none",
            )
            uniform_result = None
            reverse_result = None
            if isinstance(model.core, IntervenableTemporalMoECore) and variant != "uniform":
                uniform_result = evaluate_model(
                    model,
                    valid_data,
                    batch_size=train_cfg.batch_size,
                    device=device,
                    intervention="uniform",
                )
                reverse_result = evaluate_model(
                    model,
                    valid_data,
                    batch_size=train_cfg.batch_size,
                    device=device,
                    intervention="reverse_time",
                )
            rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "state_dim": (
                        matched_state_dim if variant == "base_param_matched" else model_cfg.state_dim
                    ),
                    "params_total": count_parameters(model),
                    "params_trainable": count_parameters(model, trainable_only=True),
                    "parameter_gap_to_temporal": abs(
                        count_parameters(model) - target_parameters
                    )
                    / max(1, target_parameters),
                    "training": training,
                    "normal": normal,
                    "uniform_intervention": uniform_result,
                    "reverse_time_intervention": reverse_result,
                    "routing": (
                        None
                        if variant == "uniform"
                        else routing_diagnostics(model, valid_data, device=device)
                    ),
                }
            )

    payload = {
        "experiment": "TBC-0 temporal-basis crossover smoke",
        "status": "SMOKE_ONLY_NOT_A_FORMAL_VERDICT",
        "device": device_label(device),
        "task": asdict(task),
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "variants": list(variants),
        "seeds": [int(seed) for seed in seeds],
        "parameter_match": {
            "target_temporal_params": target_parameters,
            "matched_base_state_dim": matched_state_dim,
            "matched_base_params": matched_count,
            "relative_gap": matched_gap,
        },
        "dataset": {
            "train_queries": train_data.query_count,
            "valid_queries": valid_data.query_count,
            "train_event_density": float(train_data.event_mask.float().mean()),
            "valid_event_density": float(valid_data.event_mask.float().mean()),
        },
        "rows": rows,
    }
    payload["summary"] = _aggregate(rows)
    return payload


def analyse_phase_a_grid(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the frozen paired-effect, intervention, and routing gates."""

    cell_effects: List[Dict[str, Any]] = []
    all_accuracy_advantages: List[float] = []
    all_nll_advantages: List[float] = []
    all_uniform_drops: List[float] = []
    all_reverse_drops: List[float] = []
    all_base_accuracy_advantages: List[float] = []
    all_base_nll_advantages: List[float] = []

    routing_usage: Dict[Tuple[int, float], List[List[float]]] = {}
    parameter_gaps: List[float] = []
    for cell in cells:
        rows = {
            (str(row["variant"]), int(row["seed"])): row for row in cell["rows"]
        }
        seeds = sorted({seed for _variant, seed in rows})
        accuracy_advantages = []
        nll_advantages = []
        uniform_drops = []
        reverse_drops = []
        base_accuracy_advantages = []
        base_nll_advantages = []
        for seed in seeds:
            temporal = rows[("temporal", seed)]
            homogeneous = rows[("homogeneous", seed)]
            base = rows[("base_param_matched", seed)]
            accuracy_advantages.append(
                temporal["normal"]["query_accuracy"]
                - homogeneous["normal"]["query_accuracy"]
            )
            nll_advantages.append(
                homogeneous["normal"]["query_nll"]
                - temporal["normal"]["query_nll"]
            )
            uniform_drops.append(
                temporal["normal"]["query_accuracy"]
                - temporal["uniform_intervention"]["query_accuracy"]
            )
            reverse_drops.append(
                temporal["normal"]["query_accuracy"]
                - temporal["reverse_time_intervention"]["query_accuracy"]
            )
            base_accuracy_advantages.append(
                temporal["normal"]["query_accuracy"]
                - base["normal"]["query_accuracy"]
            )
            base_nll_advantages.append(
                base["normal"]["query_nll"]
                - temporal["normal"]["query_nll"]
            )
            routing = temporal["routing"]
            routing_usage.setdefault(
                (int(cell["grid_horizon"]), float(cell["grid_event_probability"])), []
            ).append(routing["mean_usage"])
            parameter_gaps.append(float(base["parameter_gap_to_temporal"]))

        effect = {
            "horizon": int(cell["grid_horizon"]),
            "event_probability": float(cell["grid_event_probability"]),
            "realised_valid_event_density": float(cell["dataset"]["valid_event_density"]),
            "temporal_minus_homogeneous_accuracy": float(np.mean(accuracy_advantages)),
            "homogeneous_minus_temporal_nll": float(np.mean(nll_advantages)),
            "temporal_uniform_accuracy_drop": float(np.mean(uniform_drops)),
            "temporal_reverse_time_accuracy_drop": float(np.mean(reverse_drops)),
            "temporal_minus_matched_base_accuracy": float(
                np.mean(base_accuracy_advantages)
            ),
            "matched_base_minus_temporal_nll": float(np.mean(base_nll_advantages)),
        }
        effect["direction_consistent"] = bool(
            effect["temporal_minus_homogeneous_accuracy"] > 0.0
            or effect["homogeneous_minus_temporal_nll"] > 0.0
        )
        cell_effects.append(effect)
        all_accuracy_advantages.extend(accuracy_advantages)
        all_nll_advantages.extend(nll_advantages)
        all_uniform_drops.extend(uniform_drops)
        all_reverse_drops.extend(reverse_drops)
        all_base_accuracy_advantages.extend(base_accuracy_advantages)
        all_base_nll_advantages.extend(base_nll_advantages)

    horizons = sorted({int(cell["grid_horizon"]) for cell in cells})
    probabilities = sorted({float(cell["grid_event_probability"]) for cell in cells})
    mean_usage = {
        key: np.asarray(values, dtype=np.float64).mean(axis=0)
        for key, values in routing_usage.items()
    }
    short_probability_deltas = {
        str(horizon): float(
            mean_usage[(horizon, probabilities[-1])][0]
            - mean_usage[(horizon, probabilities[0])][0]
        )
        for horizon in horizons
    }
    long_horizon_deltas = {
        str(probability): float(
            mean_usage[(horizons[-1], probability)][-1]
            - mean_usage[(horizons[0], probability)][-1]
        )
        for probability in probabilities
    }

    effect_accuracy = float(np.mean(all_accuracy_advantages))
    effect_nll = float(np.mean(all_nll_advantages))
    uniform_drop = float(np.mean(all_uniform_drops))
    reverse_drop = float(np.mean(all_reverse_drops))
    consistent_cells = sum(bool(cell["direction_consistent"]) for cell in cell_effects)
    required_cells = math.ceil(len(cell_effects) * 2.0 / 3.0)
    short_semantic_cells = sum(delta >= 0.02 for delta in short_probability_deltas.values())
    long_semantic_cells = sum(delta >= 0.02 for delta in long_horizon_deltas.values())
    required_short = math.ceil(len(horizons) * 2.0 / 3.0)
    required_long = math.ceil(len(probabilities) * 2.0 / 3.0)

    gates = {
        "parameter_match": max(parameter_gaps, default=float("inf")) <= 0.05,
        "paired_effect": effect_nll >= 0.10 or effect_accuracy >= 0.05,
        "cell_consistency": consistent_cells >= required_cells,
        "router_intervention": max(uniform_drop, reverse_drop) >= 0.03,
        "routing_semantics": (
            short_semantic_cells >= required_short
            and long_semantic_cells >= required_long
        ),
    }
    if all(gates.values()):
        verdict = "GO"
    elif (
        gates["parameter_match"]
        and gates["paired_effect"]
        and gates["cell_consistency"]
        and gates["router_intervention"]
    ):
        verdict = "NARROW_ROUTER_EFFECT_WITHOUT_TIMESCALE_SEMANTICS"
    else:
        verdict = "NO_MECHANISM_SIGNAL"

    return {
        "frozen_thresholds": {
            "paired_nll_advantage": 0.10,
            "paired_accuracy_advantage": 0.05,
            "minimum_consistent_cell_fraction": 2.0 / 3.0,
            "router_accuracy_drop": 0.03,
            "usage_semantic_delta": 0.02,
            "parameter_gap": 0.05,
        },
        "grand_means": {
            "temporal_minus_homogeneous_accuracy": effect_accuracy,
            "homogeneous_minus_temporal_nll": effect_nll,
            "temporal_uniform_accuracy_drop": uniform_drop,
            "temporal_reverse_time_accuracy_drop": reverse_drop,
            "temporal_minus_matched_base_accuracy": float(
                np.mean(all_base_accuracy_advantages)
            ),
            "matched_base_minus_temporal_nll": float(
                np.mean(all_base_nll_advantages)
            ),
        },
        "cell_consistency": {
            "passing_cells": consistent_cells,
            "required_cells": required_cells,
            "total_cells": len(cell_effects),
        },
        "routing_semantics": {
            "short_usage_high_minus_low_probability": short_probability_deltas,
            "long_usage_long_minus_short_horizon": long_horizon_deltas,
            "short_passing": short_semantic_cells,
            "short_required": required_short,
            "long_passing": long_semantic_cells,
            "long_required": required_long,
        },
        "gates": gates,
        "cell_effects": cell_effects,
        "verdict": verdict,
    }


def run_phase_a_grid(
    base_task: TaskConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    horizons: Sequence[int],
    event_probabilities: Sequence[float],
    variants: Sequence[VariantName],
    seeds: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    required = {"base_param_matched", "temporal", "homogeneous"}
    if not required.issubset(set(variants)):
        raise ValueError(f"phase-A grid requires variants {sorted(required)}")
    if len(set(seeds)) < 3:
        raise ValueError("phase-A grid requires at least three distinct seeds")
    if len(set(horizons)) < 2 or len(set(event_probabilities)) < 2:
        raise ValueError("phase-A grid requires at least two horizons and event probabilities")

    cells = []
    for horizon in horizons:
        for probability in event_probabilities:
            task = TaskConfig(
                train_sequences=base_task.train_sequences,
                valid_sequences=base_task.valid_sequences,
                seq_len=base_task.seq_len,
                payload_values=base_task.payload_values,
                horizons=(int(horizon),),
                gap_min=base_task.gap_min,
                gap_max=base_task.gap_max,
                event_probability=float(probability),
            )
            cell = run_study(
                task,
                model_cfg,
                train_cfg,
                variants=variants,
                seeds=seeds,
                device=device,
            )
            cell["grid_horizon"] = int(horizon)
            cell["grid_event_probability"] = float(probability)
            cells.append(cell)
            print(
                f"cell horizon={horizon} event_p={probability:.2f} "
                f"density={cell['dataset']['valid_event_density']:.4f} complete"
            )

    return {
        "experiment": "TBC-1 factorised temporal-basis Phase-A grid",
        "status": "FORMAL_PHASE_A_GRID",
        "device": device_label(device),
        "base_task": asdict(base_task),
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        "grid_horizons": [int(horizon) for horizon in horizons],
        "grid_event_probabilities": [float(value) for value in event_probabilities],
        "variants": list(variants),
        "seeds": [int(seed) for seed in seeds],
        "cells": cells,
        "analysis": analyse_phase_a_grid(cells),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", nargs="+", default=[
        "base_same_width",
        "base_param_matched",
        "temporal",
        "homogeneous",
        "uniform",
    ])
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--valid-sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--horizons", nargs="+", type=int, default=[2, 8, 24])
    parser.add_argument("--gap-min", type=int, default=1)
    parser.add_argument("--gap-max", type=int, default=4)
    parser.add_argument("--event-probability", type=float)
    parser.add_argument("--phase-a-grid", action="store_true")
    parser.add_argument("--grid-horizons", nargs="+", type=int, default=[2, 8, 24])
    parser.add_argument(
        "--grid-event-probabilities",
        nargs="+",
        type=float,
        default=[0.05, 0.15, 0.30],
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    allowed = {
        "base_same_width",
        "base_param_matched",
        "temporal",
        "homogeneous",
        "uniform",
    }
    unknown = set(args.variants) - allowed
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")
    return args


def main() -> None:
    args = parse_args()
    task = TaskConfig(
        train_sequences=args.train_sequences,
        valid_sequences=args.valid_sequences,
        seq_len=args.seq_len,
        horizons=tuple(args.horizons),
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        event_probability=args.event_probability,
    )
    model_cfg = ModelConfig(
        d_model=args.d_model,
        state_dim=args.state_dim,
        n_experts=args.n_experts,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    device = choose_device(args.device)
    if args.phase_a_grid:
        payload = run_phase_a_grid(
            task,
            model_cfg,
            train_cfg,
            horizons=args.grid_horizons,
            event_probabilities=args.grid_event_probabilities,
            variants=args.variants,
            seeds=args.seeds,
            device=device,
        )
        output = args.out or (
            REPO_ROOT / "results/e3_scan/e3_tbc1_temporal_basis_phase_a_grid.json"
        )
    else:
        payload = run_study(
            task,
            model_cfg,
            train_cfg,
            variants=args.variants,
            seeds=args.seeds,
            device=device,
        )
        output = args.out or (
            REPO_ROOT / "results/e3_scan/e3_tbc0_temporal_basis_crossover.smoke.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={payload['device']} output={output}")
    if args.phase_a_grid:
        print(json.dumps(payload["analysis"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload["parameter_match"], ensure_ascii=False))
        for variant, row in payload["summary"].items():
            print(
                f"{variant:20s} params={int(row['params_total']):7d} "
                f"nll={row['normal_query_nll_mean']:.4f} "
                f"acc={row['normal_query_accuracy_mean']:.3f} "
                f"tok/s={row['train_tokens_per_s_mean']:.0f}"
            )


if __name__ == "__main__":
    main()
