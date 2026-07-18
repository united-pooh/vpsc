"""Deterministic CPU language-model training, evaluation, and streaming timing.

The functions in this module operate on WikiText ``LMTokenBatch`` objects or
any equivalent object exposing ``input_ids``, ``target_ids``, and
``reset_state``.  Consecutive chunks retain explicit model state unless the
batch requests a reset, and every returned state is detached to implement
stateful truncated BPTT without retaining an earlier chunk's graph.

Streaming timing accepts an already constructed model and already tokenised
IDs.  Model construction, vocabulary work, device movement, warmup, token
selection, percentile calculation, and state-size accounting all happen
outside the measured update intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
import sys
import time
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cores import state_nbytes


Tensor = torch.Tensor
DeviceLike = Union[str, torch.device]


class LMBatchLike(Protocol):
    """Structural protocol implemented by ``LMTokenBatch`` and ``LMTokenChunk``."""

    input_ids: object
    target_ids: object
    reset_state: bool


@dataclass
class LanguageModelOutput:
    """Logits and the explicit temporal state after a chunk."""

    logits: Tensor
    state: Any


class StatefulLanguageModelProtocol(Protocol):
    """Protocol consumed by the generic train/evaluate/benchmark functions."""

    training: bool

    def train(self, mode: bool = True) -> Any: ...

    def eval(self) -> Any: ...

    def to(self, device: torch.device) -> Any: ...

    def parameters(self) -> Iterable[nn.Parameter]: ...

    def __call__(
        self,
        input_ids: Tensor,
        state: Any = None,
        *,
        detach_state: bool = False,
    ) -> LanguageModelOutput: ...

    def step(
        self,
        input_id: Tensor,
        state: Any = None,
        *,
        detach_state: bool = False,
    ) -> LanguageModelOutput: ...


class _JSONDataclassMixin:
    def to_dict(self) -> Dict[str, Any]:
        """Return a recursively JSON-compatible dictionary."""

        return asdict(self)

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialise with strict finite-number checking and stable key order."""

        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )


@dataclass(frozen=True)
class TrainingConfig(_JSONDataclassMixin):
    seed: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    max_steps: Optional[int] = 100
    token_budget: Optional[int] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("token_budget must be positive or None")
        _require_cpu(self.device)


@dataclass(frozen=True)
class TrainingMetrics(_JSONDataclassMixin):
    seed: int
    steps: int
    target_count: int
    nll: float
    ppl: float
    mean_gradient_norm: float
    elapsed_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class EvaluationMetrics(_JSONDataclassMixin):
    batches: int
    target_count: int
    nll: float
    ppl: float


@dataclass(frozen=True)
class StreamingBenchmarkMetrics(_JSONDataclassMixin):
    seed: int
    warmup_steps: int
    measured_steps: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    tokens_per_second: float
    state_nbytes: int


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic CPU experiments.

    Call this before model construction to reproduce initial parameters.  The
    training and benchmark entry points call it again to reproduce stochastic
    layers and any token-stream sampling performed inside those functions.
    """

    if seed < 0:
        raise ValueError("seed cannot be negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_cpu(device: DeviceLike) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise ValueError(f"this reproducible harness is CPU-only, got {resolved}")
    return resolved


def _batch_tensors(batch: LMBatchLike, device: torch.device) -> Tuple[Tensor, Tensor]:
    inputs = torch.as_tensor(batch.input_ids, dtype=torch.long, device=device)
    targets = torch.as_tensor(batch.target_ids, dtype=torch.long, device=device)
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("LM input_ids and target_ids must be rank one or two")
    if inputs.shape != targets.shape:
        raise ValueError(
            f"input/target shapes must match, got {tuple(inputs.shape)} and "
            f"{tuple(targets.shape)}"
        )
    if inputs.numel() == 0:
        raise ValueError("LM batches cannot be empty")
    return inputs, targets


def _truncate_to_token_budget(
    inputs: Tensor,
    targets: Tensor,
    remaining_tokens: Optional[int],
) -> Optional[Tuple[Tensor, Tensor]]:
    if remaining_tokens is None or targets.numel() <= remaining_tokens:
        return inputs, targets
    # Keep complete time steps across every persistent batch lane.  Selecting a
    # partial lane would change state batch shape and break continuation.
    usable_time = remaining_tokens // targets.shape[0]
    if usable_time <= 0:
        return None
    return inputs[:, :usable_time], targets[:, :usable_time]


def _extract_output(output: Any) -> Tuple[Tensor, Any]:
    try:
        logits = output.logits
        state = output.state
    except AttributeError as error:
        raise TypeError("language model output must expose .logits and .state") from error
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("language model logits must have shape [batch, time, vocab]")
    return logits, state


def _summed_nll(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logit prefix {tuple(logits.shape[:2])} does not match targets "
            f"{tuple(targets.shape)}"
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="sum",
    )


def _finite_perplexity(nll: float) -> float:
    # Saturating at the largest finite float keeps strict JSON serialisation
    # valid while retaining the conventional exponential definition in the
    # entire numerically representable range.
    return math.exp(min(nll, math.log(sys.float_info.max)))


def train_language_model(
    model: StatefulLanguageModelProtocol,
    batches: Iterable[LMBatchLike],
    config: TrainingConfig = TrainingConfig(),
) -> TrainingMetrics:
    """Train one deterministic pass with AdamW and stateful truncated BPTT.

    The token budget is a hard upper bound.  If necessary, the final chunk is
    truncated along time while retaining all batch lanes; therefore fewer than
    ``batch_size`` budget tokens can remain unused.

    Training time starts after seed setup, model placement, and optimizer
    construction.  It includes batch iteration/conversion, reset handling,
    forward, backward, clipping, and the AdamW update.  Model construction,
    vocabulary/data preparation, device placement, and optimizer construction
    are therefore excluded from ``elapsed_seconds`` and ``tokens_per_second``.
    """

    device = _require_cpu(config.device)
    seed_everything(config.seed)
    model.to(device)
    model.train(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("training requires at least one trainable parameter")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    state: Any = None
    steps = 0
    target_count = 0
    total_nll = 0.0
    total_gradient_norm = 0.0

    training_start_ns = time.perf_counter_ns()
    for batch in batches:
        if config.max_steps is not None and steps >= config.max_steps:
            break
        remaining = None
        if config.token_budget is not None:
            remaining = config.token_budget - target_count
            if remaining <= 0:
                break

        inputs, targets = _batch_tensors(batch, device)
        truncated = _truncate_to_token_budget(inputs, targets, remaining)
        if truncated is None:
            break
        inputs, targets = truncated
        if bool(batch.reset_state):
            state = None

        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, state, detach_state=True)
        logits, state = _extract_output(output)
        loss_sum = _summed_nll(logits, targets)
        count = int(targets.numel())
        loss = loss_sum / count
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=config.gradient_clip_norm,
        )
        optimizer.step()

        steps += 1
        target_count += count
        total_nll += float(loss_sum.detach().item())
        total_gradient_norm += float(gradient_norm.detach().item())
    elapsed_seconds = (time.perf_counter_ns() - training_start_ns) / 1_000_000_000.0

    if target_count == 0:
        raise ValueError("training consumed no target tokens")
    nll = total_nll / target_count
    return TrainingMetrics(
        seed=config.seed,
        steps=steps,
        target_count=target_count,
        nll=nll,
        ppl=_finite_perplexity(nll),
        mean_gradient_norm=total_gradient_norm / steps,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=target_count / elapsed_seconds,
    )


def evaluate_language_model(
    model: StatefulLanguageModelProtocol,
    batches: Iterable[LMBatchLike],
    *,
    max_steps: Optional[int] = None,
    token_budget: Optional[int] = None,
    device: DeviceLike = "cpu",
) -> EvaluationMetrics:
    """Evaluate token-weighted NLL/PPL while retaining explicit chunk state."""

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive or None")
    if token_budget is not None and token_budget <= 0:
        raise ValueError("token_budget must be positive or None")
    resolved_device = _require_cpu(device)
    model.to(resolved_device)
    was_training = model.training
    model.eval()

    state: Any = None
    batch_count = 0
    target_count = 0
    total_nll = 0.0
    try:
        with torch.no_grad():
            for batch in batches:
                if max_steps is not None and batch_count >= max_steps:
                    break
                remaining = None
                if token_budget is not None:
                    remaining = token_budget - target_count
                    if remaining <= 0:
                        break

                inputs, targets = _batch_tensors(batch, resolved_device)
                truncated = _truncate_to_token_budget(inputs, targets, remaining)
                if truncated is None:
                    break
                inputs, targets = truncated
                if bool(batch.reset_state):
                    state = None
                output = model(inputs, state, detach_state=True)
                logits, state = _extract_output(output)
                loss_sum = _summed_nll(logits, targets)
                count = int(targets.numel())
                batch_count += 1
                target_count += count
                total_nll += float(loss_sum.item())
    finally:
        model.train(was_training)

    if target_count == 0:
        raise ValueError("evaluation consumed no target tokens")
    nll = total_nll / target_count
    return EvaluationMetrics(
        batches=batch_count,
        target_count=target_count,
        nll=nll,
        ppl=_finite_perplexity(nll),
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def benchmark_streaming_step(
    model: StatefulLanguageModelProtocol,
    token_ids: Union[Tensor, Sequence[int]],
    *,
    warmup_steps: int = 20,
    measured_steps: int = 100,
    seed: int = 0,
    device: DeviceLike = "cpu",
) -> StreamingBenchmarkMetrics:
    """Measure batch-one token-update latency for an already built model.

    Each measured interval contains only ``model.step(token, state)``.  Moving
    the model, converting IDs, selecting the next token, warmup, metrics, and
    state-size calculation are intentionally outside those intervals.
    """

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if measured_steps <= 0:
        raise ValueError("measured_steps must be positive")
    resolved_device = _require_cpu(device)
    seed_everything(seed)
    model.to(resolved_device)
    was_training = model.training
    model.eval()

    tokens = torch.as_tensor(token_ids, dtype=torch.long, device=resolved_device).reshape(-1)
    if tokens.numel() == 0:
        raise ValueError("token_ids cannot be empty")

    state: Any = None
    durations_ms = []
    try:
        with torch.inference_mode():
            for index in range(warmup_steps):
                # Selection is outside the measured update and creates [batch=1].
                token = tokens[index % tokens.numel()].view(1)
                output = model.step(token, state, detach_state=True)
                _, state = _extract_output(output)

            for index in range(measured_steps):
                token_index = (warmup_steps + index) % tokens.numel()
                token = tokens[token_index].view(1)
                start_ns = time.perf_counter_ns()
                output = model.step(token, state, detach_state=True)
                end_ns = time.perf_counter_ns()
                _, state = _extract_output(output)
                durations_ms.append((end_ns - start_ns) / 1_000_000.0)
    finally:
        model.train(was_training)

    ordered = sorted(durations_ms)
    total_seconds = sum(durations_ms) / 1_000.0
    mean_ms = sum(durations_ms) / len(durations_ms)
    return StreamingBenchmarkMetrics(
        seed=seed,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        latency_mean_ms=mean_ms,
        latency_p50_ms=_percentile(ordered, 50.0),
        latency_p95_ms=_percentile(ordered, 95.0),
        latency_p99_ms=_percentile(ordered, 99.0),
        tokens_per_second=measured_steps / total_seconds,
        state_nbytes=state_nbytes(state),
    )


__all__ = [
    "EvaluationMetrics",
    "LanguageModelOutput",
    "LMBatchLike",
    "StatefulLanguageModelProtocol",
    "StreamingBenchmarkMetrics",
    "TrainingConfig",
    "TrainingMetrics",
    "benchmark_streaming_step",
    "evaluate_language_model",
    "seed_everything",
    "train_language_model",
]
