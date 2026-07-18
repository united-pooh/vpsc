"""Real TextWorld sparse outcome-token LM for AT1, LSTM, and Transformer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _percentile,
    _sample_summary,
    _sync,
)
from experiments.e2_textworld_lm import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    _corpus_provenance,
    _manifest_provenance,
)
from experiments.e3_at0_gated_trace import STATE_DIM  # noqa: E402
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3GatedTraceScanCore,
    E3LayerState,
    E3ScanState,
    StatefulLSTMCore,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.event_corpus import (  # noqa: E402
    TextWorldEventCorpus,
    load_event_corpus,
)
from vpsc.world_model.lm import CausalLanguageModel  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


D_MODEL = 32
MAX_QUERIES = 16
SEQUENCE_LENGTH = 512
SELECTED_CHANNELS = frozenset(
    {
        "observation",
        "reward",
        "done",
        "won",
        "admissible_actions",
        "counterfactual",
    }
)
FORMAT_TOKENS = frozenset(
    {
        '"',
        "'",
        ":",
        ",",
        ".",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "true",
        "false",
        "null",
    }
)
MODEL_NAMES = ("snn_bptt", "snn_at1", "lstm", "transformer")


@dataclass(frozen=True)
class SparseEventChunk:
    split: str
    episode_index: int
    offset: int
    reset_state: bool
    input_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    sparse_query_indices: Tuple[int, ...]
    sparse_query_channels: Tuple[str, ...]
    dense_query_indices: Tuple[int, ...]
    dense_query_channels: Tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.input_ids)


def _line_payload_channels(
    token_ids: Sequence[int], vocabulary: Vocabulary
) -> Dict[int, str]:
    """Map global target-token positions to their physical-line channel."""

    tokens = vocabulary.decode(token_ids)
    positions: Dict[int, str] = {}
    line_start = 0
    while line_start < len(tokens):
        try:
            line_end = tokens.index("<eos>", line_start)
        except ValueError as error:
            raise ValueError("TextWorld episode line lacks <eos>") from error
        marker = None
        for index in range(line_start, max(line_start, line_end - 4)):
            if (
                tokens[index : index + 5]
                == ("<", "|", tokens[index + 2], "|", ">")
            ):
                marker = index
                break
        if marker is None:
            raise ValueError(
                f"cannot locate event channel marker before token {line_end}"
            )
        channel = tokens[marker + 2]
        if channel in SELECTED_CHANNELS:
            for position in range(marker + 5, line_end):
                positions[position] = channel
        line_start = line_end + 1
    return positions


def _even_subset(count: int, maximum: int) -> Tuple[int, ...]:
    if count <= maximum:
        return tuple(range(count))
    selected = tuple(
        round(index * (count - 1) / (maximum - 1))
        for index in range(maximum)
    )
    if len(set(selected)) != maximum:  # pragma: no cover
        raise AssertionError(f"even subset produced duplicate indices: {selected}")
    return selected


def build_sparse_chunks(
    corpus: TextWorldEventCorpus,
    split: str,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    max_queries: int = MAX_QUERIES,
) -> Tuple[SparseEventChunk, ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if sequence_length <= 0 or max_queries <= 0:
        raise ValueError("sequence_length and max_queries must be positive")
    chunks = []
    for episode_index, episode in enumerate(corpus.iter_episode_token_ids(split)):
        channel_positions = _line_payload_channels(episode, corpus.vocabulary)
        final_input_offset = len(episode) - 1
        for offset in range(0, final_input_offset, sequence_length):
            length = min(sequence_length, final_input_offset - offset)
            inputs = tuple(episode[offset : offset + length])
            targets = tuple(episode[offset + 1 : offset + length + 1])
            candidates = []
            for target_position in range(offset + 1, offset + length + 1):
                channel = channel_positions.get(target_position)
                if channel is not None:
                    candidates.append((target_position - offset - 1, channel))
            selected = _even_subset(len(candidates), max_queries)
            sparse = tuple(candidates[index] for index in selected)
            chunks.append(
                SparseEventChunk(
                    split=split,
                    episode_index=episode_index,
                    offset=offset,
                    reset_state=offset == 0,
                    input_ids=inputs,
                    target_ids=targets,
                    sparse_query_indices=tuple(item[0] for item in sparse),
                    sparse_query_channels=tuple(item[1] for item in sparse),
                    dense_query_indices=tuple(item[0] for item in candidates),
                    dense_query_channels=tuple(item[1] for item in candidates),
                )
            )
    return tuple(chunks)


def _is_format_token(token: str) -> bool:
    if token in FORMAT_TOKENS:
        return True
    normalised = token.replace("-", "", 1).replace(".", "", 1)
    return bool(normalised) and normalised.isdigit()


def audit_sparse_chunks(
    chunks_by_split: Mapping[str, Sequence[SparseEventChunk]],
    vocabulary: Vocabulary,
) -> Dict[str, Any]:
    splits = {}
    format_count = 0
    selected_count = 0
    token_counts: Counter[str] = Counter()
    all_valid = True
    for split, chunks in chunks_by_split.items():
        channel_counts: Counter[str] = Counter()
        sparse_queries = 0
        dense_queries = 0
        nonempty_chunks = 0
        maximum_k = 0
        for chunk in chunks:
            indices = chunk.sparse_query_indices
            valid = (
                len(indices) == len(chunk.sparse_query_channels)
                and len(indices) <= MAX_QUERIES
                and all(0 <= index < chunk.length for index in indices)
                and all(
                    right > left for left, right in zip(indices, indices[1:])
                )
                and set(chunk.sparse_query_channels) <= SELECTED_CHANNELS
                and len(chunk.dense_query_indices)
                == len(chunk.dense_query_channels)
            )
            all_valid = all_valid and valid
            if indices:
                nonempty_chunks += 1
            maximum_k = max(maximum_k, len(indices))
            sparse_queries += len(indices)
            dense_queries += len(chunk.dense_query_indices)
            channel_counts.update(chunk.sparse_query_channels)
            for index in indices:
                token = vocabulary.tokens[chunk.target_ids[index]]
                token_counts[token] += 1
                selected_count += 1
                format_count += int(_is_format_token(token))
        splits[split] = {
            "chunks": len(chunks),
            "nonempty_query_chunks": nonempty_chunks,
            "sparse_query_count": sparse_queries,
            "dense_outcome_payload_token_count": dense_queries,
            "query_density": sparse_queries
            / sum(chunk.length for chunk in chunks),
            "maximum_queries_per_chunk": maximum_k,
            "channel_counts": dict(sorted(channel_counts.items())),
        }
    format_ratio = format_count / selected_count
    return {
        "selected_channels": sorted(SELECTED_CHANNELS),
        "format_token_definition": {
            "fixed_tokens": sorted(FORMAT_TOKENS),
            "numeric_tokens_are_format": True,
        },
        "splits": splits,
        "selected_token_count_all_splits": selected_count,
        "selected_unique_tokens_all_splits": len(token_counts),
        "most_common_selected_tokens": token_counts.most_common(20),
        "format_token_count": format_count,
        "format_token_ratio": format_ratio,
        "queries_valid": all_valid,
        "passed": all_valid
        and all(record["sparse_query_count"] > 0 for record in splits.values())
        and format_ratio < 0.70,
    }


def _common_model(
    core: nn.Module,
    *,
    vocabulary: Vocabulary,
) -> CausalLanguageModel[Any]:
    return CausalLanguageModel(
        vocab_size=len(vocabulary),
        core=core,  # type: ignore[arg-type]
        dropout=0.0,
        padding_idx=vocabulary.pad_id,
        tie_weights=True,
        head_bias=True,
    )


def _shared_wrapper_tensors(
    vocabulary: Vocabulary, seed: int
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    embedding = torch.randn(
        len(vocabulary), D_MODEL, generator=generator
    ) * (D_MODEL**-0.5)
    embedding[vocabulary.pad_id].zero_()
    return {
        "embedding": embedding,
        "norm_weight": torch.ones(D_MODEL),
        "norm_bias": torch.zeros(D_MODEL),
        "head_bias": torch.zeros(len(vocabulary)),
    }


def _copy_wrapper(
    model: CausalLanguageModel[Any], shared: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        model.embedding.weight.copy_(shared["embedding"])
        model.output_norm.weight.copy_(shared["norm_weight"])
        model.output_norm.bias.copy_(shared["norm_bias"])
        model.lm_head.bias.copy_(shared["head_bias"])


def build_models(
    seed: int,
    vocabulary: Vocabulary,
    *,
    device: torch.device,
) -> Dict[str, CausalLanguageModel[Any]]:
    shared = _shared_wrapper_tensors(vocabulary, seed + 10_000)
    torch.manual_seed(seed)
    snn_bptt = _common_model(
        E3GatedTraceScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM),
        vocabulary=vocabulary,
    )
    snn_at1 = _common_model(
        E3GatedTraceScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM),
        vocabulary=vocabulary,
    )
    snn_at1.load_state_dict(snn_bptt.state_dict())
    torch.manual_seed(seed + 1)
    lstm = _common_model(
        StatefulLSTMCore(D_MODEL, D_MODEL), vocabulary=vocabulary
    )
    torch.manual_seed(seed + 2)
    transformer = _common_model(
        CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=SEQUENCE_LENGTH,
        ),
        vocabulary=vocabulary,
    )
    models = {
        "snn_bptt": snn_bptt,
        "snn_at1": snn_at1,
        "lstm": lstm,
        "transformer": transformer,
    }
    for model in models.values():
        _copy_wrapper(model, shared)
        model.to(device)
    return models


def _sparse_forward(
    model: CausalLanguageModel[Any],
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
    state: Any,
    *,
    use_eligibility: bool,
    detach_state: bool,
) -> Tuple[torch.Tensor, Any]:
    embedded = model.input_dropout(model.embedding(input_ids))
    if use_eligibility:
        if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
            raise TypeError("eligibility requires gated trace core")
        core_output = model.core.forward_multi_query_eligibility(
            embedded,
            query_indices,
            state,
            detach_state=detach_state,
        )
        sequence = core_output.sequence
    else:
        core_output = model.core(
            embedded,
            state,
            detach_state=detach_state,
        )
        sequence = core_output.sequence.index_select(1, query_indices)
    hidden = model.output_norm(model.output_dropout(sequence))
    return model.lm_head(hidden), _project_chunk_state(model, core_output.state)


def _project_chunk_state(model: CausalLanguageModel[Any], state: Any) -> Any:
    """Enforce the gated trace's mathematical bounds at detached chunk edges."""

    if not isinstance(model.core, E3GatedTraceScanCore):
        return state
    if not isinstance(state, E3ScanState) or len(state.layers) != 1:  # pragma: no cover
        raise TypeError("gated trace model returned an invalid chunk state")
    layer = state.layers[0]
    return E3ScanState(
        layers=(
            E3LayerState(
                excitatory=layer.excitatory.clamp(0.0, 1.0),
                inhibitory=layer.inhibitory.clamp(0.0, 1.0),
            ),
        )
    )


def _advance_without_query(
    model: CausalLanguageModel[Any], input_ids: torch.Tensor, state: Any
) -> Any:
    with torch.no_grad():
        embedded = model.input_dropout(model.embedding(input_ids))
        result = model.core(embedded, state, detach_state=True)
        return _project_chunk_state(model, result.state)


def _chunk_tensors(
    chunk: SparseEventChunk,
    *,
    dense: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...]]:
    indices = chunk.dense_query_indices if dense else chunk.sparse_query_indices
    channels = chunk.dense_query_channels if dense else chunk.sparse_query_channels
    input_ids = torch.tensor(
        chunk.input_ids, dtype=torch.long, device=device
    ).unsqueeze(0)
    query_indices = torch.tensor(indices, dtype=torch.long, device=device)
    targets = torch.tensor(
        tuple(chunk.target_ids[index] for index in indices),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    return input_ids, query_indices, targets, channels


def evaluate_model(
    model: CausalLanguageModel[Any],
    chunks: Sequence[SparseEventChunk],
    *,
    dense: bool,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    state = None
    total_nll = 0.0
    count = 0
    correct = 0
    channel_nll: Dict[str, float] = defaultdict(float)
    channel_count: Counter[str] = Counter()
    channel_correct: Counter[str] = Counter()
    with torch.inference_mode():
        for chunk in chunks:
            if chunk.reset_state:
                state = None
            indices = (
                chunk.dense_query_indices if dense else chunk.sparse_query_indices
            )
            input_ids = torch.tensor(
                chunk.input_ids, dtype=torch.long, device=device
            ).unsqueeze(0)
            if not indices:
                result = model(input_ids, state, detach_state=True)
                state = _project_chunk_state(model, result.state)
                continue
            query_indices = torch.tensor(indices, dtype=torch.long, device=device)
            targets = torch.tensor(
                tuple(chunk.target_ids[index] for index in indices),
                dtype=torch.long,
                device=device,
            )
            output = model(input_ids, state, detach_state=True)
            state = _project_chunk_state(model, output.state)
            logits = output.logits.index_select(1, query_indices).squeeze(0)
            losses = F.cross_entropy(logits, targets, reduction="none")
            predictions = logits.argmax(dim=-1)
            channels = (
                chunk.dense_query_channels
                if dense
                else chunk.sparse_query_channels
            )
            total_nll += float(losses.sum().item())
            count += targets.numel()
            correct += int((predictions == targets).sum().item())
            for channel, loss, is_correct in zip(
                channels, losses.tolist(), (predictions == targets).tolist()
            ):
                channel_nll[channel] += float(loss)
                channel_count[channel] += 1
                channel_correct[channel] += int(is_correct)
    if count == 0:
        raise ValueError("evaluation consumed no TextWorld query targets")
    nll = total_nll / count
    return {
        "objective": "dense_outcome_payload" if dense else "sparse_k_query",
        "nll": nll,
        "ppl": math.exp(min(nll, 80.0)),
        "accuracy": correct / count,
        "target_count": count,
        "channels": {
            channel: {
                "nll": channel_nll[channel] / channel_count[channel],
                "accuracy": channel_correct[channel] / channel_count[channel],
                "target_count": channel_count[channel],
            }
            for channel in sorted(channel_count)
        },
    }


def train_model(
    name: str,
    model: CausalLanguageModel[Any],
    chunks: Sequence[SparseEventChunk],
    *,
    epochs: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.train(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.01)
    timings = []
    losses = []
    input_tokens = 0
    query_tokens = 0
    update = 0
    epoch_records = []
    started_all = time.perf_counter_ns()
    for epoch in range(epochs):
        state = None
        epoch_nll = 0.0
        epoch_queries = 0
        for chunk in chunks:
            if chunk.reset_state:
                state = None
            input_ids, query_indices, targets, _channels = _chunk_tensors(
                chunk, dense=False, device=device
            )
            input_tokens += chunk.length
            if query_indices.numel() == 0:
                state = _advance_without_query(model, input_ids, state)
                continue
            _sync(device)
            started = time.perf_counter_ns()
            optimizer.zero_grad(set_to_none=True)
            logits, state = _sparse_forward(
                model,
                input_ids,
                query_indices,
                state,
                use_eligibility=name == "snn_at1",
                detach_state=True,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), targets.reshape(-1)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite TW0 loss for {name} at update {update + 1}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            _sync(device)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            timings.append(elapsed_ms)
            losses.append(float(loss.detach().item()))
            query_count = targets.numel()
            query_tokens += query_count
            epoch_queries += query_count
            epoch_nll += float(loss.detach().item()) * query_count
            update += 1
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "query_nll": epoch_nll / epoch_queries,
                "query_count": epoch_queries,
                "last_gradient_norm": float(gradient_norm),
            }
        )
    elapsed_seconds = (time.perf_counter_ns() - started_all) / 1e9
    warmup_updates = len(timings) // 5
    steady = timings[warmup_updates:]
    timing = _sample_summary(steady, 1)
    timing["warmup_updates_excluded"] = warmup_updates
    timing["input_tokens_per_second_total"] = input_tokens / elapsed_seconds
    timing["query_tokens_per_second_total"] = query_tokens / elapsed_seconds
    return {
        "epochs": epochs,
        "updates": update,
        "input_tokens": input_tokens,
        "query_tokens": query_tokens,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_epoch": epoch_records[-1]["query_nll"],
        "timing": timing,
        "elapsed_seconds": elapsed_seconds,
        "epoch_records": epoch_records,
    }


def evaluate_mechanism_ablation(
    model: CausalLanguageModel[Any],
    chunks: Sequence[SparseEventChunk],
    *,
    mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("mechanism ablation requires gated trace core")
    if mode not in ("spike_only", "trace_only"):
        raise ValueError(f"unknown ablation mode: {mode}")
    model.eval()
    state = None
    total_nll = 0.0
    count = 0
    correct = 0
    with torch.inference_mode():
        for chunk in chunks:
            if chunk.reset_state:
                state = None
            input_ids, query_indices, targets, _channels = _chunk_tensors(
                chunk, dense=False, device=device
            )
            embedded = model.input_dropout(model.embedding(input_ids))
            result, trace = model.core.forward_dynamics(
                embedded, state, detach_state=True
            )
            state = _project_chunk_state(model, result.state)
            if query_indices.numel() == 0:
                continue
            zeros_spike_e = torch.zeros_like(trace.excitatory_spikes)
            zeros_spike_i = torch.zeros_like(trace.inhibitory_spikes)
            if mode == "spike_only":
                raw = torch.cat(
                    (
                        trace.excitatory_spikes,
                        -trace.inhibitory_spikes,
                        torch.zeros_like(trace.excitatory_traces),
                        torch.zeros_like(trace.inhibitory_traces),
                    ),
                    dim=-1,
                )
            else:
                raw = torch.cat(
                    (
                        zeros_spike_e,
                        -zeros_spike_i,
                        trace.excitatory_traces,
                        -trace.inhibitory_traces,
                    ),
                    dim=-1,
                )
            sequence = model.core.output_projection(model.core.output_norm(raw))
            hidden = model.output_norm(
                model.output_dropout(sequence.index_select(1, query_indices))
            )
            logits = model.lm_head(hidden)
            losses = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                reduction="none",
            )
            predictions = logits.argmax(dim=-1)
            total_nll += float(losses.sum().item())
            count += targets.numel()
            correct += int((predictions == targets).sum().item())
    nll = total_nll / count
    return {
        "mode": mode,
        "nll": nll,
        "ppl": math.exp(min(nll, 80.0)),
        "accuracy": correct / count,
        "target_count": count,
    }


class _GenericLMStreamRunner:
    def __init__(
        self, model: CausalLanguageModel[Any], tokens: torch.Tensor
    ) -> None:
        self.model = model.eval()
        self.tokens = tokens
        self.state = None
        self.index = 0

    def __call__(self) -> torch.Tensor:
        result = self.model.step(
            self.tokens[self.index].view(1),
            self.state,
            detach_state=True,
        )
        self.state = result.state
        self.index += 1
        return result.logits


class _CachedSNNLMStreamRunner:
    def __init__(
        self, model: CausalLanguageModel[Any], tokens: torch.Tensor
    ) -> None:
        if not isinstance(model.core, E3GatedTraceScanCore):
            raise TypeError("cached SNN runner requires gated trace core")
        self.model = model.eval()
        self.core = model.core
        self.tokens = tokens
        with torch.inference_mode():
            state = self.core.initial_state(1, device=tokens.device)
            self.decay_e, self.decay_i = self.core.decays()
        self.excitatory = state.layers[0].excitatory
        self.inhibitory = state.layers[0].inhibitory
        self.index = 0

    def __call__(self) -> torch.Tensor:
        token = self.tokens[self.index].view(1)
        embedded = self.model.input_dropout(self.model.embedding(token))
        output = self.core.forward_step_tensors_cached_decay(
            embedded,
            self.excitatory,
            self.inhibitory,
            self.decay_e,
            self.decay_i,
        )
        self.excitatory, self.inhibitory = output[1], output[2]
        hidden = self.model.output_norm(self.model.output_dropout(output[0]))
        self.index += 1
        return self.model.lm_head(hidden)


def _runner_state_bytes(runner: Any) -> int:
    if isinstance(runner, _CachedSNNLMStreamRunner):
        return sum(
            value.numel() * value.element_size()
            for value in (runner.excitatory, runner.inhibitory)
        )
    return state_nbytes(runner.state)


def benchmark_streaming(
    models: Mapping[str, CausalLanguageModel[Any]],
    token_ids: Sequence[int],
    *,
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    total = warmup_steps + measured_steps
    tokens = torch.tensor(token_ids[:total], dtype=torch.long, device=device)
    runners: Dict[str, Any] = {
        "snn_bptt_cached": _CachedSNNLMStreamRunner(models["snn_bptt"], tokens),
        "snn_at1_cached": _CachedSNNLMStreamRunner(models["snn_at1"], tokens),
        "lstm": _GenericLMStreamRunner(models["lstm"], tokens),
        "transformer": _GenericLMStreamRunner(models["transformer"], tokens),
    }
    samples: Dict[str, list[float]] = {name: [] for name in runners}
    with torch.inference_mode():
        for _ in range(warmup_steps):
            for runner in runners.values():
                runner()
        names = list(runners)
        generator = random.Random(seed)
        for _ in range(measured_steps):
            generator.shuffle(names)
            for name in names:
                _sync(device)
                started = time.perf_counter_ns()
                logits = runners[name]()
                logits.sum().item()
                _sync(device)
                samples[name].append((time.perf_counter_ns() - started) / 1e6)
    return {
        "models": {
            name: {
                **_sample_summary(sample, 1),
                "p99_ms": _percentile(sample, 0.99),
                "state_bytes_after_stream": _runner_state_bytes(runners[name]),
            }
            for name, sample in samples.items()
        }
    }


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    return math.fsum(values) / len(values)


def _decision(
    *,
    data_audit: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if data_audit["passed"] else "FAIL",
            "task_gate": "NOT_RUN",
            "quality_gate": "NOT_RUN",
            "speed_gate": "SMOKE",
            "stream_gate": "SMOKE",
            "overall": "SMOKE",
            "run_counterfactual_generation_next": False,
        }
    task_pass = all(
        result["post"][name]["test_sparse"]["nll"]
        <= result["pre"][name]["test_sparse"]["nll"] - 0.10
        for result in seed_results
        for name in ("lstm", "transformer")
    )
    snn_improvement = all(
        result["post"][name]["test_sparse"]["nll"]
        <= result["pre"][name]["test_sparse"]["nll"] - 0.10
        for result in seed_results
        for name in ("snn_bptt", "snn_at1")
    )
    mean_test_nll = {
        name: _mean(
            result["post"][name]["test_sparse"]["nll"]
            for result in seed_results
        )
        for name in MODEL_NAMES
    }
    best_ann = min(mean_test_nll["lstm"], mean_test_nll["transformer"])
    functional_gap = abs(mean_test_nll["snn_at1"] - mean_test_nll["snn_bptt"])
    quality_pass = (
        task_pass
        and snn_improvement
        and mean_test_nll["snn_at1"] <= best_ann + 0.25
        and functional_gap <= 0.10
    )
    mean_p50 = {
        name: _mean(
            result["training"][name]["timing"]["p50_ms"]
            for result in seed_results
        )
        for name in MODEL_NAMES
    }
    speedup = mean_p50["snn_bptt"] / mean_p50["snn_at1"]
    speed_pass = speedup >= 1.25 and mean_p50["snn_at1"] <= mean_p50["lstm"]
    stream_pass = all(
        result["streaming"]["models"]["snn_at1_cached"]["p50_ms"]
        <= result["streaming"]["models"]["lstm"]["p50_ms"]
        and result["streaming"]["models"]["snn_at1_cached"]["p95_ms"]
        <= result["streaming"]["models"]["lstm"]["p95_ms"]
        for result in seed_results
    )
    gates = {
        "data_gate": bool(data_audit["passed"]),
        "task_gate": task_pass,
        "quality_gate": quality_pass,
        "speed_gate": speed_pass,
        "stream_gate": stream_pass,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": overall,
        "mean_test_sparse_nll": mean_test_nll,
        "at1_vs_bptt_mean_nll_gap": functional_gap,
        "mean_training_p50_ms": mean_p50,
        "at1_vs_bptt_training_speedup": speedup,
        "run_counterfactual_generation_next": overall == "PASS",
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = load_event_corpus(corpus_root)
    manifest_provenance = _manifest_provenance(corpus_root)
    corpus_provenance = _corpus_provenance(corpus)
    chunks_by_split = {
        split: build_sparse_chunks(
            corpus,
            split,
            sequence_length=args.sequence_length,
            max_queries=args.max_queries,
        )
        for split in SPLITS
    }
    data_audit = audit_sparse_chunks(chunks_by_split, corpus.vocabulary)
    test_episode = next(corpus.iter_episode_token_ids("test"))
    stream_total = args.stream_warmup + args.stream_steps
    if len(test_episode) < stream_total:
        raise ValueError("test episode is too short for streaming benchmark")

    seed_results = []
    for seed in args.seeds:
        models = build_models(
            9_200_000 + 100 * seed,
            corpus.vocabulary,
            device=device,
        )
        parameter_counts = {
            name: {
                "total": count_parameters(model),
                "core": count_parameters(model.core),
            }
            for name, model in models.items()
        }
        totals = [record["total"] for record in parameter_counts.values()]
        parameter_spread = (max(totals) - min(totals)) / _mean(totals)
        if parameter_spread > 0.02:
            raise AssertionError(f"TW0 parameter spread failed: {parameter_counts}")
        pre = {
            name: {
                "valid_sparse": evaluate_model(
                    model,
                    chunks_by_split["valid"],
                    dense=False,
                    device=device,
                ),
                "test_sparse": evaluate_model(
                    model,
                    chunks_by_split["test"],
                    dense=False,
                    device=device,
                ),
            }
            for name, model in models.items()
        }
        training = {
            name: train_model(
                name,
                model,
                chunks_by_split["train"],
                epochs=args.epochs,
                device=device,
            )
            for name, model in models.items()
        }
        post = {
            name: {
                "train_sparse": evaluate_model(
                    model,
                    chunks_by_split["train"],
                    dense=False,
                    device=device,
                ),
                "valid_sparse": evaluate_model(
                    model,
                    chunks_by_split["valid"],
                    dense=False,
                    device=device,
                ),
                "test_sparse": evaluate_model(
                    model,
                    chunks_by_split["test"],
                    dense=False,
                    device=device,
                ),
                "test_dense_outcome": evaluate_model(
                    model,
                    chunks_by_split["test"],
                    dense=True,
                    device=device,
                ),
            }
            for name, model in models.items()
        }
        streaming = benchmark_streaming(
            models,
            test_episode,
            warmup_steps=args.stream_warmup,
            measured_steps=args.stream_steps,
            device=device,
            seed=9_210_000 + seed,
        )
        ablation = {
            mode: evaluate_mechanism_ablation(
                models["snn_at1"],
                chunks_by_split["test"],
                mode=mode,
                device=device,
            )
            for mode in ("spike_only", "trace_only")
        }
        seed_results.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_relative_spread": parameter_spread,
                "pre": pre,
                "training": training,
                "post": post,
                "streaming": streaming,
                "mechanism_ablation": ablation,
            }
        )
    decision = _decision(
        data_audit=data_audit,
        seed_results=seed_results,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-TW0 real TextWorld sparse outcome-token LM",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "corpus_dir": str(corpus_root),
            "seeds": tuple(args.seeds),
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "sequence_length": args.sequence_length,
            "max_queries_per_chunk": args.max_queries,
            "selected_channels": sorted(SELECTED_CHANNELS),
            "epochs": args.epochs,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "threads": args.threads if device.type == "cpu" else None,
            "stream_warmup": args.stream_warmup,
            "stream_steps": args.stream_steps,
            "embedding_trainable": True,
            "tied_embedding_head": True,
            "at1_forward_mode": "segment",
        },
        "dataset": {
            "synthetic": False,
            "manifest_provenance": manifest_provenance,
            "corpus_provenance": corpus_provenance,
            "sparse_query_audit": data_audit,
        },
        "seeds": seed_results,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_tw0_sparse_event_lm.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--max-queries", type=int, default=MAX_QUERIES)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stream-warmup", type=int, default=64)
    parser.add_argument("--stream-steps", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.sequence_length, args.max_queries, args.threads) <= 0:
        parser.error("epochs, sequence length, max queries, and threads must be positive")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 1
        args.stream_warmup = 4
        args.stream_steps = 32
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
