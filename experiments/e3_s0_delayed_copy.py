"""Parameter-matched 512-token delayed retrieval for E3-S0/LSTM/Transformer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3CumulativeScanCore,
    E3LayerTrace,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
)


Tensor = torch.Tensor
PAYLOAD_VOCAB = 16
DELAYS = (64, 256)
QUERIES_PER_DELAY = 4
SEQUENCE_LENGTH = 512
D_MODEL = 32
E3_STATE_DIM = 27


@dataclass(frozen=True)
class DelayedBatch:
    tokens: Tensor
    query_positions: Tensor
    targets: Tensor
    delays: Tensor


class DelayedCopyModel(nn.Module):
    def __init__(self, core: TemporalCore[Any]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(PAYLOAD_VOCAB + len(DELAYS), D_MODEL)
        self.core = core
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.decoder = nn.Linear(D_MODEL, PAYLOAD_VOCAB)

    def forward(self, tokens: Tensor, query_positions: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        hidden = self.core(embedded).sequence
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
        query_hidden = hidden[batch_indices, query_positions]
        return self.decoder(self.output_norm(query_hidden))

    def forward_with_e3_traces(
        self, tokens: Tensor, query_positions: Tensor
    ) -> Tuple[Tensor, Tuple[E3LayerTrace, ...]]:
        if not isinstance(self.core, E3CumulativeScanCore):
            raise TypeError("forward_with_e3_traces requires an E3 core")
        embedded = self.embedding(tokens)
        core_output, traces = self.core.forward_dynamics(embedded)
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
        query_hidden = core_output.sequence[batch_indices, query_positions]
        return self.decoder(self.output_norm(query_hidden)), traces


def _select_query_pairs(seed: int) -> Sequence[Tuple[int, int, int]]:
    generator = random.Random(seed)
    used: set[int] = set()
    pairs = []
    for delay_index, delay in reversed(tuple(enumerate(DELAYS))):
        candidates = list(range(SEQUENCE_LENGTH - delay))
        generator.shuffle(candidates)
        selected = 0
        for source in candidates:
            query = source + delay
            if source in used or query in used:
                continue
            used.add(source)
            used.add(query)
            pairs.append((source, query, delay_index))
            selected += 1
            if selected == QUERIES_PER_DELAY:
                break
        if selected != QUERIES_PER_DELAY:
            raise RuntimeError("could not place non-overlapping delayed queries")
    pairs.sort(key=lambda item: item[1])
    return pairs


def generate_batch(seed: int, batch_size: int) -> DelayedBatch:
    torch_generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        PAYLOAD_VOCAB,
        (batch_size, SEQUENCE_LENGTH),
        generator=torch_generator,
    )
    positions = torch.empty(batch_size, len(DELAYS) * QUERIES_PER_DELAY, dtype=torch.long)
    targets = torch.empty_like(positions)
    delays = torch.empty_like(positions)
    for batch_index in range(batch_size):
        pairs = _select_query_pairs(seed * 1009 + batch_index)
        for query_index, (source, query, delay_index) in enumerate(pairs):
            positions[batch_index, query_index] = query
            targets[batch_index, query_index] = tokens[batch_index, source]
            delays[batch_index, query_index] = DELAYS[delay_index]
            tokens[batch_index, query] = PAYLOAD_VOCAB + delay_index
    return DelayedBatch(
        tokens=tokens,
        query_positions=positions,
        targets=targets,
        delays=delays,
    )


def _dataset_hash(batches: Sequence[DelayedBatch]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        for value in (batch.tokens, batch.query_positions, batch.targets, batch.delays):
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _shared_initialisation(seed: int) -> Dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "embedding.weight": torch.randn(
            PAYLOAD_VOCAB + len(DELAYS), D_MODEL, generator=generator
        )
        * 0.02,
        "output_norm.weight": torch.ones(D_MODEL),
        "output_norm.bias": torch.zeros(D_MODEL),
        "decoder.weight": torch.randn(PAYLOAD_VOCAB, D_MODEL, generator=generator) * 0.02,
        "decoder.bias": torch.zeros(PAYLOAD_VOCAB),
    }


def build_models(seed: int) -> Dict[str, DelayedCopyModel]:
    shared = _shared_initialisation(seed + 50_000)
    factories = {
        "e3_scan": lambda: E3CumulativeScanCore(
            D_MODEL,
            D_MODEL,
            state_dim=E3_STATE_DIM,
            num_layers=2,
            execution_mode="scan",
        ),
        "lstm": lambda: StatefulLSTMCore(D_MODEL, D_MODEL),
        "transformer": lambda: CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=SEQUENCE_LENGTH,
        ),
    }
    models = {}
    for index, (name, factory) in enumerate(factories.items()):
        torch.manual_seed(seed + 1000 + index)
        model = DelayedCopyModel(factory())
        with torch.no_grad():
            model.embedding.weight.copy_(shared["embedding.weight"])
            model.output_norm.weight.copy_(shared["output_norm.weight"])
            model.output_norm.bias.copy_(shared["output_norm.bias"])
            model.decoder.weight.copy_(shared["decoder.weight"])
            model.decoder.bias.copy_(shared["decoder.bias"])
        models[name] = model
    return models


def train_model(
    model: DelayedCopyModel,
    batches: Sequence[DelayedBatch],
    *,
    learning_rate: float,
    timing_warmup: int,
) -> Dict[str, Any]:
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    timings = []
    losses = []
    checkpoints = []
    wall_started = time.perf_counter_ns()
    for update, batch in enumerate(batches, start=1):
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.tokens, batch.query_positions)
        loss = F.cross_entropy(logits.reshape(-1, PAYLOAD_VOCAB), batch.targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite delayed-copy loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if update > timing_warmup:
            timings.append(elapsed_ms)
        losses.append(float(loss.detach().item()))
        if update == 1 or update % 100 == 0 or update == len(batches):
            checkpoints.append(
                {
                    "update": update,
                    "loss": losses[-1],
                    "gradient_norm": float(gradient_norm),
                }
            )
    wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
    timing = _sample_summary(
        timings,
        batches[0].tokens.numel(),
    )
    timing["query_supervision_per_second_at_p50"] = (
        batches[0].targets.numel() * 1000.0 / timing["p50_ms"]
    )
    return {
        "updates": len(batches),
        "timing_warmup_updates": timing_warmup,
        "wall_ms": wall_ms,
        "timing": timing,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_100_mean": sum(losses[-100:]) / min(100, len(losses)),
        "checkpoints": checkpoints,
    }


def evaluate_model(
    model: DelayedCopyModel, batches: Sequence[DelayedBatch]
) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    nll_sum = 0.0
    by_delay = {
        str(delay): {"count": 0, "correct": 0, "nll_sum": 0.0} for delay in DELAYS
    }
    spike_sums: list[Dict[str, float]] | None = None
    with torch.inference_mode():
        for batch in batches:
            traces: Tuple[E3LayerTrace, ...] = ()
            if isinstance(model.core, E3CumulativeScanCore):
                logits, traces = model.forward_with_e3_traces(
                    batch.tokens, batch.query_positions
                )
            else:
                logits = model(batch.tokens, batch.query_positions)
            losses = F.cross_entropy(
                logits.reshape(-1, PAYLOAD_VOCAB),
                batch.targets.reshape(-1),
                reduction="none",
            ).reshape_as(batch.targets)
            predictions = logits.argmax(dim=-1)
            matches = predictions == batch.targets
            total += int(matches.numel())
            correct += int(matches.sum().item())
            nll_sum += float(losses.sum().item())
            for delay in DELAYS:
                mask = batch.delays == delay
                bucket = by_delay[str(delay)]
                bucket["count"] += int(mask.sum().item())
                bucket["correct"] += int(matches[mask].sum().item())
                bucket["nll_sum"] += float(losses[mask].sum().item())
            if traces:
                if spike_sums is None:
                    spike_sums = [
                        {"e_spikes": 0.0, "i_spikes": 0.0, "elements": 0.0}
                        for _ in traces
                    ]
                for index, trace in enumerate(traces):
                    spike_sums[index]["e_spikes"] += float(
                        trace.excitatory_spikes.sum().item()
                    )
                    spike_sums[index]["i_spikes"] += float(
                        trace.inhibitory_spikes.sum().item()
                    )
                    spike_sums[index]["elements"] += float(
                        trace.excitatory_spikes.numel()
                    )
    result = {
        "count": total,
        "accuracy": correct / total,
        "nll": nll_sum / total,
        "by_delay": {
            delay: {
                "count": values["count"],
                "accuracy": values["correct"] / values["count"],
                "nll": values["nll_sum"] / values["count"],
            }
            for delay, values in by_delay.items()
        },
    }
    if spike_sums is not None:
        result["spike_rates"] = [
            {
                "layer": index,
                "excitatory": values["e_spikes"] / values["elements"],
                "inhibitory": values["i_spikes"] / values["elements"],
            }
            for index, values in enumerate(spike_sums)
        ]
    return result


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


def aggregate(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    models = ("e3_scan", "lstm", "transformer")
    aggregate_models = {}
    for name in models:
        selected = [run["models"][name] for run in runs]
        aggregate_models[name] = {
            "test_accuracy": _mean_std([item["test"]["accuracy"] for item in selected]),
            "test_nll": _mean_std([item["test"]["nll"] for item in selected]),
            "delay_accuracy": {
                str(delay): _mean_std(
                    [item["test"]["by_delay"][str(delay)]["accuracy"] for item in selected]
                )
                for delay in DELAYS
            },
            "train_p50_ms": _mean_std(
                [item["train"]["timing"]["p50_ms"] for item in selected]
            ),
            "train_tokens_per_second": _mean_std(
                [
                    item["train"]["timing"]["tokens_per_second_at_p50"]
                    for item in selected
                ]
            ),
            "train_wall_ms": _mean_std([item["train"]["wall_ms"] for item in selected]),
        }
    lstm_valid = aggregate_models["lstm"]["test_accuracy"]["mean"] >= 0.80
    quality_checks = {
        "overall_absolute": aggregate_models["e3_scan"]["test_accuracy"]["mean"] >= 0.80,
        "overall_noninferior": aggregate_models["e3_scan"]["test_accuracy"]["mean"]
        >= aggregate_models["lstm"]["test_accuracy"]["mean"] - 0.02,
    }
    for delay in DELAYS:
        quality_checks[f"delay_{delay}_absolute"] = (
            aggregate_models["e3_scan"]["delay_accuracy"][str(delay)]["mean"] >= 0.80
        )
        quality_checks[f"delay_{delay}_noninferior"] = (
            aggregate_models["e3_scan"]["delay_accuracy"][str(delay)]["mean"]
            >= aggregate_models["lstm"]["delay_accuracy"][str(delay)]["mean"] - 0.02
        )
    speed_pass = (
        aggregate_models["e3_scan"]["train_p50_ms"]["mean"]
        < aggregate_models["lstm"]["train_p50_ms"]["mean"]
    )
    if not lstm_valid:
        status = "INVALID"
    elif all(quality_checks.values()) and speed_pass:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "models": aggregate_models,
        "decision": {
            "status": status,
            "lstm_task_validation": "PASS" if lstm_valid else "FAIL",
            "quality_checks": quality_checks,
            "speed_check": speed_pass,
            "chance_accuracy": 1.0 / PAYLOAD_VOCAB,
            "next": (
                "TextWorld next-event"
                if status == "PASS"
                else "S1 dynamic decay/selective memory"
                if status == "FAIL"
                else "pre-register a learnable diagnostic budget"
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_s0_delayed_copy.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.updates = min(args.updates, 2)
        args.test_batches = min(args.test_batches, 1)
        args.seeds = args.seeds[:1]
        args.timing_warmup = 1
    if (
        any(seed < 0 for seed in args.seeds)
        or args.updates <= 1
        or args.batch_size <= 0
        or args.test_batches <= 0
        or args.learning_rate <= 0.0
        or args.threads <= 0
        or not 0 <= args.timing_warmup < args.updates
    ):
        parser.error("invalid seeds, budget, batch size, learning rate, threads, or warmup")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    runs = []
    parameter_counts: Dict[str, int] | None = None
    for seed in args.seeds:
        train_batches = tuple(
            generate_batch(1_000_000 + seed * 100_000 + update, args.batch_size)
            for update in range(args.updates)
        )
        test_batches = tuple(
            generate_batch(2_000_000 + seed * 100_000 + index, args.batch_size)
            for index in range(args.test_batches)
        )
        models = build_models(seed)
        current_counts = {name: count_parameters(model) for name, model in models.items()}
        if parameter_counts is None:
            parameter_counts = current_counts
        elif current_counts != parameter_counts:
            raise AssertionError("parameter counts changed across seeds")
        lstm_parameters = current_counts["lstm"]
        if any(abs(value - lstm_parameters) / lstm_parameters > 0.02 for value in current_counts.values()):
            raise AssertionError(f"parameter fairness failed: {current_counts}")

        order = ("e3_scan", "lstm", "transformer")
        rotation = seed % len(order)
        order = order[rotation:] + order[:rotation]
        model_results = {}
        for name in order:
            train = train_model(
                models[name],
                train_batches,
                learning_rate=args.learning_rate,
                timing_warmup=args.timing_warmup,
            )
            test = evaluate_model(models[name], test_batches)
            model_results[name] = {"train": train, "test": test}
        runs.append(
            {
                "seed": seed,
                "model_order": order,
                "train_data_sha256": _dataset_hash(train_batches),
                "test_data_sha256": _dataset_hash(test_batches),
                "models": model_results,
            }
        )
    assert parameter_counts is not None
    aggregated = aggregate(runs)
    result = {
        "schema_version": 1,
        "experiment": "E3-S0 512-token delayed retrieval",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(torch.device("cpu")),
        "configuration": {
            "seeds": tuple(args.seeds),
            "updates": args.updates,
            "batch_size": args.batch_size,
            "test_batches": args.test_batches,
            "sequence_length": SEQUENCE_LENGTH,
            "payload_vocab": PAYLOAD_VOCAB,
            "delays": DELAYS,
            "queries_per_delay": QUERIES_PER_DELAY,
            "d_model": D_MODEL,
            "e3_state_dim": E3_STATE_DIM,
            "learning_rate": args.learning_rate,
            "threads": args.threads,
            "timing_warmup": args.timing_warmup,
        },
        "parameter_counts": parameter_counts,
        "runs": runs,
        "aggregate": aggregated["models"],
        "decision": aggregated["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
