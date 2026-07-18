"""Marked WRITE(payload)/READ(delay) diagnostic for E3-S0 memory quality."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments.e3_s0_delayed_copy import (  # noqa: E402
    DELAYS,
    D_MODEL,
    E3_STATE_DIM,
    PAYLOAD_VOCAB,
    SEQUENCE_LENGTH,
    DelayedBatch,
    DelayedCopyModel,
    _dataset_hash,
    _mean_std,
    evaluate_model,
    train_model,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3CumulativeScanCore,
    StatefulLSTMCore,
    count_parameters,
)


DISTRACTOR_VOCAB = 4
WRITE_BASE = DISTRACTOR_VOCAB
QUERY_BASE = WRITE_BASE + len(DELAYS) * PAYLOAD_VOCAB
INPUT_VOCAB = QUERY_BASE + len(DELAYS)


class MarkedDelayedCopyModel(DelayedCopyModel):
    def __init__(self, core: Any) -> None:
        super().__init__(core)
        self.embedding = nn.Embedding(INPUT_VOCAB, D_MODEL)


def _pairs(seed: int) -> Sequence[Tuple[int, int, int]]:
    generator = random.Random(seed)
    used: set[int] = set()
    pairs = []
    for delay_index, delay in reversed(tuple(enumerate(DELAYS))):
        candidates = list(range(SEQUENCE_LENGTH - delay))
        generator.shuffle(candidates)
        for source in candidates:
            query = source + delay
            if source in used or query in used:
                continue
            used.add(source)
            used.add(query)
            pairs.append((source, query, delay_index))
            break
        else:  # pragma: no cover - far more valid positions than requested
            raise RuntimeError("could not place marked delayed event")
    pairs.sort(key=lambda item: item[1])
    return pairs


def generate_marked_batch(seed: int, batch_size: int) -> DelayedBatch:
    torch_generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        DISTRACTOR_VOCAB,
        (batch_size, SEQUENCE_LENGTH),
        generator=torch_generator,
    )
    positions = torch.empty(batch_size, len(DELAYS), dtype=torch.long)
    targets = torch.empty_like(positions)
    delay_values = torch.empty_like(positions)
    payloads = torch.randint(
        PAYLOAD_VOCAB, (batch_size, len(DELAYS)), generator=torch_generator
    )
    for batch_index in range(batch_size):
        for query_index, (source, query, delay_index) in enumerate(
            _pairs(seed * 1009 + batch_index)
        ):
            payload = int(payloads[batch_index, delay_index])
            tokens[batch_index, source] = (
                WRITE_BASE + delay_index * PAYLOAD_VOCAB + payload
            )
            tokens[batch_index, query] = QUERY_BASE + delay_index
            positions[batch_index, query_index] = query
            targets[batch_index, query_index] = payload
            delay_values[batch_index, query_index] = DELAYS[delay_index]
    return DelayedBatch(
        tokens=tokens,
        query_positions=positions,
        targets=targets,
        delays=delay_values,
    )


def _shared_initialisation(seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "embedding.weight": torch.randn(INPUT_VOCAB, D_MODEL, generator=generator) * 0.02,
        "output_norm.weight": torch.ones(D_MODEL),
        "output_norm.bias": torch.zeros(D_MODEL),
        "decoder.weight": torch.randn(PAYLOAD_VOCAB, D_MODEL, generator=generator) * 0.02,
        "decoder.bias": torch.zeros(PAYLOAD_VOCAB),
    }


def build_models(seed: int) -> Dict[str, MarkedDelayedCopyModel]:
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
        model = MarkedDelayedCopyModel(factory())
        with torch.no_grad():
            model.embedding.weight.copy_(shared["embedding.weight"])
            model.output_norm.weight.copy_(shared["output_norm.weight"])
            model.output_norm.bias.copy_(shared["output_norm.bias"])
            model.decoder.weight.copy_(shared["decoder.weight"])
            model.decoder.bias.copy_(shared["decoder.bias"])
        models[name] = model
    return models


def aggregate(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate_models = {}
    for name in ("e3_scan", "lstm", "transformer"):
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
    lstm_checks = {
        "overall": aggregate_models["lstm"]["test_accuracy"]["mean"] >= 0.90,
        **{
            f"delay_{delay}": aggregate_models["lstm"]["delay_accuracy"][str(delay)][
                "mean"
            ]
            >= 0.90
            for delay in DELAYS
        },
    }
    e3_checks = {
        "overall_absolute": aggregate_models["e3_scan"]["test_accuracy"]["mean"] >= 0.90,
        "overall_noninferior": aggregate_models["e3_scan"]["test_accuracy"]["mean"]
        >= aggregate_models["lstm"]["test_accuracy"]["mean"] - 0.02,
    }
    for delay in DELAYS:
        e3_checks[f"delay_{delay}_absolute"] = (
            aggregate_models["e3_scan"]["delay_accuracy"][str(delay)]["mean"] >= 0.90
        )
        e3_checks[f"delay_{delay}_noninferior"] = (
            aggregate_models["e3_scan"]["delay_accuracy"][str(delay)]["mean"]
            >= aggregate_models["lstm"]["delay_accuracy"][str(delay)]["mean"] - 0.02
        )
    speed_check = (
        aggregate_models["e3_scan"]["train_p50_ms"]["mean"]
        < aggregate_models["lstm"]["train_p50_ms"]["mean"]
    )
    if not all(lstm_checks.values()):
        status = "INVALID"
    elif all(e3_checks.values()):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "models": aggregate_models,
        "decision": {
            "status": status,
            "lstm_task_checks": lstm_checks,
            "e3_quality_checks": e3_checks,
            "speed_check_report_only": speed_check,
            "chance_accuracy": 1.0 / PAYLOAD_VOCAB,
            "next": (
                "increase concurrent marked writes"
                if status == "PASS"
                else "S1 dynamic decay/gated charge"
                if status == "FAIL"
                else "overfit and short-delay runner audit"
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_s0_marked_delay.json"),
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
        parser.error("invalid seed, budget, batch, learning rate, threads, or warmup")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    runs = []
    parameter_counts: Dict[str, int] | None = None
    for seed in args.seeds:
        train_batches = tuple(
            generate_marked_batch(3_000_000 + seed * 100_000 + update, args.batch_size)
            for update in range(args.updates)
        )
        test_batches = tuple(
            generate_marked_batch(4_000_000 + seed * 100_000 + index, args.batch_size)
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
        "experiment": "E3-S0 marked WRITE/READ delayed memory",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(torch.device("cpu")),
        "configuration": {
            "seeds": tuple(args.seeds),
            "updates": args.updates,
            "batch_size": args.batch_size,
            "test_batches": args.test_batches,
            "sequence_length": SEQUENCE_LENGTH,
            "distractor_vocab": DISTRACTOR_VOCAB,
            "payload_vocab": PAYLOAD_VOCAB,
            "input_vocab": INPUT_VOCAB,
            "delays": DELAYS,
            "writes_per_delay": 1,
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
