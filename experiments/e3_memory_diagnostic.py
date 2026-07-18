"""D2 learning diagnostic: direct decode, short overfit, then short generalisation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments.e3_s0_delayed_copy import (  # noqa: E402
    D_MODEL,
    E3_STATE_DIM,
    PAYLOAD_VOCAB,
    DelayedBatch,
    DelayedCopyModel,
    _dataset_hash,
    _mean_std,
    train_model,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3CumulativeScanCore,
    StatefulLSTMCore,
    count_parameters,
)


DISTRACTOR_VOCAB = 4
SLOTS = 2
WRITE_BASE = DISTRACTOR_VOCAB
QUERY_BASE = WRITE_BASE + SLOTS * PAYLOAD_VOCAB
INPUT_VOCAB = QUERY_BASE + SLOTS


class DiagnosticModel(DelayedCopyModel):
    def __init__(self, core: Any) -> None:
        super().__init__(core)
        self.embedding = nn.Embedding(INPUT_VOCAB, D_MODEL)


def generate_batch(
    *,
    seed: int,
    batch_size: int,
    sequence_length: int,
    delays: Tuple[int, int],
    direct: bool = False,
) -> DelayedBatch:
    torch_generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        DISTRACTOR_VOCAB,
        (batch_size, sequence_length),
        generator=torch_generator,
    )
    positions = torch.empty(batch_size, SLOTS, dtype=torch.long)
    targets = torch.empty_like(positions)
    buckets = torch.empty_like(positions)
    payloads = torch.randint(PAYLOAD_VOCAB, (batch_size, SLOTS), generator=torch_generator)
    for batch_index in range(batch_size):
        generator = random.Random(seed * 1009 + batch_index)
        used: set[int] = set()
        for slot, delay in reversed(tuple(enumerate(delays))):
            candidates = list(range(sequence_length if direct else sequence_length - delay))
            generator.shuffle(candidates)
            for source in candidates:
                query = source if direct else source + delay
                if source in used or query in used:
                    continue
                used.add(source)
                used.add(query)
                break
            else:  # pragma: no cover
                raise RuntimeError("could not place diagnostic event")
            payload = int(payloads[batch_index, slot])
            tokens[batch_index, source] = WRITE_BASE + slot * PAYLOAD_VOCAB + payload
            if not direct:
                tokens[batch_index, query] = QUERY_BASE + slot
            positions[batch_index, slot] = query
            targets[batch_index, slot] = payload
            buckets[batch_index, slot] = delay
    return DelayedBatch(tokens=tokens, query_positions=positions, targets=targets, delays=buckets)


def _shared_initialisation(seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "embedding.weight": torch.randn(INPUT_VOCAB, D_MODEL, generator=generator) * 0.02,
        "output_norm.weight": torch.ones(D_MODEL),
        "output_norm.bias": torch.zeros(D_MODEL),
        "decoder.weight": torch.randn(PAYLOAD_VOCAB, D_MODEL, generator=generator) * 0.02,
        "decoder.bias": torch.zeros(PAYLOAD_VOCAB),
    }


def build_models(seed: int, sequence_length: int) -> Dict[str, DiagnosticModel]:
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
            max_cache_tokens=sequence_length,
        ),
    }
    models = {}
    for index, (name, factory) in enumerate(factories.items()):
        torch.manual_seed(seed + 1000 + index)
        model = DiagnosticModel(factory())
        with torch.no_grad():
            model.embedding.weight.copy_(shared["embedding.weight"])
            model.output_norm.weight.copy_(shared["output_norm.weight"])
            model.output_norm.bias.copy_(shared["output_norm.bias"])
            model.decoder.weight.copy_(shared["decoder.weight"])
            model.decoder.bias.copy_(shared["decoder.bias"])
        models[name] = model
    return models


def evaluate(
    model: DiagnosticModel,
    batches: Sequence[DelayedBatch],
    delays: Tuple[int, int],
) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    nll_sum = 0.0
    buckets = {str(delay): {"count": 0, "correct": 0} for delay in delays}
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch.tokens, batch.query_positions)
            prediction = logits.argmax(dim=-1)
            matches = prediction == batch.targets
            losses = F.cross_entropy(
                logits.reshape(-1, PAYLOAD_VOCAB),
                batch.targets.reshape(-1),
                reduction="none",
            )
            total += int(matches.numel())
            correct += int(matches.sum().item())
            nll_sum += float(losses.sum().item())
            for delay in delays:
                mask = batch.delays == delay
                buckets[str(delay)]["count"] += int(mask.sum().item())
                buckets[str(delay)]["correct"] += int(matches[mask].sum().item())
    return {
        "count": total,
        "accuracy": correct / total,
        "nll": nll_sum / total,
        "by_delay": {
            delay: {
                "count": values["count"],
                "accuracy": values["correct"] / values["count"],
            }
            for delay, values in buckets.items()
        },
    }


def _assert_parameter_fairness(models: Mapping[str, DiagnosticModel]) -> Dict[str, int]:
    counts = {name: count_parameters(model) for name, model in models.items()}
    lstm = counts["lstm"]
    if any(abs(value - lstm) / lstm > 0.02 for value in counts.values()):
        raise AssertionError(f"parameter fairness failed: {counts}")
    return counts


def run_stage(
    *,
    name: str,
    seed: int,
    sequence_length: int,
    delays: Tuple[int, int],
    batch_size: int,
    updates: int,
    test_batches: int,
    direct: bool,
    fixed_batch: bool,
    learning_rate: float,
) -> Dict[str, Any]:
    if fixed_batch:
        fixed = generate_batch(
            seed=seed + 10_000,
            batch_size=batch_size,
            sequence_length=sequence_length,
            delays=delays,
            direct=direct,
        )
        train_batches = (fixed,) * updates
        evaluation_batches = (fixed,)
    else:
        train_batches = tuple(
            generate_batch(
                seed=seed + 10_000 + update,
                batch_size=batch_size,
                sequence_length=sequence_length,
                delays=delays,
                direct=direct,
            )
            for update in range(updates)
        )
        evaluation_batches = tuple(
            generate_batch(
                seed=seed + 20_000 + index,
                batch_size=batch_size,
                sequence_length=sequence_length,
                delays=delays,
                direct=direct,
            )
            for index in range(test_batches)
        )
    models = build_models(seed, sequence_length)
    counts = _assert_parameter_fairness(models)
    model_results = {}
    for model_name in ("e3_scan", "lstm", "transformer"):
        train = train_model(
            models[model_name],
            train_batches,
            learning_rate=learning_rate,
            timing_warmup=min(100, updates - 1),
        )
        test = evaluate(models[model_name], evaluation_batches, delays)
        model_results[model_name] = {"train": train, "test": test}
    return {
        "stage": name,
        "seed": seed,
        "sequence_length": sequence_length,
        "delays": delays,
        "direct": direct,
        "fixed_batch": fixed_batch,
        "updates": updates,
        "parameter_counts": counts,
        "train_data_sha256": _dataset_hash(train_batches),
        "test_data_sha256": _dataset_hash(evaluation_batches),
        "models": model_results,
    }


def _aggregate_b(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    models = {}
    for name in ("e3_scan", "lstm", "transformer"):
        selected = [run["models"][name] for run in runs]
        models[name] = {
            "accuracy": _mean_std([item["test"]["accuracy"] for item in selected]),
            "delay_accuracy": {
                delay: _mean_std(
                    [item["test"]["by_delay"][delay]["accuracy"] for item in selected]
                )
                for delay in ("4", "16")
            },
            "train_p50_ms": _mean_std(
                [item["train"]["timing"]["p50_ms"] for item in selected]
            ),
        }
    lstm_valid = models["lstm"]["accuracy"]["mean"] >= 0.90 and all(
        models["lstm"]["delay_accuracy"][delay]["mean"] >= 0.90
        for delay in ("4", "16")
    )
    e3_pass = models["e3_scan"]["accuracy"]["mean"] >= 0.90 and all(
        models["e3_scan"]["delay_accuracy"][delay]["mean"] >= 0.90
        and models["e3_scan"]["delay_accuracy"][delay]["mean"]
        >= models["lstm"]["delay_accuracy"][delay]["mean"] - 0.02
        for delay in ("4", "16")
    ) and models["e3_scan"]["accuracy"]["mean"] >= models["lstm"]["accuracy"]["mean"] - 0.02
    return {
        "models": models,
        "lstm_valid": lstm_valid,
        "e3_quality_pass": e3_pass if lstm_valid else False,
        "status": "PASS" if lstm_valid and e3_pass else "FAIL" if lstm_valid else "INVALID",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_memory_diagnostic.json"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.learning_rate <= 0.0:
        parser.error("threads and learning rate must be positive")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    a0_updates = 2 if args.quick else 300
    a1_updates = 2 if args.quick else 1000
    b_updates = 2 if args.quick else 500
    test_batches = 1 if args.quick else 64
    a0 = run_stage(
        name="A0_same_position",
        seed=5_000_000,
        sequence_length=32,
        delays=(0, 0),
        batch_size=8,
        updates=a0_updates,
        test_batches=test_batches,
        direct=True,
        fixed_batch=False,
        learning_rate=args.learning_rate,
    )
    a0_checks = {
        name: values["test"]["accuracy"] >= 0.99 for name, values in a0["models"].items()
    }
    a1 = None
    a1_checks: Dict[str, bool] = {}
    b_runs = []
    b_aggregate = None
    if all(a0_checks.values()) or args.quick:
        a1 = run_stage(
            name="A1_fixed_short_delay",
            seed=6_000_000,
            sequence_length=32,
            delays=(1, 4),
            batch_size=8,
            updates=a1_updates,
            test_batches=1,
            direct=False,
            fixed_batch=True,
            learning_rate=args.learning_rate,
        )
        a1_checks = {
            name: values["test"]["accuracy"] >= 0.99
            for name, values in a1["models"].items()
        }
    if (a1 is not None and all(a1_checks.values())) or args.quick:
        for seed in (0,) if args.quick else (0, 1, 2):
            b_runs.append(
                run_stage(
                    name="B_random_short_delay",
                    seed=7_000_000 + seed * 100_000,
                    sequence_length=64,
                    delays=(4, 16),
                    batch_size=8,
                    updates=b_updates,
                    test_batches=test_batches,
                    direct=False,
                    fixed_batch=False,
                    learning_rate=args.learning_rate,
                )
            )
        b_aggregate = _aggregate_b(b_runs)
    if not all(a0_checks.values()):
        status = "A0_FAIL"
    elif not all(a1_checks.values()):
        status = "A1_FAIL"
    elif b_aggregate is None:
        status = "B_NOT_RUN"
    else:
        status = f"B_{b_aggregate['status']}"
    result = {
        "schema_version": 1,
        "experiment": "E3 memory diagnostic ladder D2",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(torch.device("cpu")),
        "configuration": {
            "threads": args.threads,
            "learning_rate": args.learning_rate,
            "a0_updates": a0_updates,
            "a1_updates": a1_updates,
            "b_updates": b_updates,
            "test_batches": test_batches,
        },
        "a0": a0,
        "a0_checks": a0_checks,
        "a1": a1,
        "a1_checks": a1_checks,
        "b_runs": b_runs,
        "b_aggregate": b_aggregate,
        "decision": {
            "status": status,
            "next": (
                "runner audit"
                if status == "A0_FAIL"
                else "short credit-assignment audit"
                if status == "A1_FAIL"
                else "S1 dynamic decay/gating"
                if status in ("B_FAIL", "B_PASS")
                else "B budget/task audit"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
