"""Training-only fixed population-code objective for the exact-reset S0-L1 core."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
)
from experiments.e3_memory_diagnostic import (  # noqa: E402
    DiagnosticModel,
    _dataset_hash,
    _shared_initialisation,
    evaluate,
    generate_batch,
)
from experiments.e3_s0_delayed_copy import DelayedBatch, train_model  # noqa: E402
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3CumulativeScanCore,
    StatefulLSTMCore,
    count_parameters,
)


STATE_DIM = 42
CODEBOOK_SEED = 19001
TEMPERATURE = 0.2
LOCAL_WEIGHT = 1.0


def _codebook() -> torch.Tensor:
    generator = torch.Generator().manual_seed(CODEBOOK_SEED)
    values = torch.randint(0, 2, (16, 4 * STATE_DIM), generator=generator).float()
    return 2.0 * values - 1.0


def build_models(seed: int) -> Dict[str, DiagnosticModel]:
    shared = _shared_initialisation(seed + 50_000)
    torch.manual_seed(seed + 1000)
    global_model = DiagnosticModel(
        E3CumulativeScanCore(
            32,
            32,
            state_dim=STATE_DIM,
            num_layers=1,
            execution_mode="scan",
        )
    )
    with torch.no_grad():
        global_model.embedding.weight.copy_(shared["embedding.weight"])
        global_model.output_norm.weight.copy_(shared["output_norm.weight"])
        global_model.output_norm.bias.copy_(shared["output_norm.bias"])
        global_model.decoder.weight.copy_(shared["decoder.weight"])
        global_model.decoder.bias.copy_(shared["decoder.bias"])
    local_model = copy.deepcopy(global_model)

    models = {"l1_global": global_model, "l1_local": local_model}
    for index, (name, core) in enumerate(
        (
            ("lstm", StatefulLSTMCore(32, 32)),
            (
                "transformer",
                CausalTransformerCore(
                    32,
                    32,
                    num_layers=1,
                    num_heads=4,
                    mlp_ratio=2.0,
                    dropout=0.0,
                    max_cache_tokens=32,
                ),
            ),
        ),
        start=1,
    ):
        torch.manual_seed(seed + 1000 + index)
        model = DiagnosticModel(core)
        with torch.no_grad():
            model.embedding.weight.copy_(shared["embedding.weight"])
            model.output_norm.weight.copy_(shared["output_norm.weight"])
            model.output_norm.bias.copy_(shared["output_norm.bias"])
            model.decoder.weight.copy_(shared["decoder.weight"])
            model.decoder.bias.copy_(shared["decoder.bias"])
        models[name] = model
    return models


def _local_logits(
    model: DiagnosticModel,
    batch: DelayedBatch,
    codebook: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(model.core, E3CumulativeScanCore):
        raise TypeError("local code objective requires E3CumulativeScanCore")
    embedded = model.embedding(batch.tokens)
    core_output, traces = model.core.forward_dynamics(embedded)
    trace = traces[0]
    batch_indices = torch.arange(batch.tokens.shape[0]).unsqueeze(1)
    query_hidden = core_output.sequence[batch_indices, batch.query_positions]
    global_logits = model.decoder(model.output_norm(query_hidden))
    raw_representation = torch.cat(
        (
            trace.excitatory_spikes,
            trace.inhibitory_spikes,
            trace.excitatory_residuals,
            trace.inhibitory_residuals,
        ),
        dim=-1,
    )[batch_indices, batch.query_positions]
    representation = F.normalize(2.0 * raw_representation - 1.0, dim=-1)
    normalised_codebook = F.normalize(codebook, dim=-1)
    local_logits = torch.matmul(representation, normalised_codebook.T) / TEMPERATURE
    return global_logits, local_logits


def train_local(
    model: DiagnosticModel,
    batches: Sequence[DelayedBatch],
    *,
    codebook: torch.Tensor,
    timing_warmup: int,
) -> Dict[str, Any]:
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    timings = []
    global_losses = []
    local_losses = []
    checkpoints = []
    wall_started = time.perf_counter_ns()
    for update, batch in enumerate(batches, start=1):
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        global_logits, local_logits = _local_logits(model, batch, codebook)
        global_loss = F.cross_entropy(
            global_logits.reshape(-1, 16), batch.targets.reshape(-1)
        )
        local_loss = F.cross_entropy(local_logits.reshape(-1, 16), batch.targets.reshape(-1))
        loss = global_loss + LOCAL_WEIGHT * local_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite LG0 loss at update {update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if update > timing_warmup:
            timings.append(elapsed_ms)
        global_losses.append(float(global_loss.detach().item()))
        local_losses.append(float(local_loss.detach().item()))
        if update == 1 or update % 100 == 0 or update == len(batches):
            checkpoints.append(
                {
                    "update": update,
                    "global_loss": global_losses[-1],
                    "local_loss": local_losses[-1],
                    "gradient_norm": float(gradient_norm),
                }
            )
    timing = _sample_summary(timings, batches[0].tokens.numel())
    return {
        "updates": len(batches),
        "wall_ms": (time.perf_counter_ns() - wall_started) / 1e6,
        "timing": timing,
        "global_loss_last_100_mean": sum(global_losses[-100:]) / min(100, len(batches)),
        "local_loss_last_100_mean": sum(local_losses[-100:]) / min(100, len(batches)),
        "checkpoints": checkpoints,
    }


def evaluate_local_code(
    model: DiagnosticModel,
    batches: Sequence[DelayedBatch],
    codebook: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        for batch in batches:
            _, logits = _local_logits(model, batch, codebook)
            prediction = logits.argmax(dim=-1)
            total += int(prediction.numel())
            correct += int((prediction == batch.targets).sum().item())
    return {"count": total, "accuracy": correct / total}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_lg0_local_code.json"),
    )
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.updates = 2
    if args.updates <= 1 or args.threads <= 0:
        parser.error("updates must exceed one and threads must be positive")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.threads)
    timing_warmup = 1 if args.quick else 100
    test_count = 1 if args.quick else 64
    train_batches = tuple(
        generate_batch(
            seed=8_000_000 + update,
            batch_size=8,
            sequence_length=32,
            delays=(0, 1),
            direct=True,
        )
        for update in range(args.updates)
    )
    test_batches = tuple(
        generate_batch(
            seed=8_100_000 + index,
            batch_size=8,
            sequence_length=32,
            delays=(0, 1),
            direct=True,
        )
        for index in range(test_count)
    )
    models = build_models(8_000_000)
    counts = {name: count_parameters(model) for name, model in models.items()}
    lstm_parameters = counts["lstm"]
    if any(abs(value - lstm_parameters) / lstm_parameters > 0.02 for value in counts.values()):
        raise AssertionError(f"LG0 parameter fairness failed: {counts}")
    codebook = _codebook()
    results = {}
    for name in ("l1_global", "lstm", "transformer"):
        train = train_model(
            models[name], train_batches, learning_rate=1e-3, timing_warmup=timing_warmup
        )
        test = evaluate(models[name], test_batches, (0, 1))
        results[name] = {"train": train, "test": test}
    local_train = train_local(
        models["l1_local"],
        train_batches,
        codebook=codebook,
        timing_warmup=timing_warmup,
    )
    local_test = evaluate(models["l1_local"], test_batches, (0, 1))
    local_code_test = evaluate_local_code(models["l1_local"], test_batches, codebook)
    results["l1_local"] = {
        "train": local_train,
        "test": local_test,
        "local_code_test": local_code_test,
    }
    checks = {
        "l1_local_global_accuracy": local_test["accuracy"] >= 0.99,
        "lstm_validation": results["lstm"]["test"]["accuracy"] >= 0.99,
        "transformer_validation": results["transformer"]["test"]["accuracy"] >= 0.99,
    }
    codebook_hash = hashlib.sha256(codebook.numpy().tobytes()).hexdigest()
    result = {
        "schema_version": 1,
        "experiment": "E3-LG0 fixed local population code",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(torch.device("cpu")),
        "configuration": {
            "updates": args.updates,
            "threads": args.threads,
            "state_dim": STATE_DIM,
            "codebook_seed": CODEBOOK_SEED,
            "codebook_sha256": codebook_hash,
            "temperature": TEMPERATURE,
            "local_weight": LOCAL_WEIGHT,
            "timing_warmup": timing_warmup,
        },
        "parameter_counts": counts,
        "train_data_sha256": _dataset_hash(train_batches),
        "test_data_sha256": _dataset_hash(test_batches),
        "models": results,
        "decision": {
            "a0_gate": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "run_short_delay_next": all(checks.values()),
            "boundary": "The fixed local code loss is training-only and is not an online eligibility rule.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
