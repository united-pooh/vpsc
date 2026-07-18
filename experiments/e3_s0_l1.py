"""Single-layer exact-reset S0 population-code A0 and speed ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _autograd_node_count,
    _core_training_runner,
    _environment,
    _interleaved_samples,
    _sample_summary,
    _sync,
)
from experiments.e3_memory_diagnostic import (  # noqa: E402
    DiagnosticModel,
    _dataset_hash,
    _shared_initialisation,
    evaluate,
    generate_batch,
)
from experiments.e3_s0_delayed_copy import train_model  # noqa: E402
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3CumulativeScanCore,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
    state_nbytes,
)


STATE_DIM = 42


def build_a0_models(seed: int) -> Dict[str, DiagnosticModel]:
    shared = _shared_initialisation(seed + 50_000)
    factories = {
        "s0_l1": lambda: E3CumulativeScanCore(
            32,
            32,
            state_dim=STATE_DIM,
            num_layers=1,
            execution_mode="scan",
        ),
        "lstm": lambda: StatefulLSTMCore(32, 32),
        "transformer": lambda: CausalTransformerCore(
            32,
            32,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=32,
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


def run_a0() -> Dict[str, Any]:
    train_batches = tuple(
        generate_batch(
            seed=8_000_000 + update,
            batch_size=8,
            sequence_length=32,
            delays=(0, 1),
            direct=True,
        )
        for update in range(300)
    )
    test_batches = tuple(
        generate_batch(
            seed=8_100_000 + index,
            batch_size=8,
            sequence_length=32,
            delays=(0, 1),
            direct=True,
        )
        for index in range(64)
    )
    models = build_a0_models(8_000_000)
    counts = {name: count_parameters(model) for name, model in models.items()}
    lstm = counts["lstm"]
    if any(abs(value - lstm) / lstm > 0.02 for value in counts.values()):
        raise AssertionError(f"A0 parameter fairness failed: {counts}")
    results = {}
    for name in ("s0_l1", "lstm", "transformer"):
        train = train_model(
            models[name], train_batches, learning_rate=1e-3, timing_warmup=100
        )
        test = evaluate(models[name], test_batches, (0, 1))
        results[name] = {"train": train, "test": test}
    checks = {name: value["test"]["accuracy"] >= 0.99 for name, value in results.items()}
    return {
        "parameter_counts": counts,
        "train_data_sha256": _dataset_hash(train_batches),
        "test_data_sha256": _dataset_hash(test_batches),
        "models": results,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _suite(device: torch.device, seed: int) -> Dict[str, TemporalCore[Any]]:
    torch.manual_seed(seed)
    cores: Dict[str, TemporalCore[Any]] = {
        "s0_l1": E3CumulativeScanCore(
            32,
            32,
            state_dim=STATE_DIM,
            num_layers=1,
            execution_mode="scan",
        ),
        "s0_l2": E3CumulativeScanCore(
            32,
            32,
            state_dim=27,
            num_layers=2,
            execution_mode="scan",
        ),
        "lstm": StatefulLSTMCore(32, 32),
        "transformer": CausalTransformerCore(
            32,
            32,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=512,
        ),
    }
    return {name: core.to(device).train(True) for name, core in cores.items()}


def benchmark_training(
    *, threads: Sequence[int], warmup: int, repeats: int, device: torch.device
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        cores = _suite(device, 8_200_000 + thread_count)
        base = torch.randn(1, 512, 32, device=device)
        values = {name: base.detach().clone().requires_grad_(True) for name in cores}
        runners = {
            name: _core_training_runner(core, values[name])
            for name, core in cores.items()
        }
        nodes = {}
        for name, core in cores.items():
            output = core(values[name])
            nodes[name] = _autograd_node_count(output.sequence)
            core.zero_grad(set_to_none=True)
            values[name].grad = None
        samples = _interleaved_samples(
            runners,
            warmup=warmup,
            repeats=repeats,
            device=device,
            seed=8_300_000 + thread_count,
        )
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "models": {
                    name: {
                        **_sample_summary(sample, 512),
                        "parameters": count_parameters(cores[name]),
                        "state_bytes": state_nbytes(cores[name].initial_state(1)),
                        "autograd_nodes": nodes[name],
                    }
                    for name, sample in samples.items()
                },
            }
        )
    return records


def benchmark_streaming(
    *, threads: Sequence[int], warmup_steps: int, measured_steps: int, device: torch.device
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        cores = _suite(device, 8_400_000 + thread_count)
        for core in cores.values():
            core.eval()
        tokens = torch.randn(warmup_steps + measured_steps, 1, 32, device=device)
        states: Dict[str, Any] = {name: None for name in cores}
        samples: Dict[str, list[float]] = {name: [] for name in cores}
        generator = random.Random(8_500_000 + thread_count)
        with torch.inference_mode():
            for index in range(warmup_steps):
                for name, core in cores.items():
                    states[name] = core.step(tokens[index], states[name]).state
            names = list(cores)
            for index in range(warmup_steps, warmup_steps + measured_steps):
                generator.shuffle(names)
                for name in names:
                    _sync(device)
                    started = time.perf_counter_ns()
                    result = cores[name].step(tokens[index], states[name])
                    _sync(device)
                    samples[name].append((time.perf_counter_ns() - started) / 1e6)
                    states[name] = result.state
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "models": {
                    name: {
                        **_sample_summary(sample, 1),
                        "state_bytes": state_nbytes(states[name]),
                    }
                    for name, sample in samples.items()
                },
            }
        )
    return records


def _decision(
    a0: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    streaming: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    stream_by_thread = {str(record["threads"]): record for record in streaming}
    checks = {}
    for record in training:
        key = str(record["threads"])
        stream = stream_by_thread[key]
        checks[key] = {
            "train_l1_ms": record["models"]["s0_l1"]["p50_ms"],
            "train_lstm_ms": record["models"]["lstm"]["p50_ms"],
            "stream_l1_p95_ms": stream["models"]["s0_l1"]["p95_ms"],
            "stream_lstm_p95_ms": stream["models"]["lstm"]["p95_ms"],
        }
        checks[key]["passed"] = (
            checks[key]["train_l1_ms"] <= checks[key]["train_lstm_ms"]
            and checks[key]["stream_l1_p95_ms"] <= checks[key]["stream_lstm_p95_ms"]
        )
    return {
        "a0_gate": "PASS" if a0["passed"] else "FAIL",
        "ann_train_and_inference_speed_gate": (
            "PASS" if any(value["passed"] for value in checks.values()) else "FAIL"
        ),
        "ann_checks": checks,
        "run_short_delay_next": bool(a0["passed"]),
        "boundary": "L1 removes signed inter-layer communication and stores 31.25% more state than LSTM.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/e3_scan/e3_s0_l1.json")
    )
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--streaming-warmup", type=int, default=16)
    parser.add_argument("--streaming-steps", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.threads = args.threads[:1]
        args.warmup = 1
        args.repeats = 1
        args.streaming_warmup = 1
        args.streaming_steps = 2
    return args


def main() -> None:
    args = _parse_args()
    threads = tuple(dict.fromkeys(args.threads))
    torch.set_num_threads(threads[0])
    device = torch.device("cpu")
    a0 = None if args.quick else run_a0()
    training = benchmark_training(
        threads=threads, warmup=args.warmup, repeats=args.repeats, device=device
    )
    streaming = benchmark_streaming(
        threads=threads,
        warmup_steps=args.streaming_warmup,
        measured_steps=args.streaming_steps,
        device=device,
    )
    decision = (
        {"a0_gate": "NOT_RUN", "ann_train_and_inference_speed_gate": "SMOKE"}
        if a0 is None
        else _decision(a0, training, streaming)
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-S0 single-layer population code",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "state_dim": STATE_DIM,
            "threads": threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "streaming_warmup": args.streaming_warmup,
            "streaming_steps": args.streaming_steps,
        },
        "a0": a0,
        "training_t512": training,
        "streaming": streaming,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
