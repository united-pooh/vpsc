"""AT2 scan-aligned eligibility with deterministic BF16 gradient projection."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
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
    _sync,
)
from experiments.e3_at0_gated_trace import (  # noqa: E402
    STATE_DIM,
    _event_diagnostics,
)
from experiments.e3_at1_trace_eligibility import (  # noqa: E402
    AT1TokenModel,
    benchmark_saved_tensors,
    benchmark_streaming,
    benchmark_training,
    run_equivalence,
)
from experiments.e3_el0_terminal_eligibility import (  # noqa: E402
    _dataset_hash,
    _parameter_max_abs,
)
from experiments.e3_el1_multi_query_eligibility import (  # noqa: E402
    D_MODEL,
    PAYLOAD_VOCAB,
    QUERY_INDICES,
    SEQUENCE_LENGTH,
    MultiQueryTokenModel,
    _evaluate_model,
    _initialise_shared_wrapper,
    _shared_wrapper_state,
    _train_model,
    generate_register_batch,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3GatedTraceScanCore,
    StatefulLSTMCore,
    count_parameters,
)


def _build_quality_models(
    seed: int, device: torch.device
) -> Dict[str, nn.Module]:
    shared = _shared_wrapper_state(8_930_001)
    torch.manual_seed(seed)
    at0 = AT1TokenModel(
        E3GatedTraceScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    )
    _initialise_shared_wrapper(at0, shared)  # type: ignore[arg-type]
    at2 = copy.deepcopy(at0)
    if not isinstance(at2.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("AT2 core mismatch")
    at2.core.eligibility_forward_mode = "scan_aligned"
    at2.use_trace_eligibility = True
    torch.manual_seed(seed + 1)
    lstm = MultiQueryTokenModel(StatefulLSTMCore(D_MODEL, D_MODEL))
    _initialise_shared_wrapper(lstm, shared)
    torch.manual_seed(seed + 2)
    transformer = MultiQueryTokenModel(
        CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=SEQUENCE_LENGTH,
        )
    )
    _initialise_shared_wrapper(transformer, shared)
    return {
        "at0_bptt_bf16_gradient": at0.to(device),
        "at2_eligibility_bf16_gradient": at2.to(device),
        "lstm": lstm.to(device),
        "transformer": transformer.to(device),
    }


def _canonical_gradients(
    model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], bool]:
    snapshots: Dict[str, torch.Tensor] = {}
    finite = True
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            finite = finite and bool(torch.isfinite(parameter.grad).all())
            parameter.grad.copy_(
                parameter.grad.to(dtype=torch.bfloat16).to(dtype=parameter.dtype)
            )
            snapshots[name] = parameter.grad.detach().clone()
    return snapshots, finite


def _compare_gradient_snapshots(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> Tuple[int, int, float]:
    if left.keys() != right.keys():  # pragma: no cover
        raise AssertionError("paired models exposed different gradient tensors")
    mismatches = 0
    elements = 0
    maximum = 0.0
    for name, left_value in left.items():
        right_value = right[name]
        mismatches += int((left_value != right_value).sum().item())
        elements += left_value.numel()
        maximum = max(
            maximum,
            float((left_value - right_value).abs().max().item()),
        )
    return mismatches, elements, maximum


def _train_canonical_pair(
    at0: nn.Module,
    at2: nn.Module,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    timing_warmup: int,
    device: torch.device,
) -> Dict[str, Any]:
    names = ("at0_bptt_bf16_gradient", "at2_eligibility_bf16_gradient")
    models = dict(zip(names, (at0, at2)))
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        for name, model in models.items()
    }
    for model in models.values():
        model.train(True)
    losses: Dict[str, list[float]] = {name: [] for name in names}
    timings: Dict[str, list[float]] = {name: [] for name in names}
    checkpoints: Dict[str, list[Dict[str, Any]]] = {name: [] for name in names}
    mismatch_count = 0
    compared_elements = 0
    maximum_quantized_gradient_difference = 0.0
    first_mismatch_update = None
    all_finite = True

    for update, (cpu_tokens, cpu_targets) in enumerate(batches, start=1):
        tokens = cpu_tokens.to(device)
        targets = cpu_targets.to(device)
        gradient_snapshots = {}
        for name in names:
            model = models[name]
            optimizer = optimizers[name]
            _sync(device)
            started = time.perf_counter_ns()
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, PAYLOAD_VOCAB), targets.reshape(-1)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite AT2 loss for {name} at update {update}"
                )
            loss.backward()
            snapshot, finite = _canonical_gradients(model)
            all_finite = all_finite and finite
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _sync(device)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            if update > timing_warmup:
                timings[name].append(elapsed_ms)
            losses[name].append(float(loss.detach().item()))
            gradient_snapshots[name] = snapshot
            if update == 1 or update % 100 == 0 or update == len(batches):
                checkpoints[name].append(
                    {
                        "update": update,
                        "loss": losses[name][-1],
                        "gradient_norm_after_projection": float(gradient_norm),
                    }
                )
        mismatches, elements, maximum = _compare_gradient_snapshots(
            gradient_snapshots[names[0]], gradient_snapshots[names[1]]
        )
        mismatch_count += mismatches
        compared_elements += elements
        maximum_quantized_gradient_difference = max(
            maximum_quantized_gradient_difference, maximum
        )
        if mismatches and first_mismatch_update is None:
            first_mismatch_update = update

    model_records = {
        name: {
            "updates": len(batches),
            "loss_first": losses[name][0],
            "loss_last": losses[name][-1],
            "loss_last_100_mean": sum(losses[name][-100:])
            / min(100, len(losses[name])),
            "timing": _sample_summary(timings[name], batches[0][0].numel()),
            "checkpoints": checkpoints[name],
        }
        for name in names
    }
    loss_max_abs = max(
        abs(left - right)
        for left, right in zip(losses[names[0]], losses[names[1]])
    )
    return {
        "models": model_records,
        "loss_max_abs": loss_max_abs,
        "quantized_gradient_mismatch_count": mismatch_count,
        "quantized_gradient_compared_elements": compared_elements,
        "quantized_gradient_mismatch_rate": mismatch_count / compared_elements,
        "quantized_gradient_max_abs": maximum_quantized_gradient_difference,
        "first_mismatch_update": first_mismatch_update,
        "all_gradients_finite": all_finite,
    }


def run_quality(*, quick: bool, device: torch.device) -> Dict[str, Any]:
    seeds = (0,) if quick else (0, 1, 2)
    updates = 3 if quick else 600
    train_batch_size = 4 if quick else 32
    test_count = 64 if quick else 4096
    records = []
    paired_names = (
        "at0_bptt_bf16_gradient",
        "at2_eligibility_bf16_gradient",
    )
    for seed in seeds:
        train_batches = tuple(
            generate_register_batch(
                seed=8_930_000 + 10_000 * seed + update,
                batch_size=train_batch_size,
            )
            for update in range(updates)
        )
        test_tokens, test_targets = generate_register_batch(
            seed=8_990_000 + seed, batch_size=test_count
        )
        models = _build_quality_models(9_150_000 + 100 * seed, device)
        parameter_counts = {
            name: count_parameters(model) for name, model in models.items()
        }
        lstm_count = parameter_counts["lstm"]
        fairness = {
            name: abs(count - lstm_count) / lstm_count <= 0.02
            for name, count in parameter_counts.items()
        }
        if not all(fairness.values()):
            raise AssertionError(f"AT2 parameter fairness failed: {parameter_counts}")
        paired = _train_canonical_pair(
            models[paired_names[0]],
            models[paired_names[1]],
            train_batches,
            timing_warmup=min(100, updates - 1),
            device=device,
        )
        control_training = {}
        for name in ("lstm", "transformer"):
            control = _train_model(
                models[name],  # type: ignore[arg-type]
                train_batches,
                timing_warmup=min(100, updates - 1),
                device=device,
            )
            control.pop("losses")
            control_training[name] = control
        evaluations = {
            name: _evaluate_model(
                model,  # type: ignore[arg-type]
                test_tokens,
                test_targets,
                batch_size=256,
                device=device,
            )
            for name, model in models.items()
        }
        parameter_max_abs, parameter_errors = _parameter_max_abs(
            models[paired_names[1]], models[paired_names[0]]
        )
        at2_model = models[paired_names[1]]
        if not isinstance(at2_model, AT1TokenModel):  # pragma: no cover
            raise TypeError("AT2 quality model mismatch")
        records.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_fairness": fairness,
                "train_data_sha256": _dataset_hash(train_batches),
                "paired_training": paired,
                "control_training": control_training,
                "test": evaluations,
                "parameter_max_abs": parameter_max_abs,
                "parameter_errors": parameter_errors,
                "event_diagnostics": _event_diagnostics(
                    at2_model, test_tokens[:256], device=device  # type: ignore[arg-type]
                ),
            }
        )
    if quick:
        task_valid = False
        canonical_pass = False
        quality_pass = False
    else:
        task_valid = all(
            record["test"][name]["accuracy"] >= 0.99
            for record in records
            for name in ("lstm", "transformer")
        )
        canonical_pass = all(
            record["paired_training"]["quantized_gradient_mismatch_rate"]
            <= 1e-5
            and record["paired_training"]["all_gradients_finite"]
            for record in records
        )
        quality_pass = task_valid and canonical_pass and all(
            record["test"][name]["accuracy"] == 1.0
            for record in records
            for name in paired_names
        ) and all(
            record["paired_training"]["loss_max_abs"] <= 1e-3
            and record["parameter_max_abs"] <= 5e-3
            for record in records
        )
    return {
        "formal": not quick,
        "task": {
            "sequence_length": SEQUENCE_LENGTH,
            "query_indices": QUERY_INDICES,
            "delay": 4,
            "payload_classes": PAYLOAD_VOCAB,
            "gradient_projection": "float32(bfloat16(gradient)) before clip",
            "optimizer_parameters_and_moments": "float32",
            "updates": updates,
            "train_batch_size": train_batch_size,
            "test_sequences_per_seed": test_count,
        },
        "seeds": records,
        "task_validation": "PASS" if task_valid else "NOT_RUN" if quick else "FAIL",
        "canonical_gate": "PASS" if canonical_pass else "NOT_RUN" if quick else "FAIL",
        "passed": quality_pass,
    }


def _decision(
    *,
    equivalence: Mapping[str, Any],
    memory: Mapping[str, Any],
    training: Mapping[str, Any],
    streaming: Mapping[str, Any],
    quality: Mapping[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
            "memory_gate": "PASS" if memory["passed"] else "FAIL",
            "canonical_gate": "NOT_RUN",
            "quality_gate": "NOT_RUN",
            "speed_gate": "SMOKE",
            "stream_gate": "SMOKE",
            "ann_gate": "SMOKE",
            "overall": "SMOKE",
            "run_action_language_next": False,
        }
    stream_by_thread = {
        record["threads"]: record["passed"] for record in streaming["records"]
    }
    ann_pass = any(
        record["passed"] and stream_by_thread.get(record["threads"], False)
        for record in training["records"]
    )
    gates = {
        "equivalence_gate": bool(equivalence["passed"]),
        "memory_gate": bool(memory["passed"]),
        "canonical_gate": quality.get("canonical_gate") == "PASS",
        "quality_gate": bool(quality["passed"]),
        "speed_gate": bool(training["passed"]),
        "stream_gate": bool(streaming["passed"]),
        "ann_gate": ann_pass,
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "overall": overall,
        "run_action_language_next": overall == "PASS",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_at2_bf16_canonical.json"),
    )
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--stream-warmup", type=int, default=64)
    parser.add_argument("--stream-steps", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.threads = args.threads[:1]
        args.warmup = 1
        args.repeats = 1
        args.stream_warmup = 4
        args.stream_steps = 32
    return args


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    threads = tuple(dict.fromkeys(args.threads))
    if device.type == "cpu":
        torch.set_num_threads(threads[0])
    equivalence = run_equivalence(
        device, eligibility_forward_mode="scan_aligned"
    )
    if equivalence["passed"]:
        memory = benchmark_saved_tensors(
            device, eligibility_forward_mode="scan_aligned"
        )
        training = benchmark_training(
            threads=threads,
            lengths=(512,) if args.quick else (512, 2048),
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
            eligibility_forward_mode="scan_aligned",
        )
        streaming = benchmark_streaming(
            threads=threads,
            warmup_steps=args.stream_warmup,
            measured_steps=args.stream_steps,
            device=device,
        )
        if device.type == "cpu":
            torch.set_num_threads(threads[0] if args.quick else 4)
        quality = run_quality(quick=args.quick, device=device)
    else:
        memory = {"records": [], "passed": False, "not_run": "EQ failed"}
        training = {"records": [], "passed": False, "not_run": "EQ failed"}
        streaming = {"records": [], "passed": False, "not_run": "EQ failed"}
        quality = {"passed": False, "not_run": "EQ failed"}
    decision = _decision(
        equivalence=equivalence,
        memory=memory,
        training=training,
        streaming=streaming,
        quality=quality,
        quick=args.quick,
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-AT2 scan-aligned eligibility plus BF16 gradient projection",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "threads": threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "stream_warmup": args.stream_warmup,
            "stream_steps": args.stream_steps,
            "eligibility_forward_mode": "scan_aligned",
            "gradient_projection": "bfloat16",
        },
        "equivalence": equivalence,
        "saved_tensors": memory,
        "training": training,
        "streaming": streaming,
        "quality": quality,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
