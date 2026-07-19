"""SG25A stable blocked closed-form scan and reverse-adjoint CUDA study."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _environment,
    _sample_summary,
    _sync,
)
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_ra0_reverse_adjoint import build_textworld_models  # noqa: E402
from vpsc.world_model.cores import E3GatedTraceScanCore  # noqa: E402


BLOCK_SIZES = (32, 64, 128)
LENGTHS = (48, 80, 128, 134, 512)
FORWARD_ATOL = 2e-6
FORWARD_RTOL = 2e-5
GRAD_ATOL = 3e-5
GRAD_RTOL = 3e-4
SG24_LSTM_P50_MS = 2.4325648333333336
EXPECTED_SG24_SHA256 = (
    "d940421bd0ac9c07dee623e93547ec3d17b025064e22ec26fe01a3e53f1c6067"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return math.inf
    return float((left - right).abs().max().item())


def _close(left: torch.Tensor, right: torch.Tensor, *, gradient: bool = False) -> bool:
    atol = GRAD_ATOL if gradient else FORWARD_ATOL
    rtol = GRAD_RTOL if gradient else FORWARD_RTOL
    return bool(
        torch.isfinite(left).all()
        and torch.isfinite(right).all()
        and torch.allclose(left, right, atol=atol, rtol=rtol)
    )


def _decays(profile: str, *, device: torch.device) -> torch.Tensor:
    if profile == "initial_grid":
        base = torch.linspace(0.55, 0.99, steps=tw0.STATE_DIM, device=device)
        return torch.cat((base, base.flip(0)))
    if profile == "boundary_stress":
        return torch.linspace(
            0.50, 0.995, steps=2 * tw0.STATE_DIM, device=device
        )
    raise ValueError(f"unknown decay profile: {profile}")


def _legacy_adjoint(
    impulses: torch.Tensor, decay: torch.Tensor
) -> torch.Tensor:
    coefficient = decay.view(1, 1, -1).expand_as(impulses)
    return E3GatedTraceScanCore._affine_prefix_scan(
        coefficient,
        impulses,
        torch.zeros_like(impulses[:, 0]),
    )


def primitive_audit(device: torch.device) -> Dict[str, Any]:
    records = []
    for block_size in BLOCK_SIZES:
        for length in LENGTHS:
            for profile_index, profile in enumerate(
                ("initial_grid", "boundary_stress")
            ):
                generator = torch.Generator(device=device).manual_seed(
                    25_100_000 + block_size * 10_000 + length * 10 + profile_index
                )
                decay = _decays(profile, device=device)
                write = torch.randint(
                    0,
                    2,
                    (1, length, decay.numel()),
                    generator=generator,
                    device=device,
                ).to(torch.float32)
                initial = torch.rand(
                    1, decay.numel(), generator=generator, device=device
                )
                impulses = 0.5 * torch.randn(
                    1,
                    length,
                    decay.numel(),
                    generator=generator,
                    device=device,
                )
                legacy_trace = E3GatedTraceScanCore._constant_affine_prefix_scan(
                    write, decay, initial
                )
                candidate_trace = (
                    E3GatedTraceScanCore._blocked_constant_affine_prefix_scan(
                        write,
                        decay,
                        initial,
                        block_size=block_size,
                    )
                )
                reversed_impulses = impulses.flip(1)
                legacy_adjoint = _legacy_adjoint(reversed_impulses, decay)
                candidate_adjoint = (
                    E3GatedTraceScanCore._blocked_constant_affine_prefix_scan(
                        reversed_impulses,
                        decay,
                        torch.zeros_like(initial),
                        block_size=block_size,
                        injection_scale=torch.ones_like(decay),
                    )
                )
                spike_disagreements = int(
                    ((legacy_trace >= 0.5) != (candidate_trace >= 0.5))
                    .sum()
                    .item()
                )
                trace_pass = _close(candidate_trace, legacy_trace)
                adjoint_pass = _close(candidate_adjoint, legacy_adjoint)
                finite = bool(
                    torch.isfinite(candidate_trace).all()
                    and torch.isfinite(candidate_adjoint).all()
                )
                records.append(
                    {
                        "block_size": block_size,
                        "length": length,
                        "decay_profile": profile,
                        "finite": finite,
                        "trace_max_abs": _max_abs(candidate_trace, legacy_trace),
                        "adjoint_max_abs": _max_abs(
                            candidate_adjoint, legacy_adjoint
                        ),
                        "trace_close": trace_pass,
                        "adjoint_close": adjoint_pass,
                        "spike_disagreements": spike_disagreements,
                        "passed": finite
                        and trace_pass
                        and adjoint_pass
                        and spike_disagreements == 0,
                    }
                )
    by_block = {
        str(block_size): {
            "passed": all(
                record["passed"]
                for record in records
                if record["block_size"] == block_size
            ),
            "maximum_trace_abs": max(
                record["trace_max_abs"]
                for record in records
                if record["block_size"] == block_size
            ),
            "maximum_adjoint_abs": max(
                record["adjoint_max_abs"]
                for record in records
                if record["block_size"] == block_size
            ),
        }
        for block_size in BLOCK_SIZES
    }
    return {"records": records, "by_block": by_block}


def primitive_benchmark(
    device: torch.device, *, warmup: int, repeats: int
) -> Dict[str, Any]:
    records = []
    for length in LENGTHS:
        generator = torch.Generator(device=device).manual_seed(25_200_000 + length)
        decay = _decays("initial_grid", device=device)
        write = torch.randint(
            0,
            2,
            (1, length, decay.numel()),
            generator=generator,
            device=device,
        ).to(torch.float32)
        initial = torch.rand(1, decay.numel(), generator=generator, device=device)
        reversed_impulses = torch.randn(
            1,
            length,
            decay.numel(),
            generator=generator,
            device=device,
        )

        def legacy() -> None:
            E3GatedTraceScanCore._constant_affine_prefix_scan(
                write, decay, initial
            )
            _legacy_adjoint(reversed_impulses, decay)

        runners = {"legacy": legacy}
        for block_size in BLOCK_SIZES:

            def candidate(size: int = block_size) -> None:
                E3GatedTraceScanCore._blocked_constant_affine_prefix_scan(
                    write, decay, initial, block_size=size
                )
                E3GatedTraceScanCore._blocked_constant_affine_prefix_scan(
                    reversed_impulses,
                    decay,
                    torch.zeros_like(initial),
                    block_size=size,
                    injection_scale=torch.ones_like(decay),
                )

            runners[f"block_{block_size}"] = candidate
        for _ in range(warmup):
            for runner in runners.values():
                runner()
        samples = {name: [] for name in runners}
        order = list(runners)
        randomizer = random.Random(25_210_000 + length)
        for _ in range(repeats):
            randomizer.shuffle(order)
            for name in order:
                _sync(device)
                started = time.perf_counter_ns()
                runners[name]()
                _sync(device)
                samples[name].append((time.perf_counter_ns() - started) / 1e6)
        summaries = {
            name: _sample_summary(values, length) for name, values in samples.items()
        }
        for block_size in BLOCK_SIZES:
            summaries[f"block_{block_size}"]["versus_legacy_speedup"] = (
                summaries["legacy"]["p50_ms"]
                / summaries[f"block_{block_size}"]["p50_ms"]
            )
        records.append({"length": length, "models": summaries})
    return {"warmup": warmup, "repeats": repeats, "records": records}


def full_core_equivalence(device: torch.device) -> Dict[str, Any]:
    records = []
    for block_size in BLOCK_SIZES:
        torch.manual_seed(25_300_000 + block_size)
        legacy = E3GatedTraceScanCore(
            tw0.D_MODEL,
            tw0.D_MODEL,
            state_dim=tw0.STATE_DIM,
            eligibility_backward_mode="reverse_adjoint",
        ).to(device)
        candidate = E3GatedTraceScanCore(
            tw0.D_MODEL,
            tw0.D_MODEL,
            state_dim=tw0.STATE_DIM,
            eligibility_backward_mode="reverse_adjoint",
            scan_math_mode="blocked_cumsum",
            scan_block_size=block_size,
        ).to(device)
        candidate.load_state_dict(legacy.state_dict())
        legacy_x = torch.randn(
            2, 134, tw0.D_MODEL, device=device, requires_grad=True
        )
        candidate_x = legacy_x.detach().clone().requires_grad_(True)
        queries = torch.tensor(
            (0, 7, 31, 63, 96, 119, 133), dtype=torch.long, device=device
        )
        legacy_output = legacy.forward_multi_query_eligibility(
            legacy_x, queries, _unchecked=True
        )
        candidate_output = candidate.forward_multi_query_eligibility(
            candidate_x, queries, _unchecked=True
        )
        probe = torch.randn_like(legacy_output.sequence)
        legacy_loss = (legacy_output.sequence * probe).sum() + 0.1 * (
            legacy_output.state.layers[0].excitatory.sum()
            - legacy_output.state.layers[0].inhibitory.sum()
        )
        candidate_loss = (candidate_output.sequence * probe).sum() + 0.1 * (
            candidate_output.state.layers[0].excitatory.sum()
            - candidate_output.state.layers[0].inhibitory.sum()
        )
        legacy_loss.backward()
        candidate_loss.backward()
        checks: Dict[str, Dict[str, Any]] = {
            "sequence": {
                "max_abs": _max_abs(
                    candidate_output.sequence, legacy_output.sequence
                ),
                "passed": _close(
                    candidate_output.sequence, legacy_output.sequence
                ),
            },
            "state_e": {
                "max_abs": _max_abs(
                    candidate_output.state.layers[0].excitatory,
                    legacy_output.state.layers[0].excitatory,
                ),
                "passed": _close(
                    candidate_output.state.layers[0].excitatory,
                    legacy_output.state.layers[0].excitatory,
                ),
            },
            "state_i": {
                "max_abs": _max_abs(
                    candidate_output.state.layers[0].inhibitory,
                    legacy_output.state.layers[0].inhibitory,
                ),
                "passed": _close(
                    candidate_output.state.layers[0].inhibitory,
                    legacy_output.state.layers[0].inhibitory,
                ),
            },
            "input_gradient": {
                "max_abs": _max_abs(candidate_x.grad, legacy_x.grad),
                "passed": _close(candidate_x.grad, legacy_x.grad, gradient=True),
            },
        }
        for (legacy_name, legacy_parameter), (
            candidate_name,
            candidate_parameter,
        ) in zip(legacy.named_parameters(), candidate.named_parameters()):
            if legacy_name != candidate_name:  # pragma: no cover
                raise AssertionError("parameter order mismatch")
            checks[f"gradient:{legacy_name}"] = {
                "max_abs": _max_abs(
                    candidate_parameter.grad, legacy_parameter.grad
                ),
                "passed": _close(
                    candidate_parameter.grad,
                    legacy_parameter.grad,
                    gradient=True,
                ),
            }
        with torch.inference_mode():
            _, legacy_trace = legacy.forward_dynamics(legacy_x.detach())
            _, candidate_trace = candidate.forward_dynamics(candidate_x.detach())
        spike_disagreements = int(
            (
                legacy_trace.excitatory_spikes
                != candidate_trace.excitatory_spikes
            ).sum().item()
            + (
                legacy_trace.inhibitory_spikes
                != candidate_trace.inhibitory_spikes
            ).sum().item()
        )
        records.append(
            {
                "block_size": block_size,
                "checks": checks,
                "spike_disagreements": spike_disagreements,
                "passed": all(check["passed"] for check in checks.values())
                and spike_disagreements == 0,
            }
        )
    return {
        "records": records,
        "by_block": {
            str(record["block_size"]): record["passed"] for record in records
        },
    }


def _ra0_model(
    vocabulary: Any,
    *,
    device: torch.device,
    block_size: int | None,
) -> Any:
    models = build_textworld_models(9_400_000, vocabulary, device=device)
    model = models["snn_ra0"]
    del models
    if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("RA0 builder returned the wrong core")
    if block_size is not None:
        model.core.scan_math_mode = "blocked_cumsum"
        model.core.scan_block_size = block_size
    return model


def _spike_signature(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    core = model.core
    if not isinstance(core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("spike signature requires a gated trace core")
    with torch.no_grad():
        embedded = model.embedding(input_ids)
        _, _, _, _, write_e, write_i = core.input_events(embedded)
        decay_e, decay_i = core.decays()
        state = core.initial_state(input_ids.shape[0], device=input_ids.device)
        trace_e = core._trace(write_e, decay_e, state.layers[0].excitatory)
        trace_i = core._trace(write_i, decay_i, state.layers[0].inhibitory)
        return torch.cat((trace_e >= 0.5, trace_i >= 0.5), dim=-1)


def _optimizer(model: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=0.01,
        fused=True,
    )


def _update(
    model: Any,
    optimizer: torch.optim.Optimizer,
    example: Any,
    *,
    device: torch.device,
) -> float:
    input_ids, query_indices, targets = sg0._example_tensors(
        example, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    logits, _ = tw0._sparse_forward(
        model,
        input_ids,
        query_indices,
        None,
        use_eligibility=True,
        detach_state=True,
    )
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite SG25A training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        1.0,
        foreach=True,
    )
    optimizer.step()
    return float(loss.detach().item())


def trajectory_audit(
    examples: Sequence[Any], vocabulary: Any, *, device: torch.device
) -> Dict[str, Any]:
    schedule = sg0._training_schedule(len(examples), 1, 9_401_000)[:20]
    records = []
    for block_size in BLOCK_SIZES:
        legacy = _ra0_model(vocabulary, device=device, block_size=None)
        candidate = copy.deepcopy(legacy)
        candidate.core.scan_math_mode = "blocked_cumsum"
        candidate.core.scan_block_size = block_size
        legacy_optimizer = _optimizer(legacy)
        candidate_optimizer = _optimizer(candidate)
        loss_differences = []
        spike_disagreements = 0
        for example_index in schedule:
            example = examples[example_index]
            input_ids, _, _ = sg0._example_tensors(example, device=device)
            spike_disagreements += int(
                (
                    _spike_signature(legacy, input_ids)
                    != _spike_signature(candidate, input_ids)
                ).sum().item()
            )
            legacy_loss = _update(
                legacy, legacy_optimizer, example, device=device
            )
            candidate_loss = _update(
                candidate, candidate_optimizer, example, device=device
            )
            loss_differences.append(abs(candidate_loss - legacy_loss))
        parameter_max_abs = max(
            _max_abs(candidate_parameter, legacy_parameter)
            for legacy_parameter, candidate_parameter in zip(
                legacy.parameters(), candidate.parameters()
            )
        )
        records.append(
            {
                "block_size": block_size,
                "updates": len(schedule),
                "maximum_loss_abs": max(loss_differences),
                "final_parameter_max_abs": parameter_max_abs,
                "spike_disagreements": spike_disagreements,
                "passed": max(loss_differences) <= 2e-4
                and parameter_max_abs <= 3e-4
                and spike_disagreements == 0,
            }
        )
        del legacy, candidate, legacy_optimizer, candidate_optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "records": records,
        "by_block": {
            str(record["block_size"]): record["passed"] for record in records
        },
    }


def update_benchmark(
    examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    schedule = sg0._training_schedule(len(examples), epochs, 9_402_000)
    order: Tuple[Tuple[str, int | None], ...] = (
        ("legacy", None),
        ("block_32", 32),
        ("block_64", 64),
        ("block_128", 128),
    )
    records: Dict[str, Any] = {}
    for name, block_size in order:
        gc.collect()
        torch.cuda.empty_cache()
        model = _ra0_model(vocabulary, device=device, block_size=block_size)
        optimizer = _optimizer(model)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_start = torch.cuda.memory_allocated(device)
        samples = []
        losses = []
        for example_index in schedule:
            _sync(device)
            started = time.perf_counter_ns()
            loss = _update(
                model, optimizer, examples[example_index], device=device
            )
            _sync(device)
            samples.append((time.perf_counter_ns() - started) / 1e6)
            losses.append(loss)
        warmup_updates = len(samples) // 5
        timing = _sample_summary(samples[warmup_updates:], 1)
        records[name] = {
            "epochs": epochs,
            "updates": len(schedule),
            "warmup_updates_excluded": warmup_updates,
            "timing": timing,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "cuda_allocated_start_bytes": allocated_start,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_additional_peak_bytes": max(
                0, torch.cuda.max_memory_allocated(device) - allocated_start
            ),
        }
        del model, optimizer
    for block_size in BLOCK_SIZES:
        name = f"block_{block_size}"
        records[name]["versus_legacy_speedup"] = (
            records["legacy"]["timing"]["p50_ms"]
            / records[name]["timing"]["p50_ms"]
        )
        records[name]["versus_sg24_lstm_ratio"] = (
            records[name]["timing"]["p50_ms"] / SG24_LSTM_P50_MS
        )
        legacy_peak = records["legacy"]["cuda_additional_peak_bytes"]
        candidate_peak = records[name]["cuda_additional_peak_bytes"]
        records[name]["additional_peak_to_legacy_ratio"] = (
            candidate_peak / legacy_peak if legacy_peak else None
        )
    return {"records": records}


def profiler_audit(
    examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
    block_size: int | None,
) -> Dict[str, Any]:
    model = _ra0_model(vocabulary, device=device, block_size=block_size)
    optimizer = _optimizer(model)
    try:
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            )
        ) as profile:
            _update(model, optimizer, examples[0], device=device)
            _sync(device)
        events = profile.events()
        cuda_events = [
            event
            for event in events
            if "cuda" in str(getattr(event, "device_type", "")).lower()
        ]
        return {
            "status": "PASS",
            "cuda_event_count": len(cuda_events),
            "unique_cuda_event_names": len(
                {str(getattr(event, "name", "")) for event in cuda_events}
            ),
            "cpu_operator_event_count": len(events) - len(cuda_events),
        }
    except Exception as error:  # pragma: no cover - environment dependent
        return {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        del model, optimizer


def _decision(
    *,
    primitive: Mapping[str, Any],
    full_core: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    update: Mapping[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    candidates = []
    block_records = {}
    for block_size in BLOCK_SIZES:
        key = str(block_size)
        timing = update["records"][f"block_{block_size}"]
        memory_ratio = timing["additional_peak_to_legacy_ratio"]
        numerical_pass = bool(primitive["by_block"][key]["passed"])
        full_core_pass = bool(full_core["by_block"][key])
        trajectory_pass = bool(trajectory["by_block"][key])
        memory_pass = memory_ratio is not None and memory_ratio <= 1.25
        eligible = numerical_pass and full_core_pass and trajectory_pass and memory_pass
        block_records[key] = {
            "primitive_pass": numerical_pass,
            "full_core_pass": full_core_pass,
            "trajectory_pass": trajectory_pass,
            "memory_pass": memory_pass,
            "eligible": eligible,
            "update_p50_ms": timing["timing"]["p50_ms"],
            "versus_legacy_speedup": timing["versus_legacy_speedup"],
            "versus_sg24_lstm_ratio": timing["versus_sg24_lstm_ratio"],
            "additional_peak_to_legacy_ratio": memory_ratio,
        }
        if eligible:
            candidates.append((timing["timing"]["p50_ms"], block_size))
    selected = min(candidates)[1] if candidates else None
    if quick:
        return {
            "blocks": block_records,
            "selected_block_size": selected,
            "overall": "SMOKE" if selected is not None else "FAIL",
        }
    if selected is None:
        numerical_pass = any(
            record["primitive_pass"] and record["full_core_pass"]
            for record in block_records.values()
        )
        trajectory_pass = any(
            record["trajectory_pass"] for record in block_records.values()
        )
        memory_pass = all(
            record["memory_pass"] for record in block_records.values()
        )
        return {
            "blocks": block_records,
            "selected_block_size": None,
            "numerical_gate": "PASS" if numerical_pass else "FAIL",
            "trajectory_gate": "PASS" if trajectory_pass else "FAIL",
            "memory_gate": "PASS" if memory_pass else "FAIL",
            "mathematical_acceleration_gate": "FAIL",
            "ann_speed_target": "FAIL",
            "overall": "FAIL",
            "next_route": "native_fused_scan",
        }
    selected_record = block_records[str(selected)]
    speed_pass = selected_record["versus_legacy_speedup"] >= 1.5
    ann_speed_pass = selected_record["versus_sg24_lstm_ratio"] <= 1.0
    return {
        "blocks": block_records,
        "selected_block_size": selected,
        "numerical_gate": "PASS",
        "trajectory_gate": "PASS",
        "memory_gate": "PASS",
        "mathematical_acceleration_gate": "PASS" if speed_pass else "FAIL",
        "ann_speed_target": "PASS" if ann_speed_pass else "FAIL",
        "overall": "PASS" if speed_pass else "FAIL",
        "next_route": (
            "expanded_real_corpus" if speed_pass else "native_fused_scan"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG25A requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG25A requires the frozen V100 backend")
    reference = ROOT / "results/e3_scan/e3_sg24_cuda_counterfactual_generation.json"
    reference_sha = _sha256(reference)
    if reference_sha != EXPECTED_SG24_SHA256:
        raise AssertionError("SG25A SG24 reference hash mismatch")
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = sg0.load_event_corpus(corpus_root)
    examples, vocabulary = sg0.build_counterfactual_examples(corpus_root, corpus)
    data_audit = sg0.audit_examples(examples, vocabulary)
    if not data_audit["passed"]:
        raise AssertionError("SG25A frozen SG24 data audit failed")

    primitive = primitive_audit(device)
    primitive_speed = primitive_benchmark(
        device, warmup=args.warmup, repeats=args.repeats
    )
    full_core = full_core_equivalence(device)
    trajectory = trajectory_audit(
        examples["train"], vocabulary, device=device
    )
    update = update_benchmark(
        examples["train"],
        vocabulary,
        device=device,
        epochs=args.update_epochs,
    )
    decision = _decision(
        primitive=primitive,
        full_core=full_core,
        trajectory=trajectory,
        update=update,
        quick=args.quick,
    )
    selected = decision["selected_block_size"]
    profiler = {
        "legacy": profiler_audit(
            examples["train"], vocabulary, device=device, block_size=None
        ),
        "selected": (
            profiler_audit(
                examples["train"],
                vocabulary,
                device=device,
                block_size=int(selected),
            )
            if selected is not None
            else None
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "E3-SG25A stable blocked closed-form CUDA scan",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "block_sizes": BLOCK_SIZES,
            "lengths": LENGTHS,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "update_epochs": args.update_epochs,
            "forward_atol": FORWARD_ATOL,
            "forward_rtol": FORWARD_RTOL,
            "gradient_atol": GRAD_ATOL,
            "gradient_rtol": GRAD_RTOL,
            "sg24_lstm_p50_ms": SG24_LSTM_P50_MS,
            "dtype": "float32",
            "device": "cuda:0",
        },
        "provenance": {
            "sg24_reference": str(reference),
            "sg24_reference_sha256": reference_sha,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "core_source_sha256": _sha256(
                ROOT / "vpsc/world_model/cores.py"
            ),
            "corpus_dir": str(corpus_root),
            "data_audit": data_audit,
        },
        "primitive_equivalence": primitive,
        "primitive_speed": primitive_speed,
        "full_core_equivalence": full_core,
        "real_update_trajectory": trajectory,
        "real_update_benchmark": update,
        "profiler": profiler,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg25a_blocked_affine_scan_cuda.json"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.update_epochs) <= 0:
        parser.error("warmup, repeats, and update epochs must be positive")
    if args.quick:
        args.warmup = 1
        args.repeats = 2
        args.update_epochs = 1
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
