"""SG25C native O(T) fused gated-trace and reverse-adjoint CUDA study."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment, _sample_summary, _sync  # noqa: E402
from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from experiments.e3_ra0_reverse_adjoint import build_textworld_models  # noqa: E402
from vpsc.world_model import cores as core_module  # noqa: E402
from vpsc.world_model.cores import E3GatedTraceScanCore  # noqa: E402
from vpsc.world_model.fused_gated_trace_cuda import (  # noqa: E402
    extension_audit,
    fused_gated_trace,
    load_extension,
)


LENGTHS = (48, 80, 134, 512)
FORWARD_ATOL = 2e-6
FORWARD_RTOL = 2e-5
GRAD_ATOL = 3e-5
GRAD_RTOL = 3e-4
SG24_LSTM_P50_MS = 2.4325648333333336
SG24_LEGACY_SEED0_NLL = 2.63643016959482
SG24_LEGACY_SEED0_EDIT = 0.65938995215311
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


def _normalized_error(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left - right).norm().item()
    denominator = max(right.norm().item(), 1e-12)
    return float(numerator / denominator)


def _close(left: torch.Tensor, right: torch.Tensor, *, gradient: bool) -> bool:
    atol = GRAD_ATOL if gradient else FORWARD_ATOL
    rtol = GRAD_RTOL if gradient else FORWARD_RTOL
    return bool(
        torch.isfinite(left).all()
        and torch.isfinite(right).all()
        and torch.allclose(left, right, atol=atol, rtol=rtol)
    )


def _initial_decay_logits(device: torch.device) -> torch.Tensor:
    initial = torch.linspace(0.55, 0.99, steps=tw0.STATE_DIM, device=device)
    normalized = (initial - 0.50) / (0.995 - 0.50)
    logits = torch.logit(normalized)
    return torch.stack((logits, logits.flip(0)), dim=0)


def kernel_equivalence(device: torch.device) -> Dict[str, Any]:
    records = []
    state_dim = tw0.STATE_DIM
    drive_dim = 4 * state_dim
    for length in LENGTHS:
        torch.manual_seed(25_600_000 + length)
        query_count = min(25, length)
        queries = torch.linspace(
            0, length - 1, steps=query_count, device=device
        ).round().to(torch.long).unique(sorted=True)
        legacy_drives = torch.randn(
            1, length, drive_dim, device=device, requires_grad=True
        )
        fused_drives = legacy_drives.detach().clone().requires_grad_(True)
        legacy_logits = _initial_decay_logits(device).requires_grad_(True)
        fused_logits = legacy_logits.detach().clone().requires_grad_(True)
        legacy_initial_e = torch.rand(
            1, state_dim, device=device, requires_grad=True
        )
        legacy_initial_i = torch.rand(
            1, state_dim, device=device, requires_grad=True
        )
        fused_initial_e = legacy_initial_e.detach().clone().requires_grad_(True)
        fused_initial_i = legacy_initial_i.detach().clone().requires_grad_(True)
        identity = torch.eye(drive_dim, device=device)
        bias = torch.zeros(drive_dim, device=device)
        legacy_raw, legacy_final_e, legacy_final_i = (
            core_module._GatedTraceMultiQueryEligibility.apply(
                legacy_drives,
                queries,
                identity,
                bias,
                legacy_logits,
                legacy_initial_e,
                legacy_initial_i,
                0.50,
                0.995,
                0.50,
                5.0,
                False,
                True,
                0,
            )
        )
        fused_sigmoid = torch.sigmoid(fused_logits)
        fused_decays = 0.50 + (0.995 - 0.50) * fused_sigmoid
        fused_raw, fused_final_e, fused_final_i = fused_gated_trace(
            fused_drives,
            queries,
            fused_decays,
            fused_initial_e,
            fused_initial_i,
            spike_threshold=0.50,
            surrogate_scale=5.0,
        )
        raw_probe = torch.randn_like(legacy_raw)
        final_probe_e = torch.randn_like(legacy_final_e)
        final_probe_i = torch.randn_like(legacy_final_i)
        legacy_loss = (
            (legacy_raw * raw_probe).sum()
            + (legacy_final_e * final_probe_e).sum()
            + (legacy_final_i * final_probe_i).sum()
        )
        fused_loss = (
            (fused_raw * raw_probe).sum()
            + (fused_final_e * final_probe_e).sum()
            + (fused_final_i * final_probe_i).sum()
        )
        legacy_loss.backward()
        fused_loss.backward()
        checks = {
            "raw": {
                "max_abs": _max_abs(fused_raw, legacy_raw),
                "passed": _close(fused_raw, legacy_raw, gradient=False),
            },
            "final_e": {
                "max_abs": _max_abs(fused_final_e, legacy_final_e),
                "passed": _close(
                    fused_final_e, legacy_final_e, gradient=False
                ),
            },
            "final_i": {
                "max_abs": _max_abs(fused_final_i, legacy_final_i),
                "passed": _close(
                    fused_final_i, legacy_final_i, gradient=False
                ),
            },
            "drive_gradient": {
                "max_abs": _max_abs(fused_drives.grad, legacy_drives.grad),
                "normalized_error": _normalized_error(
                    fused_drives.grad, legacy_drives.grad
                ),
                "passed": _close(
                    fused_drives.grad, legacy_drives.grad, gradient=True
                ),
            },
            "decay_gradient": {
                "max_abs": _max_abs(fused_logits.grad, legacy_logits.grad),
                "normalized_error": _normalized_error(
                    fused_logits.grad, legacy_logits.grad
                ),
                "passed": _close(
                    fused_logits.grad, legacy_logits.grad, gradient=True
                ),
            },
            "initial_e_gradient": {
                "max_abs": _max_abs(
                    fused_initial_e.grad, legacy_initial_e.grad
                ),
                "normalized_error": _normalized_error(
                    fused_initial_e.grad, legacy_initial_e.grad
                ),
                "passed": _close(
                    fused_initial_e.grad,
                    legacy_initial_e.grad,
                    gradient=True,
                ),
            },
            "initial_i_gradient": {
                "max_abs": _max_abs(
                    fused_initial_i.grad, legacy_initial_i.grad
                ),
                "normalized_error": _normalized_error(
                    fused_initial_i.grad, legacy_initial_i.grad
                ),
                "passed": _close(
                    fused_initial_i.grad,
                    legacy_initial_i.grad,
                    gradient=True,
                ),
            },
        }
        spike_disagreements = int(
            (
                fused_raw[:, :, : 2 * state_dim]
                != legacy_raw[:, :, : 2 * state_dim]
            ).sum().item()
        )
        records.append(
            {
                "length": length,
                "query_count": queries.numel(),
                "checks": checks,
                "spike_disagreements": spike_disagreements,
                "passed": all(check["passed"] for check in checks.values())
                and spike_disagreements == 0,
            }
        )
    return {"records": records, "passed": all(record["passed"] for record in records)}


def _ra0_model(vocabulary: Any, *, device: torch.device, fused: bool) -> Any:
    models = build_textworld_models(9_400_000, vocabulary, device=device)
    model = models["snn_ra0"]
    del models
    if not isinstance(model.core, E3GatedTraceScanCore):  # pragma: no cover
        raise TypeError("RA0 builder returned the wrong core")
    if fused:
        model.core.scan_math_mode = "cuda_fused"
    return model


def _optimizer(model: Any) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=0.01,
        fused=True,
    )


def _forward_loss(model: Any, example: Any, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, query_indices, targets = sg0._example_tensors(example, device=device)
    logits, _ = tw0._sparse_forward(
        model,
        input_ids,
        query_indices,
        None,
        use_eligibility=True,
        detach_state=True,
    )
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
    return loss, logits.detach().argmax(dim=-1)


def _update(
    model: Any,
    optimizer: torch.optim.Optimizer,
    example: Any,
    *,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    loss, predictions = _forward_loss(model, example, device=device)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite SG25C loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        1.0,
        foreach=True,
    )
    optimizer.step()
    return float(loss.detach().item()), predictions


def real_core_equivalence(
    example: Any, vocabulary: Any, *, device: torch.device
) -> Dict[str, Any]:
    legacy = _ra0_model(vocabulary, device=device, fused=False)
    fused = copy.deepcopy(legacy)
    legacy_control = copy.deepcopy(legacy)
    fused.core.scan_math_mode = "cuda_fused"
    legacy.zero_grad(set_to_none=True)
    fused.zero_grad(set_to_none=True)
    legacy_control.zero_grad(set_to_none=True)
    legacy_loss, legacy_predictions = _forward_loss(
        legacy, example, device=device
    )
    fused_loss, fused_predictions = _forward_loss(fused, example, device=device)
    control_loss, control_predictions = _forward_loss(
        legacy_control, example, device=device
    )
    legacy_loss.backward()
    fused_loss.backward()
    control_loss.backward()
    gradients = {}
    for (
        (legacy_name, legacy_parameter),
        (fused_name, fused_parameter),
        (control_name, control_parameter),
    ) in zip(
        legacy.named_parameters(),
        fused.named_parameters(),
        legacy_control.named_parameters(),
    ):
        if legacy_name != fused_name or legacy_name != control_name:  # pragma: no cover
            raise AssertionError("parameter order mismatch")
        candidate_max_abs = _max_abs(fused_parameter.grad, legacy_parameter.grad)
        candidate_normalized = _normalized_error(
            fused_parameter.grad, legacy_parameter.grad
        )
        control_max_abs = _max_abs(
            control_parameter.grad, legacy_parameter.grad
        )
        control_normalized = _normalized_error(
            control_parameter.grad, legacy_parameter.grad
        )
        direct_pass = _close(
            fused_parameter.grad, legacy_parameter.grad, gradient=True
        )
        atomic_control_pass = (
            candidate_max_abs
            <= control_max_abs + max(1e-7, 0.05 * control_max_abs)
            and candidate_normalized
            <= control_normalized + max(1e-6, 0.05 * control_normalized)
        )
        gradients[legacy_name] = {
            "max_abs": candidate_max_abs,
            "normalized_error": candidate_normalized,
            "legacy_control_max_abs": control_max_abs,
            "legacy_control_normalized_error": control_normalized,
            "direct_tolerance_pass": direct_pass,
            "atomic_nondeterminism_control_pass": atomic_control_pass,
            "passed": direct_pass or atomic_control_pass,
        }
    result = {
        "legacy_loss": float(legacy_loss.item()),
        "fused_loss": float(fused_loss.item()),
        "legacy_control_loss": float(control_loss.item()),
        "loss_abs": abs(float(fused_loss.item() - legacy_loss.item())),
        "prediction_disagreements": int(
            (fused_predictions != legacy_predictions).sum().item()
        ),
        "legacy_control_prediction_disagreements": int(
            (control_predictions != legacy_predictions).sum().item()
        ),
        "gradients": gradients,
        "passed": abs(float(fused_loss.item() - legacy_loss.item())) <= 2e-6
        and bool(torch.equal(fused_predictions, legacy_predictions))
        and bool(torch.equal(control_predictions, legacy_predictions))
        and all(record["passed"] for record in gradients.values()),
    }
    del legacy, fused, legacy_control
    return result


def short_stability(
    examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
    updates: int,
) -> Dict[str, Any]:
    legacy = _ra0_model(vocabulary, device=device, fused=False)
    fused = copy.deepcopy(legacy)
    fused.core.scan_math_mode = "cuda_fused"
    legacy_optimizer = _optimizer(legacy)
    fused_optimizer = _optimizer(fused)
    schedule = sg0._training_schedule(
        len(examples), math.ceil(updates / len(examples)), 9_403_000
    )[:updates]
    loss_gaps = []
    prediction_disagreements = 0
    prediction_count = 0
    losses = {"legacy": [], "fused": []}
    for example_index in schedule:
        legacy_loss, legacy_prediction = _update(
            legacy, legacy_optimizer, examples[example_index], device=device
        )
        fused_loss, fused_prediction = _update(
            fused, fused_optimizer, examples[example_index], device=device
        )
        losses["legacy"].append(legacy_loss)
        losses["fused"].append(fused_loss)
        loss_gaps.append(abs(fused_loss - legacy_loss))
        prediction_disagreements += int(
            (fused_prediction != legacy_prediction).sum().item()
        )
        prediction_count += fused_prediction.numel()
    final_parameter_max_abs = max(
        _max_abs(fused_parameter, legacy_parameter)
        for legacy_parameter, fused_parameter in zip(
            legacy.parameters(), fused.parameters()
        )
    )
    tail_count = min(20, len(loss_gaps))
    mean_gap = sum(loss_gaps) / len(loss_gaps)
    tail_gap = sum(loss_gaps[-tail_count:]) / tail_count
    disagreement_rate = prediction_disagreements / prediction_count
    result = {
        "updates": len(schedule),
        "mean_loss_abs_gap": mean_gap,
        "last_20_mean_loss_abs_gap": tail_gap,
        "maximum_loss_abs_gap": max(loss_gaps),
        "prediction_disagreements": prediction_disagreements,
        "prediction_count": prediction_count,
        "prediction_disagreement_rate": disagreement_rate,
        "final_parameter_max_abs": final_parameter_max_abs,
        "loss_first": {name: values[0] for name, values in losses.items()},
        "loss_last": {name: values[-1] for name, values in losses.items()},
        "passed": mean_gap <= 0.02
        and tail_gap <= 0.02
        and disagreement_rate <= 0.02,
    }
    del legacy, fused, legacy_optimizer, fused_optimizer
    return result


def update_benchmark(
    examples: Sequence[Any],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    schedule = sg0._training_schedule(len(examples), epochs, 9_404_000)
    records = {}
    for name, fused_mode in (("legacy", False), ("cuda_fused", True)):
        gc.collect()
        torch.cuda.empty_cache()
        model = _ra0_model(vocabulary, device=device, fused=fused_mode)
        optimizer = _optimizer(model)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_start = torch.cuda.memory_allocated(device)
        samples = []
        losses = []
        for example_index in schedule:
            _sync(device)
            started = time.perf_counter_ns()
            loss, _ = _update(
                model, optimizer, examples[example_index], device=device
            )
            _sync(device)
            samples.append((time.perf_counter_ns() - started) / 1e6)
            losses.append(loss)
        warmup_updates = len(samples) // 5
        records[name] = {
            "epochs": epochs,
            "updates": len(schedule),
            "warmup_updates_excluded": warmup_updates,
            "timing": _sample_summary(samples[warmup_updates:], 1),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "cuda_allocated_start_bytes": allocated_start,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_additional_peak_bytes": max(
                0, torch.cuda.max_memory_allocated(device) - allocated_start
            ),
        }
        del model, optimizer
    fused_record = records["cuda_fused"]
    legacy_record = records["legacy"]
    fused_record["versus_legacy_speedup"] = (
        legacy_record["timing"]["p50_ms"] / fused_record["timing"]["p50_ms"]
    )
    fused_record["versus_sg24_lstm_ratio"] = (
        fused_record["timing"]["p50_ms"] / SG24_LSTM_P50_MS
    )
    fused_record["additional_peak_to_legacy_ratio"] = (
        fused_record["cuda_additional_peak_bytes"]
        / legacy_record["cuda_additional_peak_bytes"]
    )
    return {"records": records}


def profiler_audit(
    example: Any, vocabulary: Any, *, device: torch.device, fused: bool
) -> Dict[str, Any]:
    model = _ra0_model(vocabulary, device=device, fused=fused)
    optimizer = _optimizer(model)
    try:
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            )
        ) as profile:
            _update(model, optimizer, example, device=device)
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


def seed0_quality(
    examples: Mapping[str, Sequence[Any]],
    vocabulary: Any,
    *,
    device: torch.device,
    epochs: int,
) -> Dict[str, Any]:
    model = _ra0_model(vocabulary, device=device, fused=True)
    schedule = sg0._training_schedule(
        len(examples["train"]), epochs, 9_401_000
    )
    training = sg0.train_model(
        "snn_ra0",
        model,
        examples["train"],
        schedule,
        epochs=epochs,
        device=device,
    )
    teacher = {
        split: sg0.evaluate_teacher(model, examples[split], device=device)
        for split in ("train", "valid", "test")
    }
    generation = sg0.generate_model(
        model,
        examples["test"],
        vocabulary,
        max_tokens=sg0.MAX_GENERATION_TOKENS,
        device=device,
        include_records=True,
    )
    passed = (
        teacher["test"]["nll"] <= SG24_LEGACY_SEED0_NLL + 0.10
        and generation["edit_similarity"] >= SG24_LEGACY_SEED0_EDIT - 0.05
        and generation["paired_action_sensitivity"] >= 0.50
    )
    return {
        "seed": 0,
        "epochs": epochs,
        "training": training,
        "teacher": teacher,
        "generation": generation,
        "reference": {
            "legacy_seed0_test_nll": SG24_LEGACY_SEED0_NLL,
            "legacy_seed0_edit_similarity": SG24_LEGACY_SEED0_EDIT,
        },
        "passed": passed,
    }


def _decision(
    *,
    kernel: Mapping[str, Any],
    real_core: Mapping[str, Any],
    stability: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    profiler: Mapping[str, Any],
    quality: Mapping[str, Any] | None,
    quick: bool,
) -> Dict[str, Any]:
    fused_record = benchmark["records"]["cuda_fused"]
    kernel_pass = bool(kernel["passed"] and real_core["passed"])
    stability_pass = bool(stability["passed"])
    speedup = fused_record["versus_legacy_speedup"]
    speed_pass = speedup >= 1.5
    ann_speed_pass = fused_record["versus_sg24_lstm_ratio"] <= 1.0
    memory_pass = fused_record["additional_peak_to_legacy_ratio"] <= 1.25
    legacy_events = profiler["legacy"].get("cuda_event_count")
    fused_events = profiler["cuda_fused"].get("cuda_event_count")
    event_reduction = (
        1.0 - fused_events / legacy_events
        if legacy_events and fused_events is not None
        else None
    )
    event_pass = event_reduction is not None and event_reduction >= 0.30
    if quick:
        return {
            "kernel_gradient_gate": "PASS" if kernel_pass else "FAIL",
            "short_stability_gate": "PASS" if stability_pass else "FAIL",
            "speed_gate": "SMOKE",
            "event_reduction_gate": "SMOKE",
            "memory_gate": "PASS" if memory_pass else "FAIL",
            "overall": "SMOKE" if kernel_pass and stability_pass else "FAIL",
            "update_speedup": speedup,
            "event_reduction": event_reduction,
        }
    gates = {
        "kernel_gradient_gate": kernel_pass,
        "short_stability_gate": stability_pass,
        "speed_gate": speed_pass,
        "event_reduction_gate": event_pass,
        "memory_gate": memory_pass,
    }
    overall = all(gates.values())
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "ann_speed_target": "PASS" if ann_speed_pass else "FAIL",
        "seed0_quality_gate": (
            "NOT_RUN"
            if quality is None
            else "PASS"
            if quality["passed"]
            else "FAIL"
        ),
        "overall": "PASS" if overall else "FAIL",
        "legacy_update_p50_ms": benchmark["records"]["legacy"]["timing"][
            "p50_ms"
        ],
        "fused_update_p50_ms": fused_record["timing"]["p50_ms"],
        "update_speedup": speedup,
        "fused_to_sg24_lstm_ratio": fused_record["versus_sg24_lstm_ratio"],
        "event_reduction": event_reduction,
        "additional_peak_to_legacy_ratio": fused_record[
            "additional_peak_to_legacy_ratio"
        ],
        "next_route": (
            "expanded_real_corpus"
            if overall and quality is not None and quality["passed"]
            else "segmented_batch_or_projection_fusion"
            if kernel_pass
            else "fix_native_gradient"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("SG25C requires CUDA")
    device = torch.device("cuda:0")
    if "V100" not in torch.cuda.get_device_name(device).upper():
        raise AssertionError("SG25C requires the frozen V100 backend")
    reference = ROOT / "results/e3_scan/e3_sg24_cuda_counterfactual_generation.json"
    reference_sha = _sha256(reference)
    if reference_sha != EXPECTED_SG24_SHA256:
        raise AssertionError("SG25C SG24 reference hash mismatch")
    _, compile_audit = load_extension(verbose=args.verbose_compile)
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = sg0.load_event_corpus(corpus_root)
    examples, vocabulary = sg0.build_counterfactual_examples(corpus_root, corpus)
    data_audit = sg0.audit_examples(examples, vocabulary)
    if not data_audit["passed"]:
        raise AssertionError("SG25C frozen data audit failed")

    kernel = kernel_equivalence(device)
    real_core = real_core_equivalence(
        examples["train"][0], vocabulary, device=device
    )
    stability = short_stability(
        examples["train"],
        vocabulary,
        device=device,
        updates=args.stability_updates,
    )
    benchmark = update_benchmark(
        examples["train"],
        vocabulary,
        device=device,
        epochs=args.update_epochs,
    )
    profiler = {
        "legacy": profiler_audit(
            examples["train"][0], vocabulary, device=device, fused=False
        ),
        "cuda_fused": profiler_audit(
            examples["train"][0], vocabulary, device=device, fused=True
        ),
    }
    provisional_speed_pass = (
        benchmark["records"]["cuda_fused"]["versus_legacy_speedup"] >= 1.5
    )
    quality = None
    if (
        not args.quick
        and kernel["passed"]
        and real_core["passed"]
        and provisional_speed_pass
    ):
        quality = seed0_quality(
            examples, vocabulary, device=device, epochs=args.quality_epochs
        )
    decision = _decision(
        kernel=kernel,
        real_core=real_core,
        stability=stability,
        benchmark=benchmark,
        profiler=profiler,
        quality=quality,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-SG25C native O(T) fused gated-trace CUDA",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "cuda_compute_capability": torch.cuda.get_device_capability(device),
        },
        "configuration": {
            "lengths": LENGTHS,
            "stability_updates": args.stability_updates,
            "update_epochs": args.update_epochs,
            "quality_epochs": args.quality_epochs,
            "forward_atol": FORWARD_ATOL,
            "forward_rtol": FORWARD_RTOL,
            "gradient_atol": GRAD_ATOL,
            "gradient_rtol": GRAD_RTOL,
            "dtype": "float32",
            "device": "cuda:0",
        },
        "provenance": {
            "sg24_reference": str(reference),
            "sg24_reference_sha256": reference_sha,
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "core_source_sha256": _sha256(ROOT / "vpsc/world_model/cores.py"),
            "extension_python_sha256": _sha256(
                ROOT / "vpsc/world_model/fused_gated_trace_cuda.py"
            ),
            "data_audit": data_audit,
        },
        "extension": compile_audit or extension_audit(),
        "kernel_equivalence": kernel,
        "real_core_equivalence": real_core,
        "short_optimization_stability": stability,
        "real_update_benchmark": benchmark,
        "profiler": profiler,
        "seed0_quality": quality,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg25c_native_fused_scan_cuda.json"),
    )
    parser.add_argument("--stability-updates", type=int, default=100)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--quality-epochs", type=int, default=100)
    parser.add_argument("--verbose-compile", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if min(args.stability_updates, args.update_epochs, args.quality_epochs) <= 0:
        parser.error("all counts must be positive")
    if args.quick:
        args.stability_updates = min(args.stability_updates, 20)
        args.update_epochs = 1
        args.quality_epochs = 2
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
