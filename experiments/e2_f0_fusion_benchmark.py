"""E2-F0 exact-fusion equivalence and CPU/GPU scaling benchmark.

The experiment deliberately separates implementation fusion from new SNN
mathematics.  ``reference`` and ``fused`` E2 instances share an identical state
dict; LSTM and causal Transformer are timing baselines, not substitute parts of
the E2 model.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    CoreOutput,
    E2CoreState,
    E2SignedCore,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.homegrid_model import (  # noqa: E402
    ACTION_COUNT,
    READ_CLASS_COUNT,
    REWARD_CLASS_COUNT,
    VISUAL_PATCHES,
    VISUAL_VOCAB_SIZE,
    HomeGridWorldModel,
    build_homegrid_model_suite,
)


Tensor = torch.Tensor
Runner = Callable[[], None]
F0_ATOL = 2e-6
F0_RTOL = 1e-5
FORMAL_CORE_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (1, 32, 32),
    (8, 32, 32),
    (1, 128, 32),
    (1, 512, 32),
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sample_summary(samples_ms: Sequence[float], tokens: int) -> Dict[str, float]:
    p50 = _percentile(samples_ms, 0.50)
    return {
        "count": len(samples_ms),
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "p50_ms": p50,
        "p95_ms": _percentile(samples_ms, 0.95),
        "tokens_per_second_at_p50": tokens * 1000.0 / p50,
    }


def _time_once(runner: Runner, device: torch.device) -> float:
    _sync(device)
    started = time.perf_counter_ns()
    runner()
    _sync(device)
    return (time.perf_counter_ns() - started) / 1e6


def _interleaved_samples(
    runners: Mapping[str, Runner],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Sequence[float]]:
    for _ in range(warmup):
        for runner in runners.values():
            runner()
    samples: Dict[str, list[float]] = {name: [] for name in runners}
    names = list(runners)
    generator = random.Random(seed)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            generator.shuffle(names)
            for name in names:
                samples[name].append(_time_once(runners[name], device))
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples


def _max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _close(left: Tensor, right: Tensor) -> bool:
    return bool(torch.allclose(left, right, atol=F0_ATOL, rtol=F0_RTOL))


def _gradient_error(left: Tensor | None, right: Tensor | None) -> Tuple[bool, float]:
    if left is None or right is None:
        return left is None and right is None, 0.0
    return _close(left, right), _max_abs(left, right)


def _equivalence_case(
    *,
    seed: int,
    batch: int,
    time_steps: int,
    configuration: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    reference = E2SignedCore(
        4,
        6,
        state_dim=5,
        execution_mode="reference",
        **configuration,
    ).to(device)
    fused = E2SignedCore(
        4,
        6,
        state_dim=5,
        execution_mode="fused",
        **configuration,
    ).to(device)
    fused.load_state_dict(reference.state_dict())

    reference_input = torch.randn(batch, time_steps, 4, device=device, requires_grad=True)
    fused_input = reference_input.detach().clone().requires_grad_(True)
    reference_state = E2CoreState(
        excitatory=torch.randn(batch, 5, device=device, requires_grad=True),
        inhibitory=torch.randn(batch, 5, device=device, requires_grad=True),
    )
    fused_state = E2CoreState(
        excitatory=reference_state.excitatory.detach().clone().requires_grad_(True),
        inhibitory=reference_state.inhibitory.detach().clone().requires_grad_(True),
    )
    reference_result = reference(reference_input, reference_state)
    fused_result = fused(fused_input, fused_state)
    probe = torch.linspace(
        -0.9,
        1.1,
        reference_result.sequence.numel(),
        device=device,
    ).reshape_as(reference_result.sequence)
    reference_loss = (reference_result.sequence * probe).mean() + 0.17 * (
        reference_result.state.excitatory.mean() - reference_result.state.inhibitory.mean()
    )
    fused_loss = (fused_result.sequence * probe).mean() + 0.17 * (
        fused_result.state.excitatory.mean() - fused_result.state.inhibitory.mean()
    )
    reference_loss.backward()
    fused_loss.backward()

    checks: Dict[str, bool] = {}
    errors: Dict[str, float] = {}
    tensor_pairs = {
        "sequence": (fused_result.sequence, reference_result.sequence),
        "state_excitatory": (
            fused_result.state.excitatory,
            reference_result.state.excitatory,
        ),
        "state_inhibitory": (
            fused_result.state.inhibitory,
            reference_result.state.inhibitory,
        ),
        "input_gradient": (fused_input.grad, reference_input.grad),
    }
    for name, (fused_value, reference_value) in tensor_pairs.items():
        assert fused_value is not None and reference_value is not None
        checks[name] = _close(fused_value, reference_value)
        errors[name] = _max_abs(fused_value, reference_value)

    for name, fused_value, reference_value in (
        (
            "initial_state_excitatory_gradient",
            fused_state.excitatory.grad,
            reference_state.excitatory.grad,
        ),
        (
            "initial_state_inhibitory_gradient",
            fused_state.inhibitory.grad,
            reference_state.inhibitory.grad,
        ),
    ):
        checks[name], errors[name] = _gradient_error(fused_value, reference_value)

    reference_parameters = dict(reference.named_parameters())
    fused_parameters = dict(fused.named_parameters())
    checks["parameter_keys"] = tuple(reference_parameters) == tuple(fused_parameters)
    checks["parameter_shapes"] = all(
        reference_parameters[name].shape == fused_parameters[name].shape
        for name in reference_parameters
    )
    parameter_gradient_errors: Dict[str, float] = {}
    parameter_gradient_checks: Dict[str, bool] = {}
    for name in reference_parameters:
        passed, error = _gradient_error(
            fused_parameters[name].grad, reference_parameters[name].grad
        )
        parameter_gradient_checks[name] = passed
        parameter_gradient_errors[name] = error
    checks["all_parameter_gradients"] = all(parameter_gradient_checks.values())
    errors["parameter_gradient_max"] = max(parameter_gradient_errors.values())

    with torch.no_grad():
        full = fused(fused_input.detach(), fused_state)
        stream_state = fused_state
        pieces = []
        for index in range(time_steps):
            streamed = fused.step(fused_input.detach()[:, index], stream_state)
            pieces.append(streamed.sequence)
            stream_state = streamed.state
        streamed_sequence = torch.cat(pieces, dim=1)
    checks["fused_streaming_sequence"] = _close(streamed_sequence, full.sequence)
    errors["fused_streaming_sequence"] = _max_abs(streamed_sequence, full.sequence)
    checks["fused_streaming_state_e"] = _close(
        stream_state.excitatory, full.state.excitatory
    )
    errors["fused_streaming_state_e"] = _max_abs(
        stream_state.excitatory, full.state.excitatory
    )
    checks["fused_streaming_state_i"] = _close(
        stream_state.inhibitory, full.state.inhibitory
    )
    errors["fused_streaming_state_i"] = _max_abs(
        stream_state.inhibitory, full.state.inhibitory
    )
    checks["parameter_count"] = count_parameters(reference) == count_parameters(fused)
    checks["state_bytes"] = state_nbytes(reference_result.state) == state_nbytes(
        fused_result.state
    )

    return {
        "seed": seed,
        "batch": batch,
        "time": time_steps,
        "configuration": dict(configuration),
        "checks": checks,
        "max_abs_errors": errors,
        "parameter_gradient_max_abs": parameter_gradient_errors,
        "passed": all(checks.values()),
    }


def run_equivalence(device: torch.device, seed: int) -> Dict[str, Any]:
    configurations: Tuple[Mapping[str, Any], ...] = (
        {"policy": "exact", "micro_steps": 1},
        {"policy": "margin", "no_positive": True, "micro_steps": 2},
        {"policy": "hybrid", "positive_factor": 0.8, "micro_steps": 1},
        {"policy": "exact", "state_reset": True, "micro_steps": 2},
    )
    shapes = ((1, 1), (4, 32))
    cases = []
    for configuration_index, configuration in enumerate(configurations):
        for shape_index, (batch, time_steps) in enumerate(shapes):
            cases.append(
                _equivalence_case(
                    seed=seed + 100 * configuration_index + shape_index,
                    batch=batch,
                    time_steps=time_steps,
                    configuration=configuration,
                    device=device,
                )
            )
    return {
        "atol": F0_ATOL,
        "rtol": F0_RTOL,
        "case_count": len(cases),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def _autograd_node_count(output: Tensor) -> int:
    root = output.grad_fn
    if root is None:
        return 0
    seen: set[Any] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(next_node for next_node, _ in node.next_functions if next_node is not None)
    return len(seen)


def _core_suite(dimension: int, time_steps: int, device: torch.device, seed: int) -> Dict[str, TemporalCore[Any]]:
    torch.manual_seed(seed)
    reference = E2SignedCore(
        dimension,
        dimension,
        state_dim=dimension,
        policy="hybrid",
        positive_factor=0.8,
        execution_mode="reference",
    )
    fused = E2SignedCore(
        dimension,
        dimension,
        state_dim=dimension,
        policy="hybrid",
        positive_factor=0.8,
        execution_mode="fused",
    )
    fused.load_state_dict(reference.state_dict())
    heads = 4 if dimension % 4 == 0 else 1
    cores: Dict[str, TemporalCore[Any]] = {
        "e2_reference": reference,
        "e2_fused": fused,
        "lstm": StatefulLSTMCore(dimension, dimension),
        "transformer": CausalTransformerCore(
            dimension,
            dimension,
            num_layers=1,
            num_heads=heads,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=max(128, time_steps),
        ),
    }
    return {name: core.to(device).train(True) for name, core in cores.items()}


def _core_training_runner(core: TemporalCore[Any], value: Tensor) -> Runner:
    def run() -> None:
        core.zero_grad(set_to_none=True)
        value.grad = None
        result: CoreOutput[Any] = core(value)
        loss = result.sequence.square().mean()
        loss.backward()

    return run


def _cuda_peak_bytes(runner: Runner, device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    runner()
    _sync(device)
    return int(torch.cuda.max_memory_allocated(device))


def benchmark_cores(
    *,
    shapes: Iterable[Tuple[int, int, int]],
    threads: Sequence[int],
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for shape_index, (batch, time_steps, dimension) in enumerate(shapes):
            cores = _core_suite(dimension, time_steps, device, seed + shape_index)
            base = torch.randn(batch, time_steps, dimension, device=device)
            inputs = {
                name: base.detach().clone().requires_grad_(True) for name in cores
            }
            runners = {
                name: _core_training_runner(core, inputs[name])
                for name, core in cores.items()
            }
            nodes: Dict[str, int] = {}
            for name, core in cores.items():
                result = core(inputs[name])
                nodes[name] = _autograd_node_count(result.sequence)
                core.zero_grad(set_to_none=True)
                inputs[name].grad = None
            samples = _interleaved_samples(
                runners,
                warmup=warmup,
                repeats=repeats,
                device=device,
                seed=seed + thread_count * 1000 + shape_index,
            )
            summaries = {
                name: {
                    **_sample_summary(values, batch * time_steps),
                    "parameters": count_parameters(cores[name]),
                    "state_bytes": state_nbytes(cores[name].initial_state(batch)),
                    "autograd_nodes": nodes[name],
                    "peak_memory_bytes": _cuda_peak_bytes(runners[name], device),
                }
                for name, values in samples.items()
            }
            reference_p50 = summaries["e2_reference"]["p50_ms"]
            for summary in summaries.values():
                summary["versus_e2_reference_speedup"] = reference_p50 / summary["p50_ms"]
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "batch": batch,
                    "time": time_steps,
                    "dimension": dimension,
                    "models": summaries,
                }
            )
    return records


def benchmark_streaming(
    *,
    threads: Sequence[int],
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
    seed: int,
) -> Sequence[Dict[str, Any]]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        cores = _core_suite(32, 128, device, seed + thread_count)
        for core in cores.values():
            core.eval()
        tokens = torch.randn(warmup_steps + measured_steps, 1, 32, device=device)
        states: Dict[str, Any] = {name: None for name in cores}
        samples: Dict[str, list[float]] = {name: [] for name in cores}
        generator = random.Random(seed + 2000 * thread_count)
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
        summaries = {
            name: {
                **_sample_summary(values, 1),
                "state_bytes": state_nbytes(states[name]),
            }
            for name, values in samples.items()
        }
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "warmup_steps": warmup_steps,
                "measured_steps": measured_steps,
                "models": summaries,
            }
        )
    return records


def _homegrid_loss(output: Any, targets: Mapping[str, Tensor]) -> Tensor:
    total = F.cross_entropy(
        output.next_visual_logits.reshape(-1, VISUAL_VOCAB_SIZE),
        targets["visual"].reshape(-1),
    )
    total = total + 0.25 * F.cross_entropy(
        output.next_language_logits.reshape(-1, output.next_language_logits.shape[-1]),
        targets["language"].reshape(-1),
    )
    total = total + 0.10 * F.cross_entropy(
        output.next_read_logits.reshape(-1, READ_CLASS_COUNT), targets["read"].reshape(-1)
    )
    return total + 0.10 * F.cross_entropy(
        output.reward_logits.reshape(-1, REWARD_CLASS_COUNT), targets["reward"].reshape(-1)
    )


def _homegrid_runner(
    model: HomeGridWorldModel[Any],
    optimizer: torch.optim.Optimizer,
    inputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
) -> Runner:
    def run() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(
            inputs["visual"], inputs["language"], inputs["action"], inputs["read"]
        )
        loss = _homegrid_loss(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return run


def benchmark_homegrid(
    *,
    threads: Sequence[int],
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> Sequence[Dict[str, Any]]:
    records = []
    batch, time_steps, dimension, vocabulary = 1, 32, 32, 32
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        torch.manual_seed(seed + thread_count)
        suite = build_homegrid_model_suite(vocabulary, d_model=dimension, num_heads=4)
        fused = suite.e2
        reference = copy.deepcopy(fused)
        assert isinstance(reference.core, E2SignedCore)
        assert isinstance(fused.core, E2SignedCore)
        reference.core.execution_mode = "reference"
        fused.core.execution_mode = "fused"
        models: Dict[str, HomeGridWorldModel[Any]] = {
            "e2_reference": reference,
            "e2_fused": fused,
            "lstm": suite.lstm,
            "transformer": suite.transformer,
        }
        models = {name: model.to(device).train(True) for name, model in models.items()}
        inputs = {
            "visual": torch.randint(
                VISUAL_VOCAB_SIZE, (batch, time_steps, VISUAL_PATCHES), device=device
            ),
            "language": torch.randint(vocabulary, (batch, time_steps), device=device),
            "action": torch.randint(ACTION_COUNT, (batch, time_steps), device=device),
            "read": torch.randint(READ_CLASS_COUNT, (batch, time_steps), device=device),
        }
        targets = {
            "visual": torch.randint(
                VISUAL_VOCAB_SIZE, (batch, time_steps, VISUAL_PATCHES), device=device
            ),
            "language": torch.randint(vocabulary, (batch, time_steps), device=device),
            "read": torch.randint(READ_CLASS_COUNT, (batch, time_steps), device=device),
            "reward": torch.randint(REWARD_CLASS_COUNT, (batch, time_steps), device=device),
        }
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            for name, model in models.items()
        }
        runners = {
            name: _homegrid_runner(model, optimizers[name], inputs, targets)
            for name, model in models.items()
        }
        samples = _interleaved_samples(
            runners,
            warmup=warmup,
            repeats=repeats,
            device=device,
            seed=seed + 3000 * thread_count,
        )
        summaries = {
            name: {
                **_sample_summary(values, batch * time_steps),
                "parameters": count_parameters(models[name]),
                "peak_memory_bytes": _cuda_peak_bytes(runners[name], device),
            }
            for name, values in samples.items()
        }
        reference_p50 = summaries["e2_reference"]["p50_ms"]
        for summary in summaries.values():
            summary["versus_e2_reference_speedup"] = reference_p50 / summary["p50_ms"]
        records.append(
            {
                "threads": thread_count if device.type == "cpu" else None,
                "batch": batch,
                "time": time_steps,
                "dimension": dimension,
                "includes": "forward+weighted_cross_entropy+backward+clip_grad_norm+AdamW.step",
                "models": summaries,
            }
        )
    return records


def _environment(device: torch.device) -> Dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    git_executable = "git.exe" if shutil.which("git.exe") else "git"
    git_commit = subprocess.run(
        [git_executable, "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    git_status = subprocess.run(
        [git_executable, "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "logical_cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "git_commit": git_commit,
        "git_executable": git_executable,
        "git_status_porcelain": git_status,
    }


def _decision(
    equivalence: Mapping[str, Any],
    core: Sequence[Mapping[str, Any]],
    streaming: Sequence[Mapping[str, Any]],
    homegrid: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    canonical_core = [
        record
        for record in core
        if (record["batch"], record["time"], record["dimension"]) == (1, 32, 32)
    ]
    core_speedups = {
        str(record["threads"]): record["models"]["e2_fused"][
            "versus_e2_reference_speedup"
        ]
        for record in canonical_core
    }
    homegrid_speedups = {
        str(record["threads"]): record["models"]["e2_fused"][
            "versus_e2_reference_speedup"
        ]
        for record in homegrid
    }
    implementation_pass = bool(equivalence["passed"]) and any(
        speedup >= 4.0 for speedup in core_speedups.values()
    ) and any(speedup >= 2.0 for speedup in homegrid_speedups.values())

    streaming_by_thread = {str(record["threads"]): record for record in streaming}
    ann_thread_checks: Dict[str, Dict[str, Any]] = {}
    for record in homegrid:
        thread = str(record["threads"])
        stream_record = streaming_by_thread[thread]
        train_e2 = record["models"]["e2_fused"]["p50_ms"]
        train_lstm = record["models"]["lstm"]["p50_ms"]
        stream_e2 = stream_record["models"]["e2_fused"]["p95_ms"]
        stream_lstm = stream_record["models"]["lstm"]["p95_ms"]
        ann_thread_checks[thread] = {
            "train_e2_ms": train_e2,
            "train_lstm_ms": train_lstm,
            "stream_p95_e2_ms": stream_e2,
            "stream_p95_lstm_ms": stream_lstm,
            "passed": train_e2 <= train_lstm and stream_e2 <= stream_lstm,
        }
    ann_speed_pass = bool(equivalence["passed"]) and any(
        check["passed"] for check in ann_thread_checks.values()
    )
    return {
        "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
        "implementation_speed_gate": "PASS" if implementation_pass else "FAIL",
        "ann_train_and_inference_speed_gate": "PASS" if ann_speed_pass else "FAIL",
        "canonical_core_speedups_by_threads": core_speedups,
        "homegrid_train_step_speedups_by_threads": homegrid_speedups,
        "ann_thread_checks": ann_thread_checks,
        "boundary": (
            "F0 changes only the execution graph. Passing any speed gate is not evidence "
            "that E2 is a strict SNN or that SNN quality replaces ANN baselines."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e2_acceleration/e2_f0_fusion_benchmark.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--homegrid-warmup", type=int, default=2)
    parser.add_argument("--homegrid-repeats", type=int, default=8)
    parser.add_argument("--streaming-warmup", type=int, default=16)
    parser.add_argument("--streaming-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8300)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test one small shape; never use this flag for a formal F0 result.",
    )
    args = parser.parse_args()
    positive_values = (
        *args.threads,
        args.warmup,
        args.repeats,
        args.homegrid_warmup,
        args.homegrid_repeats,
        args.streaming_warmup,
        args.streaming_steps,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("thread counts, warmups, repeats, and streaming steps must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    resolved_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(resolved_device)
    threads = tuple(dict.fromkeys(args.threads))
    shapes = ((1, 4, 8),) if args.quick else FORMAL_CORE_SHAPES
    if args.quick:
        threads = threads[:1]
    if device.type == "cpu":
        torch.set_num_threads(threads[0])
    environment = _environment(device)
    equivalence = run_equivalence(device, args.seed)
    core = benchmark_cores(
        shapes=shapes,
        threads=threads,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
        seed=args.seed,
    )
    streaming = benchmark_streaming(
        threads=threads,
        warmup_steps=args.streaming_warmup,
        measured_steps=args.streaming_steps,
        device=device,
        seed=args.seed,
    )
    homegrid = benchmark_homegrid(
        threads=threads,
        warmup=args.homegrid_warmup,
        repeats=args.homegrid_repeats,
        device=device,
        seed=args.seed,
    )
    result = {
        "schema_version": 1,
        "experiment": "E2-F0 exact signed-block fusion",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "configuration": {
            "seed": args.seed,
            "threads": threads,
            "core_shapes": shapes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "homegrid_warmup": args.homegrid_warmup,
            "homegrid_repeats": args.homegrid_repeats,
            "streaming_warmup": args.streaming_warmup,
            "streaming_steps": args.streaming_steps,
        },
        "equivalence": equivalence,
        "core_forward_backward": core,
        "streaming_inference": streaming,
        "homegrid_train_step": homegrid,
        "decision": _decision(equivalence, core, streaming, homegrid),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
