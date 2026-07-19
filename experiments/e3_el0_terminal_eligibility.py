"""Exact terminal eligibility scan: equivalence, memory, speed, and quality gates."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _autograd_node_count,
    _environment,
    _interleaved_samples,
    _sample_summary,
)
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3InputCodedScanCore,
    E3LayerState,
    E3ScanState,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
)


D_MODEL = 32
STATE_DIM = 42
PAYLOAD_VOCAB = 16
DISTRACTOR_TOKEN = 0
WRITE_BASE = 1
QUERY_TOKEN = WRITE_BASE + PAYLOAD_VOCAB
INPUT_VOCAB = QUERY_TOKEN + 1
EL0_ATOL = 2e-6
EL0_RTOL = 1e-5
Runner = Callable[[], None]


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _gradient_record(
    candidate: Optional[torch.Tensor], reference: Optional[torch.Tensor]
) -> Dict[str, Any]:
    if candidate is None or reference is None:
        passed = candidate is None and reference is None
        return {"passed": passed, "max_abs": None}
    return {
        "passed": bool(
            torch.allclose(candidate, reference, atol=EL0_ATOL, rtol=EL0_RTOL)
        ),
        "max_abs": _max_abs(candidate, reference),
    }


def _equivalence_case(
    *,
    batch: int,
    time_steps: int,
    input_gradient: bool,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    reference = E3InputCodedScanCore(4, 6, state_dim=5).to(device)
    candidate = E3InputCodedScanCore(4, 6, state_dim=5).to(device)
    candidate.load_state_dict(reference.state_dict())
    reference_input = torch.randn(
        batch, time_steps, 4, device=device, requires_grad=input_gradient
    )
    candidate_input = reference_input.detach().clone().requires_grad_(input_gradient)
    initial_e = torch.rand(batch, 5, device=device, requires_grad=True)
    initial_i = torch.rand(batch, 5, device=device, requires_grad=True)
    reference_state = E3ScanState(
        layers=(E3LayerState(excitatory=initial_e, inhibitory=initial_i),)
    )
    candidate_state = E3ScanState(
        layers=(
            E3LayerState(
                excitatory=initial_e.detach().clone().requires_grad_(True),
                inhibitory=initial_i.detach().clone().requires_grad_(True),
            ),
        )
    )
    reference_result = reference(reference_input, reference_state)
    candidate_result = candidate.forward_terminal_eligibility(
        candidate_input, candidate_state
    )
    reference_terminal = reference_result.sequence[:, -1:]
    probe = torch.linspace(
        -0.7,
        0.9,
        reference_terminal.numel(),
        device=device,
        dtype=reference_terminal.dtype,
    ).reshape_as(reference_terminal)
    reference_loss = (reference_terminal * probe).mean() + 0.13 * (
        reference_result.state.layers[0].excitatory.mean()
        - reference_result.state.layers[0].inhibitory.mean()
    )
    candidate_loss = (candidate_result.sequence * probe).mean() + 0.13 * (
        candidate_result.state.layers[0].excitatory.mean()
        - candidate_result.state.layers[0].inhibitory.mean()
    )
    reference_loss.backward()
    candidate_loss.backward()

    forward = {
        "sequence": {
            "passed": bool(
                torch.allclose(
                    candidate_result.sequence,
                    reference_terminal,
                    atol=EL0_ATOL,
                    rtol=EL0_RTOL,
                )
            ),
            "max_abs": _max_abs(candidate_result.sequence, reference_terminal),
        },
        "state_e": {
            "passed": bool(
                torch.equal(
                    candidate_result.state.layers[0].excitatory,
                    reference_result.state.layers[0].excitatory,
                )
            ),
            "max_abs": _max_abs(
                candidate_result.state.layers[0].excitatory,
                reference_result.state.layers[0].excitatory,
            ),
        },
        "state_i": {
            "passed": bool(
                torch.equal(
                    candidate_result.state.layers[0].inhibitory,
                    reference_result.state.layers[0].inhibitory,
                )
            ),
            "max_abs": _max_abs(
                candidate_result.state.layers[0].inhibitory,
                reference_result.state.layers[0].inhibitory,
            ),
        },
    }
    gradients: Dict[str, Dict[str, Any]] = {
        "input": _gradient_record(candidate_input.grad, reference_input.grad),
        "initial_e": _gradient_record(
            candidate_state.layers[0].excitatory.grad,
            reference_state.layers[0].excitatory.grad,
        ),
        "initial_i": _gradient_record(
            candidate_state.layers[0].inhibitory.grad,
            reference_state.layers[0].inhibitory.grad,
        ),
    }
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        gradients[f"parameter:{name}"] = _gradient_record(
            parameter.grad, reference_parameters[name].grad
        )
    passed = all(value["passed"] for value in forward.values()) and all(
        value["passed"] for value in gradients.values()
    )
    return {
        "batch": batch,
        "time": time_steps,
        "input_gradient": input_gradient,
        "forward": forward,
        "gradients": gradients,
        "passed": passed,
    }


def run_equivalence(device: torch.device) -> Dict[str, Any]:
    specifications = ((1, 1, True), (4, 32, True), (1, 512, False))
    cases = [
        _equivalence_case(
            batch=batch,
            time_steps=time_steps,
            input_gradient=input_gradient,
            device=device,
            seed=8_600_000 + index,
        )
        for index, (batch, time_steps, input_gradient) in enumerate(specifications)
    ]
    return {"cases": cases, "passed": all(case["passed"] for case in cases)}


class _SavedTensorCounter:
    def __init__(self) -> None:
        self.logical_bytes = 0
        self.tensor_count = 0
        self._storages: Dict[Tuple[str, int], int] = {}

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        self.logical_bytes += tensor.numel() * tensor.element_size()
        self.tensor_count += 1
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()))
        self._storages[key] = max(self._storages.get(key, 0), int(storage.nbytes()))
        return tensor

    @staticmethod
    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def unique_storage_bytes(self) -> int:
        return sum(self._storages.values())


def _terminal_result(
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
) -> torch.Tensor:
    if name.startswith("el0"):
        if not isinstance(core, E3InputCodedScanCore):  # pragma: no cover
            raise TypeError("EL0 runner requires E3InputCodedScanCore")
        return core.forward_terminal_eligibility(value).sequence
    return core(value).sequence[:, -1:]


def _saved_tensor_measurement(
    *,
    name: str,
    time_steps: int,
    input_gradient: bool,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    core = E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM).to(device)
    value = torch.randn(
        1, time_steps, D_MODEL, device=device, requires_grad=input_gradient
    )
    counter = _SavedTensorCounter()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        terminal = _terminal_result(name, core, value)
        terminal.square().mean().backward()
    return {
        "mode": name,
        "time": time_steps,
        "input_gradient": input_gradient,
        "logical_saved_bytes": counter.logical_bytes,
        "unique_storage_bytes": counter.unique_storage_bytes,
        "saved_tensor_count": counter.tensor_count,
        "autograd_nodes": _autograd_node_count(terminal),
    }


def benchmark_saved_tensors(device: torch.device) -> Dict[str, Any]:
    records = []
    for time_steps in (128, 512):
        for name, input_gradient in (
            ("bptt_core_only", False),
            ("el0_core_only", False),
            ("bptt_input_grad", True),
            ("el0_input_grad", True),
        ):
            records.append(
                _saved_tensor_measurement(
                    name=name,
                    time_steps=time_steps,
                    input_gradient=input_gradient,
                    device=device,
                    seed=8_610_000 + time_steps,
                )
            )
    lookup = {(record["mode"], record["time"]): record for record in records}
    bptt_512 = lookup[("bptt_core_only", 512)]["unique_storage_bytes"]
    el0_128 = lookup[("el0_core_only", 128)]["unique_storage_bytes"]
    el0_512 = lookup[("el0_core_only", 512)]["unique_storage_bytes"]
    checks = {
        "t512_ratio_le_25pct": el0_512 <= 0.25 * bptt_512,
        "el0_t128_to_t512_growth_le_1_25x": el0_512 <= 1.25 * el0_128,
    }
    return {
        "records": records,
        "checks": checks,
        "t512_unique_storage_ratio": el0_512 / bptt_512,
        "el0_growth_t128_to_t512": el0_512 / el0_128,
        "passed": all(checks.values()),
    }


def _benchmark_suite(
    *, device: torch.device, time_steps: int, seed: int
) -> Dict[str, TemporalCore[Any]]:
    torch.manual_seed(seed)
    bptt = E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    el0 = E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    el0.load_state_dict(bptt.state_dict())
    return {
        "ic0_bptt": bptt.to(device).train(True),
        "el0_core_only": el0.to(device).train(True),
        "lstm": StatefulLSTMCore(D_MODEL, D_MODEL).to(device).train(True),
        "transformer": CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=time_steps,
        )
        .to(device)
        .train(True),
    }


def _terminal_training_runner(
    *, name: str, core: TemporalCore[Any], value: torch.Tensor
) -> Runner:
    def run() -> None:
        core.zero_grad(set_to_none=True)
        terminal = _terminal_result(name, core, value)
        terminal.square().mean().backward()

    return run


def benchmark_speed(
    *,
    time_steps: Iterable[int],
    threads: Sequence[int],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    records = []
    for thread_count in threads:
        if device.type == "cpu":
            torch.set_num_threads(thread_count)
        for length in time_steps:
            cores = _benchmark_suite(
                device=device,
                time_steps=length,
                seed=8_620_000 + thread_count * 10000 + length,
            )
            base = torch.randn(1, length, D_MODEL, device=device)
            values = {name: base.detach().clone() for name in cores}
            runners = {
                name: _terminal_training_runner(
                    name=name, core=core, value=values[name]
                )
                for name, core in cores.items()
            }
            nodes = {
                name: _autograd_node_count(_terminal_result(name, core, values[name]))
                for name, core in cores.items()
            }
            for core in cores.values():
                core.zero_grad(set_to_none=True)
            samples = _interleaved_samples(
                runners,
                warmup=warmup,
                repeats=repeats,
                device=device,
                seed=8_630_000 + thread_count * 10000 + length,
            )
            models = {
                name: {
                    **_sample_summary(sample, length),
                    "parameters": count_parameters(cores[name]),
                    "autograd_nodes": nodes[name],
                }
                for name, sample in samples.items()
            }
            speedup = models["ic0_bptt"]["p50_ms"] / models["el0_core_only"][
                "p50_ms"
            ]
            passed = (
                speedup >= 1.5
                and models["el0_core_only"]["p50_ms"]
                <= models["lstm"]["p50_ms"]
            )
            records.append(
                {
                    "threads": thread_count if device.type == "cpu" else None,
                    "time": length,
                    "models": models,
                    "el0_vs_ic0_speedup": speedup,
                    "passed": passed,
                }
            )
    return {"records": records, "passed": any(record["passed"] for record in records)}


def generate_terminal_batch(
    *, seed: int, batch_size: int, sequence_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    targets = torch.randint(PAYLOAD_VOCAB, (batch_size,), generator=generator)
    tokens = torch.full(
        (batch_size, sequence_length), DISTRACTOR_TOKEN, dtype=torch.long
    )
    tokens[:, 0] = WRITE_BASE + targets
    tokens[:, -1] = QUERY_TOKEN
    return tokens, targets


def _dataset_hash(batches: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for tokens, targets in batches:
        digest.update(tokens.numpy().tobytes())
        digest.update(targets.numpy().tobytes())
    return digest.hexdigest()


class TerminalTokenModel(nn.Module):
    def __init__(
        self,
        core: TemporalCore[Any],
        *,
        use_terminal_eligibility: bool = False,
    ) -> None:
        super().__init__()
        self.core = core
        self.use_terminal_eligibility = bool(use_terminal_eligibility)
        self.embedding = nn.Embedding(INPUT_VOCAB, D_MODEL)
        self.embedding.weight.requires_grad_(False)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.decoder = nn.Linear(D_MODEL, PAYLOAD_VOCAB)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        if self.use_terminal_eligibility and self.training:
            if not isinstance(self.core, E3InputCodedScanCore):  # pragma: no cover
                raise TypeError("terminal eligibility requires IC0")
            sequence = self.core.forward_terminal_eligibility(embedded).sequence
        else:
            sequence = self.core(embedded).sequence[:, -1:]
        return self.decoder(self.output_norm(sequence[:, -1]))


def _initialise_shared_wrapper(
    model: TerminalTokenModel, shared: Mapping[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        model.embedding.weight.copy_(shared["embedding"])
        model.output_norm.weight.copy_(shared["norm_weight"])
        model.output_norm.bias.copy_(shared["norm_bias"])
        model.decoder.weight.copy_(shared["decoder_weight"])
        model.decoder.bias.copy_(shared["decoder_bias"])


def build_quality_models(seed: int) -> Dict[str, TerminalTokenModel]:
    generator = torch.Generator().manual_seed(8_700_001)
    shared = {
        "embedding": torch.randn(
            INPUT_VOCAB, D_MODEL, generator=generator
        )
        * 0.2,
        "norm_weight": torch.ones(D_MODEL),
        "norm_bias": torch.zeros(D_MODEL),
        "decoder_weight": torch.randn(
            PAYLOAD_VOCAB, D_MODEL, generator=generator
        )
        * 0.02,
        "decoder_bias": torch.zeros(PAYLOAD_VOCAB),
    }
    torch.manual_seed(seed)
    bptt = TerminalTokenModel(
        E3InputCodedScanCore(D_MODEL, D_MODEL, state_dim=STATE_DIM)
    )
    _initialise_shared_wrapper(bptt, shared)
    el0 = copy.deepcopy(bptt)
    el0.use_terminal_eligibility = True
    torch.manual_seed(seed + 1)
    lstm = TerminalTokenModel(StatefulLSTMCore(D_MODEL, D_MODEL))
    _initialise_shared_wrapper(lstm, shared)
    torch.manual_seed(seed + 2)
    transformer = TerminalTokenModel(
        CausalTransformerCore(
            D_MODEL,
            D_MODEL,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_cache_tokens=64,
        )
    )
    _initialise_shared_wrapper(transformer, shared)
    return {
        "ic0_bptt": bptt,
        "el0": el0,
        "lstm": lstm,
        "transformer": transformer,
    }


def _train_terminal_model(
    model: TerminalTokenModel,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    timing_warmup: int,
) -> Dict[str, Any]:
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    timings = []
    checkpoints = []
    for update, (tokens, targets) in enumerate(batches, start=1):
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite terminal loss at update {update}")
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
    return {
        "updates": len(batches),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last_100_mean": sum(losses[-100:]) / min(100, len(losses)),
        "losses": losses,
        "timing": _sample_summary(timings, batches[0][0].numel()),
        "checkpoints": checkpoints,
    }


def _evaluate_terminal_model(
    model: TerminalTokenModel,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    nll = 0.0
    with torch.inference_mode():
        for tokens, targets in batches:
            logits = model(tokens)
            losses = F.cross_entropy(logits, targets, reduction="none")
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            total += targets.numel()
            nll += float(losses.sum().item())
    return {"accuracy": correct / total, "nll": nll / total, "count": total}


def _parameter_max_abs(
    left: nn.Module, right: nn.Module
) -> Tuple[float, Dict[str, float]]:
    right_parameters = dict(right.named_parameters())
    errors = {
        name: _max_abs(parameter, right_parameters[name])
        for name, parameter in left.named_parameters()
    }
    return max(errors.values()), errors


def run_quality(*, quick: bool) -> Dict[str, Any]:
    updates = 3 if quick else 300
    train_batches = tuple(
        generate_terminal_batch(
            seed=8_710_000 + update,
            batch_size=8,
            sequence_length=64,
        )
        for update in range(updates)
    )
    test_tokens = torch.full((PAYLOAD_VOCAB, 64), DISTRACTOR_TOKEN, dtype=torch.long)
    test_targets = torch.arange(PAYLOAD_VOCAB)
    test_tokens[:, 0] = WRITE_BASE + test_targets
    test_tokens[:, -1] = QUERY_TOKEN
    test_batches = ((test_tokens, test_targets),)
    models = build_quality_models(8_720_000)
    parameter_counts = {name: count_parameters(model) for name, model in models.items()}
    lstm_count = parameter_counts["lstm"]
    fairness = {
        name: abs(count - lstm_count) / lstm_count <= 0.02
        for name, count in parameter_counts.items()
    }
    if not all(fairness.values()):
        raise AssertionError(f"terminal parameter fairness failed: {parameter_counts}")
    model_results = {}
    for name in ("ic0_bptt", "el0", "lstm", "transformer"):
        model_results[name] = {
            "train": _train_terminal_model(
                models[name], train_batches, timing_warmup=min(100, updates - 1)
            ),
            "test": _evaluate_terminal_model(models[name], test_batches),
        }
    bptt_losses = model_results["ic0_bptt"]["train"]["losses"]
    el0_losses = model_results["el0"]["train"]["losses"]
    loss_max_abs = max(abs(left - right) for left, right in zip(bptt_losses, el0_losses))
    parameter_max_abs, parameter_errors = _parameter_max_abs(
        models["el0"], models["ic0_bptt"]
    )
    if quick:
        task_valid = False
        quality_pass = False
    else:
        task_valid = all(
            model_results[name]["test"]["accuracy"] >= 0.99
            for name in ("lstm", "transformer")
        )
        quality_pass = (
            task_valid
            and model_results["ic0_bptt"]["test"]["accuracy"] >= 0.99
            and model_results["el0"]["test"]["accuracy"] >= 0.99
            and loss_max_abs <= 1e-4
            and parameter_max_abs <= 1e-4
        )
    for result in model_results.values():
        result["train"].pop("losses")
    return {
        "formal": not quick,
        "parameter_counts": parameter_counts,
        "parameter_fairness": fairness,
        "train_data_sha256": _dataset_hash(train_batches),
        "models": model_results,
        "el0_vs_bptt_loss_max_abs": loss_max_abs,
        "el0_vs_bptt_parameter_max_abs": parameter_max_abs,
        "el0_vs_bptt_parameter_errors": parameter_errors,
        "task_validation": "PASS" if task_valid else "NOT_RUN" if quick else "FAIL",
        "passed": quality_pass,
    }


def _decision(
    *,
    equivalence: Mapping[str, Any],
    memory: Mapping[str, Any],
    speed: Mapping[str, Any],
    quality: Mapping[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    if quick:
        return {
            "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
            "memory_gate": "PASS" if memory["passed"] else "FAIL",
            "speed_gate": "SMOKE",
            "quality_gate": "NOT_RUN",
            "overall": "SMOKE",
        }
    binary_gates = {
        "equivalence_gate": bool(equivalence["passed"]),
        "memory_gate": bool(memory["passed"]),
        "speed_gate": bool(speed["passed"]),
    }
    task_invalid = quality.get("task_validation") == "FAIL"
    quality_gate = "INVALID" if task_invalid else "PASS" if quality["passed"] else "FAIL"
    all_binary_pass = all(binary_gates.values())
    overall = (
        "PASS"
        if all_binary_pass and quality_gate == "PASS"
        else "MIXED"
        if all_binary_pass and quality_gate == "INVALID"
        else "FAIL"
    )
    return {
        **{
            name: "PASS" if value else "FAIL"
            for name, value in binary_gates.items()
        },
        "quality_gate": quality_gate,
        "overall": overall,
        "run_textworld_k_query_next": overall == "PASS",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_el0_terminal_eligibility.json"),
    )
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.threads = args.threads[:1]
        args.warmup = 1
        args.repeats = 1
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
    equivalence = run_equivalence(device)
    if not equivalence["passed"]:
        memory = {"records": [], "passed": False, "not_run": "equivalence failed"}
        speed = {"records": [], "passed": False, "not_run": "equivalence failed"}
        quality = {"passed": False, "not_run": "equivalence failed"}
    else:
        memory = benchmark_saved_tensors(device)
        speed = benchmark_speed(
            time_steps=(512,) if args.quick else (512, 2048),
            threads=threads,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        quality = run_quality(quick=args.quick)
    decision = _decision(
        equivalence=equivalence,
        memory=memory,
        speed=speed,
        quality=quality,
        quick=args.quick,
    )
    result = {
        "schema_version": 1,
        "experiment": "E3-EL0 exact terminal eligibility scan",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(device),
        "configuration": {
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "quality_sequence_length": 64,
            "quality_updates": 3 if args.quick else 300,
            "threads": threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "atol": EL0_ATOL,
            "rtol": EL0_RTOL,
        },
        "equivalence": equivalence,
        "saved_tensors": memory,
        "speed": speed,
        "quality": quality,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
