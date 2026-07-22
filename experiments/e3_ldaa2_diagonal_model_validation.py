#!/usr/bin/env python3
"""LDAA-2A transfer to a diagonal SSM and real event-token LM updates."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import _environment  # noqa: E402
from experiments.e3_el0_terminal_eligibility import _SavedTensorCounter  # noqa: E402
from vpsc.world_model.diagonal_recurrence import (  # noqa: E402
    DiagonalLinearCore,
    diagonal_query_recurrence,
)


ATOL = 2e-5
RTOL = 1e-4
MODES = ("bptt", "segmented_adjoint")
DEFAULT_CORPUS = Path("results/e3_scan/textworld_sg22r_l5")
DEFAULT_OUTPUT = Path(
    "results/e3_scan/e3_ldaa2_diagonal_model_validation.json"
)


def query_indices(time_steps: int, density: float) -> Tensor:
    count = max(1, min(time_steps, int(round(time_steps * density))))
    values = torch.linspace(0, time_steps - 1, count + 1)[1:].round().long()
    return torch.unique(values, sorted=True)


def _gradient_snapshot(
    inputs: Tensor, initial: Tensor, decay: Tensor
) -> Dict[str, Tensor]:
    return {
        "inputs": inputs.grad.detach().clone(),
        "initial": initial.grad.detach().clone(),
        "decay": decay.grad.detach().clone(),
    }


def _operator_run(
    base_inputs: Tensor,
    base_initial: Tensor,
    base_decay: Tensor,
    indices: Tensor,
    mode: str,
    *,
    measure_storage: bool,
) -> Dict[str, Any]:
    inputs = base_inputs.detach().clone().requires_grad_(True)
    initial = base_initial.detach().clone().requires_grad_(True)
    decay = base_decay.detach().clone().requires_grad_(True)
    counter = _SavedTensorCounter()
    context = (
        torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack)
        if measure_storage
        else torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x)
    )
    with context:
        queries, final = diagonal_query_recurrence(
            inputs, initial, decay, indices, mode=mode  # type: ignore[arg-type]
        )
        probe = torch.linspace(-0.3, 0.7, queries.numel()).reshape_as(queries)
        loss = (queries * probe).mean() + 0.03 * final.square().mean()
        loss.backward()
    return {
        "queries": queries.detach(),
        "final": final.detach(),
        "gradients": _gradient_snapshot(inputs, initial, decay),
        "unique_storage_bytes": counter.unique_storage_bytes,
        "logical_saved_bytes": counter.logical_bytes,
        "saved_tensor_count": counter.tensor_count,
    }


def _max_abs(left: Tensor, right: Tensor) -> float:
    return float((left - right).abs().max())


def operator_probe(
    *,
    time_steps: int,
    state_dim: int,
    density: float,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    base_inputs = torch.randn(2, time_steps, state_dim)
    base_initial = torch.randn(2, state_dim)
    base_decay = 0.6 + 0.35 * torch.rand(state_dim)
    indices = query_indices(time_steps, density)
    measurements = {
        mode: _operator_run(
            base_inputs,
            base_initial,
            base_decay,
            indices,
            mode,
            measure_storage=True,
        )
        for mode in MODES
    }
    reference = measurements["bptt"]
    candidate = measurements["segmented_adjoint"]
    component_errors = {
        "queries": _max_abs(candidate["queries"], reference["queries"]),
        "final": _max_abs(candidate["final"], reference["final"]),
        **{
            f"gradient:{name}": _max_abs(
                candidate["gradients"][name], reference["gradients"][name]
            )
            for name in reference["gradients"]
        },
    }
    exact = all(
        torch.allclose(
            candidate["queries"] if name == "queries" else (
                candidate["final"] if name == "final" else candidate["gradients"][name.split(":", 1)[1]]
            ),
            reference["queries"] if name == "queries" else (
                reference["final"] if name == "final" else reference["gradients"][name.split(":", 1)[1]]
            ),
            atol=ATOL,
            rtol=RTOL,
        )
        for name in component_errors
    )
    samples = {mode: [] for mode in MODES}
    order = list(MODES)
    rng = random.Random(seed + 91)
    for _ in range(1):
        for mode in MODES:
            _operator_run(
                base_inputs, base_initial, base_decay, indices, mode,
                measure_storage=False,
            )
    for _ in range(repeats):
        rng.shuffle(order)
        for mode in order:
            started = time.perf_counter_ns()
            _operator_run(
                base_inputs, base_initial, base_decay, indices, mode,
                measure_storage=False,
            )
            samples[mode].append((time.perf_counter_ns() - started) / 1e6)
    latencies = {
        mode: {
            "samples_ms": values,
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
        }
        for mode, values in samples.items()
    }
    return {
        "time_steps": time_steps,
        "state_dim": state_dim,
        "density": int(indices.numel()) / time_steps,
        "query_count": int(indices.numel()),
        "exactness": {"passed": exact, "component_max_abs": component_errors},
        "backends": {
            mode: {
                "unique_storage_bytes": measurements[mode]["unique_storage_bytes"],
                "logical_saved_bytes": measurements[mode]["logical_saved_bytes"],
                "saved_tensor_count": measurements[mode]["saved_tensor_count"],
                **latencies[mode],
            }
            for mode in MODES
        },
        "segmented_speedup_vs_bptt": (
            latencies["bptt"]["p50_ms"]
            / latencies["segmented_adjoint"]["p50_ms"]
        ),
        "segmented_storage_ratio_to_bptt": (
            measurements["segmented_adjoint"]["unique_storage_bytes"]
            / measurements["bptt"]["unique_storage_bytes"]
        ),
    }


class SparseQueryLM(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.core = DiagonalLinearCore(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.decoder = nn.Linear(model_dim, vocab_size)

    def forward(
        self, tokens: Tensor, indices: Tensor, *, mode: str
    ) -> Tensor:
        hidden, _final = self.core.forward_queries(
            self.embedding(tokens), indices, mode=mode  # type: ignore[arg-type]
        )
        return self.decoder(self.norm(hidden))


def _load_tokens(corpus_root: Path, max_vocab: int) -> Tuple[Dict[str, Any], Dict[str, Tensor]]:
    raw = {
        split: (corpus_root / split / "token_events.txt")
        .read_text(encoding="utf-8")
        .split()
        for split in ("train", "valid")
    }
    counts: Dict[str, int] = {}
    for token in raw["train"]:
        counts[token] = counts.get(token, 0) + 1
    vocabulary = ["<unk>"] + [
        token
        for token, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[: max_vocab - 1]
    ]
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    encoded = {
        split: torch.tensor(
            [token_to_id.get(token, 0) for token in raw[split]],
            dtype=torch.long,
        )
        for split in raw
    }
    return {
        "vocab_size": len(vocabulary),
        "train_tokens": len(raw["train"]),
        "valid_tokens": len(raw["valid"]),
        "valid_unknown_rate": float((encoded["valid"] == 0).float().mean()),
    }, encoded


def _segments(tokens: Tensor, sequence_length: int) -> Tensor:
    width = sequence_length + 1
    count = tokens.numel() // width
    if count == 0:
        raise ValueError("corpus is shorter than one model sequence")
    return tokens[: count * width].reshape(count, width)


def _loss(model: SparseQueryLM, batch: Tensor, indices: Tensor, mode: str) -> Tensor:
    logits = model(batch[:, :-1], indices, mode=mode)
    targets = batch[:, 1:].index_select(1, indices)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _parameter_max_abs(left: nn.Module, right: nn.Module) -> float:
    return max(
        float((a.detach() - b.detach()).abs().max())
        for a, b in zip(left.parameters(), right.parameters())
    )


def train_pair(
    encoded: Mapping[str, Tensor],
    *,
    vocab_size: int,
    model_dim: int,
    sequence_length: int,
    density: float,
    batch_size: int,
    steps: int,
    eval_batches: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    reference = SparseQueryLM(vocab_size, model_dim)
    candidate = copy.deepcopy(reference)
    models = {"bptt": reference, "segmented_adjoint": candidate}
    optimizers = {
        mode: torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for mode, model in models.items()
    }
    train = _segments(encoded["train"], sequence_length)
    valid = _segments(encoded["valid"], sequence_length)
    indices = query_indices(sequence_length, density)
    generator = torch.Generator().manual_seed(seed + 1234)
    order = torch.randperm(train.shape[0], generator=generator)
    timings = {mode: [] for mode in MODES}
    losses = {mode: [] for mode in MODES}
    for step in range(steps):
        start = (step * batch_size) % max(1, train.shape[0] - batch_size + 1)
        selected = order[start : start + batch_size]
        if selected.numel() < batch_size:
            selected = torch.cat((selected, order[: batch_size - selected.numel()]))
        batch = train.index_select(0, selected)
        for mode in MODES:
            started = time.perf_counter_ns()
            optimizers[mode].zero_grad(set_to_none=True)
            value = _loss(models[mode], batch, indices, mode)
            value.backward()
            optimizers[mode].step()
            timings[mode].append((time.perf_counter_ns() - started) / 1e6)
            losses[mode].append(float(value.detach()))
    valid_nll = {}
    with torch.no_grad():
        for mode in MODES:
            values = []
            for offset in range(min(eval_batches, math.ceil(valid.shape[0] / batch_size))):
                batch = valid[offset * batch_size : (offset + 1) * batch_size]
                if batch.numel():
                    values.append(float(_loss(models[mode], batch, indices, mode)))
            valid_nll[mode] = float(np.mean(values))
    return {
        "seed": seed,
        "query_count": int(indices.numel()),
        "actual_density": int(indices.numel()) / sequence_length,
        "valid_nll": valid_nll,
        "segmented_minus_bptt_valid_nll": (
            valid_nll["segmented_adjoint"] - valid_nll["bptt"]
        ),
        "parameter_max_abs_after_training": _parameter_max_abs(reference, candidate),
        "final_train_loss": {mode: values[-1] for mode, values in losses.items()},
        "update_p50_ms": {
            mode: float(np.percentile(values, 50)) for mode, values in timings.items()
        },
        "segmented_update_speedup": (
            float(np.percentile(timings["bptt"], 50))
            / float(np.percentile(timings["segmented_adjoint"], 50))
        ),
    }


def decide(operator: Mapping[str, Any], training: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    exactness = bool(operator["exactness"]["passed"])
    transfer = bool(
        operator["segmented_speedup_vs_bptt"] >= 1.5
        and operator["segmented_storage_ratio_to_bptt"] <= 0.25
    )
    quality = all(
        abs(float(run["segmented_minus_bptt_valid_nll"])) <= 0.10
        for run in training
    )
    trajectory = all(
        float(run["parameter_max_abs_after_training"]) <= 1e-3
        for run in training
    )
    passed = exactness and transfer and quality and trajectory
    return {
        "H1_second_core_exactness": exactness,
        "H2_second_core_sparse_crossover": transfer,
        "H3_model_quality": quality,
        "H3_training_trajectory": trajectory,
        "overall": "PASS" if passed else "FAIL",
        "verdict": (
            "SECOND_CORE_MODEL_GO_DISPATCHER_REQUIRED"
            if passed
            else "NARROW_OR_NO_GO_SECOND_CORE"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    torch.set_num_threads(args.threads)
    operator = operator_probe(
        time_steps=args.operator_length,
        state_dim=args.operator_state_dim,
        density=args.density,
        repeats=args.repeats,
        seed=20260722,
    )
    corpus, encoded = _load_tokens(args.corpus_dir.resolve(), args.max_vocab)
    training = [
        train_pair(
            encoded,
            vocab_size=corpus["vocab_size"],
            model_dim=args.model_dim,
            sequence_length=args.sequence_length,
            density=args.density,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_batches=args.eval_batches,
            learning_rate=args.learning_rate,
            seed=seed,
        )
        for seed in args.seeds
    ]
    decision = decide(operator, training)
    return {
        "schema_version": 1,
        "experiment": "E3-LDAA2 diagonal SSM and event-token LM validation",
        "environment": _environment(torch.device("cpu")),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "corpus": corpus,
        "operator": operator,
        "training": training,
        "decision": decision,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--operator-length", type=int, default=2048)
    parser.add_argument("--operator-state-dim", type=int, default=32)
    parser.add_argument("--density", type=float, default=1 / 64)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-vocab", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if not 0 < args.density <= 1:
        parser.error("--density must be in (0, 1]")
    return args


def main() -> None:
    args = _parse_args()
    result = run_experiment(args)
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
