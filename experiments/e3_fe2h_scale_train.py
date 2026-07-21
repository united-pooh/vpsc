#!/usr/bin/env python3
"""Single-model FE-2H scaling runner for real catgirl language learning.

Unlike ``e3_fe2h_neuron_tile.py``, this entry point keeps exactly one model on
the GPU.  It is intended for memory/utilisation calibration and sustained real
training, not for matched three-model quality comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.e3_fe2h_neuron_tile import (  # noqa: E402
    ExperimentConfig,
    _build_fe2h_model,
    _estimate_fe2h_memory,
    _forward_model,
    _loss_breakdown,
)
from vpsc.world_model.fe2h_tile_sparse import (  # noqa: E402
    FE2HDiagnostics,
    FE2HFiniteGuardError,
    FE2HNeuronTileCore,
    run_fe2h_finite_guard,
)
from vpsc.world_model.fe2h_low_rank import LowRankLinear  # noqa: E402
from vpsc.world_model.lm import CausalLanguageModel  # noqa: E402


GIB = float(1024**3)
MEMORY_WARNING_GIB = 32.0
MEMORY_REFUSE_GIB = 32.0
PHYSICAL_HEADROOM_GIB = 0.5


@dataclass(frozen=True)
class ScaleConfig:
    out: Path
    cache_dir: Path
    checkpoint: Optional[Path]
    label: str
    d_model: int
    state_dim: int
    tile_size: int
    active_tiles: int
    block_size: int
    rank: int
    batch_size: int
    seq_len: int
    max_steps: int
    train_mode: str
    epochs: int
    warmup_steps: int
    valid_steps: int
    full_validation: bool
    log_every: int
    finite_check_every: int
    learning_rate: float
    weight_decay: float
    route_supervision_weight: float
    homeostasis_weight: float
    seed: int
    amp: bool
    amp_init_scale: float
    amp_growth_interval: int
    low_rank_output: bool
    sample_interval_ms: int
    estimator_factor: float
    estimator_overhead_gib: float
    save_optimizer: bool
    checkpoint_every_steps: int
    tqdm_progress: bool
    shared_init: Optional[Path]
    create_shared_init: bool


def _path(value: Optional[Path]) -> Optional[str]:
    return None if value is None else str(value)


def _configuration_record(cfg: ScaleConfig) -> Dict[str, Any]:
    record = asdict(cfg)
    for name in ("out", "cache_dir", "checkpoint", "shared_init"):
        record[name] = _path(getattr(cfg, name))
    return record


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _series_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": float(max(values)),
    }


class NvidiaSmiSampler:
    """Collect device-level occupancy without synchronising the training loop."""

    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = int(interval_ms)
        self.samples: List[Dict[str, Any]] = []
        self._process: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                fields = [field.strip() for field in line.strip().split(",")]
                if len(fields) != 6:
                    continue
                try:
                    memory_used, memory_total, gpu_util, memory_util, power, temperature = (
                        float(field) for field in fields
                    )
                except ValueError:
                    continue
                self.samples.append(
                    {
                        "monotonic_s": time.monotonic(),
                        "memory_used_mib": memory_used,
                        "memory_total_mib": memory_total,
                        "gpu_util_percent": gpu_util,
                        "memory_util_percent": memory_util,
                        "power_w": power,
                        "temperature_c": temperature,
                    }
                )
        except Exception as error:  # monitoring must not kill training
            self.error = f"{error.__class__.__name__}: {error}"

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        if self._thread is not None:
            self._thread.join(timeout=3)

    def summary(self, *, start: float, end: float) -> Dict[str, Any]:
        selected = [
            sample for sample in self.samples if start <= sample["monotonic_s"] <= end
        ]
        gpu_values = [sample["gpu_util_percent"] for sample in selected]
        memory_values = [sample["memory_used_mib"] for sample in selected]
        memory_util_values = [sample["memory_util_percent"] for sample in selected]
        power_values = [sample["power_w"] for sample in selected]
        return {
            "sample_count": len(selected),
            "interval_ms": self.interval_ms,
            "gpu_util_percent": _series_summary(gpu_values),
            "memory_used_mib": _series_summary(memory_values),
            "memory_util_percent": _series_summary(memory_util_values),
            "power_w": _series_summary(power_values),
            "error": self.error,
        }


def _load_token_ids(cache_dir: Path) -> Dict[str, Any]:
    train_path = cache_dir / "tok" / "catgirl_train_ids.pt"
    valid_path = cache_dir / "tok" / "catgirl_val_ids.pt"
    tokenizer_path = cache_dir / "bpe" / "catgirl_bpe_8192.json"
    missing = [str(path) for path in (train_path, valid_path, tokenizer_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing real catgirl cache files: " + ", ".join(missing))
    train_ids = torch.load(train_path, weights_only=True, map_location="cpu")
    valid_ids = torch.load(valid_path, weights_only=True, map_location="cpu")
    if train_ids.ndim != 1 or valid_ids.ndim != 1:
        raise ValueError("cached token tensors must be one-dimensional")
    vocab_size = max(int(train_ids.max().item()), int(valid_ids.max().item())) + 1
    return {
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "vocab_size": vocab_size,
        "train_path": train_path,
        "valid_path": valid_path,
        "tokenizer_path": tokenizer_path,
    }


def _make_sequences(ids: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
    sequence_count = (ids.numel() - 1) // seq_len
    if sequence_count <= 0:
        raise ValueError("token cache is shorter than seq_len")
    inputs = ids[: sequence_count * seq_len].view(sequence_count, seq_len)
    targets = ids[1 : sequence_count * seq_len + 1].view(sequence_count, seq_len)
    return inputs, targets


def _experiment_config(cfg: ScaleConfig) -> ExperimentConfig:
    return ExperimentConfig(
        mode="formal",
        device_request="cuda",
        out=cfg.out,
        cache_dir=cfg.cache_dir,
        d_model=cfg.d_model,
        state_dim=cfg.state_dim,
        tile_size=cfg.tile_size,
        active_tiles=cfg.active_tiles,
        block_size=cfg.block_size,
        rank=cfg.rank,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        epochs=1,
        seed=cfg.seed,
        vocab_size=8192,
        svd_init=False,
    )


def _build_scale_model(
    vocab_size: int, cfg: ScaleConfig
) -> tuple[CausalLanguageModel, Dict[str, Any]]:
    """Build one model while keeping non-formal scaling ranks explicit."""

    if not cfg.low_rank_output:
        model, provenance = _build_fe2h_model(
            vocab_size,
            _experiment_config(cfg),
            low_rank_output=False,
        )
        if model is None:
            raise RuntimeError(f"dense model build unsupported: {provenance['unsupported_reason']}")
        return model, provenance

    non_formal_rank = cfg.rank not in {16, 32}
    output_projection = LowRankLinear(
        4 * cfg.state_dim,
        cfg.d_model,
        rank=cfg.rank,
        allow_test_rank=non_formal_rank,
    )
    model = CausalLanguageModel(
        vocab_size,
        FE2HNeuronTileCore(
            cfg.d_model,
            cfg.d_model,
            state_dim=cfg.state_dim,
            tile_size=cfg.tile_size,
            active_tiles=cfg.active_tiles,
            block_size=cfg.block_size,
            output_projection=output_projection,
        ),
    )
    provenance = {
        "path_label": "low_rank_dense_mask_training",
        "supported": True,
        "hardware_executed_sparsity": False,
        "dense_input_projection_retained": True,
        "input_projection_kind": "dense",
        "router_projection_kind": "dense_router_mlp",
        "output_projection_kind": "low_rank",
        "output_projection_provenance": output_projection.provenance_dict(),
        "output_projection_cost": output_projection.cost_report().as_dict(),
        "non_formal_scaling_rank": non_formal_rank,
        "unsupported_reason": None,
        "notes": [
            "Training uses the differentiable dense-mask path; no hardware sparse speedup is claimed.",
            "Ranks outside {16, 32} are isolated to this scaling runner and marked non-formal.",
        ],
    }
    return model, provenance


def _apply_shared_initialization(
    model: CausalLanguageModel,
    cfg: ScaleConfig,
) -> Dict[str, Any]:
    """Create or apply a canonical CPU initialization for matched sweeps."""

    if cfg.shared_init is None:
        if cfg.create_shared_init:
            raise ValueError("--create-shared-init requires --shared-init")
        return {
            "enabled": False,
            "created": False,
            "path": None,
            "matched_parameter_count": 0,
            "matched_parameter_ratio": 0.0,
            "unmatched_keys": [],
        }

    path = cfg.shared_init
    created = False
    if cfg.create_shared_init:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "metadata": {
                    "seed": cfg.seed,
                    "label": cfg.label,
                    "d_model": cfg.d_model,
                    "state_dim": cfg.state_dim,
                    "rank": cfg.rank,
                    "vocab_size": model.vocab_size,
                },
            },
            temporary,
        )
        os.replace(temporary, path)
        created = True
    if not path.exists():
        raise FileNotFoundError(f"shared initialization does not exist: {path}")

    payload = torch.load(path, weights_only=True, map_location="cpu")
    reference = payload.get("model")
    if not isinstance(reference, dict):
        raise ValueError("shared initialization must contain a model state_dict")
    current = model.state_dict()
    matched_keys: List[str] = []
    unmatched_keys: List[Dict[str, Any]] = []
    matched_parameter_count = 0
    for name, tensor in current.items():
        source = reference.get(name)
        if isinstance(source, Tensor) and tuple(source.shape) == tuple(tensor.shape):
            current[name] = source.to(dtype=tensor.dtype)
            matched_keys.append(name)
            matched_parameter_count += int(tensor.numel())
        else:
            unmatched_keys.append(
                {
                    "name": name,
                    "current_shape": list(tensor.shape),
                    "reference_shape": (
                        list(source.shape) if isinstance(source, Tensor) else None
                    ),
                }
            )
    model.load_state_dict(current, strict=True)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "enabled": True,
        "created": created,
        "path": str(path),
        "bytes": path.stat().st_size,
        "reference_metadata": payload.get("metadata", {}),
        "matched_key_count": len(matched_keys),
        "matched_parameter_count": matched_parameter_count,
        "total_parameter_count": int(total_parameter_count),
        "matched_parameter_ratio": matched_parameter_count
        / max(1, total_parameter_count),
        "unmatched_keys": unmatched_keys,
    }


def _preflight(
    cfg: ScaleConfig,
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    estimate = _estimate_fe2h_memory(
        model,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        sparse_inference=False,
        dtype=torch.float32,
    )
    raw_gib = float(estimate["model_total_gib"])
    calibrated_gib = raw_gib * cfg.estimator_factor + cfg.estimator_overhead_gib
    properties = torch.cuda.get_device_properties(device)
    physical_total_gib = properties.total_memory / GIB
    physical_launch_limit_gib = max(0.0, physical_total_gib - PHYSICAL_HEADROOM_GIB)
    if calibrated_gib > MEMORY_REFUSE_GIB:
        status = "REFUSE"
        reason = f"calibrated estimate exceeds the {MEMORY_REFUSE_GIB:g} GiB hard-refuse threshold"
    elif calibrated_gib > MEMORY_WARNING_GIB:
        status = "PAUSE"
        reason = f"calibrated estimate exceeds the {MEMORY_WARNING_GIB:g} GiB user-warning threshold"
    elif calibrated_gib > physical_launch_limit_gib:
        status = "REFUSE_PHYSICAL"
        reason = "calibrated estimate leaves less than 0.5 GiB physical device headroom"
    else:
        status = "ALLOW"
        reason = "estimate is within both user and physical launch limits"
    return {
        "status": status,
        "can_launch": status == "ALLOW",
        "reason": reason,
        "raw_estimate_gib": raw_gib,
        "estimator_factor": cfg.estimator_factor,
        "estimator_overhead_gib": cfg.estimator_overhead_gib,
        "calibrated_estimate_gib": calibrated_gib,
        "user_warning_gib": MEMORY_WARNING_GIB,
        "user_refuse_gib": MEMORY_REFUSE_GIB,
        "physical_total_gib": physical_total_gib,
        "physical_headroom_gib": PHYSICAL_HEADROOM_GIB,
        "physical_launch_limit_gib": physical_launch_limit_gib,
        "architecture_estimate": estimate,
    }


def _diagnostics_record(diagnostics: FE2HDiagnostics) -> Dict[str, Any]:
    homeostasis = diagnostics.homeostasis
    return {
        "activation_rate": [
            float(value) for value in homeostasis.activation_rate.detach().float().cpu()
        ],
        "entropy": float(homeostasis.entropy.detach().float().item()),
        "gini": float(homeostasis.gini.detach().float().item()),
        "p99_tile_load": float(homeostasis.p99_tile_load.detach().float().item()),
        "block_fill": float(homeostasis.block_fill.detach().float().item()),
        "hotspot_share": float(homeostasis.hotspot_share.detach().float().item()),
        "dead_tile_ratio": float(homeostasis.dead_tile_ratio.detach().float().item()),
        "target_activation_rate": float(homeostasis.target_activation_rate),
    }


def _mean_homeostasis(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    tile_count = len(records[0]["activation_rate"])
    return {
        "activation_rate": [
            float(statistics.fmean(record["activation_rate"][index] for record in records))
            for index in range(tile_count)
        ],
        **{
            key: float(statistics.fmean(float(record[key]) for record in records))
            for key in (
                "entropy",
                "gini",
                "p99_tile_load",
                "block_fill",
                "hotspot_share",
                "dead_tile_ratio",
                "target_activation_rate",
            )
        },
    }


def _finite_loss_terms(loss_terms: Mapping[str, Any], step: int) -> None:
    for name in ("ce", "route_supervision", "homeostasis", "total"):
        value = loss_terms[name]
        if not bool(torch.isfinite(value.detach()).item()):
            raise FE2HFiniteGuardError(
                scope="loss", name=name, step=step, index=(0,), value=float(value.detach())
            )


def _apply_loss_weights(
    loss_terms: Mapping[str, Any],
    *,
    cfg: ScaleConfig,
) -> Dict[str, Any]:
    weighted = dict(loss_terms)
    if bool(weighted.get("aux_applied")):
        weighted["total"] = (
            weighted["ce"]
            + cfg.route_supervision_weight * weighted["route_supervision"]
            + cfg.homeostasis_weight * weighted["homeostasis"]
        )
        provenance = dict(weighted.get("provenance") or {})
        provenance["total"] = (
            "ce + "
            f"{cfg.route_supervision_weight:g}*route_supervision + "
            f"{cfg.homeostasis_weight:g}*homeostasis"
        )
        weighted["provenance"] = provenance
    return weighted


def _window_record(
    *,
    step_start: int,
    step_end: int,
    started: float,
    tokens: int,
    ce_values: Sequence[float],
    objective_values: Sequence[float],
    route_values: Sequence[float],
    homeostasis_loss_values: Sequence[float],
    grad_norm_values: Sequence[float],
    homeostasis_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    elapsed = max(time.perf_counter() - started, 1e-12)
    ce = float(statistics.fmean(ce_values))
    return {
        "step_start": step_start,
        "step_end": step_end,
        "elapsed_s": elapsed,
        "tokens": tokens,
        "tokens_per_s": tokens / elapsed,
        "ce": ce,
        "bpc": ce / math.log(2.0),
        "objective": float(statistics.fmean(objective_values)),
        "route_supervision_loss": float(statistics.fmean(route_values)),
        "homeostasis_loss": float(statistics.fmean(homeostasis_loss_values)),
        "grad_norm": _series_summary(grad_norm_values),
        "homeostasis": _mean_homeostasis(homeostasis_records),
    }


def _route_specialization(
    token_chunks: Sequence[Tensor],
    route_chunks: Sequence[Tensor],
    *,
    hard_tile_sum: Tensor,
    hard_block_count: int,
    tile_count: int,
    active_tiles: int,
) -> Dict[str, Any]:
    if not token_chunks or not route_chunks or hard_block_count <= 0:
        return {"supported": False, "reason": "no FE-2H route observations"}
    if tile_count > 62:
        return {
            "supported": False,
            "reason": "bit-coded route specialization supports at most 62 tiles",
        }

    import numpy as np

    tokens = torch.cat(list(token_chunks)).numpy().astype(np.int64, copy=False)
    routes = torch.cat(list(route_chunks)).numpy().astype(np.int64, copy=False)
    if tokens.shape != routes.shape:
        raise ValueError("token and route observations must have identical shapes")
    observation_count = int(tokens.size)
    route_ids, route_counts = np.unique(routes, return_counts=True)
    route_probabilities = route_counts.astype(np.float64) / max(1, observation_count)
    route_entropy = float(
        -(route_probabilities * np.log(np.clip(route_probabilities, 1e-300, None))).sum()
    )
    base = 1 << tile_count
    token_counts = np.bincount(tokens)

    def mutual_information(route_values: Any) -> float:
        local_route_ids, local_route_counts = np.unique(
            route_values, return_counts=True
        )
        pair_codes = tokens * base + route_values
        pair_ids, pair_counts = np.unique(pair_codes, return_counts=True)
        pair_tokens = pair_ids // base
        pair_routes = pair_ids % base
        route_positions = np.searchsorted(local_route_ids, pair_routes)
        joint_probabilities = pair_counts.astype(np.float64) / max(
            1, observation_count
        )
        return float(
            (
                joint_probabilities
                * (
                    np.log(pair_counts.astype(np.float64) * observation_count)
                    - np.log(token_counts[pair_tokens].astype(np.float64))
                    - np.log(
                        local_route_counts[route_positions].astype(np.float64)
                    )
                )
            ).sum()
        )

    token_route_mi = mutual_information(routes)
    shuffled_routes = np.random.default_rng(0).permutation(routes)
    shuffled_token_route_mi = mutual_information(shuffled_routes)
    excess_token_route_mi = token_route_mi - shuffled_token_route_mi
    order = np.argsort(route_counts)[::-1][:10]
    hard_activation_rate = (
        hard_tile_sum.to(dtype=torch.float64) / float(hard_block_count)
    )
    theoretical_route_count = math.comb(tile_count, active_tiles)
    return {
        "supported": True,
        "observation_count": observation_count,
        "tile_count": tile_count,
        "active_tiles": active_tiles,
        "theoretical_route_count": theoretical_route_count,
        "unique_route_count": int(route_ids.size),
        "observed_route_fraction": int(route_ids.size)
        / max(1, theoretical_route_count),
        "route_entropy_nats": route_entropy,
        "route_entropy_over_log_unique": (
            route_entropy / math.log(int(route_ids.size)) if route_ids.size > 1 else 0.0
        ),
        "top_route_share": float(route_counts.max() / max(1, observation_count)),
        "top_routes": [
            {
                "route_code": int(route_ids[index]),
                "count": int(route_counts[index]),
                "share": float(route_counts[index] / max(1, observation_count)),
            }
            for index in order
        ],
        "token_route_mutual_information_nats": token_route_mi,
        "shuffled_token_route_mutual_information_nats": shuffled_token_route_mi,
        "excess_token_route_mutual_information_nats": excess_token_route_mi,
        "token_route_mi_over_route_entropy": (
            token_route_mi / route_entropy if route_entropy > 0.0 else 0.0
        ),
        "excess_token_route_mi_over_route_entropy": (
            excess_token_route_mi / route_entropy if route_entropy > 0.0 else 0.0
        ),
        "hard_activation_rate": [float(value) for value in hard_activation_rate],
        "hard_hotspot_share": float(hard_activation_rate.max().item()),
        "hard_dead_tile_ratio": float(
            hard_activation_rate.eq(0.0).to(dtype=torch.float64).mean().item()
        ),
        "interpretation_boundary": (
            "Token-route MI measures statistical dependence between input token IDs "
            "and blockwise hard routes. The shuffled baseline estimates finite-sample "
            "bias; even positive excess MI does not by itself prove semantic interpretability."
        ),
    }


def _validate(
    model: torch.nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    cfg: ScaleConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    available_steps = math.ceil(inputs.shape[0] / cfg.batch_size)
    requested_steps = available_steps if cfg.full_validation else min(
        cfg.valid_steps, available_steps
    )
    completed_steps = 0
    tokens = 0
    total_ce = 0.0
    homeostasis: List[Dict[str, Any]] = []
    token_chunks: List[Tensor] = []
    route_chunks: List[Tensor] = []
    tile_count = cfg.state_dim // cfg.tile_size
    hard_tile_sum = torch.zeros(tile_count, dtype=torch.float64)
    hard_block_count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(requested_steps):
            start = step * cfg.batch_size
            batch_inputs = inputs[start : start + cfg.batch_size].to(device, non_blocking=True)
            batch_targets = targets[start : start + cfg.batch_size].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cfg.amp
            ):
                forward = _forward_model(model, batch_inputs, targets=batch_targets)
                loss_terms = _loss_breakdown(forward, training=False)
            ce = float(loss_terms["ce"].detach().float().item())
            if not math.isfinite(ce):
                raise RuntimeError(f"non-finite validation CE at step {step}")
            batch_tokens = int(batch_targets.numel())
            total_ce += ce * batch_tokens
            tokens += batch_tokens
            completed_steps += 1
            diagnostics = forward["diagnostics"]
            if diagnostics is not None:
                homeostasis.append(_diagnostics_record(diagnostics))
                hard_mask = diagnostics.route.hard_mask.detach().gt(0.5)
                hard_tile_sum += hard_mask.sum(dim=(0, 1)).to(
                    device="cpu", dtype=torch.float64
                )
                hard_block_count += int(hard_mask.shape[0] * hard_mask.shape[1])
                powers = torch.bitwise_left_shift(
                    torch.ones(tile_count, dtype=torch.int64, device=hard_mask.device),
                    torch.arange(tile_count, dtype=torch.int64, device=hard_mask.device),
                )
                block_codes = (hard_mask.to(dtype=torch.int64) * powers).sum(dim=-1)
                token_codes = block_codes.repeat_interleave(cfg.block_size, dim=1)[
                    :, : batch_inputs.shape[1]
                ]
                token_chunks.append(batch_inputs.detach().reshape(-1).cpu())
                route_chunks.append(token_codes.detach().reshape(-1).cpu())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    mean_ce = total_ce / max(1, tokens)
    return {
        "steps": completed_steps,
        "requested_steps": requested_steps,
        "available_steps": available_steps,
        "full_validation": cfg.full_validation,
        "sequences": min(int(inputs.shape[0]), completed_steps * cfg.batch_size),
        "available_sequences": int(inputs.shape[0]),
        "tokens": tokens,
        "available_tokens": int(targets.numel()),
        "coverage_fraction": tokens / max(1, int(targets.numel())),
        "elapsed_s": elapsed,
        "tokens_per_s": tokens / max(elapsed, 1e-12),
        "ce": mean_ce,
        "bpc": mean_ce / math.log(2.0),
        "finite": math.isfinite(mean_ce),
        "homeostasis": _mean_homeostasis(homeostasis),
        "route_specialization": _route_specialization(
            token_chunks,
            route_chunks,
            hard_tile_sum=hard_tile_sum,
            hard_block_count=hard_block_count,
            tile_count=tile_count,
            active_tiles=cfg.active_tiles,
        ),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ScaleConfig,
    completed_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "configuration": _configuration_record(cfg),
        "completed_steps": completed_steps,
    }
    if cfg.save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _full_epoch_batches(
    sequence_count: int,
    batch_size: int,
    epochs: int,
    *,
    generator: torch.Generator,
) -> Iterable[Tensor]:
    """Yield deterministic no-replacement batches, preserving partial tails."""

    if sequence_count <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("sequence_count, batch_size and epochs must be positive")
    for _ in range(epochs):
        order = torch.randperm(sequence_count, generator=generator)
        for start in range(0, sequence_count, batch_size):
            yield order[start : start + batch_size]


def run(cfg: ScaleConfig) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("this scaling runner requires CUDA")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")

    cache = _load_token_ids(cfg.cache_dir)
    train_inputs, train_targets = _make_sequences(cache["train_ids"], cfg.seq_len)
    valid_inputs, valid_targets = _make_sequences(cache["valid_ids"], cfg.seq_len)
    model, provenance = _build_scale_model(int(cache["vocab_size"]), cfg)
    shared_initialization = _apply_shared_initialization(model, cfg)
    parameter_stats = model.parameter_stats().as_dict()
    preflight = _preflight(cfg, model, device=device)
    steps_per_epoch = math.ceil(train_inputs.shape[0] / cfg.batch_size)
    requested_steps = (
        steps_per_epoch * cfg.epochs
        if cfg.train_mode == "full_epoch"
        else cfg.max_steps
    )
    base_artifact: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": "e3_fe2h_scale_train",
        "label": cfg.label,
        "configuration": {
            **_configuration_record(cfg),
            "precision": "fp16_autocast_fp32_master" if cfg.amp else "fp32",
            "requested_steps": requested_steps,
            "steps_per_epoch": steps_per_epoch,
        },
        "data": {
            "name": "cyberlangke/Nana-catgirl-dataset-110k",
            "vocab_size": int(cache["vocab_size"]),
            "train_tokens": int(cache["train_ids"].numel()),
            "valid_tokens": int(cache["valid_ids"].numel()),
            "train_sequences": int(train_inputs.shape[0]),
            "valid_sequences": int(valid_inputs.shape[0]),
            "train_sequence_tokens": int(train_targets.numel()),
            "valid_sequence_tokens": int(valid_targets.numel()),
            "tokenizer_path": str(cache["tokenizer_path"]),
        },
        "model": {
            "parameter_stats": parameter_stats,
            "provenance": provenance,
            "shared_initialization": shared_initialization,
        },
        "preflight": preflight,
    }
    if not preflight["can_launch"]:
        return {
            **base_artifact,
            "status": "NOT_LAUNCHED",
            "failure": {"type": preflight["status"], "message": preflight["reason"]},
        }

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.amp,
        init_scale=cfg.amp_init_scale,
        growth_interval=cfg.amp_growth_interval,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    sampler = NvidiaSmiSampler(cfg.sample_interval_ms)
    progress = None
    if cfg.tqdm_progress:
        try:
            from tqdm import tqdm
        except ImportError as error:
            raise RuntimeError("--tqdm-progress requires the tqdm package") from error
        progress = tqdm(
            total=requested_steps,
            desc=cfg.label,
            unit="step",
            mininterval=1.0,
            smoothing=0.1,
            dynamic_ncols=False,
            file=sys.stderr,
        )
    windows: List[Dict[str, Any]] = []
    all_ce: List[float] = []
    all_objective: List[float] = []
    all_homeostasis: List[Dict[str, Any]] = []
    total_training_tokens = 0
    total_training_ce = 0.0
    total_training_objective = 0.0
    periodic_checkpoint_steps: List[int] = []
    failure: Optional[Dict[str, Any]] = None
    completed_steps = 0
    training_start = time.monotonic()
    steady_start = training_start
    training_end = training_start
    sampler.start()
    generator = torch.Generator().manual_seed(cfg.seed)
    window_started = time.perf_counter()
    window_step_start = 0
    window_tokens = 0
    window_ce: List[float] = []
    window_objective: List[float] = []
    window_route: List[float] = []
    window_homeostasis_loss: List[float] = []
    window_grad_norm: List[float] = []
    window_homeostasis: List[Dict[str, Any]] = []
    full_epoch_batches = (
        iter(
            _full_epoch_batches(
                train_inputs.shape[0],
                cfg.batch_size,
                cfg.epochs,
                generator=generator,
            )
        )
        if cfg.train_mode == "full_epoch"
        else None
    )
    model.train()
    try:
        for step in range(requested_steps):
            if cfg.train_mode == "full_epoch":
                assert full_epoch_batches is not None
                selection = next(full_epoch_batches)
            else:
                selection = torch.randint(
                    0,
                    train_inputs.shape[0],
                    (cfg.batch_size,),
                    generator=generator,
                )
            batch_inputs = train_inputs.index_select(0, selection).to(
                device, non_blocking=True
            )
            batch_targets = train_targets.index_select(0, selection).to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cfg.amp
            ):
                forward = _forward_model(model, batch_inputs, targets=batch_targets)
                loss_terms = _apply_loss_weights(
                    _loss_breakdown(forward, training=True),
                    cfg=cfg,
                )
            _finite_loss_terms(loss_terms, step)
            scale_before = float(scaler.get_scale())
            scaler.scale(loss_terms["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm = float(grad_norm_tensor.detach().float().item())
            if not math.isfinite(grad_norm):
                raise FE2HFiniteGuardError(
                    scope="gradient", name="global_norm", step=step, index=(0,), value=grad_norm
                )
            full_finite_check = step == 0 or (step + 1) % cfg.finite_check_every == 0
            if full_finite_check:
                run_fe2h_finite_guard(model, loss_terms={
                    "ce": loss_terms["ce"],
                    "route_supervision": loss_terms["route_supervision"],
                    "homeostasis": loss_terms["homeostasis"],
                    "total": loss_terms["total"],
                }, step=step)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                raise FE2HFiniteGuardError(
                    scope="gradient", name="amp_scale_drop", step=step, index=(0,), value=scale_after
                )
            if full_finite_check:
                run_fe2h_finite_guard(model, optimizer=optimizer, step=step)

            completed_steps = step + 1
            if completed_steps == cfg.warmup_steps:
                steady_start = time.monotonic()
            ce = float(loss_terms["ce"].detach().float().item())
            objective = float(loss_terms["total"].detach().float().item())
            route_loss = float(loss_terms["route_supervision"].detach().float().item())
            homeostasis_loss = float(loss_terms["homeostasis"].detach().float().item())
            diagnostics = forward["diagnostics"]
            if diagnostics is None:
                raise RuntimeError("FE-2H forward did not return diagnostics")
            homeostasis_record = _diagnostics_record(diagnostics)
            token_count = int(batch_targets.numel())
            total_training_tokens += token_count
            total_training_ce += ce * token_count
            total_training_objective += objective * token_count
            all_ce.append(ce)
            all_objective.append(objective)
            all_homeostasis.append(homeostasis_record)
            window_ce.append(ce)
            window_objective.append(objective)
            window_route.append(route_loss)
            window_homeostasis_loss.append(homeostasis_loss)
            window_grad_norm.append(grad_norm)
            window_homeostasis.append(homeostasis_record)
            window_tokens += token_count
            if progress is not None:
                progress.update(1)
            if completed_steps % cfg.log_every == 0 or completed_steps == requested_steps:
                torch.cuda.synchronize(device)
                record = _window_record(
                    step_start=window_step_start,
                    step_end=completed_steps,
                    started=window_started,
                    tokens=window_tokens,
                    ce_values=window_ce,
                    objective_values=window_objective,
                    route_values=window_route,
                    homeostasis_loss_values=window_homeostasis_loss,
                    grad_norm_values=window_grad_norm,
                    homeostasis_records=window_homeostasis,
                )
                windows.append(record)
                if progress is not None:
                    progress.set_postfix(
                        bpc=f"{record['bpc']:.3f}",
                        grad=f"{record['grad_norm']['mean']:.2f}",
                        hot=f"{record['homeostasis']['hotspot_share']:.3f}",
                    )
                print(json.dumps({"label": cfg.label, "window": record}, ensure_ascii=False), flush=True)
                window_started = time.perf_counter()
                window_step_start = completed_steps
                window_tokens = 0
                window_ce = []
                window_objective = []
                window_route = []
                window_homeostasis_loss = []
                window_grad_norm = []
                window_homeostasis = []
            if (
                cfg.checkpoint is not None
                and cfg.checkpoint_every_steps > 0
                and completed_steps % cfg.checkpoint_every_steps == 0
            ):
                _save_checkpoint(
                    cfg.checkpoint,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    completed_steps=completed_steps,
                )
                periodic_checkpoint_steps.append(completed_steps)
                print(
                    json.dumps(
                        {
                            "label": cfg.label,
                            "checkpoint": {
                                "step": completed_steps,
                                "path": str(cfg.checkpoint),
                                "bytes": cfg.checkpoint.stat().st_size,
                                "includes_optimizer": cfg.save_optimizer,
                                "rolling": True,
                            },
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        torch.cuda.synchronize(device)
        training_end = time.monotonic()
    except Exception as error:
        torch.cuda.synchronize(device)
        training_end = time.monotonic()
        failure = {
            "step": completed_steps,
            "type": error.__class__.__name__,
            "message": str(error),
        }
    finally:
        if progress is not None:
            progress.close()
        sampler.stop()

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    full_gpu_summary = sampler.summary(start=training_start, end=training_end)
    steady_gpu_summary = sampler.summary(start=steady_start, end=training_end)
    training_elapsed = max(training_end - training_start, 1e-12)
    validation: Optional[Dict[str, Any]] = None
    checkpoint_record: Optional[Dict[str, Any]] = None
    if failure is None and completed_steps == requested_steps:
        validation = _validate(
            model,
            valid_inputs,
            valid_targets,
            cfg=cfg,
            device=device,
        )
        if cfg.checkpoint is not None:
            _save_checkpoint(
                cfg.checkpoint,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                completed_steps=completed_steps,
            )
            checkpoint_record = {
                "path": str(cfg.checkpoint),
                "bytes": cfg.checkpoint.stat().st_size,
                "includes_optimizer": cfg.save_optimizer,
                "periodic_interval_steps": cfg.checkpoint_every_steps,
                "periodic_steps": periodic_checkpoint_steps,
                "last_saved_step": completed_steps,
                "rolling": False,
                "atomic_replace": True,
            }
    elif cfg.checkpoint is not None and periodic_checkpoint_steps and cfg.checkpoint.exists():
        checkpoint_record = {
            "path": str(cfg.checkpoint),
            "bytes": cfg.checkpoint.stat().st_size,
            "includes_optimizer": cfg.save_optimizer,
            "periodic_interval_steps": cfg.checkpoint_every_steps,
            "periodic_steps": periodic_checkpoint_steps,
            "last_saved_step": periodic_checkpoint_steps[-1],
            "rolling": True,
            "atomic_replace": True,
        }

    mean_ce = (
        total_training_ce / total_training_tokens if total_training_tokens else None
    )
    first_window_bpc = windows[0]["bpc"] if windows else None
    last_window_bpc = windows[-1]["bpc"] if windows else None
    utilization = steady_gpu_summary["gpu_util_percent"]
    memory_used = full_gpu_summary["memory_used_mib"]
    saturation = {
        "utilization_target": "p50>=90 and p90>=95",
        "memory_target": (
            "device max stays within the 32 GiB user limit and the "
            f"{preflight['physical_launch_limit_gib']:.3f} GiB physical launch limit"
        ),
        "utilization_pass": bool(
            utilization["p50"] is not None
            and utilization["p90"] is not None
            and utilization["p50"] >= 90.0
            and utilization["p90"] >= 95.0
        ),
        "memory_pass": bool(
            memory_used["max"] is not None
            and memory_used["max"]
            <= min(
                MEMORY_REFUSE_GIB,
                float(preflight["physical_launch_limit_gib"]),
            )
            * 1024
        ),
        "learning_signal_pass": bool(
            first_window_bpc is not None
            and last_window_bpc is not None
            and last_window_bpc < first_window_bpc
            and validation is not None
            and validation["finite"]
        ),
        "homeostasis_pass": bool(
            all_homeostasis
            and max(record["hotspot_share"] for record in all_homeostasis)
            <= min(
                1.0,
                1.4 * cfg.active_tiles / (cfg.state_dim // cfg.tile_size),
            )
            and max(record["dead_tile_ratio"] for record in all_homeostasis) == 0.0
        ),
    }
    validation_complete = bool(
        validation is not None
        and validation["finite"]
        and (
            not cfg.full_validation
            or validation["tokens"] == int(valid_targets.numel())
        )
    )
    expected_full_epoch_tokens = (
        int(train_targets.numel()) * cfg.epochs
        if cfg.train_mode == "full_epoch"
        else None
    )
    scaled_homeostasis_cap = min(
        1.0,
        1.4 * cfg.active_tiles / (cfg.state_dim // cfg.tile_size),
    )
    route_specialization = (
        validation.get("route_specialization", {}) if validation is not None else {}
    )
    hard_homeostasis_pass = bool(
        route_specialization.get("supported")
        and route_specialization.get("hard_dead_tile_ratio") == 0.0
        and route_specialization.get("hard_hotspot_share", float("inf"))
        <= scaled_homeostasis_cap
    )
    specialization_pass = bool(
        route_specialization.get("supported")
        and route_specialization.get("unique_route_count", 0) > 1
        and route_specialization.get("top_route_share", 1.0) < 0.70
        and route_specialization.get(
            "excess_token_route_mutual_information_nats", 0.0
        )
        > 0.0
    )
    matched_gates = {
        "training_complete": bool(
            failure is None
            and completed_steps == requested_steps
            and (
                expected_full_epoch_tokens is None
                or total_training_tokens == expected_full_epoch_tokens
            )
        ),
        "validation_complete": validation_complete,
        "finite": failure is None and validation_complete,
        "scaled_homeostasis_cap": scaled_homeostasis_cap,
        "soft_homeostasis_pass": saturation["homeostasis_pass"],
        "hard_homeostasis_pass": hard_homeostasis_pass,
        "specialization_pass": specialization_pass,
    }
    return {
        **base_artifact,
        "status": "COMPLETED" if failure is None else "FAILED_CLOSED",
        "training": {
            "completed_steps": completed_steps,
            "requested_steps": requested_steps,
            "train_mode": cfg.train_mode,
            "epochs": cfg.epochs,
            "steps_per_epoch": steps_per_epoch,
            "tokens": total_training_tokens,
            "expected_full_epoch_tokens": expected_full_epoch_tokens,
            "coverage_fraction": (
                total_training_tokens / expected_full_epoch_tokens
                if expected_full_epoch_tokens is not None
                else None
            ),
            "elapsed_s": training_elapsed,
            "tokens_per_s": total_training_tokens / training_elapsed,
            "mean_ce": mean_ce,
            "mean_bpc": None if mean_ce is None else mean_ce / math.log(2.0),
            "mean_objective": (
                total_training_objective / total_training_tokens
                if total_training_tokens
                else None
            ),
            "loss_weights": {
                "route_supervision": cfg.route_supervision_weight,
                "homeostasis": cfg.homeostasis_weight,
            },
            "amp_scale": {
                "initial": cfg.amp_init_scale if cfg.amp else None,
                "final": float(scaler.get_scale()) if cfg.amp else None,
                "growth_interval": cfg.amp_growth_interval if cfg.amp else None,
            },
            "windows": windows,
            "homeostasis": _mean_homeostasis(all_homeostasis),
            "finite_check_protocol": {
                "loss": "every step",
                "gradient_global_norm": "every step after AMP unscale",
                "parameters_gradients_optimizer": f"step 0 and every {cfg.finite_check_every} steps",
                "amp_overflow": "fail closed on any scale drop",
            },
        },
        "validation": validation,
        "memory": {
            "torch_peak_allocated_bytes": peak_allocated,
            "torch_peak_allocated_gib": peak_allocated / GIB,
            "torch_peak_reserved_bytes": peak_reserved,
            "torch_peak_reserved_gib": peak_reserved / GIB,
            "device_samples_full_training": full_gpu_summary,
        },
        "gpu_utilization": {
            "full_training": full_gpu_summary,
            "post_warmup": steady_gpu_summary,
        },
        "saturation": saturation,
        "matched_gates": matched_gates,
        "checkpoint": checkpoint_record,
        "failure": failure,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> ScaleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=REPO_ROOT / "results" / "e3_sg29_cache"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--label", default="scale")
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--state-dim", type=int, required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--active-tiles", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--train-mode", choices=("random", "full_epoch"), default="random"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--valid-steps", type=int, default=4)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--finite-check-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--route-supervision-weight", type=float, default=0.01)
    parser.add_argument("--homeostasis-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-init-scale", type=float, default=65536.0)
    parser.add_argument("--amp-growth-interval", type=int, default=2000)
    parser.add_argument("--dense-output", action="store_true")
    parser.add_argument("--sample-interval-ms", type=int, default=200)
    parser.add_argument("--estimator-factor", type=float, default=1.5)
    parser.add_argument("--estimator-overhead-gib", type=float, default=0.75)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--tqdm-progress", action="store_true")
    parser.add_argument("--shared-init", type=Path, default=None)
    parser.add_argument("--create-shared-init", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.batch_size <= 0 or args.seq_len <= 0:
        parser.error("max-steps, batch-size and seq-len must be positive")
    if args.warmup_steps < 0 or args.warmup_steps > args.max_steps:
        parser.error("warmup-steps must lie in [0, max-steps]")
    if args.log_every <= 0 or args.finite_check_every <= 0:
        parser.error("log-every and finite-check-every must be positive")
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.route_supervision_weight < 0.0 or args.homeostasis_weight < 0.0:
        parser.error("auxiliary loss weights must be non-negative")
    if args.amp_init_scale <= 0.0 or args.amp_growth_interval <= 0:
        parser.error("AMP init scale and growth interval must be positive")
    if args.create_shared_init and args.shared_init is None:
        parser.error("--create-shared-init requires --shared-init")
    if args.checkpoint_every_steps < 0:
        parser.error("checkpoint-every-steps must be non-negative")
    if args.checkpoint_every_steps > 0 and args.checkpoint is None:
        parser.error("--checkpoint-every-steps requires --checkpoint")
    return ScaleConfig(
        out=args.out,
        cache_dir=args.cache_dir,
        checkpoint=args.checkpoint,
        label=args.label,
        d_model=args.d_model,
        state_dim=args.state_dim,
        tile_size=args.tile_size,
        active_tiles=args.active_tiles,
        block_size=args.block_size,
        rank=args.rank,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        train_mode=args.train_mode,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        valid_steps=args.valid_steps,
        full_validation=args.full_validation,
        log_every=args.log_every,
        finite_check_every=args.finite_check_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        route_supervision_weight=args.route_supervision_weight,
        homeostasis_weight=args.homeostasis_weight,
        seed=args.seed,
        amp=not args.no_amp,
        amp_init_scale=args.amp_init_scale,
        amp_growth_interval=args.amp_growth_interval,
        low_rank_output=not args.dense_output,
        sample_interval_ms=args.sample_interval_ms,
        estimator_factor=args.estimator_factor,
        estimator_overhead_gib=args.estimator_overhead_gib,
        save_optimizer=args.save_optimizer,
        checkpoint_every_steps=args.checkpoint_every_steps,
        tqdm_progress=args.tqdm_progress,
        shared_init=args.shared_init,
        create_shared_init=args.create_shared_init,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(argv)
    artifact: Dict[str, Any]
    try:
        artifact = run(cfg)
    except Exception as error:
        artifact = {
            "schema_version": 1,
            "experiment": "e3_fe2h_scale_train",
            "label": cfg.label,
            "status": "SETUP_FAILED",
            "failure": {"type": error.__class__.__name__, "message": str(error)},
        }
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = cfg.out.with_suffix(cfg.out.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, cfg.out)
    print(json.dumps({"status": artifact["status"], "failure": artifact.get("failure")}, ensure_ascii=False))
    print(f"wrote {cfg.out}")
    return 0 if artifact["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
