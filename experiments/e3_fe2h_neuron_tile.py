#!/usr/bin/env python3
"""FE-2H neuron-tile experiment driver with fail-closed artifact gates.

This worker owns only the experiment driver and its artifact-decision tests.
The implementation stays inside those boundaries: it consumes the existing
FE-2H core/low-rank helpers without modifying them, and it does not write
canonical `results/` artifacts unless the caller explicitly asks for an `--out`
path.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vpsc.world_model.catgirl_corpus import load_bpe_corpus, make_sequences  # noqa: E402
from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore,
    count_parameters,
    state_nbytes,
)
from vpsc.world_model.devices import choose_device, device_label, synchronize  # noqa: E402
from vpsc.world_model.fe2h_low_rank import (  # noqa: E402
    LowRankLinear,
    ProjectionConfig,
    build_projection,
    matched_projection_report,
)
from vpsc.world_model.fe2h_tile_sparse import (  # noqa: E402
    FE2HNeuronTileCore,
    FE2HRoute,
    FE2HUnsupportedError,
    run_fe2h_finite_guard,
)
from vpsc.world_model.lm import CausalLanguageModel  # noqa: E402


SCHEMA_VERSION = 1
REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "formal",
    "environment",
    "configuration",
    "provenance",
    "mechanism",
    "numerics",
    "memory",
    "speed",
    "quality",
    "decision",
)
GATE_ORDER = ("mechanism", "numerics", "memory", "speed", "quality")
GATE_TERMINAL = frozenset({"FAIL", "PAUSE", "REFUSE", "UNSUPPORTED", "NOT_RUN"})
MECHANISM_MAX_ABS_TOLERANCE = 1e-5
MEMORY_WARNING_GIB = 16.0
MEMORY_REFUSE_GIB = 32.0
HOMEOSTASIS_HOTSPOT_MAX = 0.70
HOMEOSTASIS_DEAD_TILE_RATIO_MAX = 0.0
HOMEOSTASIS_ACTIVATION_RATE_MAX_ABS_TOLERANCE = 0.25
ROUTE_SUPERVISION_LOSS_WEIGHT = 0.01
HOMEOSTASIS_LOSS_WEIGHT = 0.01


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    device_request: str
    out: Path
    cache_dir: Path
    d_model: int = 128
    state_dim: int = 128
    tile_size: int = 32
    active_tiles: int = 2
    block_size: int = 32
    rank: int = 16
    batch_size: int = 8
    seq_len: int = 32
    max_convs: Optional[int] = None
    epochs: int = 1
    seed: int = 0
    warmup_steps: int = 2
    benchmark_steps: int = 4
    vocab_size: int = 8192
    smoke_vocab_size: int = 64
    smoke_train_sequences: int = 24
    smoke_valid_sequences: int = 8
    svd_init: bool = False


def predicted_memory_gate(predicted_gib: float) -> Dict[str, Any]:
    """Return the conservative launch decision for predicted memory in GiB."""

    if not math.isfinite(predicted_gib):
        return {
            "status": "REFUSE",
            "can_launch": False,
            "predicted_gib": predicted_gib,
            "message": "predicted memory is non-finite",
        }
    if predicted_gib > MEMORY_REFUSE_GIB:
        return {
            "status": "REFUSE",
            "can_launch": False,
            "predicted_gib": predicted_gib,
            "message": f"predicted memory {predicted_gib:.3f} GiB exceeds {MEMORY_REFUSE_GIB:.0f} GiB refuse threshold",
        }
    if predicted_gib > MEMORY_WARNING_GIB:
        return {
            "status": "PAUSE",
            "can_launch": False,
            "predicted_gib": predicted_gib,
            "message": f"predicted memory {predicted_gib:.3f} GiB exceeds {MEMORY_WARNING_GIB:.0f} GiB warning threshold",
        }
    return {
        "status": "ALLOW",
        "can_launch": True,
        "predicted_gib": predicted_gib,
        "message": f"predicted memory {predicted_gib:.3f} GiB is within launch threshold",
    }


def _validate_path_label(
    record: Mapping[str, Any], *, variant_name: str, expected: str
) -> List[str]:
    if not record or record.get("path_label") == expected:
        return []
    return [f"{variant_name} must use path_label='{expected}'"]


def _validate_dense_mask_path(record: Mapping[str, Any]) -> List[str]:
    errors = _validate_path_label(
        record,
        variant_name="fe2h_dense_mask",
        expected="dense_mask",
    )
    if bool(record.get("hardware_executed_sparsity")):
        errors.append("dense-mask path cannot claim hardware_executed_sparsity=true")
    return errors


def _validate_sparse_path(
    record: Mapping[str, Any],
    *,
    variant_name: str,
    expected_label: str,
    display_label: str,
    require_dense_input_retained: bool = False,
) -> List[str]:
    errors = _validate_path_label(
        record,
        variant_name=variant_name,
        expected=expected_label,
    )
    if bool(record.get("supported")):
        if not bool(record.get("hardware_executed_sparsity")):
            errors.append(
                f"supported {display_label} path must report real hardware_executed_sparsity"
            )
        if require_dense_input_retained and not bool(
            record.get("dense_input_projection_retained")
        ):
            errors.append(
                "true sparse low-rank path must disclose dense_input_projection_retained=true"
            )
        return errors
    if not record.get("unsupported_reason"):
        errors.append(f"unsupported {display_label} path must carry unsupported_reason")
    return errors


def _validate_all_low_rank_dense_mask_path(record: Mapping[str, Any]) -> List[str]:
    return _validate_path_label(
        record,
        variant_name="all_lowrank_dense_mask",
        expected="all_lowrank_dense_mask",
    )


def validate_variant_paths(variant_paths: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Validate that artifact path labels stay honest about sparse execution."""

    errors: List[str] = []
    dense = dict(variant_paths.get("fe2h_dense_mask", {}))
    sparse = dict(variant_paths.get("fe2h_tile_sparse", {}))
    low_rank = dict(variant_paths.get("fe2h_low_rank_tile_sparse", {}))
    all_low_rank = dict(variant_paths.get("all_lowrank_dense_mask", {}))
    if dense:
        errors.extend(_validate_dense_mask_path(dense))
    if sparse:
        errors.extend(
            _validate_sparse_path(
                sparse,
                variant_name="fe2h_tile_sparse",
                expected_label="tile_sparse",
                display_label="tile-sparse",
            )
        )
    if low_rank:
        errors.extend(
            _validate_sparse_path(
                low_rank,
                variant_name="fe2h_low_rank_tile_sparse",
                expected_label="low_rank_tile_sparse",
                display_label="low-rank tile-sparse",
                require_dense_input_retained=True,
            )
        )
    if all_low_rank:
        errors.extend(_validate_all_low_rank_dense_mask_path(all_low_rank))
    return errors


def validate_artifact_schema(artifact: Mapping[str, Any]) -> List[str]:
    """Return a list of top-level artifact schema violations."""

    errors: List[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if artifact.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(artifact.get("formal"), bool):
        errors.append("formal must be a boolean")
    for field in (
        "environment",
        "configuration",
        "provenance",
        "mechanism",
        "numerics",
        "memory",
        "speed",
        "quality",
        "decision",
    ):
        if not isinstance(artifact.get(field), Mapping):
            errors.append(f"{field} must be a mapping")
    provenance = artifact.get("provenance")
    if isinstance(provenance, Mapping) and "variant_paths" in provenance:
        variant_paths = provenance["variant_paths"]
        if not isinstance(variant_paths, Mapping):
            errors.append("provenance.variant_paths must be a mapping")
        else:
            errors.extend(validate_variant_paths(variant_paths))
    decision = artifact.get("decision")
    if isinstance(decision, Mapping):
        if tuple(decision.get("gate_order", ())) != GATE_ORDER:
            errors.append("decision.gate_order must preserve mechanism->numerics->memory->speed->quality")
        if "overall" not in decision:
            errors.append("decision.overall is required")
    return errors


def make_gate_decision(
    *,
    mechanism: Mapping[str, Any],
    numerics: Mapping[str, Any],
    memory: Mapping[str, Any],
    speed: Mapping[str, Any],
    quality: Mapping[str, Any],
    variant_paths: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Normalise gate order, preserve negative results, and compute overall status."""

    inputs = {
        "mechanism": dict(mechanism),
        "numerics": dict(numerics),
        "memory": dict(memory),
        "speed": dict(speed),
        "quality": dict(quality),
    }
    path_errors = validate_variant_paths(variant_paths or {})
    if path_errors:
        inputs["mechanism"].setdefault("details", [])
        details = list(inputs["mechanism"].get("details", []))
        details.extend(path_errors)
        inputs["mechanism"]["details"] = details
        inputs["mechanism"]["status"] = "FAIL"

    if _is_memory_preflight_block(inputs["mechanism"], inputs["numerics"], inputs["memory"]):
        memory_status = str(inputs["memory"].get("status", "NOT_RUN")).upper()
        normalised = {
            "mechanism": "NOT_RUN",
            "numerics": "NOT_RUN",
            "memory": memory_status,
            "speed": "NOT_RUN",
            "quality": "NOT_RUN",
        }
        return {
            "gate_order": list(GATE_ORDER),
            "mechanism_gate": normalised["mechanism"],
            "numerics_gate": normalised["numerics"],
            "memory_gate": normalised["memory"],
            "speed_gate": normalised["speed"],
            "quality_gate": normalised["quality"],
            "overall": memory_status,
            "first_blocker": "memory",
            "quality_executed": False,
            "retained_negative_result": bool(speed.get("retained_negative_result")),
            "path_honesty_errors": path_errors,
            "summary": f"memory preflight blocked launch with status={memory_status}",
        }

    normalised: Dict[str, str] = {}
    blocking_seen = False
    for gate in GATE_ORDER:
        status = str(inputs[gate].get("status", "NOT_RUN")).upper()
        if blocking_seen:
            status = "NOT_RUN"
        normalised[gate] = status
        blocking_seen = status in GATE_TERMINAL

    if (
        normalised["mechanism"] == "PASS"
        and normalised["numerics"] == "PASS"
        and normalised["memory"] == "PASS"
        and normalised["speed"] == "FAIL"
    ):
        overall = "NEGATIVE"
    elif normalised["memory"] == "REFUSE":
        overall = "REFUSE"
    elif "PAUSE" in normalised.values():
        overall = "PAUSE"
    elif all(status == "PASS" for status in normalised.values()):
        overall = "PASS"
    else:
        overall = "FAIL"

    first_blocker = None
    for gate in GATE_ORDER:
        status = normalised[gate]
        if status != "PASS":
            first_blocker = gate
            break

    retained_negative_result = (
        normalised["speed"] in {"FAIL", "UNSUPPORTED"}
        or bool(speed.get("retained_negative_result"))
    )
    return {
        "gate_order": list(GATE_ORDER),
        "mechanism_gate": normalised["mechanism"],
        "numerics_gate": normalised["numerics"],
        "memory_gate": normalised["memory"],
        "speed_gate": normalised["speed"],
        "quality_gate": normalised["quality"],
        "overall": overall,
        "first_blocker": first_blocker,
        "quality_executed": normalised["quality"] == "PASS",
        "retained_negative_result": retained_negative_result,
        "path_honesty_errors": path_errors,
        "summary": _decision_summary(normalised, overall),
    }


def _decision_summary(normalised: Mapping[str, str], overall: str) -> str:
    if overall == "PASS":
        return "all gates passed"
    if overall == "NEGATIVE":
        return "mechanism, numerics, and memory passed, but sparse speedup stayed below 1.0x; preserve negative result"
    for gate in GATE_ORDER:
        status = normalised[gate]
        if status != "PASS":
            return f"{gate} gate ended the run with status={status}"
    return "run incomplete"


def _path(value: Path) -> str:
    return str(value.resolve())


def _scalar(value: Tensor) -> float:
    if not isinstance(value, Tensor):
        return float(value)
    return float(value.detach().cpu().item())


def _float_list(value: Tensor) -> List[float]:
    if not isinstance(value, Tensor):
        return [float(item) for item in value]
    return [float(item) for item in value.detach().cpu().tolist()]


def _environment(device: torch.device, requested: str) -> Dict[str, Any]:
    env = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "requested_device": requested,
        "resolved_device": str(device),
        "device_label": device_label(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        env["nvidia_device"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        }
    else:
        env["nvidia_device"] = None
    return env


def _projection_report(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    bias: bool = True,
) -> Dict[str, Any]:
    try:
        return {
            "supported": True,
            "report": matched_projection_report(
                in_features, out_features, rank, bias=bias
            ).as_dict(),
            "unsupported_reason": None,
        }
    except ValueError as error:
        return {
            "supported": False,
            "report": None,
            "unsupported_reason": str(error),
        }


def _build_all_low_rank_dense_mask_report(cfg: ExperimentConfig) -> Dict[str, Any]:
    tile_count = cfg.state_dim // cfg.tile_size
    input_report = _projection_report(cfg.d_model, 4 * cfg.state_dim, cfg.rank)
    router_first = _projection_report(4, 64, cfg.rank)
    router_second = _projection_report(64, tile_count, cfg.rank)
    output_report = _projection_report(4 * cfg.state_dim, cfg.d_model, cfg.rank)
    supported = all(
        report["supported"]
        for report in (input_report, router_first, router_second, output_report)
    )
    reasons = [
        report["unsupported_reason"]
        for report in (input_report, router_first, router_second, output_report)
        if report["unsupported_reason"] is not None
    ]
    return {
        "path_label": "all_lowrank_dense_mask",
        "requested_rank": cfg.rank,
        "supported": supported,
        "hardware_executed_sparsity": False,
        "unsupported_reason": reasons[0] if reasons else None,
        "projection_reports": {
            "input_event_projection": input_report,
            "router_projection_first_linear": router_first,
            "router_projection_second_linear": router_second,
            "output_projection": output_report,
        },
        "notes": [
            "This report is honest dense-mask accounting only; it is not claimed as true sparse execution.",
            "Router low-rank support is fail-closed when rank 16/32 exceeds the router dimensions.",
        ],
    }


def _build_base_model(vocab_size: int, cfg: ExperimentConfig) -> CausalLanguageModel:
    return CausalLanguageModel(
        vocab_size,
        E3GatedTraceScanCore(cfg.d_model, cfg.d_model, state_dim=cfg.state_dim),
    )


def _build_fe2h_model(
    vocab_size: int,
    cfg: ExperimentConfig,
    *,
    low_rank_output: bool,
) -> Tuple[Optional[CausalLanguageModel], Dict[str, Any]]:
    provenance: Dict[str, Any] = {
        "path_label": "low_rank_tile_sparse" if low_rank_output else "dense_mask",
        "supported": True,
        "hardware_executed_sparsity": False if not low_rank_output else True,
        "dense_input_projection_retained": True,
        "input_projection_kind": "dense",
        "router_projection_kind": "dense_router_mlp",
        "output_projection_kind": "low_rank" if low_rank_output else "dense",
        "output_projection_provenance": None,
        "unsupported_reason": None,
        "notes": [],
    }
    output_projection: nn.Module
    if low_rank_output:
        dense_output = nn.Linear(4 * cfg.state_dim, cfg.d_model)
        try:
            if cfg.svd_init:
                output_projection = build_projection(
                    4 * cfg.state_dim,
                    cfg.d_model,
                    config=ProjectionConfig(
                        kind="low_rank",
                        rank=cfg.rank,
                        init="svd",
                        source_name="output_projection",
                    ),
                    dense_source=dense_output,
                )
            else:
                output_projection = build_projection(
                    4 * cfg.state_dim,
                    cfg.d_model,
                    config=ProjectionConfig(kind="low_rank", rank=cfg.rank),
                )
        except ValueError as error:
            provenance["supported"] = False
            provenance["hardware_executed_sparsity"] = False
            provenance["unsupported_reason"] = str(error)
            provenance["notes"].append("low-rank output projection build failed; no fallback was applied")
            return None, provenance
        provenance["output_projection_provenance"] = (
            output_projection.provenance_dict()
            if isinstance(output_projection, LowRankLinear)
            else None
        )
        provenance["notes"].append(
            "True sparse path keeps input_event_projection dense because sparse slicing currently requires nn.Linear."
        )
    else:
        output_projection = nn.Linear(4 * cfg.state_dim, cfg.d_model)

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
    return model, provenance


def _build_models(
    vocab_size: int, cfg: ExperimentConfig
) -> Tuple[Dict[str, Optional[CausalLanguageModel]], Dict[str, Dict[str, Any]]]:
    dense_model, dense_provenance = _build_fe2h_model(
        vocab_size, cfg, low_rank_output=False
    )
    low_rank_model, low_rank_provenance = _build_fe2h_model(
        vocab_size, cfg, low_rank_output=True
    )
    variant_paths = {
        "base_e3": {
            "path_label": "base_e3",
            "supported": True,
            "hardware_executed_sparsity": False,
        },
        "fe2h_dense_mask": dense_provenance,
        "fe2h_tile_sparse": {
            "path_label": "tile_sparse",
            "supported": True,
            "hardware_executed_sparsity": True,
            "route_protocol": "eval+no_grad route_blocks, then route_override into sparse_inference",
            "dense_input_projection_retained": True,
            "unsupported_reason": None,
        },
        "fe2h_low_rank_tile_sparse": low_rank_provenance,
        "all_lowrank_dense_mask": _build_all_low_rank_dense_mask_report(cfg),
    }
    models: Dict[str, Optional[CausalLanguageModel]] = {
        "base_e3": _build_base_model(vocab_size, cfg),
        "fe2h_dense_mask": dense_model,
        "fe2h_low_rank_tile_sparse": low_rank_model,
    }
    return models, variant_paths


def _iter_tensors(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_tensors(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_tensors(child)


def _state_max_abs_diff(left: Any, right: Any) -> float:
    maximum = 0.0
    left_tensors = list(_iter_tensors(left))
    right_tensors = list(_iter_tensors(right))
    if len(left_tensors) != len(right_tensors):
        raise ValueError("state tensor counts do not match")
    for left_tensor, right_tensor in zip(left_tensors, right_tensors):
        diff = (left_tensor - right_tensor).abs().max().item()
        maximum = max(maximum, float(diff))
    return maximum


def _forward_model(
    model: CausalLanguageModel,
    input_ids: Tensor,
    *,
    targets: Optional[Tensor] = None,
    route_override: Optional[FE2HRoute] = None,
    sparse_inference: bool = False,
) -> Dict[str, Any]:
    if isinstance(model.core, FE2HNeuronTileCore):
        embedded = model.input_dropout(model.embedding(input_ids))
        core_result, diagnostics = model.core.forward_dynamics(
            embedded,
            sparse_inference=sparse_inference,
            route_override=route_override,
        )
        hidden = model.output_dropout(model.output_norm(core_result.sequence))
        logits = model.lm_head(hidden)
        loss = None
        target_count = None
        if targets is not None:
            loss, target_count = model._loss(logits, targets)
        return {
            "logits": logits,
            "hidden_states": hidden,
            "state": core_result.state,
            "loss": loss,
            "target_count": target_count,
            "diagnostics": diagnostics,
        }
    output = model(input_ids, targets=targets)
    return {
        "logits": output.logits,
        "hidden_states": output.hidden_states,
        "state": output.state,
        "loss": output.loss,
        "target_count": output.target_count,
        "diagnostics": None,
    }


def _prepare_sparse_route(
    model: CausalLanguageModel, input_ids: Tensor
) -> Tuple[Optional[FE2HRoute], Optional[str]]:
    if not isinstance(model.core, FE2HNeuronTileCore):
        return None, "sparse route override is only defined for FE2H cores"
    with torch.no_grad():
        model.eval()
        embedded = model.input_dropout(model.embedding(input_ids))
        route = model.core.route_blocks(embedded)
    return route, None


def _slice_batch(inputs: Tensor, targets: Tensor, batch_size: int) -> Iterable[Tuple[Tensor, Tensor]]:
    for start in range(0, inputs.shape[0], batch_size):
        yield inputs[start : start + batch_size], targets[start : start + batch_size]


def _synthetic_dataset(cfg: ExperimentConfig) -> Dict[str, Any]:
    def build(n_sequences: int, offset: int) -> Tuple[Tensor, Tensor]:
        sequences: List[Tensor] = []
        for index in range(n_sequences):
            stride = 1 + ((index + offset) % 5)
            base = (7 * (index + offset)) % cfg.smoke_vocab_size
            ids = (base + stride * torch.arange(cfg.seq_len + 1)) % cfg.smoke_vocab_size
            sequences.append(ids.to(dtype=torch.long))
        stacked = torch.stack(sequences, dim=0)
        return stacked[:, :-1], stacked[:, 1:]

    train_inputs, train_targets = build(cfg.smoke_train_sequences, cfg.seed)
    valid_inputs, valid_targets = build(cfg.smoke_valid_sequences, cfg.seed + 97)
    return {
        "name": "synthetic_next_token_smoke",
        "vocab_size": cfg.smoke_vocab_size,
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "valid_inputs": valid_inputs,
        "valid_targets": valid_targets,
        "metadata": {
            "train_sequences": cfg.smoke_train_sequences,
            "valid_sequences": cfg.smoke_valid_sequences,
            "seq_len": cfg.seq_len,
            "seed": cfg.seed,
        },
    }


def _formal_cache_ready(cache_dir: Path, vocab_size: int) -> Tuple[bool, List[str]]:
    expected = (
        cache_dir / "raw" / "catgirl_texts.txt",
        cache_dir / "bpe" / f"catgirl_bpe_{vocab_size}.json",
        cache_dir / "tok" / "catgirl_train_ids.pt",
        cache_dir / "tok" / "catgirl_val_ids.pt",
    )
    missing = [_path(path) for path in expected if not path.exists()]
    return not missing, missing


def _formal_dataset(cfg: ExperimentConfig) -> Dict[str, Any]:
    ready, missing = _formal_cache_ready(cfg.cache_dir, cfg.vocab_size)
    if not ready:
        raise FileNotFoundError(
            "formal mode requires an existing catgirl BPE cache; missing: "
            + ", ".join(missing)
        )
    corpus = load_bpe_corpus(
        cfg.cache_dir,
        vocab_size=cfg.vocab_size,
        max_convs=cfg.max_convs,
    )
    train_inputs, train_targets = make_sequences(corpus["train_ids"], cfg.seq_len)
    valid_inputs, valid_targets = make_sequences(corpus["val_ids"], cfg.seq_len)
    return {
        "name": "catgirl_bpe_cache",
        "vocab_size": int(corpus["vocab_size"]),
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "valid_inputs": valid_inputs,
        "valid_targets": valid_targets,
        "metadata": {
            "n_train_tokens": int(corpus["n_train_tokens"]),
            "n_val_tokens": int(corpus["n_val_tokens"]),
            "n_train_convs": int(corpus["n_train_convs"]),
            "n_val_convs": int(corpus["n_val_convs"]),
            "tokenizer_path": str(corpus["tokenizer_path"]),
        },
    }


def _dataset(cfg: ExperimentConfig) -> Dict[str, Any]:
    if cfg.mode == "smoke":
        return _synthetic_dataset(cfg)
    return _formal_dataset(cfg)


def _parameter_count(module: nn.Module) -> int:
    return count_parameters(module, trainable_only=False)


def _first_parameter_dtype(models: Mapping[str, Optional[CausalLanguageModel]]) -> torch.dtype:
    for model in models.values():
        if model is None:
            continue
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.dtype
    return torch.float32


def _estimate_generic_model_memory_gib(
    model: CausalLanguageModel,
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> float:
    element_size = torch.tensor((), dtype=dtype).element_size()
    parameter_bytes = _parameter_count(model) * element_size
    state_bytes = state_nbytes(model.initial_state(batch_size, dtype=dtype))
    embedding_bytes = batch_size * seq_len * model.embedding_dim * element_size
    hidden_bytes = batch_size * seq_len * model.output_dim * element_size
    logits_bytes = batch_size * seq_len * model.vocab_size * element_size
    total_bytes = (
        4 * parameter_bytes
        + state_bytes
        + 2 * embedding_bytes
        + 2 * hidden_bytes
        + 2 * logits_bytes
    )
    return total_bytes / float(1024 ** 3)


def _estimate_fe2h_memory(
    model: CausalLanguageModel,
    *,
    batch_size: int,
    seq_len: int,
    sparse_inference: bool,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    core = model.core
    if not isinstance(core, FE2HNeuronTileCore):
        raise TypeError("expected FE2HNeuronTileCore")
    core_bound = core.estimate_memory_upper_bound(
        batch_size=batch_size,
        time_steps=seq_len,
        dtype=dtype,
    )
    element_size = torch.tensor((), dtype=dtype).element_size()
    total_parameter_bytes = _parameter_count(model) * element_size
    core_parameter_bytes = _parameter_count(core) * element_size
    wrapper_parameter_bytes = max(0, total_parameter_bytes - core_parameter_bytes)
    projection_bytes = (
        batch_size * seq_len * 4 * core.state_dim * element_size
        if not sparse_inference
        else core_bound.active_projection_bytes
    )
    embedding_bytes = batch_size * seq_len * model.embedding_dim * element_size
    hidden_bytes = batch_size * seq_len * model.output_dim * element_size
    logits_bytes = batch_size * seq_len * model.vocab_size * element_size
    adjusted_core_total = (
        core_bound.total_bytes - core_bound.active_projection_bytes + projection_bytes
    )
    model_total_bytes = (
        adjusted_core_total
        + 4 * wrapper_parameter_bytes
        + 2 * embedding_bytes
        + 2 * hidden_bytes
        + 2 * logits_bytes
    )
    return {
        "core_total_gib": core_bound.total_gib,
        "core_forward_only_gib": core_bound.forward_only_gib,
        "model_total_gib": model_total_bytes / float(1024 ** 3),
        "projection_bytes": int(projection_bytes),
        "core": asdict(core_bound),
    }


def _mechanism_equivalence_record(
    model: CausalLanguageModel,
    input_ids: Tensor,
    targets: Tensor,
) -> Dict[str, Any]:
    route, route_error = _prepare_sparse_route(model, input_ids)
    if route_error is not None or route is None:
        return {
            "supported": False,
            "unsupported_reason": route_error,
            "status": "UNSUPPORTED",
        }
    try:
        with torch.no_grad():
            model.eval()
            dense = _forward_model(
                model,
                input_ids,
                targets=targets,
                route_override=route,
                sparse_inference=False,
            )
            sparse = _forward_model(
                model,
                input_ids,
                targets=targets,
                route_override=route,
                sparse_inference=True,
            )
        max_logit_diff = float((dense["logits"] - sparse["logits"]).abs().max().item())
        max_state_diff = _state_max_abs_diff(dense["state"], sparse["state"])
        dense_loss = dense["loss"]
        sparse_loss = sparse["loss"]
        loss_diff = (
            None
            if dense_loss is None or sparse_loss is None
            else float(abs(dense_loss.item() - sparse_loss.item()))
        )
        dense_diagnostics = dense["diagnostics"]
        homeostasis = _homeostasis_gate_result(
            dense_diagnostics.homeostasis,
            batch_windows=int(route.hard_mask.shape[0] * route.hard_mask.shape[1]),
        )
        active_per_block = route.hard_mask.sum(dim=-1)
        route_budget_ok = bool(
            active_per_block.eq(float(model.core.active_tiles)).all().item()
        )
        passed = (
            route_budget_ok
            and homeostasis["status"] == "PASS"
            and max_logit_diff <= MECHANISM_MAX_ABS_TOLERANCE
            and max_state_diff <= MECHANISM_MAX_ABS_TOLERANCE
        )
        failure_reasons: List[str] = []
        if not route_budget_ok:
            failure_reasons.append("route_budget_mismatch")
        if homeostasis["status"] != "PASS":
            failure_reasons.extend(homeostasis["failures"])
        if max_logit_diff > MECHANISM_MAX_ABS_TOLERANCE:
            failure_reasons.append("dense_vs_sparse_logits_mismatch")
        if max_state_diff > MECHANISM_MAX_ABS_TOLERANCE:
            failure_reasons.append("dense_vs_sparse_state_mismatch")
        return {
            "supported": True,
            "status": "PASS" if passed else "FAIL",
            "route_budget_ok": route_budget_ok,
            "max_abs_logit_diff": max_logit_diff,
            "max_abs_state_diff": max_state_diff,
            "loss_diff": loss_diff,
            "homeostasis": homeostasis,
            "blocks": int(route.hard_mask.shape[1]),
            "tiles": int(route.hard_mask.shape[2]),
            "active_tiles": int(model.core.active_tiles),
            "route_protocol": "eval+no_grad route_blocks then route_override into sparse_inference",
            "unsupported_reason": None,
            "failure_reasons": failure_reasons,
        }
    except FE2HUnsupportedError as error:
        return {
            "supported": False,
            "status": "UNSUPPORTED",
            "unsupported_reason": str(error),
        }


def _homeostasis_gate_result(
    homeostasis: Any,
    *,
    batch_windows: Optional[int] = None,
) -> Dict[str, Any]:
    activation_rate = _float_list(homeostasis.activation_rate)
    target = float(homeostasis.target_activation_rate)
    max_abs_deviation = max(abs(value - target) for value in activation_rate)
    metrics = {
        "status": "PASS",
        "target_activation_rate": target,
        "activation_rate": activation_rate,
        "activation_rate_mean": sum(activation_rate) / max(1, len(activation_rate)),
        "activation_rate_max_abs_deviation": max_abs_deviation,
        "activation_rate_max_abs_tolerance": HOMEOSTASIS_ACTIVATION_RATE_MAX_ABS_TOLERANCE,
        "entropy": _scalar(homeostasis.entropy),
        "gini": _scalar(homeostasis.gini),
        "p99_tile_load": _scalar(homeostasis.p99_tile_load),
        "block_fill": _scalar(homeostasis.block_fill),
        "hotspot_share": _scalar(homeostasis.hotspot_share),
        "hotspot_share_max": HOMEOSTASIS_HOTSPOT_MAX,
        "dead_tile_ratio": _scalar(homeostasis.dead_tile_ratio),
        "dead_tile_ratio_max": HOMEOSTASIS_DEAD_TILE_RATIO_MAX,
        "batch_windows": batch_windows,
        "failures": [],
    }
    if metrics["hotspot_share"] > HOMEOSTASIS_HOTSPOT_MAX:
        metrics["failures"].append("hotspot_share_exceeds_0.70")
    if metrics["dead_tile_ratio"] > HOMEOSTASIS_DEAD_TILE_RATIO_MAX:
        metrics["failures"].append("dead_tile_ratio_nonzero")
    if (
        metrics["activation_rate_max_abs_deviation"]
        > HOMEOSTASIS_ACTIVATION_RATE_MAX_ABS_TOLERANCE
    ):
        metrics["failures"].append("activation_rate_not_close_to_target")
    if metrics["failures"]:
        metrics["status"] = "FAIL"
    return metrics


def _optimizer(module: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )


def _scalar_zero(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _loss_breakdown(
    forward: Mapping[str, Any],
    *,
    training: bool,
) -> Dict[str, Any]:
    ce_loss = forward["loss"]
    if ce_loss is None:
        raise RuntimeError("forward pass did not produce a loss")

    diagnostics = forward.get("diagnostics")
    route_loss = _scalar_zero(ce_loss)
    homeostasis_loss = _scalar_zero(ce_loss)
    route_source = "not_applied_eval_or_missing_diagnostics"
    homeostasis_source = "not_applied_eval_or_missing_diagnostics"

    if training and diagnostics is not None:
        if diagnostics.route_supervision_loss is not None:
            route_loss = diagnostics.route_supervision_loss
            route_source = "diagnostics.route_supervision_loss"
        else:
            route_source = "diagnostics.route_supervision_loss_unavailable"
        homeostasis = getattr(diagnostics, "homeostasis", None)
        homeostasis_value = getattr(homeostasis, "loss", None)
        if homeostasis_value is not None:
            homeostasis_loss = homeostasis_value
            homeostasis_source = "diagnostics.homeostasis.loss"
        else:
            homeostasis_source = "diagnostics.homeostasis.loss_unavailable"

    total_loss = ce_loss
    if training:
        total_loss = (
            ce_loss
            + ROUTE_SUPERVISION_LOSS_WEIGHT * route_loss
            + HOMEOSTASIS_LOSS_WEIGHT * homeostasis_loss
        )

    return {
        "ce": ce_loss,
        "route_supervision": route_loss,
        "homeostasis": homeostasis_loss,
        "total": total_loss,
        "aux_applied": training and diagnostics is not None,
        "provenance": {
            "ce": "model._loss(logits, targets)",
            "route_supervision": route_source,
            "homeostasis": homeostasis_source,
            "total": (
                "ce_only"
                if not training
                else "ce + 0.01*route_supervision + 0.01*homeostasis"
            ),
        },
    }


def _run_epoch(
    model: CausalLanguageModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    seed: int,
    sparse_inference: bool = False,
) -> Dict[str, Any]:
    training = optimizer is not None
    generator = torch.Generator().manual_seed(seed)
    indices = (
        torch.randperm(inputs.shape[0], generator=generator)
        if training
        else torch.arange(inputs.shape[0])
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    total_ce = 0.0
    total_route_supervision = 0.0
    total_homeostasis = 0.0
    total_objective = 0.0
    total_tokens = 0
    first_failure = None
    loss_provenance = {
        "ce": "model._loss(logits, targets)",
        "route_supervision": "not_applied_eval_or_missing_diagnostics",
        "homeostasis": "not_applied_eval_or_missing_diagnostics",
        "total": "ce_only" if not training else "ce + 0.01*route_supervision + 0.01*homeostasis",
    }
    aux_applied = False
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, start in enumerate(range(0, inputs.shape[0], batch_size)):
            selection = indices[start : start + batch_size]
            batch_inputs = inputs.index_select(0, selection).to(device)
            batch_targets = targets.index_select(0, selection).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            try:
                if sparse_inference:
                    route, route_error = _prepare_sparse_route(model, batch_inputs)
                    if route_error is not None or route is None:
                        raise FE2HUnsupportedError(route_error or "missing sparse route")
                    forward = _forward_model(
                        model,
                        batch_inputs,
                        targets=batch_targets,
                        route_override=route,
                        sparse_inference=True,
                    )
                else:
                    forward = _forward_model(model, batch_inputs, targets=batch_targets)
                loss_terms = _loss_breakdown(forward, training=training)
                loss = loss_terms["ce"]
                if training:
                    run_fe2h_finite_guard(
                        model,
                        loss_terms={
                            "ce": loss_terms["ce"],
                            "route_supervision": loss_terms["route_supervision"],
                            "homeostasis": loss_terms["homeostasis"],
                            "total": loss_terms["total"],
                        },
                        step=step,
                    )
                    loss_terms["total"].backward()
                    run_fe2h_finite_guard(
                        model,
                        loss_terms={
                            "ce": loss_terms["ce"],
                            "route_supervision": loss_terms["route_supervision"],
                            "homeostasis": loss_terms["homeostasis"],
                            "total": loss_terms["total"],
                        },
                        optimizer=optimizer,
                        step=step,
                    )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    run_fe2h_finite_guard(
                        model,
                        optimizer=optimizer,
                        step=step,
                    )
            except Exception as error:  # fail closed and preserve first evidence
                first_failure = {
                    "step": step,
                    "type": error.__class__.__name__,
                    "message": str(error),
                }
                break
            batch_tokens = int(batch_targets.numel())
            total_ce += float(loss_terms["ce"].detach().item()) * batch_tokens
            total_route_supervision += (
                float(loss_terms["route_supervision"].detach().item()) * batch_tokens
            )
            total_homeostasis += (
                float(loss_terms["homeostasis"].detach().item()) * batch_tokens
            )
            total_objective += float(loss_terms["total"].detach().item()) * batch_tokens
            total_tokens += batch_tokens
            aux_applied = aux_applied or bool(loss_terms["aux_applied"])
            loss_provenance = dict(loss_terms["provenance"])
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    mean_ce = total_ce / max(1, total_tokens)
    mean_route_supervision = total_route_supervision / max(1, total_tokens)
    mean_homeostasis = total_homeostasis / max(1, total_tokens)
    mean_total = total_objective / max(1, total_tokens)
    return {
        "ce": mean_ce,
        "bpc": mean_ce / math.log(2.0) if total_tokens else None,
        "loss_breakdown": {
            "ce": mean_ce,
            "route_supervision": mean_route_supervision,
            "homeostasis": mean_homeostasis,
            "total": mean_total,
            "aux_applied": aux_applied,
        },
        "loss_provenance": loss_provenance,
        "tokens_per_s": total_tokens / max(elapsed, 1e-9),
        "elapsed_s": elapsed,
        "peak_memory_bytes": peak_memory_bytes,
        "first_failure": first_failure,
    }


def _move_models_to_device(
    models: Mapping[str, Optional[CausalLanguageModel]],
    device: torch.device,
) -> Dict[str, CausalLanguageModel]:
    moved: Dict[str, CausalLanguageModel] = {}
    for name, model in models.items():
        if model is not None:
            moved[name] = model.to(device)
    return moved


def _unsupported_numerics_record(reason: str) -> Dict[str, Any]:
    return {"status": "UNSUPPORTED", "unsupported_reason": reason}


def _record_variant_failure(
    current: Optional[Dict[str, Any]],
    *,
    variant_name: str,
    phase: str,
    run_record: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if current is not None or run_record.get("first_failure") is None:
        return current
    return {
        "variant": variant_name,
        "phase": phase,
        **run_record["first_failure"],
    }


def _train_variant_epochs(
    model: CausalLanguageModel,
    *,
    cfg: ExperimentConfig,
    train_inputs: Tensor,
    train_targets: Tensor,
    device: torch.device,
    variant_name: str,
    first_failure: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    optimizer = _optimizer(model)
    history: List[Dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        train_record = _run_epoch(
            model,
            train_inputs,
            train_targets,
            batch_size=cfg.batch_size,
            device=device,
            optimizer=optimizer,
            seed=cfg.seed + epoch,
        )
        history.append(train_record)
        first_failure = _record_variant_failure(
            first_failure,
            variant_name=variant_name,
            phase="train",
            run_record=train_record,
        )
        if train_record["first_failure"] is not None:
            break
    return history, first_failure


def _train_peak_memory_bytes(history: Sequence[Mapping[str, Any]]) -> int:
    peaks = [
        record["peak_memory_bytes"] or 0
        for record in history
        if record.get("peak_memory_bytes") is not None
    ]
    return max(peaks or [0])


def _dense_variant_record(
    *,
    variant_name: str,
    model: CausalLanguageModel,
    cfg: ExperimentConfig,
    train_inputs: Tensor,
    train_targets: Tensor,
    valid_inputs: Tensor,
    valid_targets: Tensor,
    device: torch.device,
    vocab_size: int,
    first_failure: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    train_history, first_failure = _train_variant_epochs(
        model,
        cfg=cfg,
        train_inputs=train_inputs,
        train_targets=train_targets,
        device=device,
        variant_name=variant_name,
        first_failure=first_failure,
    )
    valid_dense = _run_epoch(
        model,
        valid_inputs,
        valid_targets,
        batch_size=cfg.batch_size,
        device=device,
        optimizer=None,
        seed=cfg.seed,
        sparse_inference=False,
    )
    first_failure = _record_variant_failure(
        first_failure,
        variant_name=variant_name,
        phase="eval",
        run_record=valid_dense,
    )
    return (
        {
            "status": "PASS" if first_failure is None else "FAIL",
            "vocab_size": vocab_size,
            "train_history": train_history,
            "valid_dense": valid_dense,
        },
        {
            "train_peak_memory_bytes": _train_peak_memory_bytes(train_history),
            "valid_peak_memory_bytes": valid_dense["peak_memory_bytes"],
        },
        first_failure,
    )


def _sparse_eval_record(
    model: CausalLanguageModel,
    *,
    cfg: ExperimentConfig,
    valid_inputs: Tensor,
    valid_targets: Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return _run_epoch(
        model,
        valid_inputs,
        valid_targets,
        batch_size=cfg.batch_size,
        device=device,
        optimizer=None,
        seed=cfg.seed,
        sparse_inference=True,
    )


def _attach_sparse_variant_record(
    variant_records: Dict[str, Any],
    observed_memory: Dict[str, Any],
    *,
    key: str,
    sparse_record: Dict[str, Any],
    parent_key: Optional[str] = None,
    observed_key: str = "valid_peak_memory_bytes",
) -> None:
    if parent_key is None:
        variant_records[key] = sparse_record
    else:
        variant_records[parent_key]["valid_sparse"] = sparse_record
    observed_memory[key] = {observed_key: sparse_record["peak_memory_bytes"]}


def _numerics_records(
    cfg: ExperimentConfig,
    dataset: Mapping[str, Any],
    models: Mapping[str, Optional[CausalLanguageModel]],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_inputs = dataset["train_inputs"]
    train_targets = dataset["train_targets"]
    valid_inputs = dataset["valid_inputs"]
    valid_targets = dataset["valid_targets"]
    vocab_size = int(dataset["vocab_size"])
    moved_models = _move_models_to_device(models, device)

    variant_records: Dict[str, Any] = {}
    observed_memory: Dict[str, Any] = {}
    first_failure = None
    for variant_name in ("base_e3", "fe2h_dense_mask", "fe2h_low_rank_tile_sparse"):
        model = moved_models.get(variant_name)
        if model is None:
            variant_records[variant_name] = _unsupported_numerics_record(
                "model was not constructed"
            )
            continue
        (
            variant_records[variant_name],
            observed_memory[variant_name],
            first_failure,
        ) = _dense_variant_record(
            variant_name=variant_name,
            model=model,
            cfg=cfg,
            train_inputs=train_inputs,
            train_targets=train_targets,
            valid_inputs=valid_inputs,
            valid_targets=valid_targets,
            device=device,
            vocab_size=vocab_size,
            first_failure=first_failure,
        )

    dense_model = moved_models.get("fe2h_dense_mask")
    if dense_model is not None:
        sparse_record = _sparse_eval_record(
            dense_model,
            cfg=cfg,
            valid_inputs=valid_inputs,
            valid_targets=valid_targets,
            device=device,
        )
        _attach_sparse_variant_record(
            variant_records,
            observed_memory,
            key="fe2h_tile_sparse",
            sparse_record=sparse_record,
        )
        first_failure = _record_variant_failure(
            first_failure,
            variant_name="fe2h_tile_sparse",
            phase="sparse_eval",
            run_record=sparse_record,
        )

    low_rank_model = moved_models.get("fe2h_low_rank_tile_sparse")
    if low_rank_model is not None:
        sparse_record = _sparse_eval_record(
            low_rank_model,
            cfg=cfg,
            valid_inputs=valid_inputs,
            valid_targets=valid_targets,
            device=device,
        )
        _attach_sparse_variant_record(
            variant_records,
            observed_memory,
            key="fe2h_low_rank_tile_sparse",
            sparse_record=sparse_record,
            parent_key="fe2h_low_rank_tile_sparse",
            observed_key="valid_sparse_peak_memory_bytes",
        )
        first_failure = _record_variant_failure(
            first_failure,
            variant_name="fe2h_low_rank_tile_sparse",
            phase="sparse_eval",
            run_record=sparse_record,
        )

    status = "PASS" if first_failure is None else "FAIL"
    numerics = {
        "status": status,
        "mode": cfg.mode,
        "variants": variant_records,
        "first_failure": first_failure,
        "notes": [
            "Sparse variants are evaluated only under eval+no_grad route_override.",
            "The first non-finite or unsupported evidence is preserved verbatim.",
        ],
    }
    return numerics, observed_memory


def _benchmark_base_model(
    model: CausalLanguageModel,
    input_ids: Tensor,
    targets: Tensor,
    *,
    device: torch.device,
    warmup_steps: int,
    benchmark_steps: int,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            _forward_model(model, input_ids, targets=targets)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times: List[float] = []
        synchronize(device)
        for _ in range(benchmark_steps):
            synchronize(device)
            started = time.perf_counter()
            _forward_model(model, input_ids, targets=targets)
            synchronize(device)
            times.append(time.perf_counter() - started)
    total = sum(times)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return {
        "status": "PASS",
        "median_step_s": statistics.median(times),
        "tokens_per_s": (input_ids.numel() * benchmark_steps) / max(total, 1e-9),
        "peak_memory_bytes": peak,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
    }


def _benchmark_fe2h_model(
    model: CausalLanguageModel,
    input_ids: Tensor,
    targets: Tensor,
    *,
    device: torch.device,
    sparse_inference: bool,
    warmup_steps: int,
    benchmark_steps: int,
) -> Dict[str, Any]:
    if not isinstance(model.core, FE2HNeuronTileCore):
        raise TypeError("expected FE2H model")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            route, route_error = _prepare_sparse_route(model, input_ids)
            if route_error is not None or route is None:
                return {
                    "status": "UNSUPPORTED",
                    "unsupported_reason": route_error,
                    "retained_negative_result": True,
                }
            _forward_model(
                model,
                input_ids,
                targets=targets,
                route_override=route,
                sparse_inference=sparse_inference,
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        route_times: List[float] = []
        core_times: List[float] = []
        lm_times: List[float] = []
        full_times: List[float] = []
        for _ in range(benchmark_steps):
            synchronize(device)
            step_started = time.perf_counter()
            route_started = time.perf_counter()
            route, route_error = _prepare_sparse_route(model, input_ids)
            synchronize(device)
            route_times.append(time.perf_counter() - route_started)
            if route_error is not None or route is None:
                return {
                    "status": "UNSUPPORTED",
                    "unsupported_reason": route_error,
                    "retained_negative_result": True,
                }
            embedded = model.input_dropout(model.embedding(input_ids))
            synchronize(device)
            core_started = time.perf_counter()
            core_result, _ = model.core.forward_dynamics(
                embedded,
                sparse_inference=sparse_inference,
                route_override=route,
            )
            hidden = model.output_dropout(model.output_norm(core_result.sequence))
            synchronize(device)
            core_times.append(time.perf_counter() - core_started)
            synchronize(device)
            lm_started = time.perf_counter()
            logits = model.lm_head(hidden)
            model._loss(logits, targets)
            synchronize(device)
            lm_times.append(time.perf_counter() - lm_started)
            full_times.append(time.perf_counter() - step_started)
    total = sum(full_times)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    return {
        "status": "PASS",
        "median_route_step_s": statistics.median(route_times),
        "median_core_step_s": statistics.median(core_times),
        "median_lm_head_step_s": statistics.median(lm_times),
        "median_full_step_s": statistics.median(full_times),
        "tokens_per_s": (input_ids.numel() * benchmark_steps) / max(total, 1e-9),
        "peak_memory_bytes": peak,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "notes": [
            "route timing is measured separately before route_override execution",
            "core timing includes active projection plus the still-dense FE2H core output projection",
            "lm_head timing isolates the remaining dense wrapper output projection",
        ],
    }


def _speed_records(
    cfg: ExperimentConfig,
    dataset: Mapping[str, Any],
    models: Mapping[str, Optional[CausalLanguageModel]],
    device: torch.device,
) -> Dict[str, Any]:
    input_ids = dataset["valid_inputs"][: cfg.batch_size].to(device)
    targets = dataset["valid_targets"][: cfg.batch_size].to(device)
    benchmarks: Dict[str, Any] = {}

    base_model = models.get("base_e3")
    dense_model = models.get("fe2h_dense_mask")
    low_rank_model = models.get("fe2h_low_rank_tile_sparse")
    if base_model is None or dense_model is None:
        return {
            "status": "NOT_RUN",
            "reason": "required models were not constructed",
            "benchmarks": {},
            "retained_negative_result": True,
        }

    benchmarks["base_e3"] = _benchmark_base_model(
        base_model.to(device),
        input_ids,
        targets,
        device=device,
        warmup_steps=cfg.warmup_steps,
        benchmark_steps=cfg.benchmark_steps,
    )
    benchmarks["fe2h_dense_mask"] = _benchmark_fe2h_model(
        dense_model.to(device),
        input_ids,
        targets,
        device=device,
        sparse_inference=False,
        warmup_steps=cfg.warmup_steps,
        benchmark_steps=cfg.benchmark_steps,
    )
    benchmarks["fe2h_tile_sparse"] = _benchmark_fe2h_model(
        dense_model.to(device),
        input_ids,
        targets,
        device=device,
        sparse_inference=True,
        warmup_steps=cfg.warmup_steps,
        benchmark_steps=cfg.benchmark_steps,
    )
    if low_rank_model is not None:
        benchmarks["fe2h_low_rank_tile_sparse"] = _benchmark_fe2h_model(
            low_rank_model.to(device),
            input_ids,
            targets,
            device=device,
            sparse_inference=True,
            warmup_steps=cfg.warmup_steps,
            benchmark_steps=cfg.benchmark_steps,
        )
    else:
        benchmarks["fe2h_low_rank_tile_sparse"] = {
            "status": "UNSUPPORTED",
            "unsupported_reason": "low-rank model was not constructed",
            "retained_negative_result": True,
        }

    base_tokens = benchmarks["base_e3"].get("tokens_per_s", 0.0) or 0.0
    sparse_speedups = {}
    for variant in ("fe2h_tile_sparse", "fe2h_low_rank_tile_sparse"):
        tokens = benchmarks[variant].get("tokens_per_s")
        if tokens is not None and base_tokens > 0.0:
            sparse_speedups[variant] = float(tokens) / base_tokens
            benchmarks[variant]["speedup_over_base_e3"] = sparse_speedups[variant]

    best_variant = None
    best_speedup = 0.0
    if sparse_speedups:
        best_variant = max(sparse_speedups, key=sparse_speedups.get)
        best_speedup = sparse_speedups[best_variant]
    passed = best_speedup > 1.0
    return {
        "status": "PASS" if passed else "FAIL",
        "benchmarks": benchmarks,
        "best_sparse_variant": best_variant,
        "best_sparse_speedup_over_base_e3": best_speedup,
        "retained_negative_result": True,
        "notes": [
            "A speed result below 1.0x is retained as a negative conclusion rather than hidden.",
            "Dense-mask timing is reported separately and is never labelled as sparse speed evidence.",
        ],
    }


def _quality_records(
    cfg: ExperimentConfig,
    numerics: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    if cfg.mode == "smoke":
        return {
            "status": "PASS",
            "dataset": dataset["name"],
            "mode": "smoke",
            "variants": {
                "base_e3": numerics["variants"].get("base_e3", {}).get("valid_dense"),
                "fe2h_dense_mask": numerics["variants"].get("fe2h_dense_mask", {}).get(
                    "valid_dense"
                ),
                "fe2h_tile_sparse": numerics["variants"].get("fe2h_tile_sparse"),
                "fe2h_low_rank_tile_sparse": numerics["variants"]
                .get("fe2h_low_rank_tile_sparse", {})
                .get("valid_sparse"),
            },
            "notes": [
                "Smoke quality is the synthetic next-token BPC collected after the numerics gate.",
            ],
        }
    return {
        "status": "PASS",
        "dataset": dataset["name"],
        "mode": "formal",
        "variants": {
            "base_e3": numerics["variants"].get("base_e3", {}).get("valid_dense"),
            "fe2h_dense_mask": numerics["variants"].get("fe2h_dense_mask", {}).get(
                "valid_dense"
            ),
            "fe2h_tile_sparse": numerics["variants"].get("fe2h_tile_sparse"),
            "fe2h_low_rank_tile_sparse": numerics["variants"]
            .get("fe2h_low_rank_tile_sparse", {})
            .get("valid_sparse"),
        },
        "notes": [
            "Formal quality uses the existing catgirl BPE cache and only runs after earlier gates pass.",
        ],
    }


def _not_run_mechanism(reason: str) -> Dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "variant_checks": {},
        "tolerance": {
            "max_abs_logit": MECHANISM_MAX_ABS_TOLERANCE,
            "max_abs_state": MECHANISM_MAX_ABS_TOLERANCE,
        },
        "details": [reason],
    }


def _not_run_numerics(cfg: ExperimentConfig, reason: str) -> Dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "mode": cfg.mode,
        "variants": {},
        "first_failure": None,
        "reason": reason,
    }


def _not_run_speed(reason: str) -> Dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "benchmarks": {},
        "reason": reason,
        "retained_negative_result": True,
    }


def _not_run_quality(cfg: ExperimentConfig, reason: str) -> Dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "mode": cfg.mode,
        "reason": reason,
        "variants": {},
    }


def _is_memory_preflight_block(
    mechanism: Mapping[str, Any],
    numerics: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> bool:
    memory_status = str(memory.get("status", "NOT_RUN")).upper()
    return (
        memory_status in {"PAUSE", "REFUSE"}
        and str(mechanism.get("status", "NOT_RUN")).upper() == "NOT_RUN"
        and str(numerics.get("status", "NOT_RUN")).upper() == "NOT_RUN"
    )


def _artifact_sections(
    cfg: ExperimentConfig,
    device: torch.device,
    dataset: Mapping[str, Any],
    variant_paths: Mapping[str, Mapping[str, Any]],
    *,
    mechanism: Mapping[str, Any],
    numerics: Mapping[str, Any],
    memory: Mapping[str, Any],
    speed: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> Dict[str, Any]:
    decision = make_gate_decision(
        mechanism=mechanism,
        numerics=numerics,
        memory=memory,
        speed=speed,
        quality=quality,
        variant_paths=variant_paths,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "formal": cfg.mode == "formal",
        "environment": _environment(device, cfg.device_request),
        "configuration": {
            **asdict(cfg),
            "out": _path(cfg.out),
            "cache_dir": _path(cfg.cache_dir),
        },
        "provenance": {
            "base_ref": str(
                REPO_ROOT / ".pipeline-workspace" / "bases" / "wave-3-group-5-base.json"
            ),
            "worker_group": "GROUP-5",
            "training_loss": {
                "weights": {
                    "ce": 1.0,
                    "route_supervision": ROUTE_SUPERVISION_LOSS_WEIGHT,
                    "homeostasis": HOMEOSTASIS_LOSS_WEIGHT,
                },
                "provenance": {
                    "ce": "model._loss(logits, targets)",
                    "route_supervision": "diagnostics.route_supervision_loss during FE2H training",
                    "homeostasis": "diagnostics.homeostasis.loss during FE2H training",
                    "total": "ce + 0.01*route_supervision + 0.01*homeostasis for FE2H training; eval remains ce only",
                },
                "metric_reporting": "artifact ce and bpc remain cross_entropy only",
            },
            "variant_paths": variant_paths,
            "dataset": {
                "name": dataset["name"],
                "metadata": dataset["metadata"],
            },
        },
        "mechanism": mechanism,
        "numerics": numerics,
        "memory": memory,
        "speed": speed,
        "quality": quality,
        "decision": decision,
    }


def _memory_records(
    cfg: ExperimentConfig,
    dataset: Mapping[str, Any],
    models: Mapping[str, Optional[CausalLanguageModel]],
    variant_paths: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    observed_memory: Mapping[str, Any],
) -> Dict[str, Any]:
    dtype = _first_parameter_dtype(models)
    variants: Dict[str, Any] = {}
    for name, model in models.items():
        if model is None:
            variants[name] = {
                "status": "UNSUPPORTED",
                "unsupported_reason": "model was not constructed",
            }
            continue
        if isinstance(model.core, FE2HNeuronTileCore):
            dense_sparse = name == "fe2h_dense_mask"
            estimate = _estimate_fe2h_memory(
                model,
                batch_size=cfg.batch_size,
                seq_len=cfg.seq_len,
                sparse_inference=not dense_sparse,
                dtype=dtype,
            )
        else:
            estimate = {
                "core_total_gib": None,
                "core_forward_only_gib": None,
                "model_total_gib": _estimate_generic_model_memory_gib(
                    model,
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    dtype=dtype,
                ),
                "core": None,
            }
        gate = predicted_memory_gate(float(estimate["model_total_gib"]))
        variants[name] = {
            **estimate,
            "launch_gate": gate,
            "observed": observed_memory.get(name),
        }

    dense_model = models.get("fe2h_dense_mask")
    if dense_model is not None:
        variants["fe2h_tile_sparse"] = {
            **_estimate_fe2h_memory(
                dense_model,
                batch_size=cfg.batch_size,
                seq_len=cfg.seq_len,
                sparse_inference=True,
                dtype=dtype,
            ),
            "launch_gate": predicted_memory_gate(
                float(
                    _estimate_fe2h_memory(
                        dense_model,
                        batch_size=cfg.batch_size,
                        seq_len=cfg.seq_len,
                        sparse_inference=True,
                        dtype=dtype,
                    )["model_total_gib"]
                )
            ),
            "observed": observed_memory.get("fe2h_tile_sparse"),
        }

    tile_sparse_gate = predicted_memory_gate(
        float(variants.get("fe2h_tile_sparse", variants["fe2h_dense_mask"])["model_total_gib"])
    )
    low_rank_gate = predicted_memory_gate(
        float(
            variants.get("fe2h_low_rank_tile_sparse", variants["fe2h_dense_mask"])[
                "model_total_gib"
            ]
        )
    )
    overall_gate = tile_sparse_gate
    if low_rank_gate["status"] == "REFUSE" or (
        low_rank_gate["status"] == "PAUSE" and overall_gate["status"] == "ALLOW"
    ):
        overall_gate = low_rank_gate
    status = (
        "PASS"
        if overall_gate["status"] == "ALLOW"
        else overall_gate["status"]
    )
    return {
        "status": status,
        "launch_gate": overall_gate,
        "variants": variants,
        "nvidia_device": _environment(device, cfg.device_request)["nvidia_device"],
        "notes": [
            "Thresholds are evaluated in GiB, not MiB.",
            "Core estimates come from FE2H's conservative upper bound; model estimates add wrapper parameters and logits.",
        ],
    }


def _run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    device = choose_device(cfg.device_request)
    dataset = _dataset(cfg)
    models, variant_paths = _build_models(int(dataset["vocab_size"]), cfg)
    memory = _memory_records(
        cfg,
        dataset,
        models,
        variant_paths,
        device,
        observed_memory={},
    )
    if memory["status"] in {"PAUSE", "REFUSE"}:
        artifact = _artifact_sections(
            cfg,
            device,
            dataset,
            variant_paths,
            mechanism=_not_run_mechanism(
                "memory preflight blocked launch before any device transfer or mechanism forward"
            ),
            numerics=_not_run_numerics(
                cfg,
                "memory preflight blocked numerics before launch",
            ),
            memory=memory,
            speed=_not_run_speed("memory preflight blocked speed benchmark before launch"),
            quality=_not_run_quality(
                cfg,
                "memory preflight blocked quality before launch",
            ),
        )
        errors = validate_artifact_schema(artifact)
        if errors:
            raise ValueError("artifact schema validation failed: " + "; ".join(errors))
        return artifact

    mechanism_input = dataset["valid_inputs"][
        : min(cfg.batch_size, dataset["valid_inputs"].shape[0])
    ].to(device)
    mechanism_targets = dataset["valid_targets"][
        : min(cfg.batch_size, dataset["valid_targets"].shape[0])
    ].to(device)

    mechanism = {
        "status": "PASS",
        "variant_checks": {},
        "homeostasis": None,
        "tolerance": {
            "max_abs_logit": MECHANISM_MAX_ABS_TOLERANCE,
            "max_abs_state": MECHANISM_MAX_ABS_TOLERANCE,
        },
        "details": [],
    }
    dense_model = models.get("fe2h_dense_mask")
    if dense_model is not None:
        dense_model = dense_model.to(device)
        dense_equivalence = _mechanism_equivalence_record(
            dense_model, mechanism_input, mechanism_targets
        )
        mechanism["variant_checks"]["fe2h_tile_sparse"] = dense_equivalence
        mechanism["homeostasis"] = dense_equivalence.get("homeostasis")
        if dense_equivalence["status"] != "PASS":
            mechanism["status"] = dense_equivalence["status"]
    else:
        mechanism["status"] = "FAIL"
        mechanism["details"].append("dense FE2H model was not constructed")

    low_rank_model = models.get("fe2h_low_rank_tile_sparse")
    if low_rank_model is not None:
        low_rank_model = low_rank_model.to(device)
        low_rank_equivalence = _mechanism_equivalence_record(
            low_rank_model, mechanism_input, mechanism_targets
        )
        mechanism["variant_checks"]["fe2h_low_rank_tile_sparse"] = low_rank_equivalence
        if low_rank_equivalence["status"] != "PASS" and mechanism["status"] == "PASS":
            mechanism["status"] = low_rank_equivalence["status"]
    else:
        mechanism["variant_checks"]["fe2h_low_rank_tile_sparse"] = {
            "status": "UNSUPPORTED",
            "supported": False,
            "unsupported_reason": variant_paths["fe2h_low_rank_tile_sparse"].get(
                "unsupported_reason", "low-rank model was not constructed"
            ),
        }
        if mechanism["status"] == "PASS":
            mechanism["status"] = "UNSUPPORTED"

    if mechanism["status"] == "PASS" and memory["status"] == "PASS":
        numerics, observed_memory = _numerics_records(cfg, dataset, models, device)
        memory = _memory_records(
            cfg,
            dataset,
            models,
            variant_paths,
            device,
            observed_memory=observed_memory,
        )
    else:
        numerics = _not_run_numerics(
            cfg,
            "mechanism or memory prelaunch gate blocked numerics",
        )

    if (
        mechanism["status"] == "PASS"
        and numerics["status"] == "PASS"
        and memory["status"] == "PASS"
    ):
        speed = _speed_records(cfg, dataset, models, device)
    else:
        speed = _not_run_speed("earlier gate blocked speed benchmark")

    if (
        mechanism["status"] == "PASS"
        and numerics["status"] == "PASS"
        and memory["status"] == "PASS"
        and speed["status"] == "PASS"
    ):
        quality = _quality_records(cfg, numerics, dataset)
    else:
        quality = _not_run_quality(
            cfg,
            "quality is gated on mechanism, numerics, memory, and speed passing first",
        )

    artifact = _artifact_sections(
        cfg,
        device,
        dataset,
        variant_paths,
        mechanism=mechanism,
        numerics=numerics,
        memory=memory,
        speed=speed,
        quality=quality,
    )
    errors = validate_artifact_schema(artifact)
    if errors:
        raise ValueError("artifact schema validation failed: " + "; ".join(errors))
    return artifact


def _parse_args(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "e3_scan" / "e3_fe2h_neuron_tile.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "results" / "e3_sg29_cache",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--active-tiles", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--rank", type=int, choices=(16, 32), default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-convs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--benchmark-steps", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--smoke-vocab-size", type=int, default=64)
    parser.add_argument("--smoke-train-sequences", type=int, default=24)
    parser.add_argument("--smoke-valid-sequences", type=int, default=8)
    parser.add_argument("--svd-init", action="store_true")
    args = parser.parse_args(argv)
    return ExperimentConfig(
        mode=args.mode,
        device_request=args.device,
        out=args.out,
        cache_dir=args.cache_dir,
        d_model=args.d_model,
        state_dim=args.state_dim,
        tile_size=args.tile_size,
        active_tiles=args.active_tiles,
        block_size=args.block_size,
        rank=args.rank,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_convs=args.max_convs,
        epochs=args.epochs,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        vocab_size=args.vocab_size,
        smoke_vocab_size=args.smoke_vocab_size,
        smoke_train_sequences=args.smoke_train_sequences,
        smoke_valid_sequences=args.smoke_valid_sequences,
        svd_init=args.svd_init,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(argv)
    artifact = _run_experiment(cfg)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {cfg.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
