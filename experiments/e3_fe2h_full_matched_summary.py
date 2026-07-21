#!/usr/bin/env python3
"""Audit and summarize the four full-epoch FE-2H matched artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


EXPECTED_TRAIN_TOKENS = 15_008_896
EXPECTED_VALID_TOKENS = 810_240
EXPECTED_TRAIN_STEPS = 1_047
EXPECTED_VALID_STEPS = 57
VARIANTS = (
    ("coarse_k2", 4, 2),
    ("micro_k16", 32, 16),
    ("micro_k8", 32, 8),
    ("micro_k4", 32, 4),
)


def _load_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError(f"artifact is not an object: {path}")
    return artifact


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _audit_variant(
    name: str,
    tile_count: int,
    active_tiles: int,
    artifact: Mapping[str, Any],
    *,
    path: Path,
) -> tuple[Dict[str, Any], list[str]]:
    errors: list[str] = []
    configuration = artifact.get("configuration") or {}
    training = artifact.get("training") or {}
    validation = artifact.get("validation") or {}
    model = artifact.get("model") or {}
    parameter_stats = model.get("parameter_stats") or {}
    shared_init = model.get("shared_initialization") or {}
    matched_gates = artifact.get("matched_gates") or {}
    specialization = validation.get("route_specialization") or {}
    homeostasis = validation.get("homeostasis") or {}
    gpu = (artifact.get("gpu_utilization") or {}).get("post_warmup") or {}
    gpu_util = gpu.get("gpu_util_percent") or {}
    power = gpu.get("power_w") or {}
    memory = artifact.get("memory") or {}
    device_memory = (memory.get("device_samples_full_training") or {}).get(
        "memory_used_mib"
    ) or {}
    checkpoint = artifact.get("checkpoint") or {}

    actual_tile_count = configuration.get("state_dim", 0) // max(
        1, configuration.get("tile_size", 1)
    )
    _require(artifact.get("status") == "COMPLETED", f"{name}: status not COMPLETED", errors)
    _require(artifact.get("failure") is None, f"{name}: failure is not null", errors)
    _require(actual_tile_count == tile_count, f"{name}: tile_count mismatch", errors)
    _require(configuration.get("active_tiles") == active_tiles, f"{name}: active_tiles mismatch", errors)
    for key, expected in {
        "d_model": 8192,
        "state_dim": 8192,
        "rank": 512,
        "batch_size": 112,
        "seq_len": 128,
        "block_size": 32,
        "seed": 0,
        "epochs": 1,
    }.items():
        _require(configuration.get(key) == expected, f"{name}: {key} mismatch", errors)
    _require(configuration.get("learning_rate") == 3e-4, f"{name}: learning_rate mismatch", errors)
    _require(configuration.get("weight_decay") == 0.01, f"{name}: weight_decay mismatch", errors)
    _require(configuration.get("route_supervision_weight") == 0.01, f"{name}: route supervision weight mismatch", errors)
    _require(configuration.get("homeostasis_weight") == 1.0, f"{name}: homeostasis weight mismatch", errors)
    _require(configuration.get("amp") is True, f"{name}: AMP disabled", errors)
    _require(configuration.get("amp_init_scale") == 256.0, f"{name}: AMP init scale mismatch", errors)
    _require(configuration.get("amp_growth_interval") == 100_000, f"{name}: AMP growth interval mismatch", errors)
    _require(configuration.get("checkpoint_every_steps") == 1_000, f"{name}: checkpoint interval mismatch", errors)
    _require(configuration.get("tqdm_progress") is True, f"{name}: tqdm progress disabled", errors)
    _require(configuration.get("train_mode") == "full_epoch", f"{name}: not full_epoch", errors)
    _require(configuration.get("full_validation") is True, f"{name}: not full validation", errors)
    _require(training.get("tokens") == EXPECTED_TRAIN_TOKENS, f"{name}: train token coverage mismatch", errors)
    _require(validation.get("tokens") == EXPECTED_VALID_TOKENS, f"{name}: validation token coverage mismatch", errors)
    _require(training.get("completed_steps") == EXPECTED_TRAIN_STEPS, f"{name}: train step coverage mismatch", errors)
    _require(validation.get("steps") == EXPECTED_VALID_STEPS, f"{name}: validation step coverage mismatch", errors)
    _require(training.get("coverage_fraction") == 1.0, f"{name}: train coverage is not 1.0", errors)
    _require(validation.get("coverage_fraction") == 1.0, f"{name}: validation coverage is not 1.0", errors)
    _require(validation.get("finite") is True, f"{name}: validation is non-finite", errors)
    _require(matched_gates.get("training_complete") is True, f"{name}: training gate failed", errors)
    _require(matched_gates.get("validation_complete") is True, f"{name}: validation gate failed", errors)
    _require(shared_init.get("enabled") is True, f"{name}: shared init disabled", errors)
    _require(shared_init.get("matched_parameter_ratio", 0.0) > 0.999, f"{name}: shared init match <=99.9%", errors)
    _require(specialization.get("supported") is True, f"{name}: specialization unsupported", errors)
    _require(
        isinstance(homeostasis.get("activation_rate"), list)
        and len(homeostasis["activation_rate"]) == tile_count
        and all(_is_finite_number(value) for value in homeostasis["activation_rate"]),
        f"{name}: soft activation telemetry missing or malformed",
        errors,
    )
    _require(
        isinstance(specialization.get("hard_activation_rate"), list)
        and len(specialization["hard_activation_rate"]) == tile_count
        and all(_is_finite_number(value) for value in specialization["hard_activation_rate"]),
        f"{name}: hard activation telemetry missing or malformed",
        errors,
    )
    for key, value in {
        "validation_bpc": validation.get("bpc"),
        "soft_hotspot_share": homeostasis.get("hotspot_share"),
        "soft_dead_tile_ratio": homeostasis.get("dead_tile_ratio"),
        "hard_hotspot_share": specialization.get("hard_hotspot_share"),
        "hard_dead_tile_ratio": specialization.get("hard_dead_tile_ratio"),
        "route_entropy_nats": specialization.get("route_entropy_nats"),
        "top_route_share": specialization.get("top_route_share"),
        "token_route_mi_nats": specialization.get("token_route_mutual_information_nats"),
        "shuffled_token_route_mi_nats": specialization.get("shuffled_token_route_mutual_information_nats"),
        "excess_token_route_mi_nats": specialization.get("excess_token_route_mutual_information_nats"),
        "train_tokens_per_s": training.get("tokens_per_s"),
        "validation_tokens_per_s": validation.get("tokens_per_s"),
        "gpu_util_mean": gpu_util.get("mean"),
        "gpu_util_p50": gpu_util.get("p50"),
        "gpu_util_p90": gpu_util.get("p90"),
        "power_mean_w": power.get("mean"),
        "power_p90_w": power.get("p90"),
        "device_memory_peak_mib": device_memory.get("max"),
        "torch_peak_allocated_gib": memory.get("torch_peak_allocated_gib"),
        "torch_peak_reserved_gib": memory.get("torch_peak_reserved_gib"),
    }.items():
        _require(_is_finite_number(value), f"{name}: {key} telemetry missing or non-finite", errors)
    _require(
        isinstance(specialization.get("unique_route_count"), int)
        and specialization["unique_route_count"] >= 1,
        f"{name}: unique route count missing or invalid",
        errors,
    )
    _require(
        isinstance(specialization.get("theoretical_route_count"), int)
        and specialization["theoretical_route_count"] >= specialization.get("unique_route_count", 0),
        f"{name}: theoretical route count missing or invalid",
        errors,
    )
    _require(
        isinstance(checkpoint.get("path"), str) and bool(checkpoint["path"]),
        f"{name}: checkpoint path missing",
        errors,
    )
    _require(
        isinstance(checkpoint.get("bytes"), int) and checkpoint["bytes"] > 0,
        f"{name}: checkpoint byte size missing",
        errors,
    )
    _require(checkpoint.get("includes_optimizer") is True, f"{name}: optimizer checkpoint missing", errors)
    _require(checkpoint.get("periodic_interval_steps") == 1_000, f"{name}: periodic checkpoint interval missing", errors)
    _require(checkpoint.get("periodic_steps") == [1_000], f"{name}: step-1000 checkpoint evidence missing", errors)
    _require(checkpoint.get("last_saved_step") == EXPECTED_TRAIN_STEPS, f"{name}: final checkpoint step mismatch", errors)
    _require(checkpoint.get("atomic_replace") is True, f"{name}: checkpoint was not atomic", errors)

    row = {
        "name": name,
        "artifact": str(path),
        "tile_count": tile_count,
        "active_tiles": active_tiles,
        "active_fraction": active_tiles / tile_count,
        "parameters": parameter_stats.get("model_total"),
        "shared_init_matched_parameter_ratio": shared_init.get(
            "matched_parameter_ratio"
        ),
        "train_steps": training.get("completed_steps"),
        "train_tokens": training.get("tokens"),
        "train_elapsed_s": training.get("elapsed_s"),
        "train_tokens_per_s": training.get("tokens_per_s"),
        "train_mean_bpc": training.get("mean_bpc"),
        "train_first_window_bpc": (
            training.get("windows") or [{}]
        )[0].get("bpc"),
        "train_last_window_bpc": (
            training.get("windows") or [{}]
        )[-1].get("bpc"),
        "validation_steps": validation.get("steps"),
        "validation_tokens": validation.get("tokens"),
        "validation_elapsed_s": validation.get("elapsed_s"),
        "validation_tokens_per_s": validation.get("tokens_per_s"),
        "validation_bpc": validation.get("bpc"),
        "soft_activation_rate": homeostasis.get("activation_rate"),
        "soft_hotspot_share": homeostasis.get("hotspot_share"),
        "soft_dead_tile_ratio": homeostasis.get("dead_tile_ratio"),
        "hard_activation_rate": specialization.get("hard_activation_rate"),
        "hard_hotspot_share": specialization.get("hard_hotspot_share"),
        "hard_dead_tile_ratio": specialization.get("hard_dead_tile_ratio"),
        "unique_route_count": specialization.get("unique_route_count"),
        "theoretical_route_count": specialization.get("theoretical_route_count"),
        "route_entropy_nats": specialization.get("route_entropy_nats"),
        "top_route_share": specialization.get("top_route_share"),
        "token_route_mi_nats": specialization.get(
            "token_route_mutual_information_nats"
        ),
        "shuffled_token_route_mi_nats": specialization.get(
            "shuffled_token_route_mutual_information_nats"
        ),
        "excess_token_route_mi_nats": specialization.get(
            "excess_token_route_mutual_information_nats"
        ),
        "excess_token_route_mi_over_route_entropy": specialization.get(
            "excess_token_route_mi_over_route_entropy"
        ),
        "gpu_util_mean": gpu_util.get("mean"),
        "gpu_util_p50": gpu_util.get("p50"),
        "gpu_util_p90": gpu_util.get("p90"),
        "power_mean_w": power.get("mean"),
        "power_p90_w": power.get("p90"),
        "device_memory_peak_mib": device_memory.get("max"),
        "torch_peak_allocated_gib": memory.get("torch_peak_allocated_gib"),
        "torch_peak_reserved_gib": memory.get("torch_peak_reserved_gib"),
        "soft_homeostasis_pass": matched_gates.get("soft_homeostasis_pass"),
        "hard_homeostasis_pass": matched_gates.get("hard_homeostasis_pass"),
        "specialization_pass": matched_gates.get("specialization_pass"),
        "checkpoint": checkpoint,
    }
    return row, errors


def build_summary(result_dir: Path) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    errors: list[str] = []
    for name, tile_count, active_tiles in VARIANTS:
        path = result_dir / f"e3_fe2h_full_{name}.json"
        try:
            artifact = _load_artifact(path)
        except Exception as error:
            errors.append(f"{name}: {error.__class__.__name__}: {error}")
            continue
        row, row_errors = _audit_variant(
            name,
            tile_count,
            active_tiles,
            artifact,
            path=path,
        )
        rows.append(row)
        errors.extend(row_errors)

    if errors or len(rows) != len(VARIANTS):
        return {
            "schema_version": 1,
            "experiment": "e3_fe2h_full_matched",
            "status": "INCOMPLETE",
            "audit_errors": errors,
            "variants": rows,
        }

    by_name = {row["name"]: row for row in rows}
    coarse = by_name["coarse_k2"]
    micro16 = by_name["micro_k16"]
    micro8 = by_name["micro_k8"]
    micro4 = by_name["micro_k4"]
    coarse_bpc = float(coarse["validation_bpc"])
    micro16_bpc = float(micro16["validation_bpc"])

    for row in rows:
        row["validation_bpc_delta_vs_coarse"] = float(row["validation_bpc"]) - coarse_bpc
        row["validation_bpc_relative_vs_coarse"] = float(row["validation_bpc"]) / coarse_bpc - 1.0
        row["train_throughput_ratio_vs_coarse"] = float(row["train_tokens_per_s"]) / float(coarse["train_tokens_per_s"])

    hypothesis = {
        "h1_micro50_improves": micro16_bpc < coarse_bpc,
        "h1_micro50_non_inferior_0_5pct": micro16_bpc <= coarse_bpc * 1.005,
        "h2_micro25_within_1pct_of_micro50": float(micro8["validation_bpc"])
        <= micro16_bpc * 1.01,
        "h2_micro12_5_within_1pct_of_micro50": float(micro4["validation_bpc"])
        <= micro16_bpc * 1.01,
        "h2_micro25_all_hard_tiles_alive": micro8["hard_dead_tile_ratio"] == 0.0,
        "h2_micro12_5_all_hard_tiles_alive": micro4["hard_dead_tile_ratio"] == 0.0,
        "h3_any_micro_specialization_pass": any(
            row["specialization_pass"] is True
            for row in (micro16, micro8, micro4)
        ),
        "h4_all_micro_slower_than_coarse": all(
            float(row["train_tokens_per_s"]) < float(coarse["train_tokens_per_s"])
            for row in (micro16, micro8, micro4)
        ),
    }
    quality_winner = min(rows, key=lambda row: float(row["validation_bpc"]))["name"]
    sparse_candidates = [
        row
        for row in (micro8, micro4)
        if float(row["validation_bpc"]) <= micro16_bpc * 1.01
        and row["hard_dead_tile_ratio"] == 0.0
    ]
    decision = {
        "quality_winner": quality_winner,
        "lowest_active_fraction_within_1pct_and_no_hard_dead_tiles": (
            min(sparse_candidates, key=lambda row: row["active_fraction"])["name"]
            if sparse_candidates
            else None
        ),
        "micro50_quality_verdict": (
            "IMPROVES"
            if hypothesis["h1_micro50_improves"]
            else (
                "NON_INFERIOR"
                if hypothesis["h1_micro50_non_inferior_0_5pct"]
                else "INFERIOR"
            )
        ),
        "hard_route_specialization_verdict": (
            "SUPPORTED"
            if hypothesis["h3_any_micro_specialization_pass"]
            else "NOT_SUPPORTED"
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "e3_fe2h_full_matched",
        "status": "COMPLETE",
        "protocol": {
            "train_tokens_per_variant": EXPECTED_TRAIN_TOKENS,
            "validation_tokens_per_variant": EXPECTED_VALID_TOKENS,
            "shared": {
                "d_model": 8192,
                "state_dim": 8192,
                "rank": 512,
                "batch_size": 112,
                "seq_len": 128,
                "block_size": 32,
                "seed": 0,
            },
        },
        "audit_errors": [],
        "variants": rows,
        "hypothesis": hypothesis,
        "decision": decision,
        "interpretation_boundaries": [
            "All quality comparisons are one-seed, one-epoch results.",
            "Dense-mask training computes every tile; active fraction is a routing semantic, not measured training FLOP sparsity.",
            "Positive excess token-route MI is statistical specialization evidence, not a semantic interpretation by itself.",
        ],
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_summary(args.result_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps({"status": summary["status"], "audit_errors": summary["audit_errors"]}, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
