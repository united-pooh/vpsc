"""SG24 same-V100 raw-language world-transition architecture comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import e3_sg0_counterfactual_generation as sg0  # noqa: E402


EXPECTED_BASE_RUNNER_SHA256 = (
    "360054a294a8ffb4905eb819545749dd379905aa7ceec2fb507b16f232266f30"
)
EXPECTED_PARENT_ARTIFACT_SHA256 = (
    "734a095b984aac495a06329565b59783116eec421942640e269aab60b0eff05d"
)
EXPECTED_CORPUS_SHA256 = {
    "train/episodes.jsonl": (
        "5938045cf8e93fb2e1863aeefbe058e73e4ede8e62cd887db89c663d93444fd3"
    ),
    "valid/episodes.jsonl": (
        "1437f6800372658fdf48db2f27a1ce6c1308953cafd9a3d85c3e3bdc6fd502d4"
    ),
    "test/episodes.jsonl": (
        "52d6a96c310a23395aafb0999aaeb7cbf572c099c8ec88bc3550c96209ea6962"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_input_audit(corpus_root: Path) -> Dict[str, Any]:
    base_runner = Path(sg0.__file__).resolve()
    parent_artifact = ROOT / "results/e3_scan/e3_sg0_counterfactual_generation.json"
    records: Dict[str, Dict[str, Any]] = {}
    for relative, expected in EXPECTED_CORPUS_SHA256.items():
        path = corpus_root / relative
        actual = _sha256(path) if path.is_file() else None
        records[relative] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }
    base_actual = _sha256(base_runner)
    parent_actual = _sha256(parent_artifact) if parent_artifact.is_file() else None
    return {
        "base_runner": {
            "path": str(base_runner),
            "expected_sha256": EXPECTED_BASE_RUNNER_SHA256,
            "actual_sha256": base_actual,
            "passed": base_actual == EXPECTED_BASE_RUNNER_SHA256,
        },
        "parent_cpu_artifact": {
            "path": str(parent_artifact),
            "expected_sha256": EXPECTED_PARENT_ARTIFACT_SHA256,
            "actual_sha256": parent_actual,
            "passed": parent_actual == EXPECTED_PARENT_ARTIFACT_SHA256,
            "gate": False,
        },
        "corpus": records,
        "passed": base_actual == EXPECTED_BASE_RUNNER_SHA256
        and all(record["passed"] for record in records.values()),
    }


def _augment_decision(
    decision: Mapping[str, Any],
    *,
    frozen_inputs_passed: bool,
    backend_passed: bool,
    quick: bool,
) -> Dict[str, Any]:
    original_overall = str(decision["overall"])
    if quick:
        overall = (
            "SMOKE"
            if frozen_inputs_passed and backend_passed and original_overall == "SMOKE"
            else "FAIL"
        )
    else:
        overall = (
            "PASS"
            if frozen_inputs_passed
            and backend_passed
            and original_overall == "PASS"
            else "FAIL"
        )
    return {
        **dict(decision),
        "frozen_input_gate": "PASS" if frozen_inputs_passed else "FAIL",
        "backend_gate": "PASS" if backend_passed else "FAIL",
        "sg0_architecture_gates_overall": original_overall,
        "overall": overall,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.device != "cuda":
        raise ValueError("SG24 is a CUDA-only same-hardware experiment")
    if not torch.cuda.is_available():
        raise RuntimeError("SG24 requires CUDA")

    corpus_root = args.corpus_dir.expanduser().resolve()
    input_audit = frozen_input_audit(corpus_root)
    if not input_audit["passed"]:
        raise AssertionError("SG24 frozen input audit failed; refusing model run")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    backend_passed = "V100" in device_name.upper()
    if not backend_passed:
        raise AssertionError(f"SG24 requires the frozen V100 backend, got {device_name}")

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    allocated_start = torch.cuda.memory_allocated(device)
    reserved_start = torch.cuda.memory_reserved(device)
    started = time.perf_counter_ns()
    result = sg0.run_experiment(args)
    torch.cuda.synchronize(device)
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9

    result["schema_version"] = 2
    result["experiment"] = (
        "E3-SG24 same-V100 raw-language action-conditioned world transition"
    )
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["source_provenance"] = {
        "wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "frozen_inputs": input_audit,
    }
    result["configuration"].update(
        {
            "comparison_backend": "same_v100_cuda_fp32",
            "required_device": "cuda:0",
            "architecture_scope": tuple(sg0.MODEL_NAMES),
            "parent_protocol": "E3-SG0",
        }
    )
    result["runtime_audit"] = {
        "total_wall_seconds": elapsed_seconds,
        "cuda_device_name": device_name,
        "cuda_compute_capability": torch.cuda.get_device_capability(device),
        "cuda_peak_scope": "all_five_models_in_one_process",
        "cuda_allocated_start_bytes": allocated_start,
        "cuda_reserved_start_bytes": reserved_start,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "cuda_allocated_end_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_end_bytes": torch.cuda.memory_reserved(device),
    }
    result["decision"] = _augment_decision(
        result["decision"],
        frozen_inputs_passed=bool(input_audit["passed"]),
        backend_passed=backend_passed,
        quick=bool(args.quick),
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=sg0.tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_sg24_cuda_counterfactual_generation.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--max-generation-tokens", type=int, default=sg0.MAX_GENERATION_TOKENS
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.device = "cuda"
    if min(args.epochs, args.threads, args.max_generation_tokens) <= 0:
        parser.error("epochs, threads, and max generation tokens must be positive")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.max_generation_tokens = min(args.max_generation_tokens, 32)
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
