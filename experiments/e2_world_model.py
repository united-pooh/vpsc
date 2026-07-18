"""CPU-only pilot entry point for the E2 world-model research track.

Subcommands:

``probe``
    Record Python/PyTorch/platform versions and non-mutating probes for the
    official TextWorld, HomeGrid, and Messenger adapters.

``wikitext-pilot``
    Load a SHA256-verified WikiText-2 raw cache, build a train-only vocabulary,
    and compare the frozen LSTM/Transformer/E2 model suite with identical data,
    seeds, step limits, and CPU measurement code.

This is deliberately a pilot harness.  It writes measurements and provenance,
never an automatic scientific PASS/FAIL decision.  TextWorld execution belongs
to its separate experiment and is not implemented here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from itertools import islice
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from vpsc.world_model.factory import (
    FairLMConfig,
    FrozenE2Config,
    assert_parameter_budget,
    build_model_suite,
)
from vpsc.world_model.homegrid import probe_homegrid, probe_messenger
from vpsc.world_model.textworld import probe_textworld
from vpsc.world_model.training import (
    TrainingConfig,
    benchmark_streaming_step,
    evaluate_language_model,
    seed_everything,
    train_language_model,
)
from vpsc.world_model.wikitext import (
    WIKITEXT2_RAW_SOURCE,
    SPECIAL_TOKENS,
    WikiText2Source,
    load_wikitext2,
)


SCHEMA_VERSION = 1
PILOT_SCOPE = "pilot_not_confirmatory"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / "data"
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results"


def _installed_version(distribution: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _probe_record(value: Any, *, available: bool) -> Dict[str, Any]:
    record = asdict(value)
    record["available"] = bool(available)
    return record


def collect_probe(*, verify_imports: bool = False) -> Dict[str, Any]:
    """Collect environment and official-adapter status without network access."""

    textworld = probe_textworld()
    homegrid = probe_homegrid(verify_import=verify_imports)
    messenger = probe_messenger(verify_import=verify_imports)
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "probe",
        "scope": PILOT_SCOPE,
        "confirmatory": False,
        "automatic_decision": None,
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "pilot_device": "cpu",
            "torch_cpu_threads": int(torch.get_num_threads()),
        },
        "package_versions": {
            "textworld": _installed_version("textworld"),
            "homegrid": _installed_version("homegrid"),
            "messenger": _installed_version("messenger"),
            "gym": _installed_version("gym"),
        },
        "official_adapter_probes": {
            "textworld": _probe_record(
                textworld,
                available=textworld.installed and textworld.platform_supported,
            ),
            "homegrid": _probe_record(homegrid, available=homegrid.available),
            "messenger": _probe_record(messenger, available=messenger.available),
        },
        "wikitext2_raw_source": WIKITEXT2_RAW_SOURCE.metadata(),
        "notes": [
            "Adapter probes do not create fallback environments.",
            "verify_imports is disabled by default to keep this probe non-mutating.",
            "No scientific pass/fail decision is produced by this command.",
        ],
    }


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write strict JSON to a sibling temporary file, then atomically replace."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                ensure_ascii=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    payload = collect_probe(verify_imports=bool(args.verify_imports))
    payload["verify_imports_requested"] = bool(args.verify_imports)
    write_json_atomic(args.output, payload)
    return payload


def _batch_stream(corpus: Any, split: str, args: argparse.Namespace) -> Any:
    return corpus.iter_batches(
        split,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        drop_last=False,
        as_tensors=False,
    )


def _read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"WikiText manifest must be a JSON object: {path}")
    return value


def run_wikitext_pilot(
    args: argparse.Namespace,
    *,
    source: Optional[WikiText2Source] = None,
) -> Dict[str, Any]:
    """Run the real-data CPU pilot and atomically write one complete JSON record."""

    source = WIKITEXT2_RAW_SOURCE if source is None else source
    corpus = load_wikitext2(
        args.cache_dir,
        download=bool(args.download),
        force_download=False,
        source=source,
        max_vocab_size=args.vocab_size,
    )
    manifest = _read_manifest(corpus.paths.manifest)
    if manifest.get("source") != source.metadata():
        raise ValueError("verified WikiText manifest source does not match requested source")

    # Every model and every seed receives this exact deterministic token cycle.
    stream_token_count = max(1, args.streaming_warmup_steps + args.streaming_steps)
    streaming_tokens = tuple(
        islice(corpus.iter_token_ids("test"), stream_token_count)
    )
    if not streaming_tokens:
        raise ValueError("WikiText test split yielded no streaming tokens")

    e2_config = FrozenE2Config(
        policy=args.e2_policy,
        positive_factor=args.positive_factor,
    )
    common_factory_config = {
        "vocab_size": len(corpus.vocabulary),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "padding_idx": corpus.vocabulary.pad_id,
        "transformer_max_cache_tokens": args.cache_window,
        "e2": e2_config,
        "parameter_tolerance": 0.02,
        "auto_match_parameters": True,
    }

    seed_results = []
    for seed in args.seeds:
        seed_everything(seed)
        suite = build_model_suite(FairLMConfig(**common_factory_config))
        parameter_report = assert_parameter_budget(suite)
        model_results: Dict[str, Any] = {}

        for model_name, model in suite.models.items():
            train_metrics = train_language_model(
                model,
                _batch_stream(corpus, "train", args),
                TrainingConfig(
                    seed=seed,
                    learning_rate=args.learning_rate,
                    max_steps=args.steps,
                    device="cpu",
                ),
            )
            valid_metrics = evaluate_language_model(
                model,
                _batch_stream(corpus, "valid", args),
                max_steps=args.eval_steps,
                device="cpu",
            )
            test_metrics = evaluate_language_model(
                model,
                _batch_stream(corpus, "test", args),
                max_steps=args.eval_steps,
                device="cpu",
            )
            streaming_metrics = benchmark_streaming_step(
                model,
                streaming_tokens,
                warmup_steps=args.streaming_warmup_steps,
                measured_steps=args.streaming_steps,
                seed=seed,
                device="cpu",
            )
            model_results[model_name] = {
                "parameters": model.parameter_stats().as_dict(),
                "train": asdict(train_metrics),
                "valid": asdict(valid_metrics),
                "test": asdict(test_metrics),
                "streaming": asdict(streaming_metrics),
            }

        observed_steps = {
            record["train"]["steps"] for record in model_results.values()
        }
        observed_targets = {
            record["train"]["target_count"] for record in model_results.values()
        }
        if len(observed_steps) != 1 or len(observed_targets) != 1:
            raise RuntimeError(
                "fairness invariant failed: models consumed different train steps/tokens"
            )

        effective_gains = asdict(suite.e2.core.effective_gains())
        seed_results.append(
            {
                "seed": seed,
                "parameter_budget": parameter_report.as_dict(),
                "e2_effective_gains": effective_gains,
                "models": model_results,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "command": "wikitext-pilot",
        "scope": PILOT_SCOPE,
        "confirmatory": False,
        "automatic_decision": None,
        "device": "cpu",
        "dataset": {
            "name": source.name,
            "version": source.version,
            "verified_archive": True,
            "synthetic": False,
            "download_requested": bool(args.download),
            "source": source.metadata(),
            "cache_root": str(corpus.paths.root),
            "archive": str(corpus.paths.archive),
            "manifest_path": str(corpus.paths.manifest),
            "manifest": manifest,
            "vocabulary": {
                "requested_max_size": args.vocab_size,
                "actual_size": len(corpus.vocabulary),
                "fingerprint_sha256": corpus.vocabulary.fingerprint,
                "special_tokens": list(SPECIAL_TOKENS),
                "built_from": "train_only",
                "tokenizer": corpus.tokenizer.metadata(),
            },
        },
        "config": {
            "seeds": list(args.seeds),
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "train_steps": args.steps,
            "eval_steps": args.eval_steps,
            "learning_rate": args.learning_rate,
            "transformer_cache_window": args.cache_window,
            "streaming_warmup_steps": args.streaming_warmup_steps,
            "streaming_steps": args.streaming_steps,
            "e2_policy": args.e2_policy,
            "e2_positive_factor": args.positive_factor,
            "same_data_order_for_all_models": True,
            "same_step_limits_for_all_models": True,
        },
        "results": seed_results,
        "interpretation_boundary": (
            "Pilot measurements only. No automatic pass/fail or confirmatory "
            "claim is produced."
        ),
    }
    write_json_atomic(args.output, payload)
    return payload


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _cache_window(value: str) -> Optional[int]:
    if value.lower() in {"none", "unbounded"}:
        return None
    return _positive_int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E2 world-model CPU pilot (never confirmatory)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe",
        help="write environment/version/official-adapter probes",
    )
    probe.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "e2_world_model_probe.json",
    )
    probe.add_argument(
        "--verify-imports",
        action="store_true",
        help="also import installed HomeGrid/Messenger packages to verify registration",
    )
    probe.set_defaults(handler=run_probe)

    pilot = subparsers.add_parser(
        "wikitext-pilot",
        help="run verified WikiText-2 raw LSTM/Transformer/E2 CPU pilot",
    )
    pilot.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    pilot.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "e2_wikitext_pilot.json",
    )
    pilot.add_argument(
        "--download",
        action="store_true",
        help="explicitly allow downloading the pinned WikiText-2 raw archive",
    )
    pilot.add_argument("--seeds", nargs="+", type=_nonnegative_int, default=[0])
    pilot.add_argument("--d-model", type=_positive_int, default=64)
    pilot.add_argument("--num-heads", type=_positive_int, default=4)
    pilot.add_argument("--vocab", "--vocab-size", dest="vocab_size", type=_positive_int, default=4096)
    pilot.add_argument("--batch", "--batch-size", dest="batch_size", type=_positive_int, default=8)
    pilot.add_argument(
        "--seq",
        "--sequence-length",
        dest="sequence_length",
        type=_positive_int,
        default=64,
    )
    pilot.add_argument("--steps", type=_positive_int, default=10)
    pilot.add_argument("--eval-steps", type=_positive_int, default=5)
    pilot.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    pilot.add_argument("--dropout", type=float, default=0.0)
    pilot.add_argument("--cache-window", type=_cache_window, default=256)
    pilot.add_argument(
        "--e2-policy",
        choices=("exact", "margin", "hybrid"),
        default="exact",
    )
    pilot.add_argument("--positive-factor", type=_positive_float, default=1.0)
    pilot.add_argument(
        "--streaming-warmup-steps",
        type=_nonnegative_int,
        default=5,
    )
    pilot.add_argument("--streaming-steps", type=_positive_int, default=20)
    pilot.set_defaults(handler=run_wikitext_pilot)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    print(f"wrote {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
