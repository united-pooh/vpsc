"""RA0 exact reverse adjoint on core and real TextWorld sparse-event LM tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e2_f0_fusion_benchmark import (  # noqa: E402
    _autograd_node_count,
    _environment,
    _sample_summary,
    _sync,
)
from experiments.e3_at0_gated_trace import STATE_DIM  # noqa: E402
from experiments.e3_at1_trace_eligibility import (  # noqa: E402
    AT1_ATOL,
    AT1_RTOL,
    _equivalence_case,
    _even_queries,
)
from experiments.e3_el0_terminal_eligibility import (  # noqa: E402
    _SavedTensorCounter,
)
from experiments import e3_tw0_sparse_event_lm as tw0  # noqa: E402
from vpsc.world_model.cores import (  # noqa: E402
    CausalTransformerCore,
    E3GatedTraceScanCore,
    StatefulLSTMCore,
    TemporalCore,
    count_parameters,
)
from vpsc.world_model.event_corpus import load_event_corpus  # noqa: E402
from vpsc.world_model.lm import CausalLanguageModel  # noqa: E402
from vpsc.world_model.wikitext import SPLITS, Vocabulary  # noqa: E402


D_MODEL = tw0.D_MODEL
SEQUENCE_LENGTH = tw0.SEQUENCE_LENGTH
QUERY_COUNT = tw0.MAX_QUERIES
CORE_NAMES = ("snn_bptt", "snn_at1", "snn_ra0", "lstm", "transformer")
Runner = Callable[[], None]


def run_equivalence(device: torch.device) -> Dict[str, Any]:
    formal_queries = tuple(
        int(value)
        for value in _even_queries(SEQUENCE_LENGTH, QUERY_COUNT, device).tolist()
    )
    specifications = (
        (1, 1, (0,), True),
        (2, 32, (0, 7, 18, 31), True),
        (1, 512, (0, 1, 63, 127, 255, 383, 510, 511), False),
        (1, 512, formal_queries, True),
    )
    cases = [
        _equivalence_case(
            batch=batch,
            time_steps=time_steps,
            queries=queries,
            input_gradient=input_gradient,
            device=device,
            seed=9_300_000 + index,
            eligibility_backward_mode="reverse_adjoint",
        )
        for index, (batch, time_steps, queries, input_gradient) in enumerate(
            specifications
        )
    ]
    return {
        "atol": AT1_ATOL,
        "rtol": AT1_RTOL,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def _snn_core(mode: str, device: torch.device) -> E3GatedTraceScanCore:
    if mode not in ("snn_bptt", "snn_at1", "snn_ra0"):
        raise ValueError(f"unknown SNN mode: {mode}")
    return E3GatedTraceScanCore(
        D_MODEL,
        D_MODEL,
        state_dim=STATE_DIM,
        execution_mode="scan",
        eligibility_backward_mode=(
            "reverse_adjoint" if mode == "snn_ra0" else "forward_eligibility"
        ),
    ).to(device)


def _query_output(
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    if name in ("snn_at1", "snn_ra0"):
        if not isinstance(core, E3GatedTraceScanCore):  # pragma: no cover
            raise TypeError("sparse eligibility requires a gated trace core")
        return core.forward_multi_query_eligibility(
            value, query_indices, _unchecked=True
        ).sequence
    return core(value).sequence.index_select(1, query_indices)


def _saved_tensor_measurement(
    *,
    name: str,
    time_steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    core = _snn_core(name, device)
    value = torch.randn(
        1, time_steps, D_MODEL, device=device, requires_grad=True
    )
    queries = _even_queries(time_steps, QUERY_COUNT, device)
    counter = _SavedTensorCounter()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
        output = _query_output(name, core, value, queries)
        output.square().mean().backward()
    return {
        "mode": name,
        "time": time_steps,
        "query_count": QUERY_COUNT,
        "input_gradient": True,
        "logical_saved_bytes": counter.logical_bytes,
        "unique_storage_bytes": counter.unique_storage_bytes,
        "saved_tensor_count": counter.tensor_count,
        "autograd_nodes": _autograd_node_count(output),
    }


def benchmark_saved_tensors(device: torch.device) -> Dict[str, Any]:
    records = [
        _saved_tensor_measurement(
            name=name,
            time_steps=time_steps,
            device=device,
            seed=9_310_000 + time_steps * 10 + index,
        )
        for time_steps in (512, 2048)
        for index, name in enumerate(("snn_bptt", "snn_at1", "snn_ra0"))
    ]
    lookup = {
        (record["mode"], record["time"]): record for record in records
    }
    ratios = {
        str(time_steps): (
            lookup[("snn_ra0", time_steps)]["unique_storage_bytes"]
            / lookup[("snn_bptt", time_steps)]["unique_storage_bytes"]
        )
        for time_steps in (512, 2048)
    }
    growth = (
        lookup[("snn_ra0", 2048)]["unique_storage_bytes"]
        / lookup[("snn_ra0", 512)]["unique_storage_bytes"]
    )
    return {
        "records": records,
        "ra0_to_bptt_unique_storage_ratio": ratios,
        "ra0_t512_to_t2048_growth": growth,
        "passed": all(ratio <= 0.25 for ratio in ratios.values()),
    }


def _training_runner(
    *,
    name: str,
    core: TemporalCore[Any],
    value: torch.Tensor,
    query_indices: torch.Tensor,
) -> Runner:
    def run() -> None:
        core.zero_grad(set_to_none=True)
        value.grad = None
        output = _query_output(name, core, value, query_indices)
        output.square().mean().backward()

    return run


def _interleaved_samples(
    runners: Mapping[str, Runner],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> Dict[str, list[float]]:
    for _ in range(warmup):
        for runner in runners.values():
            runner()
    samples: Dict[str, list[float]] = {name: [] for name in runners}
    names = list(runners)
    generator = random.Random(seed)
    for _ in range(repeats):
        generator.shuffle(names)
        for name in names:
            _sync(device)
            started = time.perf_counter_ns()
            runners[name]()
            _sync(device)
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    return samples


def benchmark_core_training(
    *,
    threads: Sequence[int],
    lengths: Iterable[int],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, Any]:
    records = []
    thread_lanes: Tuple[int | None, ...] = (
        tuple(threads) if device.type == "cpu" else (None,)
    )
    for thread_count in thread_lanes:
        if thread_count is not None:
            torch.set_num_threads(thread_count)
        for length in lengths:
            torch.manual_seed(9_320_000 + (thread_count or 0) * 10_000 + length)
            bptt = _snn_core("snn_bptt", device)
            at1 = _snn_core("snn_at1", device)
            ra0 = _snn_core("snn_ra0", device)
            at1.load_state_dict(bptt.state_dict())
            ra0.load_state_dict(bptt.state_dict())
            cores: Dict[str, TemporalCore[Any]] = {
                "snn_bptt": bptt,
                "snn_at1": at1,
                "snn_ra0": ra0,
                "lstm": StatefulLSTMCore(D_MODEL, D_MODEL).to(device),
                "transformer": CausalTransformerCore(
                    D_MODEL,
                    D_MODEL,
                    num_layers=1,
                    num_heads=4,
                    mlp_ratio=2.0,
                    dropout=0.0,
                    max_cache_tokens=length,
                ).to(device),
            }
            value = torch.randn(
                1, length, D_MODEL, device=device, requires_grad=True
            )
            query_indices = _even_queries(length, QUERY_COUNT, device)
            runners = {
                name: _training_runner(
                    name=name,
                    core=core,
                    value=value,
                    query_indices=query_indices,
                )
                for name, core in cores.items()
            }
            nodes = {
                name: _autograd_node_count(
                    _query_output(name, core, value, query_indices)
                )
                for name, core in cores.items()
            }
            value.grad = None
            for core in cores.values():
                core.zero_grad(set_to_none=True)
            samples = _interleaved_samples(
                runners,
                warmup=warmup,
                repeats=repeats,
                device=device,
                seed=9_321_000 + (thread_count or 0) * 10_000 + length,
            )
            models = {
                name: {
                    **_sample_summary(sample, length),
                    "parameters": count_parameters(cores[name]),
                    "autograd_nodes": nodes[name],
                }
                for name, sample in samples.items()
            }
            at1_speedup = (
                models["snn_at1"]["p50_ms"] / models["snn_ra0"]["p50_ms"]
            )
            bptt_speedup = (
                models["snn_bptt"]["p50_ms"] / models["snn_ra0"]["p50_ms"]
            )
            passed = (
                at1_speedup >= 1.25
                and bptt_speedup >= 1.25
                and models["snn_ra0"]["p50_ms"] <= models["lstm"]["p50_ms"]
            )
            records.append(
                {
                    "threads": thread_count,
                    "time": length,
                    "query_count": QUERY_COUNT,
                    "input_gradient": True,
                    "models": models,
                    "ra0_vs_at1_speedup": at1_speedup,
                    "ra0_vs_bptt_speedup": bptt_speedup,
                    "passed": passed,
                }
            )
    return {"records": records, "passed": any(item["passed"] for item in records)}


def build_textworld_models(
    seed: int,
    vocabulary: Vocabulary,
    *,
    device: torch.device,
) -> Dict[str, CausalLanguageModel[Any]]:
    models = tw0.build_models(seed, vocabulary, device=device)
    ra0 = tw0._common_model(
        E3GatedTraceScanCore(
            D_MODEL,
            D_MODEL,
            state_dim=STATE_DIM,
            eligibility_backward_mode="reverse_adjoint",
        ),
        vocabulary=vocabulary,
    )
    ra0.load_state_dict(models["snn_bptt"].state_dict())
    ra0.to(device)
    return {
        "snn_bptt": models["snn_bptt"],
        "snn_at1": models["snn_at1"],
        "snn_ra0": ra0,
        "lstm": models["lstm"],
        "transformer": models["transformer"],
    }


def _benchmark_ra0_streaming(
    models: Mapping[str, CausalLanguageModel[Any]],
    token_ids: Sequence[int],
    *,
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    aliases = {
        "snn_bptt": models["snn_bptt"],
        "snn_at1": models["snn_ra0"],
        "lstm": models["lstm"],
        "transformer": models["transformer"],
    }
    result = tw0.benchmark_streaming(
        aliases,
        token_ids,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        device=device,
        seed=seed,
    )
    result["models"]["snn_ra0_cached"] = result["models"].pop(
        "snn_at1_cached"
    )
    return result


def run_textworld(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    corpus_root = args.corpus_dir.expanduser().resolve()
    corpus = load_event_corpus(corpus_root)
    chunks_by_split = {
        split: tw0.build_sparse_chunks(
            corpus,
            split,
            sequence_length=args.sequence_length,
            max_queries=args.max_queries,
        )
        for split in SPLITS
    }
    data_audit = tw0.audit_sparse_chunks(chunks_by_split, corpus.vocabulary)
    test_episode = next(corpus.iter_episode_token_ids("test"))
    stream_total = args.stream_warmup + args.stream_steps
    if len(test_episode) < stream_total:
        raise ValueError("test episode is too short for streaming benchmark")

    seed_results = []
    for seed in args.seeds:
        models = build_textworld_models(
            9_330_000 + 100 * seed, corpus.vocabulary, device=device
        )
        parameter_counts = {
            name: {
                "total": count_parameters(model),
                "core": count_parameters(model.core),
            }
            for name, model in models.items()
        }
        totals = tuple(record["total"] for record in parameter_counts.values())
        parameter_spread = (max(totals) - min(totals)) / tw0._mean(totals)
        if parameter_spread > 0.02:
            raise AssertionError(f"RA0 parameter spread failed: {parameter_counts}")
        pre = {
            name: {
                "valid_sparse": tw0.evaluate_model(
                    model, chunks_by_split["valid"], dense=False, device=device
                ),
                "test_sparse": tw0.evaluate_model(
                    model, chunks_by_split["test"], dense=False, device=device
                ),
            }
            for name, model in models.items()
        }
        training = {
            name: tw0.train_model(
                name,
                model,
                chunks_by_split["train"],
                epochs=args.epochs,
                device=device,
                optimizer_foreach=True,
                optimizer_fused=True,
            )
            for name, model in models.items()
        }
        post = {
            name: {
                "train_sparse": tw0.evaluate_model(
                    model, chunks_by_split["train"], dense=False, device=device
                ),
                "valid_sparse": tw0.evaluate_model(
                    model, chunks_by_split["valid"], dense=False, device=device
                ),
                "test_sparse": tw0.evaluate_model(
                    model, chunks_by_split["test"], dense=False, device=device
                ),
                "test_dense_outcome": tw0.evaluate_model(
                    model, chunks_by_split["test"], dense=True, device=device
                ),
            }
            for name, model in models.items()
        }
        streaming = _benchmark_ra0_streaming(
            models,
            test_episode,
            warmup_steps=args.stream_warmup,
            measured_steps=args.stream_steps,
            device=device,
            seed=9_331_000 + seed,
        )
        ablation = {
            mode: tw0.evaluate_mechanism_ablation(
                models["snn_ra0"],
                chunks_by_split["test"],
                mode=mode,
                device=device,
            )
            for mode in ("spike_only", "trace_only")
        }
        seed_results.append(
            {
                "seed": seed,
                "parameter_counts": parameter_counts,
                "parameter_relative_spread": parameter_spread,
                "pre": pre,
                "training": training,
                "post": post,
                "streaming": streaming,
                "mechanism_ablation": ablation,
            }
        )
    return {
        "corpus_dir": str(corpus_root),
        "manifest_provenance": tw0._manifest_provenance(corpus_root),
        "corpus_provenance": tw0._corpus_provenance(corpus),
        "data_audit": data_audit,
        "seeds": seed_results,
    }


def _textworld_decision(
    result: Mapping[str, Any], *, quick: bool
) -> Dict[str, Any]:
    if quick:
        return {
            "data_gate": "PASS" if result["data_audit"]["passed"] else "FAIL",
            "task_gate": "SMOKE",
            "quality_gate": "SMOKE",
            "speed_gate": "SMOKE",
            "stream_gate": "SMOKE",
            "overall": "SMOKE",
        }
    seeds = result["seeds"]
    task_pass = all(
        seed["post"][name]["test_sparse"]["nll"]
        <= seed["pre"][name]["test_sparse"]["nll"] - 0.10
        for seed in seeds
        for name in ("lstm", "transformer")
    )
    ra0_improvement = all(
        seed["post"]["snn_ra0"]["test_sparse"]["nll"]
        <= seed["pre"]["snn_ra0"]["test_sparse"]["nll"] - 0.10
        for seed in seeds
    )
    mean_nll = {
        name: tw0._mean(
            seed["post"][name]["test_sparse"]["nll"] for seed in seeds
        )
        for name in CORE_NAMES
    }
    best_ann = min(mean_nll["lstm"], mean_nll["transformer"])
    gap_at1 = abs(mean_nll["snn_ra0"] - mean_nll["snn_at1"])
    gap_bptt = abs(mean_nll["snn_ra0"] - mean_nll["snn_bptt"])
    quality_pass = (
        task_pass
        and ra0_improvement
        and mean_nll["snn_ra0"] <= best_ann + 0.25
        and gap_at1 <= 0.10
        and gap_bptt <= 0.10
    )
    mean_p50 = {
        name: tw0._mean(
            seed["training"][name]["timing"]["p50_ms"] for seed in seeds
        )
        for name in CORE_NAMES
    }
    at1_speedup = mean_p50["snn_at1"] / mean_p50["snn_ra0"]
    bptt_speedup = mean_p50["snn_bptt"] / mean_p50["snn_ra0"]
    speed_pass = (
        at1_speedup >= 1.25
        and bptt_speedup >= 1.25
        and mean_p50["snn_ra0"] <= mean_p50["lstm"]
    )
    stream_pass = all(
        seed["streaming"]["models"]["snn_ra0_cached"]["p50_ms"]
        <= seed["streaming"]["models"]["lstm"]["p50_ms"]
        and seed["streaming"]["models"]["snn_ra0_cached"]["p95_ms"]
        <= seed["streaming"]["models"]["lstm"]["p95_ms"]
        for seed in seeds
    )
    gates = {
        "data_gate": bool(result["data_audit"]["passed"]),
        "task_gate": task_pass,
        "quality_gate": quality_pass,
        "speed_gate": speed_pass,
        "stream_gate": stream_pass,
    }
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "overall": "PASS" if all(gates.values()) else "FAIL",
        "mean_test_sparse_nll": mean_nll,
        "best_ann_mean_nll": best_ann,
        "ra0_vs_at1_mean_nll_gap": gap_at1,
        "ra0_vs_bptt_mean_nll_gap": gap_bptt,
        "mean_training_p50_ms": mean_p50,
        "ra0_vs_at1_training_speedup": at1_speedup,
        "ra0_vs_bptt_training_speedup": bptt_speedup,
    }


def _decision(
    *,
    equivalence: Mapping[str, Any],
    memory: Mapping[str, Any],
    core_training: Mapping[str, Any],
    textworld: Mapping[str, Any],
    quick: bool,
) -> Dict[str, Any]:
    tw_decision = _textworld_decision(textworld, quick=quick)
    if quick:
        return {
            "equivalence_gate": "PASS" if equivalence["passed"] else "FAIL",
            "memory_gate": "PASS" if memory["passed"] else "FAIL",
            "core_speed_gate": "SMOKE",
            "textworld": tw_decision,
            "overall": "SMOKE",
            "next_route": "formal_ra0",
        }
    gates = {
        "equivalence_gate": bool(equivalence["passed"]),
        "memory_gate": bool(memory["passed"]),
        "core_speed_gate": bool(core_training["passed"]),
        "textworld_gate": tw_decision["overall"] == "PASS",
    }
    overall = "PASS" if all(gates.values()) else "FAIL"
    return {
        **{name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "textworld": tw_decision,
        "overall": overall,
        "next_route": (
            "counterfactual_sequence_generation"
            if overall == "PASS"
            else "embedding_scatter_or_native_fused_adjoint"
        ),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
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
        raise AssertionError("RA0 equivalence gate failed; refusing benchmarks")
    memory = benchmark_saved_tensors(device)
    core_training = benchmark_core_training(
        threads=threads,
        lengths=(512,) if args.quick else (512, 2048),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    if device.type == "cpu":
        torch.set_num_threads(threads[0] if args.quick else 4)
    textworld = run_textworld(args, device=device)
    decision = _decision(
        equivalence=equivalence,
        memory=memory,
        core_training=core_training,
        textworld=textworld,
        quick=args.quick,
    )
    return {
        "schema_version": 1,
        "experiment": "E3-RA0 exact reverse adjoint on sparse-event LM",
        "formal": not args.quick,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **_environment(device),
            "torch_num_threads_at_textworld": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "configuration": {
            "d_model": D_MODEL,
            "state_dim": STATE_DIM,
            "sequence_length": args.sequence_length,
            "query_count": args.max_queries,
            "threads": threads if device.type == "cpu" else None,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seeds": tuple(args.seeds),
            "epochs": args.epochs,
            "stream_warmup": args.stream_warmup,
            "stream_steps": args.stream_steps,
            "input_gradient": True,
            "optimizer_foreach": True,
            "optimizer_fused": True,
            "eligibility_forward_mode": "segment_api_scan_forward",
            "eligibility_backward_mode": "reverse_adjoint",
        },
        "equivalence": equivalence,
        "saved_tensors": memory,
        "core_training": core_training,
        "textworld": textworld,
        "decision": decision,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=tw0.DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/e3_scan/e3_ra0_reverse_adjoint.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--max-queries", type=int, default=QUERY_COUNT)
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--stream-warmup", type=int, default=64)
    parser.add_argument("--stream-steps", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    positive = (
        args.epochs,
        args.sequence_length,
        args.max_queries,
        *args.threads,
        args.warmup,
        args.repeats,
        args.stream_steps,
    )
    if min(positive) <= 0 or args.stream_warmup < 0:
        parser.error("all sizes must be positive and stream warmup non-negative")
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 1
        args.threads = args.threads[:1]
        args.warmup = 1
        args.repeats = 1
        args.stream_warmup = 4
        args.stream_steps = 32
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
