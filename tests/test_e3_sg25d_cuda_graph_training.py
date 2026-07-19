from __future__ import annotations

from copy import deepcopy

from experiments.e3_sg25d_cuda_graph_training import _decision


def _architecture_record(
    graph_p50_ms: float, examples_per_second: float
) -> dict[str, object]:
    return {
        "capture": {
            "shape_count": 35,
            "capture_seconds": 1.0,
            "allocated_delta_bytes": 200,
        },
        "equivalence": {"passed": True},
        "eager_benchmark": {
            "timing": {"p50_ms": 2.0, "mean_ms": 2.0},
            "allocated_total_delta_bytes": 100,
        },
        "graph_benchmark": {
            "timing": {"p50_ms": graph_p50_ms, "mean_ms": graph_p50_ms},
            "examples_per_second_mean": examples_per_second,
        },
        "eager_profiler": {"host_launch_and_copy_api_count": 100},
        "graph_profiler": {"host_launch_and_copy_api_count": 10},
    }


def test_formal_decision_requires_snn_to_beat_both_graphed_anns() -> None:
    architectures = {
        "snn_ra0": _architecture_record(1.0, 1_000.0),
        "lstm": _architecture_record(1.1, 900.0),
        "transformer": _architecture_record(1.2, 800.0),
    }
    quality = {name: {"passed": True} for name in architectures}

    passing = _decision(architectures, quality=quality, quick=False)

    assert passing["graphed_ann_gate"] == "PASS"
    assert passing["overall"] == "PASS"

    slower_than_lstm = deepcopy(architectures)
    slower_than_lstm["lstm"]["graph_benchmark"] = {
        "timing": {"p50_ms": 0.9, "mean_ms": 0.9},
        "examples_per_second_mean": 1_100.0,
    }

    failing = _decision(slower_than_lstm, quality=quality, quick=False)

    assert failing["graphed_ann_gate"] == "FAIL"
    assert failing["overall"] == "FAIL"


def test_formal_decision_uses_graph_cache_increment_for_memory_gate() -> None:
    architectures = {
        "snn_ra0": _architecture_record(1.0, 1_000.0),
        "lstm": _architecture_record(1.1, 900.0),
        "transformer": _architecture_record(1.2, 800.0),
    }
    quality = {name: {"passed": True} for name in architectures}
    architectures["snn_ra0"]["capture"]["allocated_delta_bytes"] = 401

    result = _decision(architectures, quality=quality, quick=False)

    assert result["snn_graph_memory_to_eager_ratio"] == 4.01
    assert result["memory_gate"] == "FAIL"
    assert result["overall"] == "FAIL"
