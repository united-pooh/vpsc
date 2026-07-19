from __future__ import annotations

import torch

from experiments import e3_sg23_backend_capability as sg23h


def _operation_record(seconds: float, error: float = 0.0):
    timing = {
        "minimum_seconds": seconds,
        "median_seconds": seconds,
        "p95_seconds": seconds,
        "maximum_seconds": seconds,
        "sample_count": 3,
    }
    return {
        "supported": True,
        "max_abs_error_vs_cpu": error,
        "resident_scalar_sync": timing,
        "full_input_output_transfer": timing,
    }


def test_timing_summary_uses_observed_p95() -> None:
    summary = sg23h.timing_summary([5.0, 1.0, 4.0, 2.0, 3.0])
    assert summary["median_seconds"] == 3.0
    assert summary["p95_seconds"] == 5.0
    assert summary["sample_count"] == 5


def test_cpu_benchmark_preserves_operation_correctness() -> None:
    result = sg23h._benchmark_operation(
        name="matmul",
        operation=lambda left, right: left @ right,
        cpu_inputs=(torch.eye(4), torch.arange(16.0).reshape(4, 4)),
        device=torch.device("cpu"),
        warmups=1,
        repetitions=3,
    )
    assert result["supported"]
    assert result["max_abs_error_vs_cpu"] == 0.0
    assert result["output_device"] == "cpu"


def test_cpu_thread_sweep_records_each_requested_count() -> None:
    original_threads = torch.get_num_threads()
    try:
        result = sg23h.benchmark_cpu_thread_sweep(
            thread_counts=(1, 2),
            size=16,
            warmups=0,
            repetitions=1,
        )
    finally:
        torch.set_num_threads(original_threads)
    assert result["thread_counts"] == (1, 2)
    assert set(result["results"]) == {"1", "2"}
    assert all(
        record["supported"]
        for thread in result["results"].values()
        for record in thread.values()
    )


def test_decision_requires_adapter_correctness_and_speed() -> None:
    directml = {
        "vector_add": _operation_record(0.5),
        "sizes": {
            "443": {
                "readout": _operation_record(0.5),
                "gram": _operation_record(0.5),
                "matrix_free_normal": _operation_record(0.5),
            }
        },
    }
    decision = sg23h.make_decision(
        adapter_names=("AMD Radeon RX 7800 XT",),
        directml=directml,
        comparison={
            "any_resident_speedup": True,
            "any_full_transfer_speedup": False,
        },
    )
    assert decision["backend_available"]
    assert decision["overall"] == "PASS"
    assert decision["deployment_boundary"] == (
        "directml_batch_or_matrix_free_candidate"
    )


def test_decision_fails_closed_on_bad_matmul() -> None:
    bad = _operation_record(0.5, error=1e-2)
    directml = {
        "vector_add": _operation_record(0.5),
        "sizes": {
            "443": {
                "readout": bad,
                "gram": _operation_record(0.5),
                "matrix_free_normal": _operation_record(0.5),
            }
        },
    }
    decision = sg23h.make_decision(
        adapter_names=("AMD Radeon RX 7800 XT",),
        directml=directml,
        comparison={
            "any_resident_speedup": True,
            "any_full_transfer_speedup": True,
        },
    )
    assert not decision["fp32_correctness_gate"]
    assert not decision["backend_available"]
    assert decision["overall"] == "FAIL"


def test_wsl_rocm_gate_requires_gpu_and_hip_runtime() -> None:
    unavailable = {
        "torch": {"stdout": '{"hip": null, "cuda_available": false}'},
        "runtime": {"stdout": "Marketing Name: Ryzen 9 7950X"},
        "opencl": {"stdout": "Number of devices: 0"},
    }
    available = {
        "torch": {"stdout": '{"hip": "7.2", "cuda_available": true}'},
        "runtime": {"stdout": "Name: gfx1101 RX 7800 XT"},
        "opencl": {"stdout": "Number of devices: 1"},
    }
    assert not sg23h._wsl_rocm_gate(unavailable)
    assert sg23h._wsl_rocm_gate(available)
