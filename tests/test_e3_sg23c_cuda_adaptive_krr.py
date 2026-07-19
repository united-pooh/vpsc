from __future__ import annotations

import pytest
import torch

from experiments import e3_sg23c_cuda_adaptive_krr as sg23c


def _toy_problem(device: torch.device):
    torch.manual_seed(230719)
    dense = torch.randn(9, 6, dtype=torch.float64)
    dense[dense.abs() < 0.6] = 0.0
    coo = dense.to_sparse_coo().coalesce().to(device)
    counts = torch.arange(1, 10, dtype=torch.float64, device=device) % 3 + 1
    targets = torch.randn(9, 4, dtype=torch.float64, device=device)
    return dense, coo, counts, targets


def test_timing_summary_uses_observed_p95() -> None:
    summary = sg23c.timing_summary((0.5, 0.1, 0.4, 0.2, 0.3))
    assert summary["median_seconds"] == 0.3
    assert summary["p95_seconds"] == 0.5
    assert summary["sample_count"] == 5


def test_dense_dual_and_full_rank_prediction_space_match() -> None:
    dense, coo, counts, targets = _toy_problem(torch.device("cpu"))
    dual = sg23c._dense_dual_operation(coo, counts, targets)
    rank = sg23c._rank_operation(dense, counts, targets)
    assert torch.allclose(
        dual["scores"], rank["scores"], atol=1e-8, rtol=0.0
    )


def test_benchmark_operation_records_cold_and_resident_samples() -> None:
    value = torch.ones(2, 2, dtype=torch.float64)
    result, metrics = sg23c.benchmark_operation(
        lambda: {"value": value + 1.0},
        device=torch.device("cpu"),
        warmups=1,
        repetitions=3,
    )
    assert torch.equal(result["value"], torch.full((2, 2), 2.0))
    assert metrics["cold_seconds"] >= 0.0
    assert metrics["resident"]["sample_count"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_dense_dual_matches_cpu_fp64() -> None:
    _dense, cpu_coo, cpu_counts, cpu_targets = _toy_problem(
        torch.device("cpu")
    )
    cpu = sg23c._dense_dual_operation(cpu_coo, cpu_counts, cpu_targets)
    cuda = sg23c._dense_dual_operation(
        cpu_coo.to("cuda"), cpu_counts.to("cuda"), cpu_targets.to("cuda")
    )
    assert cuda["scores"].device.type == "cuda"
    assert torch.allclose(
        cuda["scores"].cpu(), cpu["scores"], atol=1e-8, rtol=0.0
    )
    hybrid = sg23c._hybrid_dense_dual_operation(
        cpu_coo.to("cuda"), cpu_counts, cpu_targets
    )
    assert torch.allclose(
        hybrid["scores"], cpu["scores"], atol=1e-8, rtol=0.0
    )
    assert hybrid["cuda_gram_seconds"] >= 0.0
    assert hybrid["cpu_solve_seconds"] >= 0.0
