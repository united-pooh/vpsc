import unittest

import torch

from experiments.e3_at1_trace_eligibility import _equivalence_case
from experiments.e3_ldaa1_operator_crossover import (
    analyse_matrix,
    frozen_dispatch_backend,
    query_indices_for_density,
)
from vpsc.world_model.cores import E3GatedTraceScanCore


class LossDensityAdaptiveAdjointTests(unittest.TestCase):
    def test_segmented_adjoint_matches_bptt_with_and_without_input_gradient(self) -> None:
        for input_gradient in (False, True):
            with self.subTest(input_gradient=input_gradient):
                result = _equivalence_case(
                    batch=2,
                    time_steps=96,
                    queries=(0, 7, 31, 63, 95),
                    input_gradient=input_gradient,
                    device=torch.device("cpu"),
                    seed=20260722,
                    eligibility_backward_mode="segmented_adjoint",
                )
                self.assertTrue(result["passed"])

    def test_segmented_adjoint_supports_query_at_last_and_before_last(self) -> None:
        for queries in ((0,), (0, 47), (3, 11, 46)):
            with self.subTest(queries=queries):
                result = _equivalence_case(
                    batch=1,
                    time_steps=48,
                    queries=queries,
                    input_gradient=True,
                    device=torch.device("cpu"),
                    seed=20260723,
                    eligibility_backward_mode="segmented_adjoint",
                )
                self.assertTrue(result["passed"])

    def test_query_density_rounding_and_frozen_dispatch(self) -> None:
        sparse = query_indices_for_density(
            512, 1 / 1024, device=torch.device("cpu")
        )
        dense = query_indices_for_density(512, 1.0, device=torch.device("cpu"))

        self.assertEqual(sparse.tolist(), [0])
        self.assertEqual(dense.numel(), 512)
        self.assertEqual(frozen_dispatch_backend(1 / 64), "segmented_adjoint")
        self.assertEqual(frozen_dispatch_backend(1 / 16), "reverse_adjoint")

    def test_cuda_fused_rejects_segmented_backend(self) -> None:
        core = E3GatedTraceScanCore(
            4,
            4,
            state_dim=4,
            scan_math_mode="cuda_fused",
            eligibility_backward_mode="segmented_adjoint",
        )
        with self.assertRaisesRegex(ValueError, "cuda_fused requires"):
            core.forward_multi_query_eligibility(
                torch.randn(1, 8, 4), torch.tensor([0, 7])
            )

    def test_analysis_requires_exactness_and_sparse_speed_memory(self) -> None:
        def cell(input_gradient: bool):
            backend = {
                "exactness": {"passed": True},
                "speedup_vs_bptt": 2.0,
                "storage_ratio_to_bptt": 0.20,
                "p50_ms": 1.0,
            }
            return {
                "actual_density": 1 / 64,
                "input_gradient": input_gradient,
                "oracle_backend": "segmented_adjoint",
                "frozen_dispatch_regret": 1.0,
                "backends": {
                    "bptt": {**backend, "p50_ms": 2.0},
                    "forward_eligibility": backend.copy(),
                    "reverse_adjoint": {**backend, "p50_ms": 1.5},
                    "segmented_adjoint": backend.copy(),
                },
            }

        analysis = analyse_matrix([cell(False), cell(True)])

        self.assertEqual(analysis["verdict"], "OPERATOR_GO_MODEL_VALIDATION_REQUIRED")
        self.assertTrue(analysis["gates"]["H1_exactness"])
        self.assertTrue(analysis["gates"]["H2_input_gradient_on"])


if __name__ == "__main__":
    unittest.main()
