import unittest

import torch
import torch.nn as nn

from vpsc.world_model.fe2h_low_rank import (
    LowRankLinear,
    ProjectionConfig,
    build_projection,
    matched_projection_report,
)


class LowRankLinearTests(unittest.TestCase):
    def test_forward_shape_grad_and_random_provenance_for_rank16_and_32(self) -> None:
        cases = (
            (64, 48, 16, 21000),
            (128, 96, 32, 21001),
        )
        for in_features, out_features, rank, seed in cases:
            with self.subTest(rank=rank):
                torch.manual_seed(seed)
                layer = LowRankLinear(in_features, out_features, rank)
                inputs = torch.randn(3, 5, in_features, requires_grad=True)
                outputs = layer(inputs)
                self.assertEqual(tuple(outputs.shape), (3, 5, out_features))
                loss = outputs.square().mean() + 0.05 * outputs.mean()
                loss.backward()
                self.assertIsNotNone(inputs.grad)
                self.assertIsNotNone(layer.left_factor.grad)
                self.assertIsNotNone(layer.right_factor.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all().item())
                self.assertTrue(torch.isfinite(layer.left_factor.grad).all().item())
                self.assertTrue(torch.isfinite(layer.right_factor.grad).all().item())
                provenance = layer.provenance.as_dict()
                self.assertEqual(provenance["init_mode"], "random")
                self.assertIsNone(provenance["source_name"])
                report = layer.cost_report().as_dict()
                self.assertLess(report["low_rank_parameters"], report["dense_parameters"])
                self.assertLess(report["low_rank_macs"], report["dense_macs"])

    def test_from_dense_exact_reconstruction_and_provenance_for_rank16_and_32(self) -> None:
        cases = (
            (64, 48, 16, "input_event_projection", 21010),
            (128, 96, 32, "output_projection", 21011),
        )
        for in_features, out_features, rank, source_name, seed in cases:
            with self.subTest(rank=rank):
                torch.manual_seed(seed)
                dense = nn.Linear(in_features, out_features).double()
                left = torch.randn(out_features, rank, dtype=torch.float64)
                right = torch.randn(rank, in_features, dtype=torch.float64)
                bias = torch.randn(out_features, dtype=torch.float64)
                with torch.no_grad():
                    dense.weight.copy_(left @ right)
                    dense.bias.copy_(bias)
                layer = LowRankLinear.from_dense(
                    dense,
                    rank=rank,
                    source_name=source_name,
                )
                torch.testing.assert_close(
                    layer.equivalent_weight(),
                    dense.weight,
                    atol=1e-5,
                    rtol=1e-5,
                )
                torch.testing.assert_close(layer.bias, dense.bias, atol=0.0, rtol=0.0)
                probe = torch.randn(2, 4, in_features, dtype=torch.float64)
                torch.testing.assert_close(
                    layer(probe),
                    dense(probe),
                    atol=1e-10,
                    rtol=1e-10,
                )
                provenance = layer.provenance.as_dict()
                self.assertEqual(provenance["init_mode"], "svd")
                self.assertEqual(provenance["source_name"], source_name)
                self.assertEqual(provenance["rank"], rank)
                self.assertEqual(
                    provenance["source_shape"],
                    [out_features, in_features],
                )

    def test_from_dense_matches_truncated_svd_approximation(self) -> None:
        torch.manual_seed(21020)
        dense = nn.Linear(96, 80)
        layer = LowRankLinear.from_dense(
            dense,
            rank=16,
            source_name="router_projection",
        )
        u, s, vh = torch.linalg.svd(dense.weight.detach(), full_matrices=False)
        manual = (u[:, :16] * s[:16].unsqueeze(0)) @ vh[:16, :]
        torch.testing.assert_close(
            layer.equivalent_weight(),
            manual,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_build_projection_switches_dense_and_low_rank(self) -> None:
        dense_projection = build_projection(
            32,
            64,
            config=ProjectionConfig(kind="dense", bias=False),
        )
        self.assertIsInstance(dense_projection, nn.Linear)
        self.assertIsNone(dense_projection.bias)

        torch.manual_seed(21030)
        dense_source = nn.Linear(64, 48)
        low_rank_projection = build_projection(
            64,
            48,
            config=ProjectionConfig(
                kind="low_rank",
                rank=16,
                init="svd",
                source_name="input_event_projection",
            ),
            dense_source=dense_source,
        )
        self.assertIsInstance(low_rank_projection, LowRankLinear)
        self.assertEqual(low_rank_projection.provenance.init_mode, "svd")
        self.assertEqual(
            low_rank_projection.provenance.source_name,
            "input_event_projection",
        )

    def test_report_matches_expected_parameter_and_mac_formulas(self) -> None:
        report = matched_projection_report(64, 48, 16)
        report_dict = report.as_dict()
        self.assertEqual(report_dict["dense_parameters"], 64 * 48 + 48)
        self.assertEqual(report_dict["low_rank_parameters"], 16 * 64 + 16 * 48 + 48)
        self.assertEqual(report_dict["dense_macs"], 64 * 48)
        self.assertEqual(report_dict["low_rank_macs"], 16 * (64 + 48))
        self.assertGreater(report.parameter_reduction, 0)
        self.assertGreater(report.mac_reduction, 0)

    def test_invalid_rank_and_fail_closed_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank must be one of"):
            LowRankLinear(64, 48, 8)

        with self.assertRaisesRegex(ValueError, "strictly fewer parameters"):
            LowRankLinear(17, 17, 16, allow_test_rank=True)

        with self.assertRaisesRegex(ValueError, "strictly smaller than min"):
            LowRankLinear(64, 48, 48, allow_test_rank=True)

        with self.assertRaisesRegex(ValueError, "dense_source"):
            build_projection(
                64,
                48,
                config=ProjectionConfig(
                    kind="low_rank",
                    rank=16,
                    init="svd",
                    source_name="router_projection",
                ),
            )

        dense = nn.Linear(64, 48)
        with self.assertRaisesRegex(ValueError, "source_name is required"):
            LowRankLinear.from_dense(dense, rank=16, source_name="")

        with self.assertRaisesRegex(ValueError, "additive_lora is rejected"):
            ProjectionConfig(kind="additive_lora", rank=16)


if __name__ == "__main__":
    unittest.main()
