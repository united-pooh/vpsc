import unittest

import torch

from vpsc.world_model.fe2h_low_rank import LowRankLinear
from vpsc.world_model.fe2h_tile_sparse import (
    FE2HFiniteGuardError,
    FE2HNeuronTileCore,
    FE2HRoute,
    FE2HUnsupportedError,
)


class FE2HNeuronTileCoreTests(unittest.TestCase):
    def _build_core(self, seed: int = 0, **kwargs: object) -> FE2HNeuronTileCore:
        torch.manual_seed(30000 + seed)
        return FE2HNeuronTileCore(64, 128, **kwargs)

    def _build_inputs(self, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(31000 + seed)
        return torch.randn(2, 64, 64)

    def _assert_any_nonzero_grad(self, named_parameters) -> None:
        grads = []
        for _, parameter in named_parameters:
            if parameter.grad is not None:
                grads.append(parameter.grad.abs().sum().item())
        self.assertTrue(grads, "expected at least one populated gradient")
        self.assertGreater(max(grads), 0.0)

    def test_main_config_validation_rejects_single_or_all_tile_activation(self) -> None:
        core = self._build_core(seed=1)
        self.assertEqual(core.state_dim, 128)
        self.assertEqual(core.tile_size, 32)
        self.assertEqual(core.tile_count, 4)
        self.assertEqual(core.active_tiles, 2)
        for active_tiles in (1, 4):
            with self.subTest(active_tiles=active_tiles):
                with self.assertRaisesRegex(
                    ValueError,
                    "active_tiles must be >= 2 and < tile_count",
                ):
                    self._build_core(seed=2, active_tiles=active_tiles)

    def test_routes_use_exactly_two_of_four_tiles_and_st_grad_reaches_router(self) -> None:
        core = self._build_core(seed=3)
        core.train()
        tokens = self._build_inputs(seed=3)

        result, diagnostics = core.forward_dynamics(tokens)
        self.assertTrue(diagnostics.route.hard_mask.sum(dim=-1).eq(2.0).all().item())
        self.assertEqual(tuple(diagnostics.route.hard_mask.shape), (2, 2, 4))
        self.assertTrue(
            diagnostics.route.selected_indices.ge(0).all().item()
            and diagnostics.route.selected_indices.lt(4).all().item()
        )

        loss = result.sequence.square().mean()
        loss.backward()
        self._assert_any_nonzero_grad(core.router_projection.named_parameters())
        self.assertTrue(torch.isfinite(result.sequence).all().item())
        self.assertTrue(torch.isfinite(diagnostics.stable_local_log_energy).all().item())
        self.assertTrue(torch.isfinite(diagnostics.homeostasis.loss).item())
        self.assertTrue(torch.isfinite(diagnostics.route_supervision_loss).item())
        self.assertGreaterEqual(
            diagnostics.memory_upper_bound.total_bytes,
            diagnostics.memory_upper_bound.forward_only_bytes,
        )
        self.assertGreaterEqual(
            diagnostics.memory_upper_bound.total_gib,
            diagnostics.memory_upper_bound.forward_only_gib,
        )

    def test_homeostasis_depends_only_on_soft_probabilities_and_is_differentiable(self) -> None:
        core = self._build_core(seed=4)
        soft_probs = torch.tensor(
            [
                [
                    [0.70, 0.20, 0.05, 0.05],
                    [0.10, 0.30, 0.30, 0.30],
                ]
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        route_a = FE2HRoute(
            scores=torch.zeros_like(soft_probs),
            soft_probs=soft_probs,
            hard_mask=torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]),
            selected_indices=torch.tensor([[[0, 1], [1, 2]]]),
            active_tiles=2,
            tile_count=4,
            block_size=32,
        )
        route_b = FE2HRoute(
            scores=torch.ones_like(soft_probs),
            soft_probs=soft_probs,
            hard_mask=torch.tensor([[[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]]),
            selected_indices=torch.tensor([[[2, 3], [0, 3]]]),
            active_tiles=2,
            tile_count=4,
            block_size=32,
        )

        homeostasis_a = core._homeostasis(route_a)
        homeostasis_b = core._homeostasis(route_b)
        for name in (
            "activation_rate",
            "entropy",
            "gini",
            "p99_tile_load",
            "block_fill",
            "hotspot_share",
            "dead_tile_ratio",
        ):
            with self.subTest(metric=name):
                value = getattr(homeostasis_a, name)
                self.assertTrue(torch.isfinite(value).all().item(), name)
                torch.testing.assert_close(
                    value,
                    getattr(homeostasis_b, name),
                    atol=0.0,
                    rtol=0.0,
                )

        total = (
            homeostasis_a.loss
            + homeostasis_a.entropy
            + homeostasis_a.gini
            + homeostasis_a.p99_tile_load
            + homeostasis_a.block_fill
            + homeostasis_a.hotspot_share
            + homeostasis_a.dead_tile_ratio
        )
        total.backward()
        self.assertIsNotNone(soft_probs.grad)
        self.assertTrue(torch.isfinite(soft_probs.grad).all().item())
        self.assertGreater(soft_probs.grad.abs().sum().item(), 0.0)

    def test_zero_write_trace_matches_stepwise_lazy_decay(self) -> None:
        core = self._build_core(seed=5)
        torch.manual_seed(32005)
        initial = torch.rand(3, 32)
        decay = torch.linspace(0.60, 0.95, 32)

        trace = core._zero_write_trace(initial, decay, 13)
        manual_states = []
        running = initial.clone()
        for _ in range(13):
            running = running * decay
            manual_states.append(running.clone())
        manual = torch.stack(manual_states, dim=1)
        torch.testing.assert_close(trace, manual, atol=1e-5, rtol=1e-5)

    def test_dense_mask_and_true_sparse_match_with_frozen_route(self) -> None:
        core = self._build_core(seed=6).eval()
        tokens = self._build_inputs(seed=6)
        route = core.route_blocks(tokens)

        with torch.inference_mode():
            dense_result, _ = core.forward_dynamics(
                tokens,
                sparse_inference=False,
                route_override=route,
            )
            sparse_result, _ = core.forward_dynamics(
                tokens,
                sparse_inference=True,
                route_override=route,
            )

        torch.testing.assert_close(
            sparse_result.sequence,
            dense_result.sequence,
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(sparse_result.sequence.dtype, torch.float32)
        for tile_index, (dense_state, sparse_state) in enumerate(
            zip(dense_result.state, sparse_result.state)
        ):
            with self.subTest(tile=tile_index, tensor="excitatory"):
                torch.testing.assert_close(
                    sparse_state.layers[0].excitatory,
                    dense_state.layers[0].excitatory,
                    atol=1e-5,
                    rtol=1e-5,
                )
                self.assertEqual(sparse_state.layers[0].excitatory.dtype, torch.float32)
            with self.subTest(tile=tile_index, tensor="inhibitory"):
                torch.testing.assert_close(
                    sparse_state.layers[0].inhibitory,
                    dense_state.layers[0].inhibitory,
                    atol=1e-5,
                    rtol=1e-5,
                )
                self.assertEqual(sparse_state.layers[0].inhibitory.dtype, torch.float32)

    def test_sparse_fail_closed_modes_and_low_rank_dense_path_support(self) -> None:
        tokens = self._build_inputs(seed=7)

        dense_low_rank_core = FE2HNeuronTileCore(
            64,
            128,
            input_event_projection=LowRankLinear(64, 512, 16),
            output_projection=LowRankLinear(512, 128, 16),
        )
        dense_low_rank_core.train()
        dense_result, dense_diagnostics = dense_low_rank_core.forward_dynamics(tokens)
        self.assertTrue(torch.isfinite(dense_result.sequence).all().item())
        self.assertTrue(torch.isfinite(dense_diagnostics.homeostasis.loss).item())

        low_rank_route = dense_low_rank_core.route_blocks(tokens)
        dense_low_rank_core.eval()
        with torch.inference_mode():
            with self.assertRaisesRegex(
                FE2HUnsupportedError,
                "sliceable nn.Linear input projection",
            ):
                dense_low_rank_core.forward_dynamics(
                    tokens,
                    sparse_inference=True,
                    route_override=low_rank_route,
                )

        core = self._build_core(seed=8)
        route = core.route_blocks(tokens)
        with self.assertRaisesRegex(FE2HUnsupportedError, "eval-only"):
            core.forward_dynamics(
                tokens,
                sparse_inference=True,
                route_override=route,
            )

        core.eval()
        with self.assertRaisesRegex(
            FE2HUnsupportedError,
            "requires gradients to be disabled",
        ):
            core.forward_dynamics(
                tokens,
                sparse_inference=True,
                route_override=route,
            )

        with torch.inference_mode():
            with self.assertRaisesRegex(
                FE2HUnsupportedError,
                "freeze the route first with route_override",
            ):
                core.forward_dynamics(tokens, sparse_inference=True)

    def test_route_supervision_uses_detached_energy_target_and_default_epsilon(self) -> None:
        core = self._build_core(seed=9)
        core.train()
        tokens = self._build_inputs(seed=9)
        _, diagnostics = core.forward_dynamics(tokens)

        self.assertEqual(diagnostics.energy_epsilon, 1e-8)
        self.assertIsNotNone(diagnostics.route_supervision_loss)

        core.zero_grad(set_to_none=True)
        diagnostics.route_supervision_loss.backward()
        self._assert_any_nonzero_grad(core.router_projection.named_parameters())

        self.assertIsNone(core.decay_logits.grad)
        self.assertIsNone(core.input_event_projection.weight.grad)
        if core.input_event_projection.bias is not None:
            self.assertIsNone(core.input_event_projection.bias.grad)
        self.assertIsNone(core.output_projection.weight.grad)
        if core.output_projection.bias is not None:
            self.assertIsNone(core.output_projection.bias.grad)

    def test_finite_guard_reports_first_non_finite_loss_gradient_parameter_and_optimizer_state(self) -> None:
        cases = (
            ("loss", self._expect_loss_guard_metadata),
            ("gradient", self._expect_gradient_guard_metadata),
            ("parameter", self._expect_parameter_guard_metadata),
            ("optimizer_state", self._expect_optimizer_state_guard_metadata),
        )
        for scope, check in cases:
            with self.subTest(scope=scope):
                check()

    def _expect_loss_guard_metadata(self) -> None:
        core = self._build_core(seed=10)
        with self.assertRaises(FE2HFiniteGuardError) as caught:
            core.finite_guard(
                loss_terms={"main": torch.tensor([1.0, float("nan")])},
                step=11,
            )
        self.assertEqual(caught.exception.scope, "loss")
        self.assertEqual(caught.exception.name, "main")
        self.assertEqual(caught.exception.step, 11)
        self.assertEqual(caught.exception.index, (1,))

    def _expect_gradient_guard_metadata(self) -> None:
        core = self._build_core(seed=11)
        tokens = self._build_inputs(seed=11)
        result = core(tokens)
        result.sequence.square().mean().backward()
        for parameter in core.parameters():
            if parameter.grad is not None:
                parameter.grad.view(-1)[0] = float("inf")
                break
        with self.assertRaises(FE2HFiniteGuardError) as caught:
            core.finite_guard(step=12)
        self.assertEqual(caught.exception.scope, "gradient")
        self.assertEqual(caught.exception.step, 12)
        self.assertEqual(caught.exception.index, (0,))

    def _expect_parameter_guard_metadata(self) -> None:
        core = self._build_core(seed=12)
        parameter = next(core.parameters())
        with torch.no_grad():
            parameter.view(-1)[0] = float("nan")
        with self.assertRaises(FE2HFiniteGuardError) as caught:
            core.finite_guard(step=13)
        self.assertEqual(caught.exception.scope, "parameter")
        self.assertEqual(caught.exception.step, 13)
        self.assertEqual(caught.exception.index, (0,))

    def _expect_optimizer_state_guard_metadata(self) -> None:
        core = self._build_core(seed=13)
        optimizer = torch.optim.Adam(core.parameters(), lr=1e-3)
        tokens = self._build_inputs(seed=13)
        result = core(tokens)
        result.sequence.square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        first_parameter = next(iter(optimizer.state))
        optimizer.state[first_parameter]["exp_avg"].view(-1)[0] = float("inf")
        with self.assertRaises(FE2HFiniteGuardError) as caught:
            core.finite_guard(optimizer=optimizer, step=14)
        self.assertEqual(caught.exception.scope, "optimizer_state")
        self.assertEqual(caught.exception.step, 14)
        self.assertEqual(caught.exception.index, (0,))


if __name__ == "__main__":
    unittest.main()
