import unittest

import torch

from vpsc.world_model.cores import E3FixedPointScanCore, E3LayerState, E3ScanState


class E3FixedPointScanCoreTests(unittest.TestCase):
    def test_affine_prefix_scan_matches_serial_affine_recurrence_and_gradient(self) -> None:
        torch.manual_seed(10100)
        coefficient_scan = (0.85 * torch.rand(2, 17, 3)).requires_grad_(True)
        bias_scan = torch.randn(2, 17, 3, requires_grad=True)
        initial_scan = torch.randn(2, 3, requires_grad=True)
        coefficient_serial = coefficient_scan.detach().clone().requires_grad_(True)
        bias_serial = bias_scan.detach().clone().requires_grad_(True)
        initial_serial = initial_scan.detach().clone().requires_grad_(True)

        scanned = E3FixedPointScanCore._affine_prefix_scan(
            coefficient_scan, bias_scan, initial_scan
        )
        current = initial_serial
        pieces = []
        for index in range(coefficient_serial.shape[1]):
            current = coefficient_serial[:, index] * current + bias_serial[:, index]
            pieces.append(current)
        serial = torch.stack(pieces, dim=1)
        torch.testing.assert_close(scanned, serial, atol=2e-6, rtol=1e-5)

        probe = torch.linspace(-0.5, 0.8, scanned.numel()).reshape_as(scanned)
        (scanned * probe).mean().backward()
        (serial * probe).mean().backward()
        torch.testing.assert_close(
            coefficient_scan.grad, coefficient_serial.grad, atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(bias_scan.grad, bias_serial.grad, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(
            initial_scan.grad, initial_serial.grad, atol=2e-6, rtol=1e-5
        )

    def test_one_token_fixed_point_matches_exact_serial_hard_reset(self) -> None:
        torch.manual_seed(10200)
        serial = E3FixedPointScanCore(4, 6, state_dim=5, execution_mode="serial")
        fixed = E3FixedPointScanCore(
            4,
            6,
            state_dim=5,
            execution_mode="fixed_point",
            fixed_point_iterations=1,
        )
        fixed.load_state_dict(serial.state_dict())
        value = torch.randn(3, 1, 4)
        state = E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=torch.rand(3, 5),
                    inhibitory=torch.rand(3, 5),
                ),
            )
        )
        with torch.no_grad():
            serial_result, serial_trace = serial.forward_dynamics(value, state)
            fixed_result, fixed_trace = fixed.forward_dynamics(value, state)
        torch.testing.assert_close(fixed_result.sequence, serial_result.sequence)
        torch.testing.assert_close(
            fixed_trace.excitatory_spikes,
            serial_trace.excitatory_spikes,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            fixed_trace.inhibitory_spikes,
            serial_trace.inhibitory_spikes,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            fixed_result.state.layers[0].excitatory,
            serial_result.state.layers[0].excitatory,
        )
        torch.testing.assert_close(
            fixed_result.state.layers[0].inhibitory,
            serial_result.state.layers[0].inhibitory,
        )

    def test_fixed_point_forward_is_binary_bounded_and_backward_is_finite(self) -> None:
        torch.manual_seed(10300)
        core = E3FixedPointScanCore(4, 6, state_dim=5, fixed_point_iterations=4)
        value = torch.randn(2, 32, 4, requires_grad=True)
        result, trace = core.forward_dynamics(value)
        for spikes in (trace.excitatory_spikes, trace.inhibitory_spikes):
            self.assertTrue(torch.all((spikes == 0.0) | (spikes == 1.0)))
        for residuals in (
            trace.excitatory_residuals,
            trace.inhibitory_residuals,
        ):
            self.assertTrue(torch.all(residuals >= 0.0))
            self.assertTrue(torch.all(residuals < 1.0))
        result.sequence.square().mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in core.parameters()
            )
        )

    def test_constructor_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "decay bounds"):
            E3FixedPointScanCore(4, 4, min_decay=0.9, max_decay=0.8)
        with self.assertRaisesRegex(ValueError, "fixed_point_iterations"):
            E3FixedPointScanCore(4, 4, fixed_point_iterations=0)
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            E3FixedPointScanCore(4, 4, execution_mode="bad")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
