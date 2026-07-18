import unittest

import torch

from vpsc.world_model.cores import (
    E3OscillatorState,
    E3OscillatoryScanCore,
    state_nbytes,
)


class E3OscillatoryScanCoreTests(unittest.TestCase):
    def test_scan_matches_serial_outputs_spikes_state_and_gradients(self) -> None:
        torch.manual_seed(12100)
        serial = E3OscillatoryScanCore(
            4, 6, state_dim=5, execution_mode="serial"
        )
        scan = E3OscillatoryScanCore(4, 6, state_dim=5, execution_mode="scan")
        scan.load_state_dict(serial.state_dict())
        serial_input = torch.randn(2, 33, 4, requires_grad=True)
        scan_input = serial_input.detach().clone().requires_grad_(True)
        initial_value = torch.complex(torch.randn(2, 5), torch.randn(2, 5)) * 0.1
        serial_state = E3OscillatorState(value=initial_value.detach().clone().requires_grad_(True))
        scan_state = E3OscillatorState(value=initial_value.detach().clone().requires_grad_(True))

        serial_result, serial_trace = serial.forward_dynamics(serial_input, serial_state)
        scan_result, scan_trace = scan.forward_dynamics(scan_input, scan_state)
        torch.testing.assert_close(
            scan_result.sequence, serial_result.sequence, atol=3e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            scan_trace.excitatory_spikes,
            serial_trace.excitatory_spikes,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            scan_trace.inhibitory_spikes,
            serial_trace.inhibitory_spikes,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            scan_result.state.value,
            serial_result.state.value,
            atol=3e-5,
            rtol=1e-4,
        )

        probe = torch.linspace(-0.7, 0.9, scan_result.sequence.numel()).reshape_as(
            scan_result.sequence
        )
        (scan_result.sequence * probe).mean().backward()
        (serial_result.sequence * probe).mean().backward()
        torch.testing.assert_close(
            scan_input.grad, serial_input.grad, atol=3e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            scan_state.value.grad, serial_state.value.grad, atol=3e-5, rtol=1e-4
        )
        serial_parameters = dict(serial.named_parameters())
        for name, parameter in scan.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                serial_parameters[name].grad,
                atol=3e-5,
                rtol=1e-4,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )

    def test_full_scan_matches_streaming_and_emits_binary_events(self) -> None:
        torch.manual_seed(12200)
        core = E3OscillatoryScanCore(4, 6, state_dim=5).eval()
        value = torch.randn(1, 512, 4)
        with torch.no_grad():
            full, trace = core.forward_dynamics(value)
            state = None
            pieces = []
            for index in range(value.shape[1]):
                stepped = core.step(value[:, index], state)
                pieces.append(stepped.sequence)
                state = stepped.state
        self.assertTrue(
            torch.all(
                (trace.excitatory_spikes == 0.0) | (trace.excitatory_spikes == 1.0)
            )
        )
        self.assertTrue(
            torch.all(
                (trace.inhibitory_spikes == 0.0) | (trace.inhibitory_spikes == 1.0)
            )
        )
        torch.testing.assert_close(
            torch.cat(pieces, dim=1), full.sequence, atol=3e-5, rtol=1e-4
        )
        torch.testing.assert_close(state.value, full.state.value, atol=3e-5, rtol=1e-4)

    def test_radius_bounds_state_bytes_and_detach(self) -> None:
        core = E3OscillatoryScanCore(4, 6, state_dim=5)
        value = torch.randn(3, 7, 4, requires_grad=True)
        coefficient, _ = core._coefficient_and_drive(value)
        self.assertTrue(torch.all(coefficient.abs() >= core.min_radius))
        self.assertTrue(torch.all(coefficient.abs() <= core.max_radius))
        state = core.initial_state(3)
        self.assertEqual(state_nbytes(state), 3 * 5 * 8)
        result = core(value, detach_state=True)
        self.assertTrue(result.sequence.requires_grad)
        self.assertFalse(result.state.value.requires_grad)

    def test_constructor_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "radius bounds"):
            E3OscillatoryScanCore(4, 4, min_radius=0.9, max_radius=0.8)
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            E3OscillatoryScanCore(4, 4, execution_mode="bad")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
