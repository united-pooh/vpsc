import unittest

import torch

from vpsc.world_model.cores import (
    E3CumulativeScanCore,
    E3InputCodedScanCore,
    E3LayerState,
    E3ScanState,
    state_nbytes,
)


class E3CumulativeScanCoreTests(unittest.TestCase):
    def test_exact_threshold_difference_and_hard_reset_identity(self) -> None:
        core = E3CumulativeScanCore(1, 1, state_dim=1, num_layers=1)
        charge = torch.tensor([[[0.50], [0.50], [0.25], [0.75]]])
        initial = torch.tensor([[0.25]])
        expected_spikes = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]])
        expected_residuals = torch.tensor([[[0.75], [0.25], [0.50], [0.25]]])

        with torch.no_grad():
            scan_spikes, scan_residuals, scan_final = core._integrate_scan(charge, initial)
            serial_spikes, serial_residuals, serial_final = core._integrate_serial(
                charge, initial
            )

        torch.testing.assert_close(scan_spikes, expected_spikes, atol=0.0, rtol=0.0)
        torch.testing.assert_close(scan_residuals, expected_residuals, atol=0.0, rtol=0.0)
        torch.testing.assert_close(scan_final, expected_residuals[:, -1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(serial_spikes, scan_spikes, atol=0.0, rtol=0.0)
        torch.testing.assert_close(serial_residuals, scan_residuals, atol=0.0, rtol=0.0)
        torch.testing.assert_close(serial_final, scan_final, atol=0.0, rtol=0.0)

    def test_scan_matches_serial_outputs_traces_states_and_gradients(self) -> None:
        cases = ((1, 1, 1), (4, 32, 2))
        for case_index, (batch, time_steps, layers) in enumerate(cases):
            with self.subTest(batch=batch, time=time_steps, layers=layers):
                torch.manual_seed(9100 + case_index)
                serial = E3CumulativeScanCore(
                    4,
                    6,
                    state_dim=5,
                    num_layers=layers,
                    execution_mode="serial",
                )
                scan = E3CumulativeScanCore(
                    4,
                    6,
                    state_dim=5,
                    num_layers=layers,
                    execution_mode="scan",
                )
                scan.load_state_dict(serial.state_dict())
                serial_input = torch.randn(batch, time_steps, 4, requires_grad=True)
                scan_input = serial_input.detach().clone().requires_grad_(True)

                def make_state() -> E3ScanState:
                    return E3ScanState(
                        layers=tuple(
                            E3LayerState(
                                excitatory=(
                                    torch.randint(4096, (batch, 5), dtype=torch.float32) / 4096
                                ).requires_grad_(True),
                                inhibitory=(
                                    torch.randint(4096, (batch, 5), dtype=torch.float32) / 4096
                                ).requires_grad_(True),
                            )
                            for _ in range(layers)
                        )
                    )

                serial_state = make_state()
                scan_state = E3ScanState(
                    layers=tuple(
                        E3LayerState(
                            excitatory=layer.excitatory.detach().clone().requires_grad_(True),
                            inhibitory=layer.inhibitory.detach().clone().requires_grad_(True),
                        )
                        for layer in serial_state.layers
                    )
                )
                serial_result, serial_traces = serial.forward_dynamics(
                    serial_input, serial_state
                )
                scan_result, scan_traces = scan.forward_dynamics(scan_input, scan_state)
                torch.testing.assert_close(
                    scan_result.sequence, serial_result.sequence, atol=2e-6, rtol=1e-5
                )
                for scan_layer, serial_layer in zip(scan_traces, serial_traces):
                    torch.testing.assert_close(
                        scan_layer.excitatory_spikes,
                        serial_layer.excitatory_spikes,
                        atol=0.0,
                        rtol=0.0,
                    )
                    torch.testing.assert_close(
                        scan_layer.inhibitory_spikes,
                        serial_layer.inhibitory_spikes,
                        atol=0.0,
                        rtol=0.0,
                    )
                    torch.testing.assert_close(
                        scan_layer.excitatory_residuals,
                        serial_layer.excitatory_residuals,
                        atol=2e-6,
                        rtol=1e-5,
                    )
                    torch.testing.assert_close(
                        scan_layer.inhibitory_residuals,
                        serial_layer.inhibitory_residuals,
                        atol=2e-6,
                        rtol=1e-5,
                    )
                for scan_layer, serial_layer in zip(
                    scan_result.state.layers, serial_result.state.layers
                ):
                    torch.testing.assert_close(
                        scan_layer.excitatory,
                        serial_layer.excitatory,
                        atol=2e-6,
                        rtol=1e-5,
                    )
                    torch.testing.assert_close(
                        scan_layer.inhibitory,
                        serial_layer.inhibitory,
                        atol=2e-6,
                        rtol=1e-5,
                    )

                probe = torch.linspace(-0.8, 1.2, serial_result.sequence.numel()).reshape_as(
                    serial_result.sequence
                )
                serial_loss = (serial_result.sequence * probe).mean()
                scan_loss = (scan_result.sequence * probe).mean()
                for serial_layer, scan_layer in zip(
                    serial_result.state.layers, scan_result.state.layers
                ):
                    serial_loss = serial_loss + 0.11 * (
                        serial_layer.excitatory.mean() - serial_layer.inhibitory.mean()
                    )
                    scan_loss = scan_loss + 0.11 * (
                        scan_layer.excitatory.mean() - scan_layer.inhibitory.mean()
                    )
                serial_loss.backward()
                scan_loss.backward()
                torch.testing.assert_close(
                    scan_input.grad, serial_input.grad, atol=2e-6, rtol=1e-5
                )
                for scan_layer, serial_layer in zip(scan_state.layers, serial_state.layers):
                    torch.testing.assert_close(
                        scan_layer.excitatory.grad,
                        serial_layer.excitatory.grad,
                        atol=2e-6,
                        rtol=1e-5,
                    )
                    torch.testing.assert_close(
                        scan_layer.inhibitory.grad,
                        serial_layer.inhibitory.grad,
                        atol=2e-6,
                        rtol=1e-5,
                    )
                serial_parameters = dict(serial.named_parameters())
                for name, scan_parameter in scan.named_parameters():
                    self.assertIsNotNone(scan_parameter.grad, name)
                    self.assertIsNotNone(serial_parameters[name].grad, name)
                    torch.testing.assert_close(
                        scan_parameter.grad,
                        serial_parameters[name].grad,
                        atol=2e-6,
                        rtol=1e-5,
                        msg=lambda message, parameter=name: f"{parameter}: {message}",
                    )

    def test_scan_spikes_are_binary_residuals_bounded_and_streaming_exact(self) -> None:
        torch.manual_seed(9200)
        core = E3CumulativeScanCore(4, 6, state_dim=5, num_layers=2).eval()
        sequence = torch.randn(1, 512, 4)
        with torch.no_grad():
            full, traces = core.forward_dynamics(sequence)
            state = None
            pieces = []
            for index in range(sequence.shape[1]):
                stepped = core.step(sequence[:, index], state)
                pieces.append(stepped.sequence)
                state = stepped.state

        for trace in traces:
            for spikes in (trace.excitatory_spikes, trace.inhibitory_spikes):
                self.assertTrue(torch.all((spikes == 0.0) | (spikes == 1.0)))
            for residuals in (
                trace.excitatory_residuals,
                trace.inhibitory_residuals,
            ):
                self.assertTrue(torch.all(residuals >= 0.0))
                self.assertTrue(torch.all(residuals < 1.0))
        torch.testing.assert_close(
            torch.cat(pieces, dim=1), full.sequence, atol=2e-6, rtol=1e-5
        )
        for streamed_layer, full_layer in zip(state.layers, full.state.layers):
            torch.testing.assert_close(
                streamed_layer.excitatory, full_layer.excitatory, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                streamed_layer.inhibitory, full_layer.inhibitory, atol=0.0, rtol=0.0
            )

    def test_state_size_detach_and_constructor_validation(self) -> None:
        core = E3CumulativeScanCore(4, 6, state_dim=5, num_layers=2)
        state = core.initial_state(3)
        self.assertEqual(state_nbytes(state), 2 * 2 * 3 * 5 * 4)
        result = core(torch.randn(3, 4, 4, requires_grad=True), detach_state=True)
        self.assertTrue(result.sequence.requires_grad)
        self.assertTrue(
            all(
                not value.requires_grad
                for layer in result.state.layers
                for value in (layer.excitatory, layer.inhibitory)
            )
        )
        with self.assertRaisesRegex(ValueError, "max_charge"):
            E3CumulativeScanCore(4, 4, max_charge=1.0)
        with self.assertRaisesRegex(ValueError, "drive_levels"):
            E3CumulativeScanCore(4, 4, drive_levels=1000)
        with self.assertRaisesRegex(ValueError, "charge_levels"):
            E3CumulativeScanCore(4, 4, charge_levels=1000)
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            E3CumulativeScanCore(4, 4, execution_mode="unknown")  # type: ignore[arg-type]

    def test_input_coded_scan_is_binary_exact_and_gradient_equivalent(self) -> None:
        torch.manual_seed(9250)
        serial = E3InputCodedScanCore(
            4, 6, state_dim=5, execution_mode="serial"
        )
        scan = E3InputCodedScanCore(4, 6, state_dim=5, execution_mode="scan")
        scan.load_state_dict(serial.state_dict())
        serial_input = torch.randn(2, 32, 4, requires_grad=True)
        scan_input = serial_input.detach().clone().requires_grad_(True)
        serial_result, serial_traces = serial.forward_dynamics(serial_input)
        scan_result, scan_traces = scan.forward_dynamics(scan_input)
        serial_events = serial.input_events(serial_input)
        for events in serial_events:
            self.assertTrue(torch.all((events == 0.0) | (events == 1.0)))
        torch.testing.assert_close(
            scan_result.sequence, serial_result.sequence, atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            scan_traces[0].excitatory_spikes,
            serial_traces[0].excitatory_spikes,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            scan_traces[0].inhibitory_spikes,
            serial_traces[0].inhibitory_spikes,
            atol=0.0,
            rtol=0.0,
        )
        probe = torch.linspace(-0.4, 0.6, scan_result.sequence.numel()).reshape_as(
            scan_result.sequence
        )
        (scan_result.sequence * probe).mean().backward()
        (serial_result.sequence * probe).mean().backward()
        torch.testing.assert_close(
            scan_input.grad, serial_input.grad, atol=2e-6, rtol=1e-5
        )
        serial_parameters = dict(serial.named_parameters())
        for name, parameter in scan.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                serial_parameters[name].grad,
                atol=2e-6,
                rtol=1e-5,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )

        with self.assertRaisesRegex(ValueError, "below threshold"):
            E3InputCodedScanCore(4, 4, base_charge=0.25, event_charge=0.75)

    def test_terminal_eligibility_matches_scan_forward_state_and_gradients(self) -> None:
        cases = ((1, 1, True), (2, 32, True), (1, 512, False))
        for case_index, (batch, time_steps, input_gradient) in enumerate(cases):
            with self.subTest(
                batch=batch,
                time=time_steps,
                input_gradient=input_gradient,
            ):
                torch.manual_seed(9300 + case_index)
                reference = E3InputCodedScanCore(4, 6, state_dim=5)
                eligibility = E3InputCodedScanCore(4, 6, state_dim=5)
                eligibility.load_state_dict(reference.state_dict())
                reference_input = torch.randn(
                    batch, time_steps, 4, requires_grad=input_gradient
                )
                eligibility_input = reference_input.detach().clone().requires_grad_(
                    input_gradient
                )
                initial_e = torch.rand(batch, 5, requires_grad=True)
                initial_i = torch.rand(batch, 5, requires_grad=True)
                reference_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=initial_e,
                            inhibitory=initial_i,
                        ),
                    )
                )
                eligibility_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=initial_e.detach().clone().requires_grad_(True),
                            inhibitory=initial_i.detach().clone().requires_grad_(True),
                        ),
                    )
                )

                reference_result = reference(reference_input, reference_state)
                eligibility_result = eligibility.forward_terminal_eligibility(
                    eligibility_input, eligibility_state
                )
                reference_terminal = reference_result.sequence[:, -1:]
                torch.testing.assert_close(
                    eligibility_result.sequence,
                    reference_terminal,
                    atol=2e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    eligibility_result.state.layers[0].excitatory,
                    reference_result.state.layers[0].excitatory,
                    atol=0.0,
                    rtol=0.0,
                )
                torch.testing.assert_close(
                    eligibility_result.state.layers[0].inhibitory,
                    reference_result.state.layers[0].inhibitory,
                    atol=0.0,
                    rtol=0.0,
                )

                probe = torch.linspace(
                    -0.7, 0.9, reference_terminal.numel()
                ).reshape_as(reference_terminal)
                reference_loss = (reference_terminal * probe).mean() + 0.13 * (
                    reference_result.state.layers[0].excitatory.mean()
                    - reference_result.state.layers[0].inhibitory.mean()
                )
                eligibility_loss = (
                    eligibility_result.sequence * probe
                ).mean() + 0.13 * (
                    eligibility_result.state.layers[0].excitatory.mean()
                    - eligibility_result.state.layers[0].inhibitory.mean()
                )
                reference_loss.backward()
                eligibility_loss.backward()

                if input_gradient:
                    torch.testing.assert_close(
                        eligibility_input.grad,
                        reference_input.grad,
                        atol=2e-6,
                        rtol=1e-5,
                    )
                torch.testing.assert_close(
                    eligibility_state.layers[0].excitatory.grad,
                    reference_state.layers[0].excitatory.grad,
                    atol=2e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    eligibility_state.layers[0].inhibitory.grad,
                    reference_state.layers[0].inhibitory.grad,
                    atol=2e-6,
                    rtol=1e-5,
                )
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in eligibility.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertIsNotNone(reference_parameters[name].grad, name)
                    torch.testing.assert_close(
                        parameter.grad,
                        reference_parameters[name].grad,
                        atol=2e-6,
                        rtol=1e-5,
                        msg=lambda message, parameter_name=name: (
                            f"{parameter_name}: {message}"
                        ),
                    )

    def test_multi_query_eligibility_matches_scan_and_gradients(self) -> None:
        cases = (
            (1, 1, (0,), True),
            (2, 32, (0, 7, 18, 31), True),
            (1, 512, (0, 1, 63, 127, 255, 383, 510, 511), False),
        )
        for case_index, (batch, time_steps, queries, input_gradient) in enumerate(
            cases
        ):
            with self.subTest(
                batch=batch,
                time=time_steps,
                queries=queries,
                input_gradient=input_gradient,
            ):
                torch.manual_seed(9500 + case_index)
                reference = E3InputCodedScanCore(4, 6, state_dim=5)
                eligibility = E3InputCodedScanCore(4, 6, state_dim=5)
                eligibility.load_state_dict(reference.state_dict())
                reference_input = torch.randn(
                    batch, time_steps, 4, requires_grad=input_gradient
                )
                eligibility_input = reference_input.detach().clone().requires_grad_(
                    input_gradient
                )
                initial_e = torch.rand(batch, 5, requires_grad=True)
                initial_i = torch.rand(batch, 5, requires_grad=True)
                reference_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=initial_e,
                            inhibitory=initial_i,
                        ),
                    )
                )
                eligibility_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=initial_e.detach().clone().requires_grad_(True),
                            inhibitory=initial_i.detach().clone().requires_grad_(True),
                        ),
                    )
                )
                query_indices = torch.tensor(queries, dtype=torch.long)

                reference_result = reference(reference_input, reference_state)
                eligibility_result = eligibility.forward_multi_query_eligibility(
                    eligibility_input,
                    query_indices,
                    eligibility_state,
                )
                reference_queries = reference_result.sequence.index_select(
                    1, query_indices
                )
                torch.testing.assert_close(
                    eligibility_result.sequence,
                    reference_queries,
                    atol=2e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    eligibility_result.state.layers[0].excitatory,
                    reference_result.state.layers[0].excitatory,
                    atol=0.0,
                    rtol=0.0,
                )
                torch.testing.assert_close(
                    eligibility_result.state.layers[0].inhibitory,
                    reference_result.state.layers[0].inhibitory,
                    atol=0.0,
                    rtol=0.0,
                )

                probe = torch.linspace(
                    -0.8, 0.7, reference_queries.numel()
                ).reshape_as(reference_queries)
                reference_loss = (reference_queries * probe).mean() + 0.17 * (
                    reference_result.state.layers[0].excitatory.mean()
                    - reference_result.state.layers[0].inhibitory.mean()
                )
                eligibility_loss = (
                    eligibility_result.sequence * probe
                ).mean() + 0.17 * (
                    eligibility_result.state.layers[0].excitatory.mean()
                    - eligibility_result.state.layers[0].inhibitory.mean()
                )
                reference_loss.backward()
                eligibility_loss.backward()

                if input_gradient:
                    torch.testing.assert_close(
                        eligibility_input.grad,
                        reference_input.grad,
                        atol=2e-6,
                        rtol=1e-5,
                    )
                torch.testing.assert_close(
                    eligibility_state.layers[0].excitatory.grad,
                    reference_state.layers[0].excitatory.grad,
                    atol=2e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    eligibility_state.layers[0].inhibitory.grad,
                    reference_state.layers[0].inhibitory.grad,
                    atol=2e-6,
                    rtol=1e-5,
                )
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in eligibility.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertIsNotNone(reference_parameters[name].grad, name)
                    torch.testing.assert_close(
                        parameter.grad,
                        reference_parameters[name].grad,
                        atol=2e-6,
                        rtol=1e-5,
                        msg=lambda message, parameter_name=name: (
                            f"{parameter_name}: {message}"
                        ),
                    )

    def test_multi_query_eligibility_rejects_invalid_indices(self) -> None:
        core = E3InputCodedScanCore(4, 6, state_dim=5)
        sequence = torch.randn(2, 8, 4)
        invalid = (
            ([], TypeError, "torch.Tensor"),
            (torch.tensor([[0, 1]]), ValueError, "one-dimensional"),
            (torch.tensor([0, 1], dtype=torch.int32), ValueError, "torch.long"),
            (torch.empty(0, dtype=torch.long), ValueError, "non-empty"),
            (torch.tensor([-1, 2]), ValueError, "lie in"),
            (torch.tensor([0, 8]), ValueError, "lie in"),
            (torch.tensor([1, 1]), ValueError, "strictly increasing"),
            (torch.tensor([3, 2]), ValueError, "strictly increasing"),
        )
        for indices, error_type, pattern in invalid:
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(error_type, pattern):
                    core.forward_multi_query_eligibility(sequence, indices)  # type: ignore[arg-type]

    def test_input_coded_tensor_step_matches_generic_streaming(self) -> None:
        torch.manual_seed(9400)
        core = E3InputCodedScanCore(4, 6, state_dim=5).eval()
        tokens = torch.randn(4, 64, 4)
        initial_e = torch.rand(4, 5)
        initial_i = torch.rand(4, 5)
        generic_state = E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=initial_e.clone(),
                    inhibitory=initial_i.clone(),
                ),
            )
        )
        tensor_e = initial_e.clone()
        tensor_i = initial_i.clone()
        with torch.inference_mode():
            for index in range(tokens.shape[1]):
                generic = core.step(tokens[:, index], generic_state)
                output, tensor_e, tensor_i, spike_e, spike_i = (
                    core.forward_step_tensors(tokens[:, index], tensor_e, tensor_i)
                )
                torch.testing.assert_close(
                    output.unsqueeze(1), generic.sequence, atol=2e-6, rtol=1e-5
                )
                torch.testing.assert_close(
                    tensor_e,
                    generic.state.layers[0].excitatory,
                    atol=0.0,
                    rtol=0.0,
                )
                torch.testing.assert_close(
                    tensor_i,
                    generic.state.layers[0].inhibitory,
                    atol=0.0,
                    rtol=0.0,
                )
                self.assertTrue(torch.all((spike_e == 0.0) | (spike_e == 1.0)))
                self.assertTrue(torch.all((spike_i == 0.0) | (spike_i == 1.0)))
                self.assertTrue(torch.all((tensor_e >= 0.0) & (tensor_e < 1.0)))
                self.assertTrue(torch.all((tensor_i >= 0.0) & (tensor_i < 1.0)))
                generic_state = generic.state


if __name__ == "__main__":
    unittest.main()
