import unittest

import torch

from vpsc.world_model.cores import (
    E3GatedTraceScanCore,
    E3LayerState,
    E3ScanState,
    count_parameters,
    state_nbytes,
)


class E3GatedTraceScanCoreTests(unittest.TestCase):
    def test_scan_matches_serial_forward_events_state_and_gradients(self) -> None:
        for case_index, (batch, time_steps) in enumerate(((1, 1), (4, 32), (1, 512))):
            with self.subTest(batch=batch, time=time_steps):
                torch.manual_seed(11000 + case_index)
                serial = E3GatedTraceScanCore(
                    4, 6, state_dim=5, execution_mode="serial"
                )
                scan = E3GatedTraceScanCore(
                    4, 6, state_dim=5, execution_mode="scan"
                )
                scan.load_state_dict(serial.state_dict())
                serial_input = torch.randn(batch, time_steps, 4, requires_grad=True)
                scan_input = serial_input.detach().clone().requires_grad_(True)
                initial_e = torch.rand(batch, 5, requires_grad=True)
                initial_i = torch.rand(batch, 5, requires_grad=True)
                serial_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=initial_e,
                            inhibitory=initial_i,
                        ),
                    )
                )
                scan_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=initial_e.detach().clone().requires_grad_(True),
                            inhibitory=initial_i.detach().clone().requires_grad_(True),
                        ),
                    )
                )

                serial_result, serial_trace = serial.forward_dynamics(
                    serial_input, serial_state
                )
                scan_result, scan_trace = scan.forward_dynamics(scan_input, scan_state)
                torch.testing.assert_close(
                    scan_result.sequence,
                    serial_result.sequence,
                    atol=2e-5,
                    rtol=1e-4,
                )
                for name in (
                    "excitatory_content",
                    "inhibitory_content",
                    "excitatory_gate",
                    "inhibitory_gate",
                    "excitatory_writes",
                    "inhibitory_writes",
                    "excitatory_spikes",
                    "inhibitory_spikes",
                ):
                    torch.testing.assert_close(
                        getattr(scan_trace, name),
                        getattr(serial_trace, name),
                        atol=0.0,
                        rtol=0.0,
                    )
                for name in ("excitatory_traces", "inhibitory_traces"):
                    torch.testing.assert_close(
                        getattr(scan_trace, name),
                        getattr(serial_trace, name),
                        atol=2e-5,
                        rtol=1e-4,
                    )
                torch.testing.assert_close(
                    scan_result.state.layers[0].excitatory,
                    serial_result.state.layers[0].excitatory,
                    atol=2e-5,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    scan_result.state.layers[0].inhibitory,
                    serial_result.state.layers[0].inhibitory,
                    atol=2e-5,
                    rtol=1e-4,
                )

                probe = torch.linspace(
                    -0.6, 0.8, serial_result.sequence.numel()
                ).reshape_as(serial_result.sequence)
                serial_loss = (serial_result.sequence * probe).mean() + 0.11 * (
                    serial_result.state.layers[0].excitatory.mean()
                    - serial_result.state.layers[0].inhibitory.mean()
                )
                scan_loss = (scan_result.sequence * probe).mean() + 0.11 * (
                    scan_result.state.layers[0].excitatory.mean()
                    - scan_result.state.layers[0].inhibitory.mean()
                )
                serial_loss.backward()
                scan_loss.backward()
                torch.testing.assert_close(
                    scan_input.grad,
                    serial_input.grad,
                    atol=2e-5,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    scan_state.layers[0].excitatory.grad,
                    serial_state.layers[0].excitatory.grad,
                    atol=2e-5,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    scan_state.layers[0].inhibitory.grad,
                    serial_state.layers[0].inhibitory.grad,
                    atol=2e-5,
                    rtol=1e-4,
                )
                serial_parameters = dict(serial.named_parameters())
                for name, parameter in scan.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        serial_parameters[name].grad,
                        atol=2e-5,
                        rtol=1e-4,
                        msg=lambda message, parameter_name=name: (
                            f"{parameter_name}: {message}"
                        ),
                    )

    def test_multi_query_eligibility_matches_scan_forward_and_gradients(self) -> None:
        cases = (
            (1, 1, (0,), True),
            (2, 32, (3, 11, 19, 31), True),
            (1, 512, (7, 71, 135, 199, 263, 327, 391, 511), False),
        )
        for case_index, (batch, time_steps, positions, input_grad) in enumerate(cases):
            with self.subTest(
                batch=batch,
                time=time_steps,
                queries=len(positions),
                input_grad=input_grad,
            ):
                torch.manual_seed(11050 + case_index)
                reference = E3GatedTraceScanCore(
                    4, 6, state_dim=5, execution_mode="scan"
                )
                eligibility = E3GatedTraceScanCore(
                    4, 6, state_dim=5, execution_mode="scan"
                )
                eligibility.load_state_dict(reference.state_dict())
                reference_input = torch.randn(
                    batch, time_steps, 4, requires_grad=input_grad
                )
                eligibility_input = (
                    reference_input.detach().clone().requires_grad_(input_grad)
                )
                reference_e = torch.rand(batch, 5, requires_grad=True)
                reference_i = torch.rand(batch, 5, requires_grad=True)
                eligibility_e = reference_e.detach().clone().requires_grad_(True)
                eligibility_i = reference_i.detach().clone().requires_grad_(True)
                reference_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=reference_e,
                            inhibitory=reference_i,
                        ),
                    )
                )
                eligibility_state = E3ScanState(
                    layers=(
                        E3LayerState(
                            excitatory=eligibility_e,
                            inhibitory=eligibility_i,
                        ),
                    )
                )
                query_indices = torch.tensor(positions, dtype=torch.long)

                full = reference(reference_input, reference_state)
                sparse = eligibility.forward_multi_query_eligibility(
                    eligibility_input,
                    query_indices,
                    eligibility_state,
                )
                expected_sequence = full.sequence.index_select(1, query_indices)
                torch.testing.assert_close(
                    sparse.sequence, expected_sequence, atol=2e-5, rtol=1e-4
                )
                torch.testing.assert_close(
                    sparse.state.layers[0].excitatory,
                    full.state.layers[0].excitatory,
                    atol=2e-5,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    sparse.state.layers[0].inhibitory,
                    full.state.layers[0].inhibitory,
                    atol=2e-5,
                    rtol=1e-4,
                )

                probe = torch.linspace(-0.7, 0.9, sparse.sequence.numel()).reshape_as(
                    sparse.sequence
                )
                reference_loss = (expected_sequence * probe).mean() + 0.13 * (
                    full.state.layers[0].excitatory.square().mean()
                    - full.state.layers[0].inhibitory.square().mean()
                )
                eligibility_loss = (sparse.sequence * probe).mean() + 0.13 * (
                    sparse.state.layers[0].excitatory.square().mean()
                    - sparse.state.layers[0].inhibitory.square().mean()
                )
                reference_loss.backward()
                eligibility_loss.backward()
                if input_grad:
                    torch.testing.assert_close(
                        eligibility_input.grad,
                        reference_input.grad,
                        atol=2e-5,
                        rtol=1e-4,
                    )
                else:
                    self.assertIsNone(reference_input.grad)
                    self.assertIsNone(eligibility_input.grad)
                torch.testing.assert_close(
                    eligibility_e.grad,
                    reference_e.grad,
                    atol=2e-5,
                    rtol=1e-4,
                )
                torch.testing.assert_close(
                    eligibility_i.grad,
                    reference_i.grad,
                    atol=2e-5,
                    rtol=1e-4,
                )
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in eligibility.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        reference_parameters[name].grad,
                        atol=2e-5,
                        rtol=1e-4,
                        msg=lambda message, parameter_name=name: (
                            f"{parameter_name}: {message}"
                        ),
                    )

    def test_full_scan_matches_continuous_streaming(self) -> None:
        torch.manual_seed(11100)
        core = E3GatedTraceScanCore(4, 6, state_dim=5, execution_mode="scan")
        tokens = torch.randn(3, 64, 4)
        initial_state = E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=torch.rand(3, 5),
                    inhibitory=torch.rand(3, 5),
                ),
            )
        )
        with torch.no_grad():
            full = core(tokens, initial_state)
            state = initial_state
            pieces = []
            for index in range(tokens.shape[1]):
                step = core.step(tokens[:, index], state)
                pieces.append(step.sequence)
                state = step.state
            streamed = torch.cat(pieces, dim=1)
        torch.testing.assert_close(streamed, full.sequence, atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(
            state.layers[0].excitatory,
            full.state.layers[0].excitatory,
            atol=2e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            state.layers[0].inhibitory,
            full.state.layers[0].inhibitory,
            atol=2e-5,
            rtol=1e-4,
        )

    def test_tensor_step_matches_full_scan_events_and_state(self) -> None:
        torch.manual_seed(11150)
        core = E3GatedTraceScanCore(4, 6, state_dim=5, execution_mode="scan").eval()
        tokens = torch.randn(3, 64, 4)
        initial_e = torch.rand(3, 5)
        initial_i = torch.rand(3, 5)
        initial = E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=initial_e.clone(),
                    inhibitory=initial_i.clone(),
                ),
            )
        )
        with torch.inference_mode():
            full, trace = core.forward_dynamics(tokens, initial)
            tensor_e = initial_e
            tensor_i = initial_i
            outputs = []
            spikes_e = []
            spikes_i = []
            writes_e = []
            writes_i = []
            for index in range(tokens.shape[1]):
                step = core.forward_step_tensors(tokens[:, index], tensor_e, tensor_i)
                output, tensor_e, tensor_i, spike_e, spike_i, write_e, write_i = step
                outputs.append(output)
                spikes_e.append(spike_e)
                spikes_i.append(spike_i)
                writes_e.append(write_e)
                writes_i.append(write_i)
        torch.testing.assert_close(
            torch.stack(outputs, dim=1), full.sequence, atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            tensor_e,
            full.state.layers[0].excitatory,
            atol=2e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            tensor_i,
            full.state.layers[0].inhibitory,
            atol=2e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            torch.stack(spikes_e, dim=1), trace.excitatory_spikes, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            torch.stack(spikes_i, dim=1), trace.inhibitory_spikes, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            torch.stack(writes_e, dim=1), trace.excitatory_writes, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            torch.stack(writes_i, dim=1), trace.inhibitory_writes, atol=0.0, rtol=0.0
        )

    def test_cached_decay_tensor_step_matches_uncached_step(self) -> None:
        torch.manual_seed(11175)
        core = E3GatedTraceScanCore(4, 6, state_dim=5).eval()
        token = torch.randn(3, 4)
        initial_e = torch.rand(3, 5)
        initial_i = torch.rand(3, 5)
        with torch.inference_mode():
            decay_e, decay_i = core.decays()
            uncached = core.forward_step_tensors(token, initial_e, initial_i)
            cached = core.forward_step_tensors_cached_decay(
                token, initial_e, initial_i, decay_e, decay_i
            )
        for cached_value, uncached_value in zip(cached, uncached):
            torch.testing.assert_close(cached_value, uncached_value, atol=0.0, rtol=0.0)

    def test_scan_aligned_query_forward_matches_full_scan_bit_exact(self) -> None:
        torch.manual_seed(11185)
        reference = E3GatedTraceScanCore(4, 6, state_dim=5, execution_mode="scan")
        aligned = E3GatedTraceScanCore(
            4,
            6,
            state_dim=5,
            execution_mode="scan",
            eligibility_forward_mode="scan_aligned",
        )
        aligned.load_state_dict(reference.state_dict())
        value = torch.randn(2, 32, 4)
        queries = torch.tensor([0, 7, 18, 31], dtype=torch.long)
        captured = []
        handle = aligned.output_norm.register_forward_pre_hook(
            lambda _module, arguments: captured.append(arguments[0].detach())
        )
        try:
            expected, trace = reference.forward_dynamics(value)
            actual = aligned.forward_multi_query_eligibility(value, queries)
        finally:
            handle.remove()
        expected_raw = torch.cat(
            (
                trace.excitatory_spikes.index_select(1, queries),
                -trace.inhibitory_spikes.index_select(1, queries),
                trace.excitatory_traces.index_select(1, queries),
                -trace.inhibitory_traces.index_select(1, queries),
            ),
            dim=-1,
        )
        torch.testing.assert_close(captured[0], expected_raw, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            actual.sequence,
            expected.sequence.index_select(1, queries),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            actual.state.layers[0].excitatory,
            expected.state.layers[0].excitatory,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            actual.state.layers[0].inhibitory,
            expected.state.layers[0].inhibitory,
            atol=0.0,
            rtol=0.0,
        )

    def test_reverse_adjoint_matches_bptt_with_input_gradient(self) -> None:
        torch.manual_seed(11195)
        reference = E3GatedTraceScanCore(4, 6, state_dim=5)
        reverse = E3GatedTraceScanCore(
            4,
            6,
            state_dim=5,
            eligibility_backward_mode="reverse_adjoint",
        )
        reverse.load_state_dict(reference.state_dict())
        reference_input = torch.randn(2, 64, 4, requires_grad=True)
        reverse_input = reference_input.detach().clone().requires_grad_(True)
        reference_e = torch.rand(2, 5, requires_grad=True)
        reference_i = torch.rand(2, 5, requires_grad=True)
        reverse_e = reference_e.detach().clone().requires_grad_(True)
        reverse_i = reference_i.detach().clone().requires_grad_(True)
        reference_state = E3ScanState(
            layers=(
                E3LayerState(excitatory=reference_e, inhibitory=reference_i),
            )
        )
        reverse_state = E3ScanState(
            layers=(E3LayerState(excitatory=reverse_e, inhibitory=reverse_i),)
        )
        queries = torch.tensor(
            [0, 7, 15, 23, 31, 39, 47, 63], dtype=torch.long
        )
        expected = reference(reference_input, reference_state)
        actual = reverse.forward_multi_query_eligibility(
            reverse_input, queries, reverse_state
        )
        expected_queries = expected.sequence.index_select(1, queries)
        torch.testing.assert_close(
            actual.sequence, expected_queries, atol=2e-5, rtol=1e-4
        )
        probe = torch.linspace(-0.8, 0.6, actual.sequence.numel()).reshape_as(
            actual.sequence
        )
        expected_loss = (expected_queries * probe).mean() + 0.17 * (
            expected.state.layers[0].excitatory.square().mean()
            - expected.state.layers[0].inhibitory.square().mean()
        )
        actual_loss = (actual.sequence * probe).mean() + 0.17 * (
            actual.state.layers[0].excitatory.square().mean()
            - actual.state.layers[0].inhibitory.square().mean()
        )
        expected_loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(
            reverse_input.grad, reference_input.grad, atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            reverse_e.grad, reference_e.grad, atol=2e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            reverse_i.grad, reference_i.grad, atol=2e-5, rtol=1e-4
        )
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in reverse.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference_parameters[name].grad,
                atol=2e-5,
                rtol=1e-4,
                msg=lambda message, parameter_name=name: (
                    f"{parameter_name}: {message}"
                ),
            )

    def test_multi_query_and_cached_decay_validation(self) -> None:
        core = E3GatedTraceScanCore(4, 4)
        value = torch.randn(1, 4, 4)
        invalid_queries = (
            [],
            torch.tensor([], dtype=torch.long),
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([0.0]),
            torch.tensor([-1], dtype=torch.long),
            torch.tensor([4], dtype=torch.long),
            torch.tensor([1, 1], dtype=torch.long),
            torch.tensor([2, 1], dtype=torch.long),
        )
        for query in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises((TypeError, ValueError)):
                    core.forward_multi_query_eligibility(value, query)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "cached decay shapes"):
            core.forward_step_tensors_cached_decay(
                torch.randn(1, 4),
                torch.zeros(1, 4),
                torch.zeros(1, 4),
                torch.ones(1, 4),
                torch.ones(4),
            )

    def test_events_are_binary_traces_bounded_and_gradients_finite(self) -> None:
        torch.manual_seed(11200)
        core = E3GatedTraceScanCore(4, 6, state_dim=5)
        value = torch.randn(2, 64, 4, requires_grad=True)
        result, trace = core.forward_dynamics(value)
        for name in (
            "excitatory_content",
            "inhibitory_content",
            "excitatory_gate",
            "inhibitory_gate",
            "excitatory_writes",
            "inhibitory_writes",
            "excitatory_spikes",
            "inhibitory_spikes",
        ):
            events = getattr(trace, name)
            self.assertTrue(torch.all((events == 0.0) | (events == 1.0)), name)
        for name in ("excitatory_traces", "inhibitory_traces"):
            traces = getattr(trace, name)
            self.assertTrue(torch.all((traces >= 0.0) & (traces <= 1.0)), name)
        result.sequence.square().mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in core.parameters()
            )
        )

    def test_parameter_and_state_budget(self) -> None:
        core = E3GatedTraceScanCore(32, 32, state_dim=31)
        self.assertEqual(count_parameters(core), 8402)
        self.assertEqual(state_nbytes(core.initial_state(1)), 248)

    def test_constructor_and_state_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "decay bounds"):
            E3GatedTraceScanCore(4, 4, min_decay=0.9, max_decay=0.8)
        with self.assertRaisesRegex(ValueError, "initial decay"):
            E3GatedTraceScanCore(4, 4, min_initial_decay=0.4)
        with self.assertRaisesRegex(ValueError, "spike_threshold"):
            E3GatedTraceScanCore(4, 4, spike_threshold=1.0)
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            E3GatedTraceScanCore(4, 4, execution_mode="bad")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "eligibility_forward_mode"):
            E3GatedTraceScanCore(  # type: ignore[arg-type]
                4, 4, eligibility_forward_mode="bad"
            )
        with self.assertRaisesRegex(ValueError, "eligibility_backward_mode"):
            E3GatedTraceScanCore(  # type: ignore[arg-type]
                4, 4, eligibility_backward_mode="bad"
            )
        core = E3GatedTraceScanCore(4, 4)
        invalid = E3ScanState(
            layers=(
                E3LayerState(
                    excitatory=torch.full((1, 4), 1.1),
                    inhibitory=torch.zeros(1, 4),
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "trace must lie"):
            core(torch.randn(1, 2, 4), invalid)


if __name__ == "__main__":
    unittest.main()
