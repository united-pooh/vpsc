from __future__ import annotations

import torch

from experiments import e3_sg25a_blocked_affine_scan_cuda as sg25a
from vpsc.world_model.cores import E3GatedTraceScanCore


def test_blocked_closed_form_matches_serial_trace() -> None:
    torch.manual_seed(25_500_001)
    write = torch.randint(0, 2, (2, 134, 9)).to(torch.float32)
    decay = torch.linspace(0.55, 0.99, steps=9)
    initial = torch.rand(2, 9)
    serial = E3GatedTraceScanCore._serial_trace(write, decay, initial)

    for block_size in (32, 64, 128):
        candidate = E3GatedTraceScanCore._blocked_constant_affine_prefix_scan(
            write, decay, initial, block_size=block_size
        )
        torch.testing.assert_close(candidate, serial, atol=2e-6, rtol=2e-5)
        assert torch.equal(candidate >= 0.5, serial >= 0.5)


def test_blocked_closed_form_supports_unscaled_reverse_adjoint() -> None:
    torch.manual_seed(25_500_002)
    impulses = torch.randn(1, 80, 7)
    decay = torch.linspace(0.55, 0.99, steps=7)
    coefficient = decay.view(1, 1, -1).expand_as(impulses)
    reference = E3GatedTraceScanCore._affine_prefix_scan(
        coefficient, impulses, torch.zeros_like(impulses[:, 0])
    )
    candidate = E3GatedTraceScanCore._blocked_constant_affine_prefix_scan(
        impulses,
        decay,
        torch.zeros_like(impulses[:, 0]),
        block_size=64,
        injection_scale=torch.ones_like(decay),
    )

    torch.testing.assert_close(candidate, reference, atol=2e-6, rtol=2e-5)


def test_blocked_reverse_adjoint_core_matches_legacy_gradients() -> None:
    torch.manual_seed(25_500_003)
    legacy = E3GatedTraceScanCore(
        4, 6, state_dim=5, eligibility_backward_mode="reverse_adjoint"
    )
    candidate = E3GatedTraceScanCore(
        4,
        6,
        state_dim=5,
        eligibility_backward_mode="reverse_adjoint",
        scan_math_mode="blocked_cumsum",
        scan_block_size=64,
    )
    candidate.load_state_dict(legacy.state_dict())
    legacy_x = torch.randn(2, 80, 4, requires_grad=True)
    candidate_x = legacy_x.detach().clone().requires_grad_(True)
    queries = torch.tensor((0, 7, 31, 63, 79), dtype=torch.long)
    legacy_output = legacy.forward_multi_query_eligibility(legacy_x, queries)
    candidate_output = candidate.forward_multi_query_eligibility(
        candidate_x, queries
    )
    probe = torch.randn_like(legacy_output.sequence)
    (legacy_output.sequence * probe).sum().backward()
    (candidate_output.sequence * probe).sum().backward()

    torch.testing.assert_close(
        candidate_output.sequence,
        legacy_output.sequence,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        candidate_x.grad, legacy_x.grad, atol=3e-5, rtol=3e-4
    )
    for legacy_parameter, candidate_parameter in zip(
        legacy.parameters(), candidate.parameters()
    ):
        torch.testing.assert_close(
            candidate_parameter.grad,
            legacy_parameter.grad,
            atol=3e-5,
            rtol=3e-4,
        )


def test_scan_math_mode_validation() -> None:
    try:
        E3GatedTraceScanCore(4, 6, scan_math_mode="invalid")  # type: ignore[arg-type]
    except ValueError as error:
        assert "scan_math_mode" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid scan mode was accepted")


def test_no_selected_block_does_not_erase_passing_memory_gate() -> None:
    primitive = {
        "by_block": {str(size): {"passed": False} for size in sg25a.BLOCK_SIZES}
    }
    full_core = {"by_block": {str(size): True for size in sg25a.BLOCK_SIZES}}
    trajectory = {
        "by_block": {str(size): False for size in sg25a.BLOCK_SIZES}
    }
    update = {
        "records": {
            f"block_{size}": {
                "timing": {"p50_ms": 5.0},
                "versus_legacy_speedup": 1.1,
                "versus_sg24_lstm_ratio": 2.0,
                "additional_peak_to_legacy_ratio": 1.0,
            }
            for size in sg25a.BLOCK_SIZES
        }
    }

    decision = sg25a._decision(
        primitive=primitive,
        full_core=full_core,
        trajectory=trajectory,
        update=update,
        quick=False,
    )

    assert decision["selected_block_size"] is None
    assert decision["memory_gate"] == "PASS"
    assert decision["numerical_gate"] == "FAIL"
    assert decision["overall"] == "FAIL"
