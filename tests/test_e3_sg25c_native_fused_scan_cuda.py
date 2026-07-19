from __future__ import annotations

import pytest
import torch

from vpsc.world_model.cores import E3GatedTraceScanCore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_fused_core_matches_legacy_forward_and_gradients() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(25_700_001)
    legacy = E3GatedTraceScanCore(
        4, 6, state_dim=5, eligibility_backward_mode="reverse_adjoint"
    ).to(device)
    fused = E3GatedTraceScanCore(
        4,
        6,
        state_dim=5,
        eligibility_backward_mode="reverse_adjoint",
        scan_math_mode="cuda_fused",
    ).to(device)
    fused.load_state_dict(legacy.state_dict())
    legacy_x = torch.randn(2, 80, 4, device=device, requires_grad=True)
    fused_x = legacy_x.detach().clone().requires_grad_(True)
    queries = torch.tensor((0, 7, 31, 63, 79), dtype=torch.long, device=device)
    legacy_output = legacy.forward_multi_query_eligibility(legacy_x, queries)
    fused_output = fused.forward_multi_query_eligibility(fused_x, queries)
    probe = torch.randn_like(legacy_output.sequence)
    (legacy_output.sequence * probe).sum().backward()
    (fused_output.sequence * probe).sum().backward()

    torch.testing.assert_close(
        fused_output.sequence,
        legacy_output.sequence,
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        fused_x.grad, legacy_x.grad, atol=3e-5, rtol=3e-4
    )
    for legacy_parameter, fused_parameter in zip(
        legacy.parameters(), fused.parameters()
    ):
        torch.testing.assert_close(
            fused_parameter.grad,
            legacy_parameter.grad,
            atol=3e-5,
            rtol=3e-4,
        )


def test_cuda_fused_rejects_cpu_execution() -> None:
    core = E3GatedTraceScanCore(
        4,
        6,
        state_dim=5,
        eligibility_backward_mode="reverse_adjoint",
        scan_math_mode="cuda_fused",
    )
    value = torch.randn(1, 8, 4)
    queries = torch.tensor((0, 7), dtype=torch.long)

    with pytest.raises(TypeError, match="CUDA float32"):
        core.forward_multi_query_eligibility(value, queries)
