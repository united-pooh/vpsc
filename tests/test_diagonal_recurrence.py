from __future__ import annotations

import pytest
import torch

from vpsc.world_model.diagonal_recurrence import diagonal_query_recurrence


@pytest.mark.parametrize("include_final_query", (False, True))
def test_segmented_adjoint_matches_bptt(include_final_query: bool) -> None:
    torch.manual_seed(7)
    inputs = torch.randn(2, 37, 5)
    initial = torch.randn(2, 5)
    decay = (0.6 + 0.3 * torch.rand(5)).requires_grad_(True)
    indices = [2, 11, 23]
    if include_final_query:
        indices.append(36)
    query_indices = torch.tensor(indices, dtype=torch.long)

    def run(mode: str):
        x = inputs.clone().requires_grad_(True)
        h0 = initial.clone().requires_grad_(True)
        d = decay.detach().clone().requires_grad_(True)
        queries, final = diagonal_query_recurrence(
            x, h0, d, query_indices, mode=mode  # type: ignore[arg-type]
        )
        probe = torch.linspace(-0.7, 0.8, queries.numel()).reshape_as(queries)
        loss = (queries * probe).sum() + 0.17 * final.square().sum()
        loss.backward()
        return queries, final, x.grad, h0.grad, d.grad

    reference = run("bptt")
    candidate = run("segmented_adjoint")
    for actual, expected in zip(candidate, reference):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)


def test_query_validation_is_fail_closed() -> None:
    inputs = torch.randn(1, 8, 3)
    initial = torch.zeros(1, 3)
    decay = torch.full((3,), 0.8)
    with pytest.raises(ValueError):
        diagonal_query_recurrence(
            inputs,
            initial,
            decay,
            torch.tensor([3, 3]),
            mode="segmented_adjoint",
        )
    with pytest.raises(ValueError):
        diagonal_query_recurrence(
            inputs,
            initial,
            decay,
            torch.tensor([8]),
            mode="segmented_adjoint",
        )
