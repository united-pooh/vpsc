from __future__ import annotations

import torch

from experiments.e3_sg19_plan_edge_spikes import (
    _plan_slots,
    parse_objective_plan,
    plan_edge_kernel,
    return_edge_spike,
)


def test_public_objective_compiles_to_ordered_action_spike_tape() -> None:
    objective = (
        "First travel west. Then take a trip south. And then go east. "
        "Finally move north and retrieve the coin."
    )
    assert parse_objective_plan(objective) == (
        "go west",
        "go south",
        "go east",
        "go north",
        "take coin",
    )
    assert _plan_slots(parse_objective_plan(objective), 3) == (
        "go north",
        "take coin",
    )


def test_return_edge_spike_detects_inverse_physical_move_only() -> None:
    assert return_edge_spike("go west", "go east") == 1
    assert return_edge_spike("go west", "go north") == 0
    assert return_edge_spike(None, "go east") == 0


def test_plan_edge_product_kernel_is_symmetric_psd() -> None:
    state = {
        "keys": torch.tensor(
            [[4, 4, 4, 0], [4, 4, 4, 1], [4, 4, 0, 1]],
            dtype=torch.long,
        ),
        "phases": torch.tensor([0, 0, 1], dtype=torch.long),
        "masks": torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        ),
        "plan_current": torch.tensor([0, 0, 1], dtype=torch.long),
        "plan_next": torch.tensor([1, 2, 2], dtype=torch.long),
        "return_edges": torch.tensor([0, 1, 0], dtype=torch.long),
    }
    kernel = plan_edge_kernel(state, state)
    assert torch.allclose(kernel, kernel.T)
    assert float(torch.linalg.eigvalsh(kernel).min().item()) >= -1e-10
