"""Exact sparse-loss adjoint for a non-spiking diagonal recurrence."""

from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch import Tensor, nn


DiagonalBackwardMode = Literal["bptt", "segmented_adjoint"]


def _validate_query_indices(query_indices: Tensor, time_steps: int) -> None:
    if query_indices.ndim != 1 or query_indices.numel() == 0:
        raise ValueError("query_indices must be a non-empty rank-1 tensor")
    if query_indices.dtype != torch.long:
        raise TypeError("query_indices must use torch.long")
    if int(query_indices[0]) < 0 or int(query_indices[-1]) >= time_steps:
        raise ValueError("query_indices are outside the sequence")
    if query_indices.numel() > 1 and not bool(
        torch.all(query_indices[1:] > query_indices[:-1])
    ):
        raise ValueError("query_indices must be strictly increasing")


def _serial_recurrence(
    inputs: Tensor, initial: Tensor, decay: Tensor
) -> Tensor:
    state = initial
    states = []
    for step in range(inputs.shape[1]):
        state = decay * state + inputs[:, step]
        states.append(state)
    return torch.stack(states, dim=1)


class _SegmentedDiagonalQueries(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        inputs: Tensor,
        initial: Tensor,
        decay: Tensor,
        query_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        _validate_query_indices(query_indices, inputs.shape[1])
        state = initial
        queried = []
        query_cursor = 0
        indices = query_indices.detach().cpu().tolist()
        for step in range(inputs.shape[1]):
            state = decay * state + inputs[:, step]
            if query_cursor < len(indices) and step == indices[query_cursor]:
                queried.append(state)
                query_cursor += 1
        ctx.save_for_backward(inputs, initial, decay, query_indices)  # type: ignore[attr-defined]
        return torch.stack(queried, dim=1), state

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: object,
        grad_queries: Tensor,
        grad_final: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, None]:
        inputs, initial, decay, query_indices = ctx.saved_tensors  # type: ignore[attr-defined]
        batch, time_steps, state_dim = inputs.shape
        indices = [int(value) for value in query_indices.detach().cpu().tolist()]
        anchor_gradients = [grad_queries[:, index] for index in range(len(indices))]
        if grad_final is None:
            grad_final = torch.zeros_like(initial)
        if indices[-1] == time_steps - 1:
            anchor_gradients[-1] = anchor_gradients[-1] + grad_final
        elif bool(torch.any(grad_final != 0)):
            indices.append(time_steps - 1)
            anchor_gradients.append(grad_final)

        anchor_adjoints = [torch.empty(0, device=inputs.device)] * len(indices)
        following_index = None
        following_adjoint = None
        for anchor in range(len(indices) - 1, -1, -1):
            index = indices[anchor]
            impulse = anchor_gradients[anchor]
            if following_index is None:
                adjoint = impulse
            else:
                distance = following_index - index
                adjoint = impulse + decay.pow(distance) * following_adjoint
            anchor_adjoints[anchor] = adjoint
            following_index = index
            following_adjoint = adjoint

        grad_inputs = torch.zeros_like(inputs)
        previous_anchor = -1
        for index, adjoint in zip(indices, anchor_adjoints):
            steps = torch.arange(
                previous_anchor + 1,
                index + 1,
                device=inputs.device,
                dtype=decay.dtype,
            )
            powers = (index - steps).unsqueeze(1)
            segment = decay.unsqueeze(0).pow(powers)
            grad_inputs[:, previous_anchor + 1 : index + 1] = (
                adjoint.unsqueeze(1) * segment.unsqueeze(0)
            )
            previous_anchor = index

        states = torch.empty(
            batch,
            time_steps,
            state_dim,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        state = initial
        for step in range(time_steps):
            state = decay * state + inputs[:, step]
            states[:, step] = state
        previous_states = torch.cat((initial.unsqueeze(1), states[:, :-1]), dim=1)
        grad_decay = (grad_inputs * previous_states).sum(dim=(0, 1))
        grad_initial = decay * grad_inputs[:, 0]
        return grad_inputs, grad_initial, grad_decay, None


def diagonal_query_recurrence(
    inputs: Tensor,
    initial: Tensor,
    decay: Tensor,
    query_indices: Tensor,
    *,
    mode: DiagonalBackwardMode,
) -> Tuple[Tensor, Tensor]:
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, time, state]")
    if initial.shape != (inputs.shape[0], inputs.shape[2]):
        raise ValueError("initial shape does not match inputs")
    if decay.shape != (inputs.shape[2],):
        raise ValueError("decay shape does not match state dimension")
    _validate_query_indices(query_indices, inputs.shape[1])
    if mode == "bptt":
        states = _serial_recurrence(inputs, initial, decay)
        return states.index_select(1, query_indices), states[:, -1]
    if mode == "segmented_adjoint":
        return _SegmentedDiagonalQueries.apply(
            inputs, initial, decay, query_indices
        )
    raise ValueError(f"unsupported diagonal backward mode: {mode}")


class DiagonalLinearCore(nn.Module):
    """A minimal real diagonal SSM core with no VPSC-specific mechanism."""

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, state_dim)
        self.decay_logits = nn.Parameter(torch.zeros(state_dim))

    @property
    def decay(self) -> Tensor:
        return 0.5 + 0.49 * torch.sigmoid(self.decay_logits)

    def forward_queries(
        self,
        inputs: Tensor,
        query_indices: Tensor,
        *,
        mode: DiagonalBackwardMode,
        initial: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        projected = self.input_projection(inputs)
        if initial is None:
            initial = projected.new_zeros(projected.shape[0], projected.shape[2])
        return diagonal_query_recurrence(
            projected,
            initial,
            self.decay,
            query_indices,
            mode=mode,
        )


__all__ = [
    "DiagonalBackwardMode",
    "DiagonalLinearCore",
    "diagonal_query_recurrence",
]
