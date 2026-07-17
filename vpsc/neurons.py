"""Mean-field LIF neuron for VPSC.

Implements the mean-field forward of Theorem 3:
    m = tanh(beta * (u - theta))          # differentiable magnetization
    s = 1[u >= theta]                     # hard spike (zero-temp limit, monitoring)

The membrane potential u follows standard LIF dynamics
    tau_m * du/dt = -u + I(t)
discretised as a first-order filter. u is the local field `a` in the theory;
the magnetization m serves as the continuous latent state x_l (posterior mean),
related to u by the mean-field inverse  u = (1/beta) atanh(m).

No surrogate gradient is used: the gradient flows through tanh(beta * .),
which is the *thermodynamically meaningful* activation, not a heuristic
approximation of a step. beta -> infinity recovers the hard spike.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LIFConfig:
    n: int
    tau_m: float = 20.0       # membrane time constant (timesteps)
    beta: float = 1.0         # inverse temperature (annealed during training)
    threshold: float = 1.0    # firing threshold theta
    reset: float = 0.0        # post-spike reset potential
    decay_reset: bool = True  # hard reset (True) vs soft reset (subtraction)


class MeanFieldLIF(nn.Module):
    """A bank of mean-field LIF neurons with per-call state.

    State (u) is stored as a buffer and reset between sequences via `reset_state`.
    The forward returns the differentiable magnetization `m` and the hard spike
    train `s` (for readout / monitoring only).
    """

    def __init__(self, cfg: LIFConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("u", torch.zeros(cfg.n))

    @property
    def beta(self) -> float:
        return self.cfg.beta

    def set_beta(self, beta: float) -> None:
        self.cfg.beta = float(beta)

    def reset_state(self, batch_size: Optional[int] = None) -> None:
        if batch_size is None:
            self.u.zero_()
        else:
            self.u = torch.zeros(batch_size, self.cfg.n, device=self.u.device)

    def forward(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # current: [B, N] instantaneous synaptic input at this timestep
        alpha = 1.0 / self.cfg.tau_m
        # Align state shape to input batch.
        if self.u.shape != current.shape:
            self.u = torch.zeros_like(current)
        self.u = (1.0 - alpha) * self.u + alpha * current

        # Differentiable mean-field activation. This is the surrogate-free
        # forward: its derivative is exact for the relaxed model at finite beta.
        m = torch.tanh(self.cfg.beta * (self.u - self.cfg.threshold))

        # Hard spike (zero-temperature limit). Detached — never carries gradient.
        spike = (self.u >= self.cfg.threshold).to(current.dtype)

        # Reset membrane after a spike.
        if self.cfg.decay_reset:
            self.u = torch.where(
                self.u >= self.cfg.threshold,
                torch.full_like(self.u, self.cfg.reset),
                self.u,
            )
        else:
            self.u = self.u - self.cfg.threshold * spike

        return m, spike


def spectral_radius(W: torch.Tensor) -> float:
    """Largest singular value of a weight matrix — the coupling scale J
    entering Theorem 3's critical point  beta_c ~ 1 / rho(W)."""
    with torch.no_grad():
        s = torch.linalg.svdvals(W)
        return float(s[0].item())
