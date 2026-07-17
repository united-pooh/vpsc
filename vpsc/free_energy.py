"""Training objective and beta annealing for VPSC.

The training objective is the generative free energy F ALONE (Section 1.2):
    L = F = sum_l F_l
with the task entering as the top-layer class-conditioned prior (mu_top =
prior[label]), NOT as a competing cross-entropy. This is the regime in which
Theorem 2's monotone-F guarantee holds: every learnable parameter is driven by
F, so nothing distorts the generative objective. (A joint F + CE objective
breaks the guarantee — see experiments/toy_verify.py note on the first attempt.)

Theorem 2 prediction to verify: F is monotonically non-increasing over training.
Theorem 3 prediction to verify: accuracy vs beta peaks near beta_c ~ 1/rho(W).

Beta annealing (Section 1.4): train at finite beta (differentiable mean field),
annealing toward the critical beta_c — NOT toward infinity. Stopping at beta_c
is the adaptive-termination conjecture of Section 3.7.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as Fnn

from .network import VPSCNet


def free_energy_loss(net: VPSCNet, out: dict, labels: torch.Tensor):
    """Pure generative objective (P1 / Theorem 2). Returns (loss, diagnostics)."""
    F = net.total_free_energy(out["traj"], labels=labels)
    return F, {"F": float(F.detach().item())}


def ce_loss(net: VPSCNet, out: dict, labels: torch.Tensor, lam_F: float = 0.0):
    """Cross-entropy objective for P2 (Theorem 3) training. Theorem 3 concerns
    inference-time beta given fixed weights, so the training rule is immaterial;
    CE reliably yields a network that classifies above chance. An optional small
    F term keeps the generative model from collapsing."""
    ce = Fnn.cross_entropy(out["logits"], labels)
    if lam_F > 0:
        F = net.total_free_energy(out["traj"], labels=labels)
        loss = ce + lam_F * F
        return loss, {"ce": float(ce.detach().item()), "F": float(F.detach().item())}
    return ce, {"ce": float(ce.detach().item())}


class BetaAnnealer:
    """Linearly anneal beta from `start` toward `target` over `steps` iterations.

    If target is None it is set to the network's predicted beta_c at construction
    time (Section 3.7: stop at the critical point, not at infinity).
    """

    def __init__(self, net: VPSCNet, start: float, target: Optional[float], steps: int):
        self.net = net
        self.start = start
        self.target = float(net.critical_beta()) if target is None else float(target)
        self.steps = max(1, steps)
        self._t = 0

    @property
    def beta_c(self) -> float:
        return self.target

    def step(self) -> float:
        self._t = min(self._t + 1, self.steps)
        frac = self._t / self.steps
        beta = self.start + (self.target - self.start) * frac
        self.net.set_beta(beta)
        return beta


# Backwards-compat alias used by older snippets.
@dataclass
class VPSCObjective:
    lambda_task: float = 0.0   # kept for API stability; F-only is the default
