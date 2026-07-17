"""VPSC hierarchical network.

Architecture (Theorem 2 — predictive coding as a potential game):
    layers l = 1..L
    x_l   : continuous latent state of layer l  == mean-field magnetization m_l
    mu_l  : top-down prediction of x_l from the layer above
            mu_l = g(phi_l, x_{l+1})   (linear here: mu_l = x_{l+1} W_down_l)
    eps_l : prediction error  eps_l = x_l - mu_l

The forward pass is bottom-up recognition (LIF dynamics on each layer);
the free energy is evaluated top-down (predictions flow downward). Both
share the same weights, tied by the predictive-coding symmetric convention
declared in Theorem 2 (sec 2.4) — without it the potential-game proof breaks.
"""

from dataclasses import dataclass
from typing import List, Optional, cast

import torch
import torch.nn as nn

from .neurons import LIFConfig, MeanFieldLIF, spectral_radius


@dataclass
class LayerSpec:
    n_in: int                 # dim of the input fed to this layer (sizes[l])
    n_out: int                # dim of this layer's state x_l  (sizes[l+1])
    n_next: Optional[int]     # dim of the layer above's state x_{l+1} (sizes[l+2]);
                              # None for the top layer (zero prior, no W_down).


@dataclass
class VPSCConfig:
    sizes: List[int]               # [n_in, h1, h2, ..., n_top]
    n_classes: int
    beta: float = 1.0
    tau_m: float = 20.0
    threshold: float = 1.0
    sigma: float = 1.0             # prediction-error precision Sigma (isotropic)
    sparsity: float = 0.0          # Omega(x) = sparsity * ||x||_1  (energy cost)


class VPSCLayer(nn.Module):
    """One VPSC layer: a recognition LIF bank + a top-down generative weight."""

    def __init__(self, spec: LayerSpec, cfg: VPSCConfig):
        super().__init__()
        self.cfg = cfg
        self.n_next = spec.n_next
        # Recognition (bottom-up): maps lower-layer state to synaptic current.
        self.W_up = nn.Parameter(torch.empty(spec.n_in, spec.n_out))
        nn.init.kaiming_uniform_(self.W_up, a=5 ** 0.5)
        # Generative (top-down): predicts x_l (dim n_out) from x_{l+1} (dim n_next).
        # Shape [n_next, n_out]. None for the top layer (uses a zero prior).
        if spec.n_next is not None:
            self.W_down = nn.Parameter(torch.empty(spec.n_next, spec.n_out))
            nn.init.kaiming_uniform_(self.W_down, a=5 ** 0.5)
        else:
            self.W_down = None
        self.lif = MeanFieldLIF(
            LIFConfig(
                n=spec.n_out,
                tau_m=cfg.tau_m,
                beta=cfg.beta,
                threshold=cfg.threshold,
            )
        )

    def set_beta(self, beta: float) -> None:
        self.lif.set_beta(beta)

    def predict(self, x_upper: torch.Tensor) -> torch.Tensor:
        """Top-down prediction mu_l = x_{l+1} @ W_down_l,  W_down: [n_next, n_out].
        x_upper [B, n_next] @ W_down -> [B, n_out]."""
        assert self.W_down is not None, "top layer has no top-down prediction"
        return x_upper @ self.W_down

    def forward(self, x_lower: torch.Tensor):
        current = x_lower @ self.W_up
        return self.lif(current)


class VPSCNet(nn.Module):
    def __init__(self, cfg: VPSCConfig):
        super().__init__()
        self.cfg = cfg
        L = len(cfg.sizes) - 1
        self.layers = nn.ModuleList(
            [
                VPSCLayer(
                    LayerSpec(
                        n_in=cfg.sizes[i],
                        n_out=cfg.sizes[i + 1],
                        n_next=(cfg.sizes[i + 2] if i + 2 < len(cfg.sizes) else None),
                    ),
                    cfg,
                )
                for i in range(L)
            ]
        )
        # Class-conditioned top-layer prior. The task enters the free energy HERE
        # — as the top-down prediction of x_top given the label — not as a separate
        # cross-entropy. This is the predictive-coding classification setup, and it
        # is what keeps Theorem 2's monotone-F guarantee in force: every learnable
        # parameter is driven by F, nothing competes with it.
        #
        # The prior is FIXED (non-learnable) and distinct per class. A learnable
        # prior admits a degenerate minimum where all class priors collapse to the
        # same value and x_top -> 0 (no separability). Fixed distinct targets force
        # the recognition path to produce class-dependent x_top.
        prior = torch.randn(cfg.n_classes, cfg.sizes[-1], generator=torch.Generator().manual_seed(0))
        # Orthogonalise so class targets are maximally separated.
        if cfg.n_classes <= cfg.sizes[-1]:
            q, _ = torch.linalg.qr(prior.t())
            prior = q.t()
        self.register_buffer("class_prior", prior)

        # Discriminative readout for Theorem-3 (P2) training. P2 needs a network
        # that actually classifies above chance; how it is trained is immaterial
        # to Theorem 3 (which concerns inference-time beta given fixed weights),
        # so a standard CE readout is used there. P1 does not use this readout.
        self.readout = nn.Linear(cfg.sizes[-1], cfg.n_classes)

    def set_beta(self, beta: float) -> None:
        for layer in cast(List[VPSCLayer], self.layers):
            layer.set_beta(beta)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        for layer in cast(List[VPSCLayer], self.layers):
            layer.lif.reset_state(batch_size)
            layer.lif.u = layer.lif.u.to(device)

    def forward(self, x_seq: torch.Tensor, return_all_logits: bool = False) -> dict:
        """x_seq: [T, B, n_in]. Returns per-timestep states (traj) and the final
        top-layer state (x_top). Classification is by nearest class prior.
        If return_all_logits, also returns logits at EVERY timestep (stacked
        [T, B, n_classes]) — used by the fixation experiment to read out the
        evolving classification over prolonged fixation of one image."""
        T, B, _ = x_seq.shape
        self.reset_state(B, x_seq.device)

        traj: List[List[torch.Tensor]] = []   # traj[t][l] = x_l at time t
        x_top = None
        logits = None
        all_logits: List[torch.Tensor] = []
        for t in range(T):
            x = x_seq[t]
            states_t: List[torch.Tensor] = []
            for layer in cast(List[VPSCLayer], self.layers):
                m, _spike = layer(x)
                x = m                       # magnetization is the propagated state
                states_t.append(m)
            traj.append(states_t)
            x_top = states_t[-1]
            logits = self.readout(states_t[-1])   # read at every step; use last
            if return_all_logits:
                all_logits.append(logits)
        assert x_top is not None and logits is not None
        out = {"traj": traj, "x_top": x_top, "logits": logits}
        if return_all_logits:
            out["all_logits"] = torch.stack(all_logits, dim=0)  # [T, B, C]
        return out

    @torch.no_grad()
    def classify(self, x_top: torch.Tensor) -> torch.Tensor:
        """Nearest class-prior: argmin_c ||x_top - prior_c||^2. Cheap proxy for
        full PC inference (which would re-run recognition per hypothesis)."""
        # dists: [B, C]
        dists = ((x_top.unsqueeze(1) - self.class_prior.unsqueeze(0)) ** 2).sum(dim=-1)
        return dists.argmin(dim=-1)

    # ---- Free energy (Theorem 2 / Section 1.2) ----

    def layer_free_energy(self, x_l: torch.Tensor, mu_l: torch.Tensor) -> torch.Tensor:
        """F_l = 0.5 * ||x_l - mu_l||^2 / sigma^2  +  Omega(x_l).  Per-sample, summed over units."""
        err = x_l - mu_l
        prec = 1.0 / (self.cfg.sigma ** 2)
        quad = 0.5 * prec * (err ** 2).sum(dim=-1)
        omega = self.cfg.sparsity * x_l.abs().sum(dim=-1)
        return quad + omega

    def total_free_energy(self, traj: List[List[torch.Tensor]],
                          labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum of F_l over layers and timesteps. The generative objective whose
        monotone decrease is Theorem 2's prediction.

        Top layer: prediction is the class-conditioned prior (mu_top = prior[label])
        when labels are given (training); zero prior otherwise.
        """
        F = torch.zeros((), device=traj[0][0].device)
        layers = cast(List[VPSCLayer], self.layers)
        L = len(layers)
        T = len(traj)
        for t_idx, states_t in enumerate(traj):
            for l in range(L):
                x_l = states_t[l]
                if l < L - 1:
                    mu_l = layers[l].predict(states_t[l + 1])
                else:
                    # Top layer: the class prior is the readout target. For a
                    # temporal task the class evidence is only available at the
                    # sequence end, so apply the prior ONLY at the final timestep;
                    # intermediate top-layer states use a zero prior (free dynamics).
                    if labels is not None and t_idx == T - 1:
                        mu_l = self.class_prior[labels]
                    else:
                        mu_l = torch.zeros_like(x_l)
                F = F + self.layer_free_energy(x_l, mu_l).mean()
        return F

    # ---- Diagnostics for Theorem 3 ----

    def spectral_radius(self) -> float:
        """rho(W_up) of the first layer — the coupling scale J for beta_c ~ 1/J."""
        layers = cast(List[VPSCLayer], self.layers)
        return spectral_radius(layers[0].W_up.data)

    def critical_beta(self) -> float:
        """Predicted peak location of accuracy-vs-beta: beta_c ~ 1/rho(W)."""
        rho = self.spectral_radius()
        return 1.0 / rho if rho > 0 else float("inf")
