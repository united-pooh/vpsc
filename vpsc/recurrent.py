"""Recurrent mean-field VPSC layers — the deep-network extension.

WHY THIS MODULE EXISTS
----------------------
The feedforward VPSC layers in network.py have no within-layer feedback, so the
Curie transition of Theorem 3 (beta_c = 1 / rho(W)) is NOT meaningful there — it
was verified only on an isolated recurrent layer in toy_verify.py P2. To make
the criticality result meaningful INSIDE a deep network, each layer here is a
recurrent mean-field layer: its state is the fixed point of

    m = tanh( beta * (W_rec @ m + I) )

where I is the feedforward drive from the layer below. The J*m feedback is the
interaction that creates the phase transition; per-layer beta_c = 1/rho(W_rec).

This is the genuine (and genuinely open) extension: does the criticality peak
survive stacking recurrent layers into a deep network that drives a task
readout? Tested in experiments/deep_critical.py.

FREE ENERGY (training objective)
--------------------------------
F_l = 0.5 * ||x_l - mu_l||^2 / sigma^2          # prediction error (top-down)
      - 0.5 * x_l^T Ws x_l                      # Ising interaction (Ws = sym(W_rec))
      + (1/beta) * sum_i H_bin(x_l_i)           # mean-field entropy (bounds F below)
      + 0.5 * wd * ||W_rec||^2                  # weight decay
      + lam_spec * max(0, rho(Ws) - rho_max)^2  # one-sided spectral cap

The interaction term is within-layer, so the layer-wise decomposition (and thus
Theorem 2's potential-game / monotone-F argument) is preserved in structure.
The entropy and spectral cap keep F bounded below — WITHOUT them, the Ising
interaction -0.5 x^T W_rec x is unbounded below and pure-F training drives
rho(W_rec) -> infinity (a degenerate minimum, observed in the first run). W_rec
is symmetrised (Ws = 0.5 (W + W^T)) so eigenvalues are real and differentiable.
CAVEAT: the interaction still makes F_l non-convex in x_l, so the strong-
convexity assumption of Theorem 2.3 is weakened; Theorem 2.2 (monotone decrease
under small-enough gradient steps) is what D1 tests.
"""

from dataclasses import dataclass
from typing import List, Optional
import math

import torch
import torch.nn as nn


def spectral_radius_square(W: torch.Tensor) -> float:
    """Max |eigenvalue| of a square matrix — the stability/criticality radius
    for the recurrent mean-field map. (For symmetric W this equals the largest
    singular value.) Detached: used for monitoring the beta_c prediction, not
    in the gradient path."""
    with torch.no_grad():
        ev = torch.linalg.eigvals(W)
        return float(ev.abs().max().item())


def _sym(W: torch.Tensor) -> torch.Tensor:
    return 0.5 * (W + W.t())


def _rho_sym(W: torch.Tensor) -> torch.Tensor:
    """Differentiable spectral radius for a (symmetrised) real matrix."""
    Ws = _sym(W)
    eig = torch.linalg.eigvalsh(Ws)
    return eig.abs().max()


def _binary_entropy(m: torch.Tensor) -> torch.Tensor:
    """Per-element binary entropy H(m) for m in (-1, 1), in nats. = log2 - ...
    Stable via log1p with clamping. Shape: same as m."""
    mc = m.clamp(-0.999999, 0.999999)
    # H = log2 - 0.5*[(1+m)log(1+m) + (1-m)log(1-m)]
    term = 0.5 * ((1 + mc) * torch.log1p(mc) + (1 - mc) * torch.log1p(-mc))
    return math.log(2.0) - term


def init_symmetric(n: int, target_rho: float, generator: torch.Generator) -> torch.Tensor:
    """A symmetric matrix scaled to spectral radius `target_rho` (< 1 keeps the
    fixed point convergent at beta ~ 1 during training)."""
    A = torch.randn(n, n, generator=generator)
    W = 0.5 * (A + A.t())
    rho = spectral_radius_square(W)
    if rho > 0:
        W = W * (target_rho / rho)
    return W


@dataclass
class RecurrentLayerSpec:
    n_in: int
    n_out: int
    n_next: Optional[int]   # dim of layer above; None for the top layer


class RecurrentMeanFieldLayer(nn.Module):
    """One recurrent mean-field VPSC layer."""

    def __init__(self, spec: RecurrentLayerSpec, beta: float, threshold: float,
                 sigma: float, n_relax: int, rec_rho0: float, wd: float,
                 lam_spec: float, rho_max: float, gen: torch.Generator,
                 leak: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.sigma = sigma
        self.n_relax = n_relax
        self.wd = wd
        self.lam_spec = lam_spec
        self.rho_max = rho_max
        self.n_next = spec.n_next
        # Leak factor for the membrane trace. leak=1.0 -> full relaxation to the
        # fixed point each step (no persistent trace). leak<1.0 -> leaky LIF-style
        # integration  m <- (1-leak) m_old + leak tanh(...), which preserves a
        # decaying PSP trace. Theorem 1's STDP envelope REQUIRES such a trace
        # (it is the LIF tau_m dynamics, per docs/theorem1.md sec 4), so the
        # STDP-timing experiment uses leak<1.0.
        self.leak = leak
        # If True, detach the persistent state between timesteps (default; needed
        # for training stability and memory). If False, the state stays graph-
        # connected across time — required for spike-timing credit assignment
        # (the STDP test), so that dF/dw at t_post can flow back to a pre pulse
        # at t_pre through the leaky membrane trace.
        self.detach_state = True

        self.W_up = nn.Parameter(torch.empty(spec.n_in, spec.n_out))
        nn.init.kaiming_uniform_(self.W_up, a=5 ** 0.5)
        # Recurrent within-layer coupling. Symmetric init at controlled radius;
        # symmetrised on use so eigenvalues stay real and differentiable.
        self.W_rec = nn.Parameter(init_symmetric(spec.n_out, rec_rho0, gen))
        if spec.n_next is not None:
            self.W_down = nn.Parameter(torch.empty(spec.n_next, spec.n_out))
            nn.init.kaiming_uniform_(self.W_down, a=5 ** 0.5)
        else:
            self.W_down = None

        self.register_buffer("m", torch.zeros(spec.n_out))

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.m = torch.zeros(batch_size, self.W_rec.shape[0], device=device)

    def predict(self, x_upper: torch.Tensor) -> torch.Tensor:
        assert self.W_down is not None
        return x_upper @ self.W_down

    def forward(self, x_lower: torch.Tensor, I_ext: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean-field relaxation to (approximate) fixed point, warm-started from
        the previous timestep's state. Unrolled -> differentiable. Uses the
        symmetrised recurrent coupling Ws. Optional `I_ext` adds an external
        current directly to this layer's neurons (used for controlled spike-timing
        experiments; not used in normal training)."""
        I = x_lower @ self.W_up
        if I_ext is not None:
            I = I + I_ext
        if self.m.shape != I.shape:
            self.m = torch.zeros_like(I)
        Ws = _sym(self.W_rec)
        m = self.m
        for _ in range(self.n_relax):
            m_new = torch.tanh(self.beta * (m @ Ws + I - self.threshold))
            m = (1.0 - self.leak) * m + self.leak * m_new
        self.m = m.detach() if self.detach_state else m
        return m

    def free_energy(self, x_l: torch.Tensor, mu_l: torch.Tensor) -> torch.Tensor:
        Ws = _sym(self.W_rec)
        err = x_l - mu_l
        prec = 1.0 / (self.sigma ** 2)
        quad = 0.5 * prec * (err ** 2).sum(dim=-1)
        interaction = -0.5 * (x_l * (x_l @ Ws)).sum(dim=-1)
        entropy = (1.0 / self.beta) * _binary_entropy(x_l).sum(dim=-1)
        reg = 0.5 * self.wd * (self.W_rec ** 2).sum()
        # One-sided spectral cap: penalise rho(Ws) exceeding rho_max. Keeps the
        # mean-field fixed point well-defined (convergent) and F bounded below.
        rho = _rho_sym(self.W_rec)
        spec_pen = self.lam_spec * torch.clamp(rho - self.rho_max, min=0.0) ** 2
        return quad + interaction + entropy + reg + spec_pen

    def free_energy_phi(self, x_l: torch.Tensor, mu_l: torch.Tensor) -> torch.Tensor:
        """Dimensionless free energy Phi = beta*E - S (Fix1, RC1).
        E = quad + interaction + wd; S = sum H_bin (NOT (1/beta)*S). All energy
        terms beta-scaled => coherent homotopy under beta-annealing. Barrier B(W)
        and Tikhonov are no-ops by default (Fix2/Fix4 enable them via flags)."""
        Ws = _sym(self.W_rec)
        err = x_l - mu_l
        quad = 0.5 * (1.0 / self.sigma ** 2) * (err ** 2).sum(dim=-1)
        interaction = -0.5 * (x_l * (x_l @ Ws)).sum(dim=-1)
        wd = 0.5 * self.wd * (self.W_rec ** 2).sum()
        energy = quad + interaction + wd
        entropy = _binary_entropy(x_l).sum(dim=-1)
        phi = self.beta * energy - entropy
        if getattr(self, "use_log_det_barrier", False):
            phi = phi + self.log_det_barrier(getattr(self, "gamma", 1.0))
        if getattr(self, "tikhonov_eps", 0.0) > 0:
            phi = phi + 0.5 * self.tikhonov_eps * (x_l ** 2).sum()
        return phi

    def log_det_barrier(self, gamma: float, eps: float = 1e-8) -> torch.Tensor:
        """B(W) = -(gamma/2) * sum_i log(1 - beta^2 * lambda_i^2) over eigenvalues
        lambda_i of Ws. Diverges to +inf as |lambda| -> 1/beta (= beta_c boundary),
        giving a natural spectral barrier that EMERGES from the free energy (Fix2,
        RC3) — replacing the external project_spectral hard cap. Stable via clamp."""
        Ws = _sym(self.W_rec)
        lam = torch.linalg.eigvalsh(Ws)  # real eigenvalues (Ws symmetric)
        val = torch.clamp(1.0 - (self.beta ** 2) * lam ** 2, min=eps)
        return -0.5 * gamma * torch.log(val).sum()

    def critical_beta(self) -> float:
        rho = spectral_radius_square(_sym(self.W_rec.data))
        return 1.0 / rho if rho > 0 else float("inf")

    @torch.no_grad()
    def project_spectral(self, rho_max: float) -> None:
        """Hard spectral cap: rescale W_rec so rho(sym(W_rec)) <= rho_max.
        Keeps the mean-field fixed point well-defined and F bounded below. The
        soft penalty alone is insufficient — the Ising interaction's gain
        outweighs it, and the entropy term vanishes at saturation."""
        Ws = _sym(self.W_rec.data)
        ev = torch.linalg.eigvalsh(Ws)
        rho = float(ev.abs().max().item())
        if rho > rho_max and rho > 0:
            self.W_rec.data.mul_(rho_max / rho)


class RecurrentVPSCNet(nn.Module):
    """Stack of recurrent mean-field layers + a discriminative readout (for P2)
    and a class-prior top layer (for P1, pure generative F)."""

    def __init__(self, sizes: List[int], n_classes: int, beta: float = 1.0,
                 threshold: float = 0.0, sigma: float = 1.0, n_relax: int = 8,
                 rec_rho0: float = 0.7, wd: float = 1e-4, lam_spec: float = 1.0,
                 rho_max: float = 0.9, leak: float = 1.0, seed: int = 0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.sigma = sigma
        self.n_relax = n_relax
        self.wd = wd
        self.rho_max = rho_max
        gen = torch.Generator().manual_seed(seed)

        L = len(sizes) - 1
        self.layers = nn.ModuleList([
            RecurrentMeanFieldLayer(
                RecurrentLayerSpec(
                    n_in=sizes[i], n_out=sizes[i + 1],
                    n_next=(sizes[i + 2] if i + 2 < len(sizes) else None),
                ),
                beta=beta, threshold=threshold, sigma=sigma, n_relax=n_relax,
                rec_rho0=rec_rho0, wd=wd, lam_spec=lam_spec, rho_max=rho_max, gen=gen,
                leak=leak,
            )
            for i in range(L)
        ])
        self.readout = nn.Linear(sizes[-1], n_classes)
        prior = torch.randn(n_classes, sizes[-1], generator=gen)
        if n_classes <= sizes[-1]:
            q, _ = torch.linalg.qr(prior.t())
            prior = q.t()
        self.register_buffer("class_prior", prior)

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)
        for layer in self.layers:
            layer.set_beta(beta)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        for layer in self.layers:
            layer.reset_state(batch_size, device)

    def forward(self, x_seq: torch.Tensor,
                I_ext_seq: Optional[List[List[Optional[torch.Tensor]]]] = None) -> dict:
        """x_seq: [T, B, n_in]. I_ext_seq (optional): per-timestep, per-layer
        external currents of shape [B, n_out_l]; None entries are skipped. Used
        for controlled spike-timing experiments."""
        T, B, _ = x_seq.shape
        self.reset_state(B, x_seq.device)
        traj: List[List[torch.Tensor]] = []
        logits = None
        x_top = None
        for t in range(T):
            x = x_seq[t]
            states_t: List[torch.Tensor] = []
            for li, layer in enumerate(self.layers):
                I_ext = None
                if I_ext_seq is not None and I_ext_seq[t][li] is not None:
                    I_ext = I_ext_seq[t][li]
                m = layer(x, I_ext=I_ext)
                x = m
                states_t.append(m)
            traj.append(states_t)
            x_top = states_t[-1]
            logits = self.readout(states_t[-1])
        return {"traj": traj, "x_top": x_top, "logits": logits}

    def pc_inference(self, x_seq: torch.Tensor, K: int = 8, tol: float = 1e-4,
                     eta: float = 0.3, labels: Optional[torch.Tensor] = None) -> dict:
        """Predictive-coding inference loop (Fix3, RC2). Bottom-up init, then K
        gradient-descent steps on sum_l Phi_l w.r.t. the states (true PC: minimize
        free energy over states given top-down predictions). The top mu is the
        class_prior[label] when labels are given (training: reconcile bottom-up
        state with top-down prior instead of forcing a single feedforward pass),
        else zeros (eval). This breaks the RC2 mismatch — the saturated state is
        pulled toward a self-consistent prediction rather than left to saturate."""
        T, B, _ = x_seq.shape
        self.reset_state(B, x_seq.device)
        traj = []
        L = len(self.layers)
        for t in range(T):
            x = x_seq[t]
            states = [None] * L
            cur = x
            for li, layer in enumerate(self.layers):
                cur = layer(cur).clone().detach().requires_grad_(True)
                states[li] = cur
            for _ in range(K):
                loss = torch.zeros((), device=x.device)
                for li, layer in enumerate(self.layers):
                    if li < L - 1:
                        mu = layer.predict(states[li + 1].detach())
                    else:
                        mu = self.class_prior[labels] if (labels is not None) else torch.zeros_like(states[li])
                    loss = loss + layer.free_energy_phi(states[li], mu).sum()
                grads = torch.autograd.grad(loss, states, allow_unused=True)
                max_delta = 0.0
                for li in range(L):
                    g = grads[li]
                    if g is not None:
                        new = states[li] - eta * g
                        max_delta = max(max_delta, (new - states[li]).abs().max().item())
                        states[li] = new.detach().requires_grad_(True)
                if max_delta < tol:
                    break
            traj.append([s.detach() for s in states])
        x_top = traj[-1][-1]
        return {"traj": traj, "x_top": x_top, "logits": self.readout(x_top)}

    @torch.no_grad()
    def classify(self, x_top: torch.Tensor) -> torch.Tensor:
        dists = ((x_top.unsqueeze(1) - self.class_prior.unsqueeze(0)) ** 2).sum(dim=-1)
        return dists.argmin(dim=-1)

    def total_free_energy(self, traj: List[List[torch.Tensor]],
                          labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        F = torch.zeros((), device=traj[0][0].device)
        L = len(self.layers)
        T = len(traj)
        for t_idx, states_t in enumerate(traj):
            for l in range(L):
                x_l = states_t[l]
                if l < L - 1:
                    mu_l = self.layers[l].predict(states_t[l + 1])
                else:
                    if labels is not None and t_idx == T - 1:
                        mu_l = self.class_prior[labels]
                    else:
                        mu_l = torch.zeros_like(x_l)
                F = F + self.layers[l].free_energy(x_l, mu_l).mean()
        return F

    def total_free_energy_phi(self, traj: List[List[torch.Tensor]],
                              labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum of per-layer Phi (Fix1). Same top-layer prior convention as F."""
        Phi = torch.zeros((), device=traj[0][0].device)
        L = len(self.layers); T = len(traj)
        for t_idx, states_t in enumerate(traj):
            for l in range(L):
                x_l = states_t[l]
                if l < L - 1:
                    mu_l = self.layers[l].predict(states_t[l + 1])
                else:
                    if labels is not None and t_idx == T - 1:
                        mu_l = self.class_prior[labels]
                    else:
                        mu_l = torch.zeros_like(x_l)
                Phi = Phi + self.layers[l].free_energy_phi(x_l, mu_l).mean()
        return Phi

    # ---- Theorem 3 diagnostics ----

    def per_layer_critical_beta(self) -> List[float]:
        return [layer.critical_beta() for layer in self.layers]

    @torch.no_grad()
    def project_spectral(self) -> None:
        """Hard-cap every layer's rho(W_rec) to self.rho_max. Call after each
        optimizer step when training the pure generative F (D1)."""
        for layer in self.layers:
            layer.project_spectral(self.rho_max)

    def critical_beta(self) -> float:
        """Network critical beta: the SMALLEST 1/rho(W_rec) across layers — the
        layer that goes critical first as beta increases governs the onset."""
        return min(self.per_layer_critical_beta())
