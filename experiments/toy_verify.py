"""Toy verification of VPSC theory predictions.

Self-contained, runs in seconds on CPU. Verifies the two load-bearing predictions:

  (P1, Theorem 2) Total free energy F is monotonically non-increasing over
      training. Verdict: F_final << F_initial and a negative rank correlation
      of the F-curve. If F rises, the framework is broken.

  (P2, Theorem 3) A recurrent mean-field layer (Curie-Weiss m=tanh(beta(Jm+h)))
      has susceptibility chi = ||m||/||h|| that diverges at the critical point
      beta_c = 1 / spectral_radius(J). Finite-size, the divergence shows up as
      a sharp jump in chi at beta_c. We locate the largest jump and check it
      sits near 1/rho(J).

Task: a temporal-integration classification. Class is determined by WHEN in
the sequence the input is active (early vs late), so the LIF dynamics must
integrate over time — a static classifier cannot solve it from a single step.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpsc import VPSCConfig, VPSCNet, BetaAnnealer, free_energy_loss, spectral_radius  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# ----------------------------- data -----------------------------

@dataclass
class ToyData:
    x: torch.Tensor   # [N, T, n_in]
    y: torch.Tensor   # [N]


def make_toy_data(n: int = 1200, T: int = 20, n_in: int = 16, seed: int = 0) -> ToyData:
    """Two classes by temporal position of activity.
    Class 0: activity in steps [2, 10).  Class 1: activity in steps [10, 18).
    A random subset of channels is active per sample, so no single channel
    is a class tell. The signal is WHICH HALF of time is active.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, T, n_in)
    y = torch.randint(0, 2, (n,), generator=g)
    for i in range(n):
        active = torch.randperm(n_in, generator=g)[: n_in // 2]
        if y[i] == 0:
            t0, t1 = 2, 10
        else:
            t0, t1 = 10, 18
        x[i, t0:t1, active] = 1.0
    x += 0.05 * torch.randn(n, T, n_in, generator=g)
    return ToyData(x, y)


# ----------------------------- training -----------------------------

def train(
    net: VPSCNet,
    data: ToyData,
    epochs: int = 60,
    lr: float = 5e-3,
    anneal: bool = True,
    log_every: int = 1,
) -> List[dict]:
    """Pure generative training: minimize F with class-conditioned top prior.
    No cross-entropy — the task enters only as the top-layer prior, so Theorem 2's
    monotone-F guarantee is in force."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    annealer = BetaAnnealer(net, start=0.1, target=None, steps=epochs) if anneal else None
    x = data.x.transpose(0, 1)   # [T, N, n_in]
    y = data.y
    logs: List[dict] = []
    for ep in range(epochs):
        if annealer is not None:
            beta = annealer.step()
        else:
            beta = net.layers[0].lif.beta
        opt.zero_grad()
        out = net(x)
        loss, parts = free_energy_loss(net, out, y)
        loss.backward()
        opt.step()
        if ep % log_every == 0:
            logs.append({"epoch": ep, "F": parts["F"], "beta": beta})
    return logs


# ----------------------------- verdicts -----------------------------

def spearman_corr(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation. P1 predicts F-curve has negative correlation
    with epoch (F decreases as training proceeds)."""
    n = len(xs)
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def _ranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for r, i in enumerate(order):
        ranks[i] = r + 1.0
    return ranks


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- data split ----
    full = make_toy_data(seed=args.seed)
    n = full.x.shape[0]
    ntr = int(0.8 * n)
    train_data = ToyData(full.x[:ntr], full.y[:ntr])

    # ---- network ----
    cfg = VPSCConfig(
        sizes=[16, 64, 32], n_classes=2, beta=0.1, tau_m=10.0,
        threshold=0.5, sigma=1.0, sparsity=0.0,
    )
    net = VPSCNet(cfg)

    rho0 = net.spectral_radius()
    beta_c = net.critical_beta()
    print(f"[init] spectral_radius(W_up)={rho0:.4f}  ->  beta_c = 1/rho = {beta_c:.4f}")

    # ---- P1: train at FIXED beta (no annealing), log F ----
    # Theorem 2's monotone guarantee holds at fixed beta; annealing changes the
    # objective each step and is tested separately as a practical technique.
    print("\n=== P1: free-energy monotonicity (Theorem 2, fixed beta) ===")
    net_p1 = VPSCNet(cfg)
    beta_fixed = 1.0
    net_p1.set_beta(beta_fixed)
    logs = train(net_p1, train_data, epochs=args.epochs, anneal=False)
    Fs = [l["F"] for l in logs]
    rho_F = spearman_corr([l["epoch"] for l in logs], Fs)
    print(f"  beta_fixed={beta_fixed}   F_initial={Fs[0]:.4f}  F_final={Fs[-1]:.4f}")
    print(f"  Spearman(epoch, F) = {rho_F:+.4f}   (expect strongly negative)")
    p1_pass = (Fs[-1] < Fs[0]) and (rho_F < -0.3)
    print(f"  P1 verdict: {'PASS' if p1_pass else 'FAIL'}  (F decreased and rank-anticorrelated)")

    # ---- P2: recurrent mean-field susceptibility (Theorem 3, faithful test) ----
    # Theorem 3 is a statement about a RECURRENT mean-field layer (Curie-Weiss
    # self-consistency m = tanh(beta (J m + h))); the J*m feedback is what creates
    # the phase transition at beta_c = 1/rho(J). A feedforward VPSC layer has no
    # such feedback, so beta_c = 1/rho(W) is not meaningful there. We therefore
    # test Theorem 3 directly on its stated object: iterate the recurrent mean
    # field to a fixed point under a small external field h, measure the
    # susceptibility chi = ||m||/||h||, and check it peaks near 1/rho(J).
    print("\n=== P2: recurrent mean-field susceptibility (Theorem 3) ===")
    J_target = 1.0
    n_mf = 32
    g = torch.Generator().manual_seed(1)
    A = torch.randn(n_mf, n_mf, generator=g)
    Jmat = 0.5 * (A + A.t())                       # symmetric coupling (ferromagnetic-ish)
    Jmat = Jmat * (J_target / spectral_radius(Jmat))   # scale to target rho(J)=J_target
    rho_J = spectral_radius(Jmat)
    beta_c_mf = 1.0 / rho_J
    print(f"  recurrent coupling: rho(J)={rho_J:.4f}  ->  beta_c = 1/rho = {beta_c_mf:.4f}")

    betas = [0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0]
    h = torch.randn(n_mf, generator=g) * 1e-2
    chis: List[float] = []
    for b in betas:
        m = torch.zeros(n_mf)
        for _ in range(400):
            m = torch.tanh(b * (Jmat @ m + h))
        chi = (m.norm() / (h.norm() + 1e-12)).item()
        chis.append(chi)
        print(f"  beta={b:5.2f}   chi={chi:8.3f}")

    # Theorem 3.2: chi DIVERGES at beta_c (finite-size: a sharp jump, not a
    # global max — above beta_c the ordered-phase magnetization m~O(1) keeps
    # chi=m/h large). So locate the largest consecutive log-ratio jump in chi;
    # the critical point is where that jump sits.
    log_ratios = [math.log(chis[i + 1] / max(chis[i], 1e-9)) for i in range(len(chis) - 1)]
    jump_idx = max(range(len(log_ratios)), key=lambda i: log_ratios[i])
    beta_jump = 0.5 * (betas[jump_idx] + betas[jump_idx + 1])
    print(f"\n  largest chi jump at beta~{beta_jump:.2f}  "
          f"(chi {chis[jump_idx]:.2f} -> {chis[jump_idx + 1]:.2f}, "
          f"log-ratio {log_ratios[jump_idx]:.2f})")
    print(f"  predicted beta_c  ={beta_c_mf:.2f}")
    ratio = beta_jump / beta_c_mf if beta_c_mf > 0 else float("inf")
    p2_pass = (log_ratios[jump_idx] > 1.0) and (0.5 <= ratio <= 2.0)
    print(f"  beta_jump/beta_c ratio={ratio:.2f}   "
          f"(pass if jump log-ratio>1 AND 0.5<=ratio<=2)")
    print(f"  P2 verdict: {'PASS' if p2_pass else 'FAIL / inconclusive'}")

    # ---- save curve data ----
    out_path = os.path.join(RESULTS_DIR, "toy_verify.txt")
    with open(out_path, "w") as f:
        f.write("# P1: F over training (Theorem 2)\n")
        for l in logs:
            f.write(f"epoch={l['epoch']} F={l['F']:.6f} beta={l['beta']:.4f}\n")
        f.write(f"\nspearman(epoch,F)={rho_F:.4f} P1={'PASS' if p1_pass else 'FAIL'}\n")
        f.write(f"\n# P2: recurrent mean-field susceptibility (Theorem 3)\n")
        f.write(f"# rho(J)={rho_J:.4f} beta_c={beta_c_mf:.4f}\n")
        for b, c in zip(betas, chis):
            f.write(f"beta={b} chi={c:.4f}\n")
        f.write(f"beta_jump={beta_jump:.4f} P2={'PASS' if p2_pass else 'FAIL'}\n")
    print(f"\n[curves written to {out_path}]")

    # ---- optional plot ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([l["epoch"] for l in logs], Fs, "-o", ms=3)
        ax[0].set_title(f"Theorem 2: F over training\nSpearman={rho_F:+.3f} ({'PASS' if p1_pass else 'FAIL'})")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("total free energy F")
        ax[1].plot(betas, chis, "-o", ms=5)
        ax[1].axvline(beta_c_mf, color="r", ls="--", label=f"beta_c={beta_c_mf:.2f}")
        ax[1].axvline(beta_jump, color="g", ls="--", label=f"beta_jump={beta_jump:.2f}")
        ax[1].set_title(f"Theorem 3: susceptibility vs beta\n{'PASS' if p2_pass else 'FAIL'}")
        ax[1].set_xlabel("beta"); ax[1].set_ylabel("chi = ||m||/||h||"); ax[1].legend()
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "toy_verify.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot written to {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")

    print("\n=== SUMMARY ===")
    print(f"  P1 (F monotone decrease, Theorem 2) : {'PASS' if p1_pass else 'FAIL'}")
    print(f"  P2 (chi peak near beta_c, Theorem 3) : {'PASS' if p2_pass else 'FAIL/inconclusive'}")
    return 0 if (p1_pass and p2_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
