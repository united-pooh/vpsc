"""Deep-network extension of the VPSC theory tests.

The feedforward VPSC (network.py) made Theorem 3's beta_c = 1/rho(W) meaningless
inside the network — verified only on an isolated recurrent layer (toy_verify P2).
Here each layer is a RECURRENT mean-field layer (recurrent.py), so beta_c is
defined per layer and for the whole network.

Two predictions tested in the DEEP setting:

  (D1, Theorem 2 deep) Under the pure generative free energy F (now including
      the Ising interaction term), F is still monotonically non-increasing over
      training at fixed beta. CAVEAT: the interaction term makes F_l non-convex
      in x_l, so this is a genuine empirical test of whether the monotone-F
      structure survives adding recurrence — not a foregone conclusion.

  (D2, Theorem 3 deep) For a trained deep recurrent VPSC net (frozen weights),
      task accuracy vs beta peaks near the network critical beta
      beta_c = min_l 1/rho(W_rec_l). This is the real claim: the criticality
      peak survives stacking recurrent layers into a deep network.

Task: a harder 4-class temporal problem so accuracy is not saturated (a peak is
visible rather than a flat 100%).
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpsc.recurrent import RecurrentVPSCNet  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


@dataclass
class ToyData:
    x: torch.Tensor
    y: torch.Tensor


def make_data(n: int = 1500, T: int = 24, n_in: int = 12, seed: int = 0) -> ToyData:
    """4 classes by WHERE a brief activity burst sits in time (4 quarters).
    A random channel subset is active per sample; the tell is the burst's time
    bin, which the recurrent dynamics must localize."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, T, n_in)
    y = torch.randint(0, 4, (n,), generator=g)
    for i in range(n):
        active = torch.randperm(n_in, generator=g)[: n_in // 3]
        q = y[i].item()
        t0 = 1 + q * (T // 4)
        t1 = t0 + max(2, T // 8)
        x[i, t0:t1, active] = 1.0
    x += 0.08 * torch.randn(n, T, n_in, generator=g)
    return ToyData(x, y)


def train_F(net, data, epochs, lr, beta):
    """Pure generative F (Theorem 2 deep). Fixed beta. Hard spectral projection
    after each step keeps rho(W_rec) <= rho_max so the model stays well-defined
    and F bounded below (without it, the Ising interaction drives rho->inf)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.set_beta(beta)
    x = data.x.transpose(0, 1)
    y = data.y
    logs = []
    for ep in range(epochs):
        opt.zero_grad()
        out = net(x)
        F = net.total_free_energy(out["traj"], labels=y)
        F.backward()
        opt.step()
        net.project_spectral()
        logs.append({"epoch": ep, "F": float(F.detach().item())})
    return logs


def train_CE(net, data, epochs, lr, beta, lam_F=0.0):
    """CE training for D2 (need discriminative features for a meaningful
    accuracy-vs-beta sweep). Theorem 3 concerns inference-time beta given fixed
    weights, so the training rule is immaterial."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.set_beta(beta)
    x = data.x.transpose(0, 1)
    y = data.y
    for _ in range(epochs):
        opt.zero_grad()
        out = net(x)
        ce = Fnn.cross_entropy(out["logits"], y)
        loss = ce
        if lam_F > 0:
            loss = ce + lam_F * net.total_free_energy(out["traj"], labels=y)
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def accuracy(net, data) -> float:
    net.eval()
    x = data.x.transpose(0, 1)
    out = net(x)
    pred = out["logits"].argmax(dim=-1)
    acc = (pred == data.y).float().mean().item()
    net.train()
    return acc


def spearman(xs, ys):
    n = len(xs)
    rx = [sorted(range(n), key=lambda i: xs[i]).index(i) + 1 for i in range(n)]
    ry = [sorted(range(n), key=lambda i: ys[i]).index(i) + 1 for i in range(n)]
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full = make_data(seed=args.seed)
    n = full.x.shape[0]
    ntr = int(0.8 * n)
    train_data = ToyData(full.x[:ntr], full.y[:ntr])
    test_data = ToyData(full.x[ntr:], full.y[ntr:])
    n_classes = 4

    sizes = [12, 40, 24]
    common = dict(sizes=sizes, n_classes=n_classes, threshold=0.0, sigma=1.0,
                  n_relax=8, rec_rho0=0.7, wd=1e-4, seed=args.seed)

    # ---- D1: deep F monotonicity (Theorem 2 + recurrence) ----
    print("=== D1: deep free-energy monotonicity (Theorem 2 + recurrence) ===")
    net1 = RecurrentVPSCNet(beta=1.0, **common)
    per_layer = net1.per_layer_critical_beta()
    print(f"  per-layer beta_c (init): {[f'{b:.2f}' for b in per_layer]}")
    print(f"  network beta_c (init)  : {net1.critical_beta():.3f}")
    logs = train_F(net1, train_data, epochs=args.epochs, lr=3e-3, beta=1.0)
    Fs = [l["F"] for l in logs]
    rho_F = spearman([l["epoch"] for l in logs], Fs)
    print(f"  F_initial={Fs[0]:.3f}  F_final={Fs[-1]:.3f}  Spearman(epoch,F)={rho_F:+.3f}")
    per_layer_trained = net1.per_layer_critical_beta()
    trained_rhos = [1.0 / b for b in per_layer_trained if b > 0]
    rho_bounded = all(r <= net1.rho_max * 1.05 for r in trained_rhos) if trained_rhos else False
    print(f"  per-layer beta_c (trained): {[f'{b:.2f}' for b in per_layer_trained]}")
    print(f"  trained rho(W_rec): {[f'{r:.2f}' for r in trained_rhos]}  "
          f"(cap={net1.rho_max}; bounded={rho_bounded})")
    # D1 is a genuine PASS only if F decreases AND rho stays bounded. An
    # unbounded rho makes F -> -inf trivially (degenerate), not a real result.
    d1_pass = (Fs[-1] < Fs[0]) and (rho_F < -0.3) and rho_bounded
    print(f"  D1 verdict: {'PASS' if d1_pass else 'FAIL'}  "
          f"(monotone-F survives recurrence, rho bounded)")

    # ---- D2: accuracy vs beta peak near network beta_c (Theorem 3 deep) ----
    print("\n=== D2: accuracy vs beta peak (Theorem 3, deep recurrent net) ===")
    net2 = RecurrentVPSCNet(beta=1.0, **common)
    # Train at a moderate beta (below critical) for discriminative features.
    train_CE(net2, train_data, epochs=args.epochs, lr=3e-3, beta=0.8, lam_F=0.0)
    net2.eval()
    for p in net2.parameters():
        p.requires_grad_(False)

    bc = net2.critical_beta()
    pl = net2.per_layer_critical_beta()
    print(f"  per-layer beta_c (trained): {[f'{b:.2f}' for b in pl]}")
    print(f"  network beta_c = min_l 1/rho(W_rec) = {bc:.3f}")

    betas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.8, 3.6]
    accs: List[float] = []
    for b in betas:
        net2.set_beta(b)
        a = accuracy(net2, test_data)
        accs.append(a)
        print(f"  beta={b:5.2f}   acc={a:.3f}")

    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    beta_star = betas[best_idx]
    print(f"\n  peak accuracy at beta*={beta_star:.2f}  (acc={accs[best_idx]:.3f})")
    print(f"  predicted network beta_c ={bc:.2f}")
    ratio = beta_star / bc if bc > 0 else float("inf")
    # Need a genuine interior peak (not flat-saturated). Check that accuracy
    # rises into beta* and then falls off by at least 0.05 somewhere beyond.
    rises = any(accs[i] < accs[best_idx] - 0.03 for i in range(best_idx))
    falls = any(accs[i] < accs[best_idx] - 0.05 for i in range(best_idx, len(accs)))
    interior = rises and falls
    d2_pass = interior and (0.5 <= ratio <= 2.0)
    print(f"  interior peak? {interior}   beta*/beta_c={ratio:.2f}   "
          f"(pass if interior AND 0.5<=ratio<=2)")
    print(f"  D2 verdict: {'PASS' if d2_pass else 'FAIL / inconclusive'}")

    # ---- save + plot ----
    out_path = os.path.join(RESULTS_DIR, "deep_critical.txt")
    with open(out_path, "w") as f:
        f.write("# D1: deep F over training (Theorem 2 + recurrence)\n")
        for l in logs:
            f.write(f"epoch={l['epoch']} F={l['F']:.6f}\n")
        f.write(f"\nspearman(epoch,F)={rho_F:.4f} D1={'PASS' if d1_pass else 'FAIL'}\n")
        f.write(f"\n# D2: accuracy vs beta (deep recurrent net)\n")
        f.write(f"# per-layer beta_c={pl}  network beta_c={bc:.4f}\n")
        for b, a in zip(betas, accs):
            f.write(f"beta={b} acc={a:.4f}\n")
        f.write(f"beta_star={beta_star} D2={'PASS' if d2_pass else 'FAIL'}\n")
    print(f"\n[curves written to {out_path}]")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot([l["epoch"] for l in logs], Fs, "-o", ms=3)
        ax[0].set_title(f"D1: deep F over training\nSpearman={rho_F:+.3f} ({'PASS' if d1_pass else 'FAIL'})")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("total free energy F")
        ax[1].plot(betas, accs, "-o", ms=5)
        ax[1].axvline(bc, color="r", ls="--", label=f"beta_c={bc:.2f}")
        ax[1].axvline(beta_star, color="g", ls="--", label=f"beta*={beta_star:.2f}")
        ax[1].set_title(f"D2: acc vs beta (deep)\n{'PASS' if d2_pass else 'FAIL'}")
        ax[1].set_xlabel("beta"); ax[1].set_ylabel("test accuracy"); ax[1].legend()
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "deep_critical.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot written to {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")

    print("\n=== SUMMARY ===")
    print(f"  D1 (deep F monotone, Thm 2 + recurrence) : {'PASS' if d1_pass else 'FAIL'}")
    print(f"  D2 (acc-beta peak near beta_c, Thm 3 deep): {'PASS' if d2_pass else 'FAIL/inconclusive'}")
    return 0 if (d1_pass and d2_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
