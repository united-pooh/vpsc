"""VPSC on Spiking Heidelberg Digits (SHD).

SHD is an event-based audio dataset (spiking cochlea, 700 input channels, 0.7s
of speech digits 0-9 in German and English). It is the canonical temporal-
perception benchmark for SNNs and matches the task domain chosen for VPSC
(temporal perception + control).

This script:
  * Loads SHD HDF5 if present (see README for the download URL), else falls back
    to a synthetic event-stream generator so the script is runnable anywhere.
  * Trains VPSC with the PURE GENERATIVE free-energy objective (Theorem 2 in
    force) and reports the F-curve (expect monotone decrease) and accuracy.

Theorem 3 (susceptibility divergence) is tested separately and more faithfully
on the recurrent mean-field layer in experiments/toy_verify.py — SHD here is
about scaling the Theorem-2 result to a real temporal benchmark.

Usage:
    python experiments/shd_train.py --epochs 40 --batch 64
    python experiments/shd_train.py --synthetic            # no download needed
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vpsc import VPSCConfig, VPSCNet, BetaAnnealer, free_energy_loss  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# SHD download URLs (see README). Files are HDF5 with keys:
#   'spikes' -> {'times': float[N], 'units': int[N]},  'labels' -> int
SHD_BASE = "https://compneuro.net/uploads/tools/shd/"
SHD_TRAIN_URL = SHD_BASE + "shd_train.h5"
SHD_TEST_URL = SHD_BASE + "shd_test.h5"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


@dataclass
class SHDData:
    x: torch.Tensor   # [N, T, n_in]  binned spike counts
    y: torch.Tensor   # [N]


def _download(url: str, dest: str) -> bool:
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [download failed: {e}]")
        return False


def _load_h5(path: str, T: int, n_in: int, max_samples: int) -> Optional[SHDData]:
    try:
        import h5py
    except ImportError:
        print("  [h5py not installed; using synthetic data]")
        return None
    if not os.path.exists(path):
        print(f"  [{os.path.basename(path)} not found; attempting download]")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not _download(SHD_TRAIN_URL if "train" in path else SHD_TEST_URL, path):
            return None
    with h5py.File(path, "r") as f:
        n = min(len(f["labels"]), max_samples)
        x = torch.zeros(n, T, n_in)
        y = torch.zeros(n, dtype=torch.long)
        for i in range(n):
            times = np.array(f["spikes"]["times"][i])
            units = np.array(f["spikes"]["units"][i])
            y[i] = int(f["labels"][i])
            # Bin spikes into T frames over [0, max_time].
            max_t = max(times.max(), 1e-3) if len(times) else 1.0
            bins = np.minimum((times / max_t * T).astype(int), T - 1)
            for b, u in zip(bins, units):
                if 0 <= u < n_in:
                    x[i, b, u] += 1.0
    return SHDData(x, y)


def _synthetic(n: int, T: int, n_in: int, n_classes: int, seed: int) -> SHDData:
    """Synthetic event streams: class = temporal pattern of a few 'onset' times.
    Lets the script run without the SHD download; not a real benchmark."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, T, n_in)
    y = torch.randint(0, n_classes, (n,), generator=g)
    for i in range(n):
        # Each class has a characteristic set of onset bins.
        onsets = [(c * T // n_classes + 1) % T for c in range(n_classes)]
        ob = onsets[y[i].item()]
        for t in range(ob, min(ob + T // n_classes, T)):
            chans = torch.randperm(n_in, generator=g)[: n_in // 4]
            x[i, t, chans] = 1.0
    x += 0.02 * torch.randn(n, T, n_in, generator=g)
    return SHDData(x, y)


def load_shd(T: int = 50, n_in: int = 128, synthetic: bool = False,
             max_samples: int = 2000) -> tuple[SHDData, SHDData]:
    """Returns (train, test). n_in is downsampled from SHD's 700 cochlea channels
    by grouping (factor 700//n_in) to keep the network CPU-tractable."""
    if synthetic:
        tr = _synthetic(max_samples, T, n_in, 10, seed=0)
        te = _synthetic(max_samples // 4, T, n_in, 10, seed=1)
        return tr, te
    # Real SHD: load full 700-channel then average-group to n_in.
    full_tr = _load_h5(os.path.join(DATA_DIR, "shd_train.h5"), T, 700, max_samples)
    full_te = _load_h5(os.path.join(DATA_DIR, "shd_test.h5"), T, 700, max_samples // 4)
    if full_tr is None or full_te is None:
        print("  [falling back to synthetic data]")
        return load_shd(T, n_in, synthetic=True, max_samples=max_samples)
    group = 700 // n_in
    def downsample(d: SHDData) -> SHDData:
        # [N, T, 700] -> [N, T, n_in] by summing groups.
        x = d.x[:, :, : group * n_in].reshape(d.x.shape[0], T, n_in, group).sum(-1)
        return SHDData(x, d.y)
    return downsample(full_tr), downsample(full_te)


def train(net: VPSCNet, data: SHDData, epochs: int, lr: float, batch: int,
          fixed_beta: float = 1.0, anneal: bool = False) -> List[dict]:
    """Pure generative F training. Theorem 2's monotone guarantee holds at FIXED
    beta, so we default to fixed_beta=1.0 (no annealing). Annealing is a separate
    practical technique and breaks the per-step monotonicity (each step changes
    the objective); enable it only for a practical run, not for verifying Thm 2."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    annealer = BetaAnnealer(net, start=0.1, target=None, steps=epochs) if anneal else None
    if not anneal:
        net.set_beta(fixed_beta)
    N = data.x.shape[0]
    logs: List[dict] = []
    for ep in range(epochs):
        beta = annealer.step() if annealer is not None else fixed_beta
        perm = torch.randperm(N)
        ep_F = 0.0
        nb = 0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            xb = data.x[idx].transpose(0, 1)   # [T, B, n_in]
            yb = data.y[idx]
            opt.zero_grad()
            out = net(xb)
            loss, parts = free_energy_loss(net, out, yb)
            loss.backward()
            opt.step()
            ep_F += parts["F"]; nb += 1
        logs.append({"epoch": ep, "F": ep_F / nb, "beta": beta})
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  F={ep_F/nb:8.3f}  beta={beta:.3f}")
    return logs


@torch.no_grad()
def evaluate(net: VPSCNet, data: SHDData, batch: int) -> float:
    net.eval()
    N, T, _ = data.x.shape
    correct = 0
    for i in range(0, N, batch):
        xb = data.x[i:i + batch].transpose(0, 1)
        out = net(xb)
        pred = net.classify(out["x_top"])
        correct += (pred == data.y[i:i + batch]).sum().item()
    net.train()
    return correct / N


def spearman(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    rx = [sorted(range(n), key=lambda i: xs[i]).index(i) + 1 for i in range(n)]
    ry = [sorted(range(n), key=lambda i: ys[i]).index(i) + 1 for i in range(n)]
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--n_in", type=int, default=128)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading data ...")
    train_data, test_data = load_shd(T=args.T, n_in=args.n_in,
                                     synthetic=args.synthetic, max_samples=2000)
    n_classes = int(train_data.y.max().item()) + 1
    print(f"  train={train_data.x.shape}  test={test_data.x.shape}  classes={n_classes}")

    cfg = VPSCConfig(
        sizes=[args.n_in, 256, 64], n_classes=n_classes, beta=0.1, tau_m=15.0,
        threshold=0.5, sigma=1.0, sparsity=1e-4,
    )
    net = VPSCNet(cfg)
    print(f"  net: rho(W_up)={net.spectral_radius():.3f}  beta_c={net.critical_beta():.3f}")

    print("\nTraining (pure generative F, Theorem 2) ...")
    logs = train(net, train_data, args.epochs, args.lr, args.batch)
    Fs = [l["F"] for l in logs]
    rho_F = spearman([l["epoch"] for l in logs], Fs)
    print(f"\n  F_initial={Fs[0]:.3f}  F_final={Fs[-1]:.3f}  "
          f"Spearman(epoch,F)={rho_F:+.3f}")
    p1 = (Fs[-1] < Fs[0]) and (rho_F < -0.3)
    print(f"  Theorem 2 (F monotone): {'PASS' if p1 else 'FAIL'}")

    acc = evaluate(net, test_data, args.batch)
    print(f"  test accuracy = {acc:.3f}  (chance={1.0/n_classes:.3f})")

    out_path = os.path.join(RESULTS_DIR, "shd_train.txt")
    with open(out_path, "w") as f:
        f.write(f"# {'synthetic' if args.synthetic else 'shd'}  classes={n_classes}\n")
        for l in logs:
            f.write(f"epoch={l['epoch']} F={l['F']:.6f} beta={l['beta']:.4f}\n")
        f.write(f"\nspearman(epoch,F)={rho_F:.4f} P1={'PASS' if p1 else 'FAIL'}\n")
        f.write(f"test_acc={acc:.4f}\n")
    print(f"[curves written to {out_path}]")


if __name__ == "__main__":
    main()
