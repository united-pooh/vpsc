"""Pivot 1: re-run 1+6 vs B vs pure-F on a DEEP net (4 recurrent layers) + long
sequence (T=64), where the SNN dynamics (not the readout) are the bottleneck.
If split-beta is non-redundant anywhere, it is here: the CE gradient must traverse
4 saturated tanh layers backward, so the readout-bypass may be insufficient and
the split-beta linearization may add value.
"""
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from experiments.ccpa.val_split_ce import train_split_ce_crit
from experiments.ccpa.val_fix3_ablation import train_ce_phi, acc_readout
from experiments.ccpa.val_shd_ccpa_vs_puref import train_pure_f, acc
from vpsc.recurrent import RecurrentVPSCNet


def load_deep(seed, T=64, n_in=32, C=20, n=500):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, T, n_in); y = torch.randint(0, C, (n,), generator=g)
    for i in range(n):
        t0 = 1 + int(y[i].item()) * (T // C)  # class = burst position across T
        ch = torch.randperm(n_in, generator=g)[: n_in // 4]
        x[i, t0:t0 + 2, ch] = 1.0
    x += 0.08 * torch.randn(n, T, n_in, generator=g)
    return x, y, C, n_in


def main(seeds, epochs, lr, T, width, depth):
    rows = []
    for s in seeds:
        x, y, C, n_in = load_deep(s, T=T, n_in=width)
        chance = 1.0 / C
        sizes = [n_in] + [width] * depth
        res = {"seed": s, "chance": chance, "depth": depth, "T": T}
        torch.manual_seed(s); pf = RecurrentVPSCNet(sizes, n_classes=C, rec_rho0=0.6, beta=0.2)
        train_pure_f(pf, x, y, epochs, lr); res["pure_F"] = acc(pf, x, y, False)
        torch.manual_seed(s); b = RecurrentVPSCNet(sizes, n_classes=C, rec_rho0=0.6, beta=0.2)
        train_ce_phi(b, x, y, epochs, lr); res["B_ce_phi"] = acc_readout(b, x, y)
        torch.manual_seed(s); sp = RecurrentVPSCNet(sizes, n_classes=C, rec_rho0=0.6, beta=0.2)
        train_split_ce_crit(sp, x, y, epochs, lr); res["1p6_split_ce"] = acc_readout(sp, x, y)
        rows.append(res)
    from scipy.stats import ttest_rel
    summary = {}
    for v in ["pure_F", "B_ce_phi", "1p6_split_ce"]:
        arr = np.array([r[v] for r in rows])
        b_arr = np.array([r["B_ce_phi"] for r in rows])
        p_vs_B = float(ttest_rel(arr, b_arr).pvalue) if v != "B_ce_phi" else 1.0
        summary[v] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "p_vs_B": p_vs_B, "gt_2x_chance": bool(arr.mean() > 2 * rows[0]["chance"])}
    payload = {"rows": rows, "summary": summary, "chance": rows[0]["chance"],
               "config": {"depth": depth, "T": T, "width": width, "epochs": epochs}}
    print(json.dumps(payload, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(summary.keys()); means = [summary[n]["mean"] for n in names]
    ax.bar(names, means); ax.axhline(rows[0]["chance"], ls="--", c="r", label="chance")
    ax.set_ylabel("acc"); ax.legend(); ax.set_title(f"Pivot1 deep net (depth={depth}, T={T}): 1+6 vs B vs pure-F")
    j, pth, sha = diag_common.save("val_split_ce_deep", payload, fig)
    print(f"saved {j} {pth} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    a = ap.parse_args()
    main(tuple(a.seeds), a.epochs, 0.03, a.T, a.width, a.depth)
