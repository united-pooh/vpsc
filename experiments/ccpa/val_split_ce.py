"""Direction 1+6: split-beta (forward critical / backward linear) + CE + criticality regularizer.

Ablation: 1+6 vs B (CE+Phi, no split) vs pure-F vs chance. Claim holds only if
1+6 significantly beats B (attributing the gain to split-beta/criticality, not CE).
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as Fnn
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from experiments.ccpa.val_shd_ccpa_vs_puref import load_shd, train_pure_f, acc
from experiments.ccpa.val_fix3_ablation import train_ce_phi, acc_readout
from vpsc.recurrent import RecurrentVPSCNet
from vpsc.free_energy import ContinuationAnnealer


def train_split_ce_crit(net, x, y, epochs, lr, lam=0.1, beta_grad=0.5):
    """1+6: split-beta (beta_dyn annealed, beta_grad linear) + CE + (Phi+barrier) regularizer."""
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
        l.split_beta = True; l.beta_grad = beta_grad
    ann = ContinuationAnnealer(net, start=0.2, steps=epochs)  # anneals beta_dyn -> beta_c - delta
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = x.transpose(0, 1)
    for _ in range(epochs):
        opt.zero_grad(); out = net(xt)
        ce = Fnn.cross_entropy(out["logits"], y)
        phi = net.total_free_energy_phi(out["traj"], labels=y)
        (ce + lam * phi).backward(); opt.step(); ann.step()
    return net


def main(synthetic, seeds, epochs, lr):
    rows = []
    for s in seeds:
        x, y, C, n_in = load_shd(synthetic, s)
        chance = 1.0 / C
        res = {"seed": s, "chance": chance}
        torch.manual_seed(s); pf = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_pure_f(pf, x, y, epochs, lr); res["pure_F"] = acc(pf, x, y, False)

        torch.manual_seed(s); b = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_ce_phi(b, x, y, epochs, lr); res["B_ce_phi"] = acc_readout(b, x, y)

        torch.manual_seed(s); sp = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
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
    payload = {"rows": rows, "summary": summary, "chance": rows[0]["chance"]}
    print(json.dumps(payload, indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(summary.keys()); means = [summary[n]["mean"] for n in names]
    ax.bar(names, means); ax.axhline(rows[0]["chance"], ls="--", c="r", label="chance")
    ax.set_ylabel("acc"); ax.legend(); ax.set_title("1+6 split-beta+CE vs B vs pure-F")
    j, pth, sha = diag_common.save("val_split_ce", payload, fig)
    print(f"saved {j} {pth} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    main(a.synthetic, tuple(a.seeds), a.epochs, 0.03)
