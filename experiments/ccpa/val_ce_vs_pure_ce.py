"""Repositioning test: does the VPSC package (Phi + log-det barrier + continuation)
add value over PLAIN CE-SNN? B (CE+Phi+barrier+continuation) vs plain-CE
(CE only, fixed low beta, project_spectral). If B > plain-CE significantly, the
VPSC regularizer earns its keep; if not, the whole machinery is inert in the CE regime.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as Fnn
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from experiments.ccpa.val_shd_ccpa_vs_puref import load_shd
from experiments.ccpa.val_fix3_ablation import train_ce_phi, acc_readout
from vpsc.recurrent import RecurrentVPSCNet


def train_pure_ce(net, x, y, epochs, lr, beta=0.5):
    """Vanilla CE-SNN: CE only, fixed low beta (gradient alive), project_spectral."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.set_beta(beta)
    xt = x.transpose(0, 1)
    for _ in range(epochs):
        opt.zero_grad(); out = net(xt)
        ce = Fnn.cross_entropy(out["logits"], y)
        ce.backward(); opt.step(); net.project_spectral()
    return net


def main(synthetic, seeds, epochs, lr):
    rows = []
    for s in seeds:
        x, y, C, n_in = load_shd(synthetic, s)
        chance = 1.0 / C
        res = {"seed": s, "chance": chance}
        torch.manual_seed(s); pc = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.5)
        train_pure_ce(pc, x, y, epochs, lr); res["plain_CE"] = acc_readout(pc, x, y)

        torch.manual_seed(s); b = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_ce_phi(b, x, y, epochs, lr); res["B_ce_phi_barrier"] = acc_readout(b, x, y)
        rows.append(res)
    from scipy.stats import ttest_rel
    arr_b = np.array([r["B_ce_phi_barrier"] for r in rows])
    arr_pc = np.array([r["plain_CE"] for r in rows])
    p = float(ttest_rel(arr_b, arr_pc).pvalue)
    summary = {
        "plain_CE": {"mean": float(arr_pc.mean()), "std": float(arr_pc.std())},
        "B_ce_phi_barrier": {"mean": float(arr_b.mean()), "std": float(arr_b.std())},
        "p_B_vs_plainCE": p,
        "B_gt_2x_chance": bool(arr_b.mean() > 2 * rows[0]["chance"]),
        "B_significantly_gt_plainCE": bool(arr_b.mean() > arr_pc.mean() and p < 0.05),
    }
    payload = {"rows": rows, "summary": summary, "chance": rows[0]["chance"]}
    print(json.dumps(payload, indent=2))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["plain-CE", "B(CE+Phi+barrier)"], [arr_pc.mean(), arr_b.mean()],
           yerr=[arr_pc.std(), arr_b.std()])
    ax.axhline(rows[0]["chance"], ls="--", c="r", label="chance")
    ax.set_ylabel("acc"); ax.legend(); ax.set_title(f"VPSC regularizer vs plain-CE (p={p:.3f})")
    j, pth, sha = diag_common.save("val_ce_vs_pure_ce", payload, fig)
    print(f"saved {j} {pth} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    main(a.synthetic, tuple(a.seeds), a.epochs, 0.03)
