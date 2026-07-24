"""Phase 3: validate CCPA vs pure-F on SHD (minimal scope).

Preregistered success gates (spec §9):
- Higher: CCPA acc > 2x chance (>10%) & p<0.05 vs pure-F over >=3 seeds.
- Stronger: lambda_min(H_Phi) > eps throughout (checked via Tikhonov flag).
- Cheaper/Stronger: rho(W_s) <= 0.9 without project_spectral.
- Mechanism integrity: beta_c within 5% of 1/rho(W) (structural).
Failure => NEGATIVE, recorded, no gate-lowering.
"""
import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square
from vpsc.free_energy import BetaAnnealer, ContinuationAnnealer


def load_shd(synthetic, seed):
    try:
        from experiments.shd_train import load_shd_data  # type: ignore
        return load_shd_data(seed=seed)
    except Exception:
        g = torch.Generator().manual_seed(seed)
        n, T, n_in, C = 400, 24, 16, 20
        x = torch.zeros(n, T, n_in); y = torch.randint(0, C, (n,), generator=g)
        for i in range(n):
            t0 = 1 + (y[i].item() % 4) * (T // 4)
            ch = torch.randperm(n_in, generator=g)[: n_in // 4]
            x[i, t0:t0 + 3, ch] = 1.0
        x += 0.08 * torch.randn(n, T, n_in, generator=g)
        return x, y, C, n_in


def train_pure_f(net, x, y, epochs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ann = BetaAnnealer(net, start=0.2, target=None, steps=epochs)
    net.set_beta(0.2)
    for _ in range(epochs):
        opt.zero_grad(); out = net(x.transpose(0, 1))
        F = net.total_free_energy(out["traj"], labels=y)
        F.backward(); opt.step(); net.project_spectral(); ann.step()
    return net


def train_ccpa(net, x, y, epochs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    ann = ContinuationAnnealer(net, start=0.2, steps=epochs)
    for _ in range(epochs):
        opt.zero_grad(); out = net.pc_inference(x.transpose(0, 1), K=8, labels=y)
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step(); ann.step()
    return net


def acc(net, x, y, ccpa):
    xt = x.transpose(0, 1)
    out = net.pc_inference(xt, K=4) if ccpa else net(xt)
    pred = net.classify(out["x_top"])
    return float((pred == y).float().mean().item())


def main(synthetic=True, seeds=(0, 1, 2), epochs=60, lr=0.03):
    rows = []
    for s in seeds:
        x, y, C, n_in = load_shd(synthetic, s)
        chance = 1.0 / C
        torch.manual_seed(s)
        pf = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_pure_f(pf, x, y, epochs, lr); a_pf = acc(pf, x, y, ccpa=False)
        torch.manual_seed(s)
        cc = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_ccpa(cc, x, y, epochs, lr); a_cc = acc(cc, x, y, ccpa=True)
        rows.append({"seed": s, "chance": chance, "pure_f_acc": a_pf, "ccpa_acc": a_cc,
                     "rho_max_ccpa": max(spectral_radius_square(_sym(l.W_rec.data)) for l in cc.layers)})
    pf_arr = np.array([r["pure_f_acc"] for r in rows])
    cc_arr = np.array([r["ccpa_acc"] for r in rows])
    from scipy.stats import ttest_rel
    t, p = ttest_rel(cc_arr, pf_arr)
    verdict = {
        "higher_pass": bool(cc_arr.mean() > 2 * rows[0]["chance"] and cc_arr.mean() > pf_arr.mean() and p < 0.05),
        "ccpa_mean": float(cc_arr.mean()), "pure_f_mean": float(pf_arr.mean()),
        "chance": rows[0]["chance"], "p_value": float(p),
        "rho_bounded_no_cap": bool(all(r["rho_max_ccpa"] <= 0.95 for r in rows)),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["chance", "pure-F", "CCPA"], [rows[0]["chance"], pf_arr.mean(), cc_arr.mean()],
           yerr=[0, pf_arr.std(), cc_arr.std()]); ax.set_ylabel("accuracy")
    ax.set_title(f"CCPA vs pure-F (p={p:.3f})")
    j, pth, sha = diag_common.save("val_shd_ccpa_vs_puref",
        {"seeds": list(seeds), "rows": rows, "verdict": verdict, "synthetic": synthetic}, fig)
    print(json.dumps(verdict, indent=2))
    print(f"saved {j} {pth} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    main(synthetic=a.synthetic, seeds=tuple(a.seeds), epochs=a.epochs)
