"""Fix3 ablation: run all four Fix3 reformulations (A/B/C/D) vs pure-F on SHD.

A = bipolar top prior (saturation-compatible target).
B = CE + lam_F*Phi (discriminative readout; sacrifices Theorem 2).
C = soft-prior fixed-point forward (prior as external field on top layer).
D = per-hypothesis energy classify (full Phi per class, barrier off as
    class-independent). On the unclamped trajectory D reduces to nearest-prior
    classify; reported honestly.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as Fnn
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from experiments.ccpa.val_shd_ccpa_vs_puref import load_shd, train_pure_f, train_ccpa, acc
from vpsc.recurrent import RecurrentVPSCNet
from vpsc.free_energy import ContinuationAnnealer


def make_bipolar(net):
    with torch.no_grad():
        net.class_prior.copy_(torch.sign(net.class_prior))


def train_ce_phi(net, x, y, epochs, lr, lam_F=0.1):
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    ann = ContinuationAnnealer(net, start=0.2, steps=epochs)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = x.transpose(0, 1)
    for _ in range(epochs):
        opt.zero_grad(); out = net(xt)
        ce = Fnn.cross_entropy(out["logits"], y)
        phi = net.total_free_energy_phi(out["traj"], labels=y)
        (ce + lam_F * phi).backward(); opt.step(); ann.step()
    return net


def acc_readout(net, x, y):
    xt = x.transpose(0, 1); out = net(xt)
    return float((out["logits"].argmax(-1) == y).float().mean().item())


def build_top_field(net, labels, T, kappa):
    top = len(net.layers) - 1
    field = kappa * net.class_prior[labels]
    return [[(field if li == top else None) for li in range(len(net.layers))] for _ in range(T)]


def train_soft_prior(net, x, y, epochs, lr, kappa=1.0):
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    ann = ContinuationAnnealer(net, start=0.2, steps=epochs)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = x.transpose(0, 1); T = xt.shape[0]
    for _ in range(epochs):
        Iext = build_top_field(net, y, T, kappa)
        opt.zero_grad(); out = net(xt, I_ext_seq=Iext)
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step(); ann.step()
    return net


def acc_energy(net, x, y):
    """Per-hypothesis Phi energy classify (barrier off, class-independent)."""
    flags = [l.use_log_det_barrier for l in net.layers]
    for l in net.layers:
        l.use_log_det_barrier = False
    xt = x.transpose(0, 1); out = net(xt)
    traj = out["traj"]; L = len(net.layers); B = x.shape[0]; C = net.class_prior.shape[0]
    energies = torch.zeros(B, C)
    for c in range(C):
        Phi = torch.zeros(B)
        for states_t in traj:
            for l in range(L):
                x_l = states_t[l]
                mu = net.layers[l].predict(states_t[l + 1]) if l < L - 1 else \
                    net.class_prior[c].unsqueeze(0).expand_as(x_l)
                Phi = Phi + net.layers[l].free_energy_phi(x_l, mu)
        energies[:, c] = Phi.detach()
    for i, l in enumerate(net.layers):
        l.use_log_det_barrier = flags[i]
    return float((energies.argmin(-1) == y).float().mean().item())


def main(synthetic, seeds, epochs, lr):
    rows = []
    for s in seeds:
        x, y, C, n_in = load_shd(synthetic, s)
        chance = 1.0 / C
        res = {"seed": s, "chance": chance}
        torch.manual_seed(s); pf = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_pure_f(pf, x, y, epochs, lr); res["pure_F"] = acc(pf, x, y, False)

        torch.manual_seed(s); a = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        make_bipolar(a); train_ccpa(a, x, y, epochs, lr); res["A_bipolar"] = acc(a, x, y, False)

        torch.manual_seed(s); b = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_ce_phi(b, x, y, epochs, lr); res["B_ce_phi"] = acc_readout(b, x, y)

        torch.manual_seed(s); cn = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_soft_prior(cn, x, y, epochs, lr); res["C_soft_prior"] = acc(cn, x, y, False)

        torch.manual_seed(s); d = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        for l in d.layers:
            l.use_log_det_barrier = True; l.gamma = 1.0
        ann = ContinuationAnnealer(d, start=0.2, steps=epochs); opt = torch.optim.Adam(d.parameters(), lr=lr)
        xt = x.transpose(0, 1)
        for _ in range(epochs):
            opt.zero_grad(); out = d.pc_inference(xt, K=8, labels=y)
            loss = d.total_free_energy_phi(out["traj"], labels=y)
            loss.backward(); opt.step(); ann.step()
        res["D_energy"] = acc_energy(d, x, y)
        rows.append(res)

    from scipy.stats import ttest_rel
    summary = {}
    for v in ["pure_F", "A_bipolar", "B_ce_phi", "C_soft_prior", "D_energy"]:
        arr = np.array([r[v] for r in rows])
        pf = np.array([r["pure_F"] for r in rows])
        p = float(ttest_rel(arr, pf).pvalue) if v != "pure_F" else 1.0
        summary[v] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "p_vs_pureF": p, "gt_2x_chance": bool(arr.mean() > 2 * rows[0]["chance"])}
    payload = {"rows": rows, "summary": summary, "chance": rows[0]["chance"]}
    print(json.dumps(payload, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(summary.keys()); means = [summary[n]["mean"] for n in names]
    ax.bar(names, means); ax.axhline(rows[0]["chance"], ls="--", c="r", label="chance")
    ax.set_ylabel("acc"); ax.legend(); ax.set_title("Fix3 ablation (A/B/C/D vs pure-F)")
    j, pth, sha = diag_common.save("val_fix3_ablation", payload, fig)
    print(f"saved {j} {pth} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    main(a.synthetic, tuple(a.seeds), a.epochs, 0.03)
