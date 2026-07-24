"""D-RC3: rho(W_s) degeneracy without project_spectral (confirms RC3).

Trains a RecurrentVPSCNet with pure generative F, NO hard spectral cap
(project_spectral disabled, lam_spec=0). The Ising interaction -½ m^T W_s m
is unbounded below; the mean-field entropy vanishes at saturation, so nothing
stops rho(W_s) -> infinity (degenerate minimum). Confirms the hard cap is a
band-aid for a missing weight-space barrier.
"""
import argparse
import torch
import matplotlib.pyplot as plt
from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square
from experiments.ccpa import diag_common


def train_no_cap(net, epochs, lr, T, n_in, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, 32, n_in, generator=g)
    y = torch.randint(0, net.readout.out_features, (32,), generator=g)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rhos = []
    for _ in range(epochs):
        opt.zero_grad()
        out = net(x)
        F = net.total_free_energy(out["traj"], labels=y)
        F.backward(); opt.step()
        # NOTE: deliberately NOT calling net.project_spectral()
        rhos.append(max(spectral_radius_square(_sym(l.W_rec.data)) for l in net.layers))
    return rhos


def main(seed=0):
    net = RecurrentVPSCNet([16, 16], n_classes=4, beta=0.5, rec_rho0=0.6, lam_spec=0.0)
    rhos = train_no_cap(net, epochs=60, lr=0.03, T=16, n_in=16, seed=seed)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(rhos)), rhos, "o-")
    ax.axhline(0.9, ls="--", c="r", label="rho_max=0.9")
    ax.set_xlabel("epoch"); ax.set_ylabel("rho(W_s)"); ax.legend()
    ax.set_title("RC3: rho->inf without project_spectral (entropy vanishes at saturation)")
    j, p, sha = diag_common.save("d_rc3_rho_degeneracy",
        {"seed": seed, "rhos": rhos}, fig)
    print(f"saved {j} {p} sha={sha[:12]} degeneracy={rhos[-1] > 0.9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
