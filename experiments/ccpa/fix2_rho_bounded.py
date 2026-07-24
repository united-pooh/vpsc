"""Fix2 verify: rho(W_s) bounded without project_spectral via log-det barrier."""
import argparse
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square


def main(seed=0):
    torch.manual_seed(seed)
    net = RecurrentVPSCNet([16, 16], n_classes=4, beta=0.5, rec_rho0=0.6, lam_spec=0.0)
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    x = torch.randn(16, 32, 16); y = torch.randint(0, 4, (32,))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    rhos = []
    for _ in range(60):
        opt.zero_grad(); out = net(x)
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step()  # NO project_spectral
        rhos.append(max(spectral_radius_square(_sym(l.W_rec.data)) for l in net.layers))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rhos, "o-"); ax.axhline(0.9, ls="--", c="r", label="rho_max")
    ax.set_xlabel("step"); ax.set_ylabel("rho(W_s)"); ax.legend()
    ax.set_title(f"Fix2: rho bounded w/o hard cap (max={max(rhos):.3f})")
    j, p, sha = diag_common.save("fix2_rho_bounded",
        {"seed": seed, "rhos": rhos, "bounded": bool(max(rhos) <= 0.95),
         "beta_c": net.critical_beta()}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
