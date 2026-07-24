"""Fix1 verify: Phi monotone non-increasing at fixed beta (Theorem 2 on Phi)."""
import argparse
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet


def main(seed=0):
    torch.manual_seed(seed)
    net = RecurrentVPSCNet([6, 6], n_classes=4, beta=0.6, rec_rho0=0.5)
    x = torch.randn(8, 6, 6); y = torch.randint(0, 4, (6,))
    opt = torch.optim.SGD(net.parameters(), lr=0.02)
    phis = []
    for _ in range(30):
        opt.zero_grad(); out = net(x)
        Phi = net.total_free_energy_phi(out["traj"], labels=y)
        Phi.backward(); opt.step(); net.project_spectral()
        phis.append(float(Phi.item()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phis, "o-"); ax.set_xlabel("step"); ax.set_ylabel("Phi")
    ax.set_title(f"Fix1: Phi monotone at fixed beta (delta={phis[-1]-phis[0]:.3f})")
    j, p, sha = diag_common.save("fix1_phi_monotone",
        {"seed": seed, "phis": phis, "monotone": bool(phis[-1] <= phis[0] + 1e-3),
         "beta_c": net.critical_beta()}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
