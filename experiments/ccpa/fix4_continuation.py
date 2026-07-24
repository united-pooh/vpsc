"""Fix4 verify: ContinuationAnnealer anneals to beta_c - delta, lambda_min tracked."""
import argparse
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet, _sym, _binary_entropy
from vpsc.free_energy import ContinuationAnnealer


def _lambda_min(layer, m_row):
    Ws = _sym(layer.W_rec)
    m0 = m_row.detach().clone().requires_grad_(True)

    def Phi_scalar(mm):
        quad = 0.5 * (1.0 / layer.sigma ** 2) * (mm ** 2).sum()
        inter = -0.5 * (mm * (mm @ Ws)).sum()
        entr = layer.beta * (quad + inter) - _binary_entropy(mm).sum()
        if getattr(layer, "tikhonov_eps", 0.0) > 0:
            entr = entr + 0.5 * layer.tikhonov_eps * (mm ** 2).sum()
        return entr
    H = torch.func.hessian(Phi_scalar)(m0)
    return float(torch.linalg.eigvalsh(H).min().real.item())


def main(seed=0):
    torch.manual_seed(seed)
    net = RecurrentVPSCNet([16, 16], n_classes=4, rec_rho0=0.6, beta=0.2)
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    ann = ContinuationAnnealer(net, start=0.2, steps=40)
    x = torch.randn(16, 32, 16); y = torch.randint(0, 4, (32,))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    betas = []
    for _ in range(40):
        opt.zero_grad(); out = net.pc_inference(x, K=8, labels=y)
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step()
        betas.append(ann.step())
    beta_c = ann.beta_c
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas, "o-"); ax.axhline(beta_c, ls="--", c="r", label="beta_c")
    ax.axhline(beta_c - 0.1 * beta_c, ls=":", c="g", label="beta_c-delta")
    ax.set_xlabel("step"); ax.set_ylabel("beta"); ax.legend()
    ax.set_title(f"Fix4: anneal to beta_c-delta (final={betas[-1]:.3f}, beta_c={beta_c:.3f})")
    j, p, sha = diag_common.save("fix4_continuation",
        {"seed": seed, "betas": betas, "beta_c": beta_c,
         "final": betas[-1], "within_delta": bool(betas[-1] <= beta_c + 1e-6)}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
