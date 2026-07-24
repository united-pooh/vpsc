"""D-RC4: fixed-point stability loss at beta_c (confirms RC4 — numerical ill-conditioning).

The fixed-point map G(m) = tanh(beta*(W_s m + I - theta)) has Jacobian
DG = beta * diag(1 - m^2) * W_s. At the paramagnetic fixed point (m~0, low beta),
rho(DG) = beta * rho(W_s) = beta / beta_c -> 1 at beta = beta_c: the fixed-point
iteration loses contraction (slow/divergent relaxation) — the Curie transition.
Past beta_c the forward fixed point saturates (m -> +/-1), diag(1-m^2) -> 0, and
the H_F Hessian eigenvalue 1/sigma^2 - (1/beta)/(1-m^2) -> -inf (the fixed point
becomes a saddle). Both signal numerical ill-conditioning at/beyond beta_c.

Jacobian is the clean, unambiguous signal (rho(DG) -> 1 at beta_c).
"""
import argparse
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import _sym, _binary_entropy


def rho_DG(layer, m, x_lower_row):
    """rho(DG) at the fixed point, single sample. DG = beta * diag(1-m^2) * Ws."""
    Ws = _sym(layer.W_rec)
    D = torch.diag(1.0 - m ** 2)
    DG = layer.beta * D @ Ws
    return float(torch.linalg.svdvals(DG)[0].item())


def lambda_min_H(layer, m_row, x_lower_row):
    """Min eigenvalue of H_F at the forward fixed point (single sample).
    Goes negative past saturation (fixed point = saddle)."""
    Ws = _sym(layer.W_rec)
    m0 = m_row.detach().clone().requires_grad_(True)

    def F_scalar(mm):
        err = mm  # mu = 0
        quad = 0.5 * (1.0 / layer.sigma ** 2) * (err ** 2).sum()
        inter = -0.5 * (mm * (mm @ Ws)).sum()
        entr = (1.0 / layer.beta) * _binary_entropy(mm).sum()
        return quad + inter + entr

    H = torch.func.hessian(F_scalar)(m0)
    return float(torch.linalg.eigvalsh(H).min().real.item())


def sweep(layer, x_lower, betas):
    rows = []
    for b in betas:
        layer.set_beta(b)
        m = layer(x_lower)
        rows.append({"beta": b,
                     "rho_DG": rho_DG(layer, m[0].detach(), x_lower[0]),
                     "lambda_min": lambda_min_H(layer, m[0], x_lower[0]),
                     "m_abs_mean": float(m.abs().mean().item())})
    return rows


def main(seed=0):
    layer = diag_common.build_layer(n=8, rho=0.7, seed=seed)
    beta_c = layer.critical_beta()
    # zero input => paramagnetic m=0 branch => rho(DG) = beta*rho(Ws) = beta/beta_c
    # clean Curie signal: rho(DG) -> 1 at beta_c. Sweep across beta_c.
    betas = [0.2, 0.5, 0.8, 1.0, 1.1, 1.2, 1.3, beta_c * 0.95, beta_c, beta_c * 1.05, 1.5]
    x = torch.zeros(4, 8)
    rows = sweep(layer, x, betas)
    beta_c = layer.critical_beta()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(betas, [r["rho_DG"] for r in rows], "s-")
    ax[0].axhline(1.0, ls="--", c="r"); ax[0].axvline(beta_c, ls="--", c="k")
    ax[0].set_xlabel("beta"); ax[0].set_ylabel("rho(DG)"); ax[0].set_title("RC4b: rho(DG)->1 at beta_c (Curie)")
    ax[1].plot(betas, [r["lambda_min"] for r in rows], "o-")
    ax[1].axvline(beta_c, ls="--", c="k")
    ax[1].set_xlabel("beta"); ax[1].set_ylabel("lambda_min(H_F)"); ax[1].set_title("RC4a: H_F -> negative (saddle) past beta_c")
    j, p, sha = diag_common.save("d_rc4_hessian_jacobian",
        {"seed": seed, "beta_c": beta_c, "rows": rows}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
