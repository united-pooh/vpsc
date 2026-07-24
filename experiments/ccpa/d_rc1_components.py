"""D-RC1: F_l component decomposition vs beta (confirms RC1 — non-coherent homotopy).

Sweeps beta across [0.1, 1.2*beta_c] on a small recurrent layer with fixed W
(rho<1) and fixed input. Decomposes F_l into quad (prediction error),
interaction (Ising), entropy (mean-field, 1/beta scaled). Shows F is
non-monotone across beta => "F decreased" compares different objective
functions (Theorem 2 monotonicity only holds at fixed beta).
"""
import argparse
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common


def main(seed=0):
    layer = diag_common.build_layer(n=16, rho=0.7, seed=seed)
    beta_c = layer.critical_beta()
    betas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    torch.manual_seed(seed)
    x = torch.randn(64, 16)
    rows = diag_common.beta_sweep(layer, betas, x)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bs = [r["beta"] for r in rows]
    ax[0].plot(bs, [r["F"] for r in rows], "o-", label="F")
    ax[0].axvline(beta_c, ls="--", c="k", label=f"beta_c={beta_c:.2f}")
    ax[0].set_xlabel("beta"); ax[0].legend(); ax[0].set_title("F vs beta (non-monotone => RC1)")
    ax[1].plot(bs, [r["quad"] for r in rows], "o-", label="quad")
    ax[1].plot(bs, [r["interaction"] for r in rows], "s-", label="interaction")
    ax[1].plot(bs, [r["entropy"] for r in rows], "^-", label="entropy")
    ax[1].set_xlabel("beta"); ax[1].legend(); ax[1].set_title("components")
    j, p, sha = diag_common.save("d_rc1_components",
        {"seed": seed, "beta_c": beta_c, "rows": rows}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
