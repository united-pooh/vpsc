"""D-RC2: top-layer error floor vs beta (confirms RC2 — saturation drives prediction error up).

Sweeps beta on a recurrent layer whose top-layer target is either an orthogonal
(continuous) class prior or a bipolar (±1) coding of it. The orthogonal floor
grows with beta (m saturates toward ±1, continuous prior stays put) => the
generative fit fights discretization => non-discriminative collapse.
"""
import argparse
import torch
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common


def error_floor(layer, prior, x, betas):
    """Per-beta mean ½‖m_top − prior[label]‖² for a fixed label assignment."""
    n_classes = prior.shape[0]
    labels = torch.arange(x.shape[0]) % n_classes
    floors = []
    for b in betas:
        layer.set_beta(b)
        m = layer(x)
        mu = prior[labels]
        floors.append(float(0.5 * ((m - mu) ** 2).sum(dim=-1).mean().item()))
    return floors


def main(seed=0):
    layer = diag_common.build_layer(n=16, rho=0.7, seed=seed)
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(4, 16, generator=g)
    q, _ = torch.linalg.qr(raw.t()); ortho = q.t()
    bipolar = torch.sign(ortho)
    betas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1]
    torch.manual_seed(seed)
    x = torch.randn(64, 16)
    fo = error_floor(layer, ortho, x, betas)
    fb = error_floor(layer, bipolar, x, betas)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas, fo, "o-", label="orthogonal prior (continuous)")
    ax.plot(betas, fb, "s-", label="bipolar prior (+/-1)")
    ax.axvline(layer.critical_beta(), ls="--", c="k", label="beta_c")
    ax.set_xlabel("beta"); ax.set_ylabel("top-layer error floor"); ax.legend()
    ax.set_title("RC2: floor grows with beta for continuous prior")
    j, p, sha = diag_common.save("d_rc2_errorfloor",
        {"seed": seed, "betas": betas, "orthogonal": fo, "bipolar": fb,
         "beta_c": layer.critical_beta()}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
