"""Shared utilities for CCPA diagnostic and fix scripts.

Builds a small recurrent mean-field layer, sweeps beta, decomposes F_l into
its three components, and saves JSON+PNG+SHA-256 artifacts (repo provenance
convention). CPU small-scale only.
"""
import os, json, hashlib, sys
import matplotlib
matplotlib.use("Agg")
import torch

from vpsc.recurrent import (RecurrentMeanFieldLayer, RecurrentLayerSpec,
                            _sym, _binary_entropy)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ccpa")


def build_layer(n=16, rho=0.7, seed=0, beta=1.0, sigma=1.0, threshold=0.0, n_relax=8):
    gen = torch.Generator().manual_seed(seed)
    spec = RecurrentLayerSpec(n_in=n, n_out=n, n_next=None)
    return RecurrentMeanFieldLayer(
        spec, beta=beta, threshold=threshold, sigma=sigma, n_relax=n_relax,
        rec_rho0=rho, wd=0.0, lam_spec=0.0, rho_max=0.9, gen=gen, leak=1.0)


def _components(layer, m, mu):
    Ws = _sym(layer.W_rec)
    err = m - mu
    quad = 0.5 * (1.0 / layer.sigma ** 2) * (err ** 2).sum(dim=-1).mean().item()
    interaction = -0.5 * (m * (m @ Ws)).sum(dim=-1).mean().item()
    entropy = (1.0 / layer.beta) * _binary_entropy(m).sum(dim=-1).mean().item()
    return quad, interaction, entropy


def beta_sweep(layer, betas, x_lower, mu=None):
    """Per-beta: solve fixed point, decompose F_l. mu defaults to zeros (isolate RC1)."""
    rows = []
    for b in betas:
        layer.set_beta(b)
        m = layer(x_lower)
        if mu is None:
            mu = torch.zeros_like(m)
        quad, inter, entr = _components(layer, m, mu)
        F = quad + inter + entr
        rows.append({"beta": b, "F": F, "quad": quad, "interaction": inter,
                     "entropy": entr, "m_abs_mean": float(m.abs().mean().item())})
    return rows


def save(name, payload, fig):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jpath = os.path.join(RESULTS_DIR, name + ".json")
    ppath = os.path.join(RESULTS_DIR, name + ".png")
    payload["command"] = "python -m experiments.ccpa." + name
    payload["env"] = {"python": sys.version.split()[0], "torch": torch.__version__}
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    payload["sha256"] = hashlib.sha256(open(jpath, "rb").read()).hexdigest()
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    fig.savefig(ppath, dpi=110, bbox_inches="tight")
    return jpath, ppath, payload["sha256"]
