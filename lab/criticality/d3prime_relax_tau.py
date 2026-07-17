"""D3': relaxation-scale reverberation plateau vs exact criticality.

Pre-registered in dev/LOG.md (2026-07-17, "D3′ 预注册 — 松弛尺度的回响平台 vs 精确临界").
D3 round 1 showed cross-timestep reverberation does not exist in the leak=1.0
architecture (tau == 1 everywhere, control identical) — the only operable
"reverberation" left is the contraction of the fixed-point relaxation map
itself (critical slowing down).  D3' replaces the tau instrument with that
relaxation-scale decay and reuses every other channel, criterion and
tolerance of D3 round 1 verbatim.

New instrument tau_relax: at each (seed, beta*rho), take the top-layer state
m0 and input I of the final timestep of a normal forward pass; perturb m0 by
delta (norm 0.01, 8 fixed random directions); iterate the layer's own
relaxation map m <- tanh(beta (m Ws + I - theta)) on base and perturbed
states for 8 iterations; tau_relax = first k with d_k/d_0 < 1/e (else 8),
averaged over directions and all 300 test samples.  Instrument check:
tau_relax(W_rec = 0) must be 1.  Q2's Spearman now acts on tau_relax, with
an explicit near-zero-variance guard (constant series => "not measurable",
not a pass).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

from experiments.deep_critical import spearman  # noqa: E402
from d3_reverberation import (  # noqa: E402
    BETA_RHOS,
    DELTA_NORM,
    H_SUSC,
    LABEL_REPS,
    Q1_ACC_TOL,
    Q1_MI_TOL,
    Q2_MIN_SPEARMAN,
    Q3_ACC_TOL,
    Q3_MI_TOL,
    SEEDS,
    SUBCRITICAL,
    accuracy_from_logits,
    forward_top,
    mi_corrected,
    train_net,
)

RELAX_ITERS = 8
RELAX_DIRS = 8
RELAX_DELTA = 0.01
VAR_GUARD = 1e-12


@torch.no_grad()
def relaxation_tau(
    net,
    x_test: torch.Tensor,
    beta: float,
    dir_seed: int,
) -> tuple[float, list[float]]:
    """Critical-slowing tau of the top layer's fixed-point relaxation map."""
    net.set_beta(beta)
    top = net.layers[-1]
    # manual forward to capture the top layer's input at the final timestep
    T, B, _ = x_test.shape
    net.reset_state(B, x_test.device)
    x_lower_final = None
    x_top_final = None
    for t in range(T):
        x = x_test[t]
        for li, layer in enumerate(net.layers):
            if li == len(net.layers) - 1:
                x_lower_final = x
            m = layer(x)
            x = m
        x_top_final = x
    Ws = 0.5 * (top.W_rec.data + top.W_rec.data.t())
    I = x_lower_final @ top.W_up
    m0 = x_top_final

    rng = torch.Generator().manual_seed(dir_seed)
    ratios = np.zeros(RELAX_ITERS)
    for _ in range(RELAX_DIRS):
        d = torch.randn(m0.shape[1], generator=rng)
        d = d / d.norm() * RELAX_DELTA
        mb = m0.clone()
        mp = m0 + d
        for k in range(RELAX_ITERS):
            mb = torch.tanh(beta * (mb @ Ws + I - top.threshold))
            mp = torch.tanh(beta * (mp @ Ws + I - top.threshold))
            ratios[k] += float((mp - mb).norm(dim=-1).mean().item())
    ratios = (ratios / RELAX_DIRS) / RELAX_DELTA
    tau = RELAX_ITERS
    for k, r in enumerate(ratios, start=1):
        if r < 1.0 / np.e:
            tau = k
            break
    return float(tau), [float(r) for r in ratios]


@torch.no_grad()
def evaluate_point_prime(net, beta, x_test, y_test, h_field, mi_seed, dir_seed) -> dict:
    """D3 channels minus cross-timestep tau/drop, plus relaxation tau."""
    net.set_beta(beta)
    logits0, tops0 = forward_top(net, x_test)
    acc = accuracy_from_logits(logits0, y_test)
    x_top = tops0[-1]
    bits = (x_top > 0).cpu().numpy().astype(np.int64)
    labels = y_test.cpu().numpy()
    mi_corr, mi_raw, mi_null_max = mi_corrected(labels, bits, 4, mi_seed)
    _, tops_h = forward_top(net, x_test, h_field=h_field)
    chi = float(((tops_h[-1] - x_top).norm(dim=-1)).mean().item()) / H_SUSC
    tau, decay = relaxation_tau(net, x_test, beta, dir_seed)
    return {
        "beta": float(beta),
        "acc": acc,
        "mi_corr": float(mi_corr),
        "mi_raw": float(mi_raw),
        "mi_null_max": float(mi_null_max),
        "susceptibility": chi,
        "tau_relax": tau,
        "relax_decay": decay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== D3': relaxation-scale reverberation vs exact criticality ===")
    per_seed_results = []
    for seed in SEEDS:
        net, test_data = train_net(seed, epochs=args.epochs)
        beta_c = net.critical_beta()
        print(f"\nseed={seed} beta_c={beta_c:.4f}")
        x_test = test_data.x.transpose(0, 1)
        y_test = test_data.y
        g = torch.Generator().manual_seed(seed * 1000 + 1)
        u = torch.randn(24, generator=g)
        h_field = H_SUSC * (u / u.norm())

        points = []
        for br in BETA_RHOS:
            metrics = evaluate_point_prime(
                net, br * beta_c, x_test, y_test, h_field, seed * 100 + 7,
                seed * 300 + 11,
            )
            metrics["beta_rho"] = br
            points.append(metrics)
            print(
                f"  br={br:.2f} acc={metrics['acc']:.3f} MI={metrics['mi_corr']:.3f} "
                f"chi={metrics['susceptibility']:.3f} tau_relax={metrics['tau_relax']:.2f}"
            )

        saved = [layer.W_rec.data.clone() for layer in net.layers]
        for layer in net.layers:
            layer.W_rec.data.zero_()
        tau_ctrl, _ = relaxation_tau(net, x_test, beta_c, seed * 300 + 11)
        # relaxation_tau leaves the net at beta = beta_c
        logits_ctrl, _ = forward_top(net, x_test)
        acc_ctrl = accuracy_from_logits(logits_ctrl, y_test)
        for layer, w in zip(net.layers, saved):
            layer.W_rec.data.copy_(w)
        print(f"  control W_rec=0 @br=1.00: tau_relax={tau_ctrl:.2f} acc={acc_ctrl:.3f}")

        accs = [p["acc"] for p in points]
        mis = [p["mi_corr"] for p in points]
        max_acc, max_mi = max(accs), max(mis)
        sub_points = [p for p in points if p["beta_rho"] < 1.0]
        near = [
            p["beta_rho"]
            for p in sub_points
            if p["acc"] >= max_acc - Q1_ACC_TOL and p["mi_corr"] >= max_mi - Q1_MI_TOL
        ]
        near_set = set(near)
        longest = 0
        run = 0
        for br in SUBCRITICAL:
            run = run + 1 if br in near_set else 0
            longest = max(longest, run)
        q1 = longest >= 3
        taus_sub = [p["tau_relax"] for p in sub_points]
        if float(np.var(taus_sub)) < VAR_GUARD:
            q2_spearman = None
            q2 = False  # "not measurable" per pre-registration, not a pass
        else:
            q2_spearman = spearman([p["beta_rho"] for p in sub_points], taus_sub)
            q2 = q2_spearman >= Q2_MIN_SPEARMAN
        crit = points[BETA_RHOS.index(1.00)]
        max_sub_acc = max(p["acc"] for p in sub_points)
        max_sub_mi = max(p["mi_corr"] for p in sub_points)
        q3 = bool(
            crit["acc"] <= max_sub_acc + Q3_ACC_TOL
            and crit["mi_corr"] <= max_sub_mi + Q3_MI_TOL
        )
        strict_dominance = bool(
            crit["acc"] > max_sub_acc + Q3_ACC_TOL
            and crit["mi_corr"] > max_sub_mi + Q3_MI_TOL
        )
        chi_spearman = spearman(BETA_RHOS, [p["susceptibility"] for p in points])
        per_seed_results.append(
            {
                "seed": seed,
                "beta_c": beta_c,
                "points": points,
                "no_rec_control": {"tau_relax": tau_ctrl, "acc": acc_ctrl},
                "q1_plateau": bool(q1),
                "q1_plateau_points": near,
                "q2_reverberation": bool(q2),
                "q2_spearman": q2_spearman,
                "q3_not_dominant": q3,
                "strict_critical_dominance": strict_dominance,
                "chi_spearman": float(chi_spearman),
                "seed_pass": bool(q1 and q2 and q3),
            }
        )
        print(
            f"  Q1={q1} (plateau {near}) Q2={q2} (rho={q2_spearman}) "
            f"Q3={q3} strict_dom={strict_dominance}"
        )

    n_pass = sum(int(r["seed_pass"]) for r in per_seed_results)
    n_reject = sum(
        int(r["strict_critical_dominance"] or not r["q1_plateau"])
        for r in per_seed_results
    )
    if n_pass >= 2:
        verdict = "ADOPT"
    elif n_reject >= 2:
        verdict = "REJECT"
    else:
        verdict = "MIXED"
    print(f"\n  seeds passing Q1^Q2^Q3: {n_pass}/3; reject-rule seeds: {n_reject}/3")
    print(f"  D3' verdict: {verdict}")

    output = {
        "experiment": "D3' relaxation-scale reverberation vs exact criticality",
        "formalization_version": "D3prime-v0.1",
        "protocol": {
            "beta_rhos": BETA_RHOS,
            "seeds": SEEDS,
            "epochs": args.epochs,
            "relax_iters": RELAX_ITERS,
            "relax_dirs": RELAX_DIRS,
            "relax_delta": RELAX_DELTA,
            "h_susc": H_SUSC,
            "label_reps": LABEL_REPS,
            "var_guard": VAR_GUARD,
            "tolerances": {
                "q1_acc": Q1_ACC_TOL,
                "q1_mi": Q1_MI_TOL,
                "q3_acc": Q3_ACC_TOL,
                "q3_mi": Q3_MI_TOL,
                "q2_spearman": Q2_MIN_SPEARMAN,
            },
        },
        "per_seed": per_seed_results,
        "verdict": {"n_pass": n_pass, "n_reject": n_reject, "d3prime": verdict},
    }
    json_path = os.path.join(RESULTS_DIR, "d3prime_relax_tau.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def plot_metric(ax, key, title, ylabel):
        for r in per_seed_results:
            xs = [p["beta_rho"] for p in r["points"]]
            ys = [p[key] for p in r["points"]]
            ax.plot(xs, ys, "-o", ms=4, alpha=0.8, label=f"seed {r['seed']}")
        ax.axvline(1.0, color="black", ls="--", lw=0.8)
        ax.set(title=title, xlabel="beta*rho", ylabel=ylabel)
        ax.legend(fontsize=8)

    plot_metric(axes[0, 0], "acc", "task accuracy", "acc")
    plot_metric(axes[0, 1], "mi_corr", "label <-> state MI (corrected)", "bits")
    plot_metric(axes[1, 0], "tau_relax", "relaxation tau (critical slowing)", "iterations")
    plot_metric(axes[1, 1], "susceptibility", "susceptibility chi", "chi")
    for r in per_seed_results:
        axes[1, 0].axhline(
            r["no_rec_control"]["tau_relax"], color="gray", ls=":", lw=0.8
        )
    fig.suptitle(f"D3': {verdict}", fontsize=14)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "d3prime_relax_tau.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
