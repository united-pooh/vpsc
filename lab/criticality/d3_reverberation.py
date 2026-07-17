"""D3: subcritical reverberating plateau vs exact criticality.

Pre-registered in dev/LOG.md (2026-07-17, "D3 预注册 — 略次临界回响态 vs 精确临界").
Wilting & Priesemann (2018) found cortical recordings match a slightly
subcritical, reverberating regime rather than exact criticality; this
project's own deep_critical run also peaked slightly below beta_c.  D3
compares the two competing hypotheses directly on a trained deep recurrent
VPSC net.

Design (fixed by pre-registration, do not tune after seeing results):

* RecurrentVPSCNet(sizes=[12,40,24], 4-class temporal task) with the exact
  make_data split and hyperparameters of experiments/deep_critical.py;
  CE-trained at beta=0.8 (100 epochs), weights frozen; seeds {0,1,2};
* inference-time sweep beta*rho = beta / beta_c in
  {0.80,0.85,0.90,0.95,1.00,1.05,1.10}, beta_c measured per trained net;
* metrics per (seed, beta*rho) on the 300-sample test set: accuracy,
  permutation-corrected MI between class label and top-layer state
  (sign-bit quantization, 16 label-shuffle nulls), finite-difference
  susceptibility chi = mean||dx_top||/h (h=0.05), reverberation time
  tau_rev of a norm-0.1 perturbation of the top layer's persistent state at
  t=8 (8 probe samples), and perturbation recovery drop in accuracy;
* pass (D3 adopted) requires >= 2/3 seeds with all of:
  Q1 plateau: >= 3 contiguous subcritical points with
  acc >= max_acc - 0.03 and MI >= max_MI - 0.25 bit;
  Q2 reverberation: Spearman(beta*rho in {0.80..0.95}, tau_rev) >= 0.8;
  Q3 exact criticality not strictly dominant:
  acc(1.00) <= max_sub acc + 0.01 and MI(1.00) <= max_sub MI + 0.1;
* REJECT if >= 2/3 seeds show strict beta*rho=1 dominance or no plateau;
  controls: no-recurrence (W_rec=0) tau_rev / accuracy at beta*rho=1,
  susceptibility monotonicity as instrument sanity.
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

from experiments.deep_critical import ToyData, make_data, spearman, train_CE  # noqa: E402
from vpsc.recurrent import RecurrentVPSCNet  # noqa: E402

BETA_RHOS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
SUBCRITICAL = [b for b in BETA_RHOS if b < 1.0]
SEEDS = [0, 1, 2]
H_SUSC = 0.05
T_PERTURB = 8
DELTA_NORM = 0.1
N_PROBE = 8
LABEL_REPS = 16
EPOCHS = 100
Q1_ACC_TOL = 0.03
Q1_MI_TOL = 0.25
Q2_MIN_SPEARMAN = 0.8
Q3_ACC_TOL = 0.01
Q3_MI_TOL = 0.1


@torch.no_grad()
def forward_top(
    net: RecurrentVPSCNet,
    x_seq: torch.Tensor,
    h_field: torch.Tensor | None = None,
    perturb_t: int | None = None,
    perturb_vec: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RecurrentVPSCNet.forward with two pre-registered hooks.

    h_field: [n_top] constant extra drive to the top layer every timestep
    (susceptibility).  perturb_t/perturb_vec: add perturb_vec to the top
    layer's persistent (warm-start) state right after timestep perturb_t
    (reverberation / perturbation recovery).
    """
    T, B, _ = x_seq.shape
    net.reset_state(B, x_seq.device)
    tops = []
    logits = None
    top_layer = len(net.layers) - 1
    for t in range(T):
        x = x_seq[t]
        for li, layer in enumerate(net.layers):
            I_ext = None
            if h_field is not None and li == top_layer:
                I_ext = h_field.unsqueeze(0).expand(B, -1)
            m = layer(x, I_ext=I_ext)
            x = m
        if perturb_t is not None and t == perturb_t:
            net.layers[top_layer].m = net.layers[top_layer].m + perturb_vec.unsqueeze(0)
        tops.append(x)
        logits = net.readout(x)
    return logits, torch.stack(tops)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def _entropy_bits(counts: np.ndarray) -> float:
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def _mi(labels: np.ndarray, bits: np.ndarray, n_classes: int) -> float:
    n = len(labels)
    h_k = _entropy_bits(np.bincount(labels, minlength=n_classes).astype(float))
    _, inverse = np.unique(bits, axis=0, return_inverse=True)
    h_cond = 0.0
    for r in np.unique(inverse):
        sel = inverse == r
        h_cond += sel.sum() / n * _entropy_bits(
            np.bincount(labels[sel], minlength=n_classes).astype(float)
        )
    return float(h_k - h_cond)


def mi_corrected(
    labels: np.ndarray, bits: np.ndarray, n_classes: int, seed: int
) -> tuple[float, float, float]:
    raw = _mi(labels, bits, n_classes)
    rng = np.random.default_rng(seed)
    nulls = [
        _mi(labels[rng.permutation(len(labels))], bits, n_classes)
        for _ in range(LABEL_REPS)
    ]
    return raw - float(np.mean(nulls)), raw, float(np.max(nulls))


def evaluate_point(
    net: RecurrentVPSCNet,
    beta: float,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    h_field: torch.Tensor,
    perturb_vec: torch.Tensor,
    mi_seed: int,
) -> dict:
    net.set_beta(beta)
    logits0, tops0 = forward_top(net, x_test)
    acc = accuracy_from_logits(logits0, y_test)
    x_top = tops0[-1]
    bits = (x_top > 0).cpu().numpy().astype(np.int64)
    labels = y_test.cpu().numpy()
    mi_corr, mi_raw, mi_null_max = mi_corrected(labels, bits, 4, mi_seed)

    _, tops_h = forward_top(net, x_test, h_field=h_field)
    chi = float(((tops_h[-1] - x_top).norm(dim=-1)).mean().item()) / H_SUSC

    logits_p, _ = forward_top(
        net, x_test, perturb_t=T_PERTURB, perturb_vec=perturb_vec
    )
    acc_pert = accuracy_from_logits(logits_p, y_test)

    probes = x_test[:, :N_PROBE, :]
    _, tops_b = forward_top(net, probes)
    _, tops_p = forward_top(net, probes, perturb_t=T_PERTURB, perturb_vec=perturb_vec)
    d0 = float(perturb_vec.norm().item())
    taus = []
    for sample in range(N_PROBE):
        diffs = [
            float((tops_p[T_PERTURB + k, sample] - tops_b[T_PERTURB + k, sample]).norm().item())
            for k in range(1, tops_b.shape[0] - T_PERTURB)
        ]
        tau = len(diffs)
        for k, d in enumerate(diffs, start=1):
            if d < d0 / np.e:
                tau = k
                break
        taus.append(float(tau))
    return {
        "beta": float(beta),
        "acc": acc,
        "acc_perturbed": acc_pert,
        "drop": acc - acc_pert,
        "mi_corr": float(mi_corr),
        "mi_raw": float(mi_raw),
        "mi_null_max": float(mi_null_max),
        "susceptibility": chi,
        "tau_rev": float(np.mean(taus)),
        "tau_rev_per_probe": taus,
    }


def train_net(seed: int, epochs: int = EPOCHS) -> tuple[RecurrentVPSCNet, ToyData]:
    torch.manual_seed(seed)
    full = make_data(seed=seed)
    n = full.x.shape[0]
    ntr = int(0.8 * n)
    train_data = ToyData(full.x[:ntr], full.y[:ntr])
    test_data = ToyData(full.x[ntr:], full.y[ntr:])
    net = RecurrentVPSCNet(
        sizes=[12, 40, 24], n_classes=4, beta=1.0, threshold=0.0, sigma=1.0,
        n_relax=8, rec_rho0=0.7, wd=1e-4, seed=seed,
    )
    train_CE(net, train_data, epochs=epochs, lr=3e-3, beta=0.8)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, test_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== D3: subcritical reverberation vs exact criticality ===")
    per_seed_results = []
    for seed in SEEDS:
        net, test_data = train_net(seed, epochs=args.epochs)
        beta_c = net.critical_beta()
        per_layer_bc = net.per_layer_critical_beta()
        print(f"\nseed={seed} beta_c={beta_c:.4f} per-layer={['%.3f' % b for b in per_layer_bc]}")
        x_test = test_data.x.transpose(0, 1)
        y_test = test_data.y

        g = torch.Generator().manual_seed(seed * 1000 + 1)
        u = torch.randn(24, generator=g)
        u = u / u.norm()
        h_field = H_SUSC * u
        g2 = torch.Generator().manual_seed(seed * 1000 + 2)
        delta = torch.randn(24, generator=g2)
        delta = delta / delta.norm() * DELTA_NORM

        points = []
        for br in BETA_RHOS:
            metrics = evaluate_point(
                net, br * beta_c, x_test, y_test, h_field, delta, seed * 100 + 7
            )
            metrics["beta_rho"] = br
            points.append(metrics)
            print(
                f"  br={br:.2f} acc={metrics['acc']:.3f} MI={metrics['mi_corr']:.3f} "
                f"chi={metrics['susceptibility']:.3f} tau={metrics['tau_rev']:.2f} "
                f"drop={metrics['drop']:.3f}"
            )

        # no-recurrence control at beta_rho = 1.00
        saved = [layer.W_rec.data.clone() for layer in net.layers]
        for layer in net.layers:
            layer.W_rec.data.zero_()
        ctrl = evaluate_point(
            net, beta_c, x_test, y_test, h_field, delta, seed * 100 + 7
        )
        for layer, w in zip(net.layers, saved):
            layer.W_rec.data.copy_(w)
        print(
            f"  control W_rec=0 @br=1.00: tau={ctrl['tau_rev']:.2f} "
            f"acc={ctrl['acc']:.3f}"
        )

        # ---- judge criteria ----
        accs = [p["acc"] for p in points]
        mis = [p["mi_corr"] for p in points]
        max_acc, max_mi = max(accs), max(mis)
        sub_points = [p for p in points if p["beta_rho"] < 1.0]
        near = [
            p["beta_rho"]
            for p in sub_points
            if p["acc"] >= max_acc - Q1_ACC_TOL and p["mi_corr"] >= max_mi - Q1_MI_TOL
        ]
        # contiguity on the BETA_RHOS lattice
        near_set = set(near)
        longest = 0
        run = 0
        for br in SUBCRITICAL:
            run = run + 1 if br in near_set else 0
            longest = max(longest, run)
        q1 = longest >= 3
        taus_sub = [p["tau_rev"] for p in sub_points]
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
                "per_layer_beta_c": per_layer_bc,
                "points": points,
                "no_rec_control": ctrl,
                "q1_plateau": bool(q1),
                "q1_plateau_points": near,
                "q2_reverberation": bool(q2),
                "q2_spearman": float(q2_spearman),
                "q3_not_dominant": q3,
                "strict_critical_dominance": strict_dominance,
                "chi_spearman": float(chi_spearman),
                "seed_pass": bool(q1 and q2 and q3),
            }
        )
        print(
            f"  Q1={q1} (plateau {near}) Q2={q2} (rho={q2_spearman:+.3f}) "
            f"Q3={q3} strict_dom={strict_dominance} chi_rho={chi_spearman:+.3f}"
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
    print(f"  D3 verdict: {verdict}")

    output = {
        "experiment": "D3 subcritical reverberation vs exact criticality",
        "formalization_version": "D3-v0.1",
        "protocol": {
            "beta_rhos": BETA_RHOS,
            "seeds": SEEDS,
            "epochs": args.epochs,
            "h_susc": H_SUSC,
            "t_perturb": T_PERTURB,
            "delta_norm": DELTA_NORM,
            "n_probe": N_PROBE,
            "label_reps": LABEL_REPS,
            "tolerances": {
                "q1_acc": Q1_ACC_TOL,
                "q1_mi": Q1_MI_TOL,
                "q3_acc": Q3_ACC_TOL,
                "q3_mi": Q3_MI_TOL,
                "q2_spearman": Q2_MIN_SPEARMAN,
            },
        },
        "per_seed": per_seed_results,
        "verdict": {"n_pass": n_pass, "n_reject": n_reject, "d3": verdict},
    }
    json_path = os.path.join(RESULTS_DIR, "d3_reverberation.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def plot_metric(ax, key, title, ylabel, control=None):
        for r in per_seed_results:
            xs = [p["beta_rho"] for p in r["points"]]
            ys = [p[key] for p in r["points"]]
            ax.plot(xs, ys, "-o", ms=4, alpha=0.8, label=f"seed {r['seed']}")
        ax.axvline(1.0, color="black", ls="--", lw=0.8)
        ax.set(title=title, xlabel="beta*rho", ylabel=ylabel)
        ax.legend(fontsize=8)

    plot_metric(axes[0, 0], "acc", "task accuracy", "acc")
    plot_metric(axes[0, 1], "mi_corr", "label <-> state MI (corrected)", "bits")
    plot_metric(axes[1, 0], "tau_rev", "reverberation time", "timesteps")
    plot_metric(axes[1, 1], "susceptibility", "susceptibility chi", "chi")
    for r in per_seed_results:
        axes[1, 0].axhline(
            r["no_rec_control"]["tau_rev"], color="gray", ls=":", lw=0.8
        )
    fig.suptitle(f"D3: {verdict}", fontsize=14)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "d3_reverberation.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
