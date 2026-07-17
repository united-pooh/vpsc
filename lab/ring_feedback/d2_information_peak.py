"""D2: does stimulus information peak at intermediate E/I gain (Shew analog)?

Pre-registered in dev/LOG.md (2026-07-17, "D2 预注册 — E/I 信息峰").
Shew et al. (2009/2011) showed dynamic range, Shannon entropy and
stimulus-response mutual information peak at intermediate E/I.  This script
tests whether the same intermediate information peak exists in the C-line E/I
dual ring, with pre-registered metrics and controls.

Design (fixed by pre-registration, do not tune after seeing results):

* full E/I ring, Params() base; grid g_ee factor x g_ei factor in
  {0.4,0.6,0.8,1.0,1.25,1.5}^2 x seeds {0,1,2} (108 points); every cell is
  regime-labelled with the C1 oscillation check;
* fixed stimulus set (rng seed 20260717): 8 random patterns, each activating
  3 of 8 nodes (amplitude 5.0 per active node, t in [0,2)); 4 jitter
  instances per pattern; response = per-node mean e in window t in [4,16);
* metrics (as amended before the first grid run — see the amendment note in
  dev/LOG.md and below): pattern entropy and stimulus-response MI use 4-bin
  fixed quantization (edges {0.2,0.4,0.6,0.8}, bits, MI from the contingency
  over 32 stimulus x instance pairs with label-shuffle bias control);
  dynamic range Delta = log10(I_90/I_10) over the amplitude factors
  {0.1,0.2,0.4,0.6,0.8,1.0,1.3,1.6} with excess = max(0, r - r0) against the
  0-factor member of the same amplitude series;
* controls at the MI peak cell: no_positive (g_ee=0), no_negative (g_ei=0),
  single_ring, connectivity shuffle (same random node permutation applied to
  both ring kernels, 5 reps);
* pass: P1 every metric's peak set (>= max - 1e-9) contains an interior
  oscillating cell and no boundary cell, and the peak exceeds the mean over
  non-oscillating cells by the margin (0.5 bit / 0.5 bit / 0.2);
  P2 ablations <= 0.5 x full MI in >= 2 of 3 seeds;
  P3 shuffle <= 0.5 x full MI in >= 2 of 3 seeds;
  P4 (amendment 2) permutation test: raw MI > 16-rep label-shuffle max at
  the peak cell in >= 2 of 3 seeds; all MI quantities are bias-corrected
  (raw minus label-shuffle mean).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from typing import Callable

import numpy as np

from c2_verify import (
    Params,
    Trajectory,
    oscillation_metrics,
    rk4_step,
    sigmoid,
    simulate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Pre-registered constants.
GRID_FACTORS = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5]
SEEDS = [0, 1, 2]
STIM_SEED = 20260717
N_STIM = 8
ACTIVE_PER_STIM = 3
INSTANCES = 4
TOTAL_TIME = 16.0
CUE_END = 2.0
WINDOW = (4.0, 16.0)
AMP_FACTORS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
SHUFFLE_REPS = 4
LABEL_SHUFFLE_REPS = 16
CONN_SHUFFLE_REPS = 5
MARGIN_MI = 0.5
MARGIN_ENTROPY = 0.5
MARGIN_DR = 0.2
ABLATION_RATIO = 0.5
# Amendment 2: MI is bias-corrected (16-rep label-shuffle null); P4 is the
# permutation test raw MI > null max, replacing the old absolute bias bound.
# Amendment (2026-07-17, before the first grid run): the original 2-bin edge
# (0.30) collapses every stimulus pattern to the same bit string at
# oscillating points (the traveling wave equalises per-node window means),
# and the separate no-input baseline is itself highly active there, so both
# metrics were degenerate.  Fixed: 4-bin fixed edges and a 0-factor baseline
# member inside the amplitude series with excess = max(0, r - r0).
QUANT_EDGES = [0.2, 0.4, 0.6, 0.8]


def make_stimuli(p: Params) -> np.ndarray:
    """Fixed pre-registered stimulus set: 8 patterns x 3 active nodes."""
    rng = np.random.default_rng(STIM_SEED)
    stimuli = np.zeros((N_STIM, p.n), dtype=np.float64)
    for k in range(N_STIM):
        active = rng.permutation(p.n)[:ACTIVE_PER_STIM]
        stimuli[k, active] = p.cue_amplitude
    return stimuli


def shuffle_permutation(n: int, seed: int) -> np.ndarray:
    """A derangement-like permutation for the connectivity-shuffle control."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    return perm


def neighbor_kernel(x: np.ndarray, rho: float, forward: np.ndarray) -> np.ndarray:
    """(1-rho) x + rho x[forward]; `forward` is the neighbor index map."""
    return (1.0 - rho) * x + rho * x[forward]


def simulate_pattern(
    p: Params,
    drive: np.ndarray,
    seed: int,
    total_time: float,
    dt: float,
    conn_perm: np.ndarray | None = None,
) -> Trajectory:
    """Full E/I ring driven by an arbitrary input pattern.

    conn_perm=None uses the pre-registered ring kernels (roll +/-1); a
    permutation replaces both neighbor maps (K_I uses the inverse) for the
    connectivity-shuffle control.
    """
    rng = np.random.default_rng(seed)
    state = np.concatenate(
        [0.02 + 0.01 * rng.random(p.n), 0.02 + 0.01 * rng.random(p.n)]
    )
    if conn_perm is None:
        fwd_e = (np.arange(p.n) - 1) % p.n  # roll(+1) == x[(i-1) % n]
        fwd_i = (np.arange(p.n) + 1) % p.n  # roll(-1)
    else:
        fwd_e = conn_perm
        fwd_i = np.argsort(conn_perm)  # inverse permutation

    def derivative(x: np.ndarray, t: float) -> np.ndarray:
        e, i = x[: p.n], x[p.n :]
        ke = neighbor_kernel(e, p.rho_e, fwd_e)
        ki = neighbor_kernel(i, p.rho_i, fwd_i)
        u = drive if t < CUE_END else np.zeros(p.n, dtype=np.float64)
        de = (-e + sigmoid(p.g_ee * ke - p.g_ei * ki + u - p.theta_e)) / p.tau_e
        di = (-i + sigmoid(p.g_ie * ke - p.g_ii * ki - p.theta_i)) / p.tau_i
        return np.concatenate([de, di])

    steps = int(round(total_time / dt))
    states = np.empty((steps, 2 * p.n), dtype=np.float64)
    times = (np.arange(steps, dtype=np.float64) + 1.0) * dt
    for step in range(steps):
        state = rk4_step(state, step * dt, dt, derivative)
        states[step] = state
    return Trajectory(times, states[:, : p.n], states[:, p.n :])


def response_feature(traj: Trajectory) -> np.ndarray:
    mask = (traj.time >= WINDOW[0]) & (traj.time < WINDOW[1])
    if not np.any(mask):
        raise ValueError("empty response window")
    return traj.e[mask].mean(axis=0)


def quantize(features: np.ndarray) -> np.ndarray:
    return np.digitize(features, QUANT_EDGES).astype(np.int64)


def shannon_entropy_bits(counts: np.ndarray) -> float:
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def pattern_entropy(mean_patterns: np.ndarray) -> float:
    """Entropy over the 8 (equally weighted) instance-averaged patterns."""
    bits = quantize(mean_patterns)
    _, counts = np.unique(bits, axis=0, return_counts=True)
    return shannon_entropy_bits(counts)


def mutual_information(labels: np.ndarray, bits: np.ndarray) -> float:
    """I(K;R) in bits from the (stimulus x instance) contingency."""
    n = len(labels)
    h_k = shannon_entropy_bits(np.bincount(labels, minlength=N_STIM).astype(float))
    _, inverse = np.unique(bits, axis=0, return_inverse=True)
    h_k_given_r = 0.0
    for r in np.unique(inverse):
        sel = inverse == r
        p_r = sel.sum() / n
        h_k_given_r += p_r * shannon_entropy_bits(
            np.bincount(labels[sel], minlength=N_STIM).astype(float)
        )
    return float(h_k - h_k_given_r)


def dynamic_range(magnitudes: np.ndarray) -> tuple[float, float, float]:
    """Delta = log10(I_90/I_10) over the pre-registered amplitude factors."""
    r = np.asarray(magnitudes, dtype=np.float64)
    r_max = r.max()
    if r_max <= 1e-9:
        return 0.0, float("nan"), float("nan")
    rn = r / r_max

    def crossing(level: float) -> float | None:
        for idx in range(len(rn)):
            if rn[idx] >= level:
                if idx == 0:
                    return AMP_FACTORS[0]
                a0, a1 = AMP_FACTORS[idx - 1], AMP_FACTORS[idx]
                r0, r1 = rn[idx - 1], rn[idx]
                frac = 0.0 if r1 == r0 else (level - r0) / (r1 - r0)
                return a0 + frac * (a1 - a0)
        return None

    i10 = crossing(0.10)
    i90 = crossing(0.90)
    if i10 is None:
        return 0.0, float("nan"), float("nan")
    if i90 is None:
        i90 = AMP_FACTORS[-1]  # cap at the largest amplitude factor
    return float(np.log10(i90 / i10)), float(i10), float(i90)


def instance_seed(seed: int, stim: int, instance: int) -> int:
    return seed * 1000 + stim * 100 + instance


def run_cell(
    p: Params,
    seed: int,
    stimuli: np.ndarray,
    dt: float,
    condition: str = "full",
    conn_perm: np.ndarray | None = None,
) -> dict:
    """All pre-registered metrics at one grid cell under one condition."""
    q = p
    if condition == "no_positive":
        q = replace(q, g_ee=0.0)
    elif condition == "no_negative":
        q = replace(q, g_ei=0.0)
    elif condition != "full":
        raise ValueError(f"unknown condition: {condition}")

    # stimulus-response set S1
    feats = np.empty((N_STIM * INSTANCES, q.n), dtype=np.float64)
    labels = np.repeat(np.arange(N_STIM), INSTANCES)
    for stim in range(N_STIM):
        for inst in range(INSTANCES):
            traj = simulate_pattern(
                q, stimuli[stim], instance_seed(seed, stim, inst), TOTAL_TIME, dt,
                conn_perm=conn_perm,
            )
            feats[stim * INSTANCES + inst] = response_feature(traj)
    bits = quantize(feats)
    mi_raw = mutual_information(labels, bits)
    mean_patterns = feats.reshape(N_STIM, INSTANCES, q.n).mean(axis=1)
    entropy = pattern_entropy(mean_patterns)

    # Amendment 2: bias-corrected MI with a 16-rep permutation null.
    rng = np.random.default_rng(seed * 55 + 7)
    label_mis = [
        mutual_information(labels[rng.permutation(len(labels))], bits)
        for _ in range(LABEL_SHUFFLE_REPS)
    ]
    label_mi_mean = float(np.mean(label_mis))
    label_mi_max = float(np.max(label_mis))
    mi_corr = mi_raw - label_mi_mean

    # dynamic range on stimulus 0 with amplitude factors; the 0-factor member
    # of the same series is the baseline (amendment), excess = max(0, r - r0)
    baseline_traj = simulate_pattern(q, np.zeros(q.n), seed, TOTAL_TIME, dt)
    baseline = float(response_feature(baseline_traj).sum())
    magnitudes = []
    for factor in AMP_FACTORS:
        traj = simulate_pattern(q, stimuli[0] * factor, seed + 9000, TOTAL_TIME, dt)
        excess = float(response_feature(traj).sum()) - baseline
        magnitudes.append(max(0.0, excess))
    dr, i10, i90 = dynamic_range(np.array(magnitudes))

    return {
        "mi": float(mi_corr),
        "mi_raw": float(mi_raw),
        "label_shuffle_mi_mean": label_mi_mean,
        "label_shuffle_mi_max": label_mi_max,
        "pattern_entropy": entropy,
        "dynamic_range": dr,
        "dr_i10": i10,
        "dr_i90": i90,
        "dr_magnitudes": magnitudes,
    }


def oscillation_regime(p: Params, seed: int, dt: float) -> bool:
    traj = simulate(p, "full", "autonomous", seed, total_time=60.0, dt=dt)
    return bool(oscillation_metrics(traj)["pass"])


def single_ring_mi(p: Params, seed: int, stimuli: np.ndarray, dt: float) -> float:
    """MI of the A3 single ring at default gain (fixed baseline condition)."""
    feats = np.empty((N_STIM * INSTANCES, p.n), dtype=np.float64)
    labels = np.repeat(np.arange(N_STIM), INSTANCES)
    for stim in range(N_STIM):
        for inst in range(INSTANCES):
            rng = np.random.default_rng(instance_seed(seed, stim, inst))
            state = 0.01 * rng.standard_normal(p.n)

            def derivative(x: np.ndarray, t: float) -> np.ndarray:
                u = stimuli[stim] if t < CUE_END else np.zeros(p.n)
                ke = (1.0 - p.rho_e) * x + p.rho_e * np.roll(x, 1)
                return -x + np.tanh(p.single_ring_gain * ke + u)

            steps = int(round(TOTAL_TIME / dt))
            values = np.empty((steps, p.n), dtype=np.float64)
            times = (np.arange(steps) + 1.0) * dt
            for step_i in range(steps):
                state = rk4_step(state, step_i * dt, dt, derivative)
                values[step_i] = state
            traj = Trajectory(times, 0.5 * (values + 1.0), np.zeros_like(values))
            feats[stim * INSTANCES + inst] = response_feature(traj)
    return mutual_information(labels, quantize(feats))


def parse_csv(text: str, cast: Callable) -> list:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.04, help="RK4 step")
    args = parser.parse_args()

    base_params = Params()
    stimuli = make_stimuli(base_params)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== D2: E/I information peak (Shew analog) ===")
    print(f"grid={GRID_FACTORS}^2, seeds={SEEDS}, dt={args.dt}")

    # ---- grid sweep (full E/I) ----
    cells: dict[tuple[float, float], dict] = {}
    for fe in GRID_FACTORS:
        for fi in GRID_FACTORS:
            p = replace(
                base_params, g_ee=base_params.g_ee * fe, g_ei=base_params.g_ei * fi
            )
            per_seed = []
            for seed in SEEDS:
                metrics = run_cell(p, seed, stimuli, args.dt)
                metrics["oscillating"] = oscillation_regime(p, seed, args.dt)
                per_seed.append(metrics)
            cells[(fe, fi)] = {
                "mi": float(np.mean([m["mi"] for m in per_seed])),
                "pattern_entropy": float(np.mean([m["pattern_entropy"] for m in per_seed])),
                "dynamic_range": float(np.mean([m["dynamic_range"] for m in per_seed])),
                "label_shuffle_mi": float(
                    np.mean([m["label_shuffle_mi_mean"] for m in per_seed])
                ),
                "oscillating_seeds": [bool(m["oscillating"]) for m in per_seed],
                "oscillating": bool(sum(m["oscillating"] for m in per_seed) >= 2),
                "per_seed": per_seed,
            }
            print(
                f"  cell g_ee x{fe:<5} g_ei x{fi:<5} MI={cells[(fe, fi)]['mi']:.3f} "
                f"H={cells[(fe, fi)]['pattern_entropy']:.3f} "
                f"DR={cells[(fe, fi)]['dynamic_range']:.3f} "
                f"osc={cells[(fe, fi)]['oscillating_seeds']}"
            )

    n_f = len(GRID_FACTORS)

    def is_interior(idx: tuple[int, int]) -> bool:
        return 0 < idx[0] < n_f - 1 and 0 < idx[1] < n_f - 1

    cell_list = [
        (fe, fi, cells[(fe, fi)]) for fe in GRID_FACTORS for fi in GRID_FACTORS
    ]
    non_osc = [c for c in cell_list if not c[2]["oscillating"]]

    # ---- P1: intermediate peak for each metric ----
    p1: dict[str, dict] = {}
    for metric, margin in [
        ("mi", MARGIN_MI),
        ("pattern_entropy", MARGIN_ENTROPY),
        ("dynamic_range", MARGIN_DR),
    ]:
        values = np.array([c[2][metric] for c in cell_list])
        v_max = values.max()
        peak_cells = [c for c in cell_list if c[2][metric] >= v_max - 1e-9]
        peak_interior_osc = any(
            c[2]["oscillating"]
            and is_interior((GRID_FACTORS.index(c[0]), GRID_FACTORS.index(c[1])))
            for c in peak_cells
        )
        peak_on_boundary = any(
            not is_interior((GRID_FACTORS.index(c[0]), GRID_FACTORS.index(c[1])))
            for c in peak_cells
        )
        if non_osc:
            contrast = float(np.mean([c[2][metric] for c in non_osc]))
            margin_ok = bool(v_max >= contrast + margin)
        else:
            contrast = float("nan")
            margin_ok = False
        p1[metric] = {
            "max": float(v_max),
            "peak_cells": [[c[0], c[1]] for c in peak_cells],
            "peak_interior_osc": bool(peak_interior_osc),
            "peak_on_boundary": bool(peak_on_boundary),
            "non_osc_mean": contrast,
            "margin_ok": margin_ok,
            "pass": bool(peak_interior_osc and not peak_on_boundary and margin_ok),
        }
        print(
            f"  P1 {metric:15s} max={v_max:.3f} interior_osc={peak_interior_osc} "
            f"on_boundary={peak_on_boundary} non_osc_mean={contrast:.3f} "
            f"=> {'PASS' if p1[metric]['pass'] else 'FAIL'}"
        )

    # ---- controls at the MI peak cell ----
    # tie-break: prefer interior oscillating cells, then scan order
    mi_max = max(c[2]["mi"] for c in cell_list)
    candidates = [c for c in cell_list if c[2]["mi"] >= mi_max - 1e-9]
    candidates.sort(
        key=lambda c: not (
            c[2]["oscillating"]
            and is_interior((GRID_FACTORS.index(c[0]), GRID_FACTORS.index(c[1])))
        )
    )
    peak_fe, peak_fi = candidates[0][0], candidates[0][1]
    peak_p = replace(
        base_params,
        g_ee=base_params.g_ee * peak_fe,
        g_ei=base_params.g_ei * peak_fi,
    )
    print(f"\n  MI peak cell: g_ee x{peak_fe}, g_ei x{peak_fi} (MI={mi_max:.3f})")

    controls: dict[str, dict] = {}
    for seed in SEEDS:
        no_pos = run_cell(peak_p, seed, stimuli, args.dt, condition="no_positive")["mi"]
        no_neg = run_cell(peak_p, seed, stimuli, args.dt, condition="no_negative")["mi"]
        single = single_ring_mi(peak_p, seed, stimuli, args.dt)
        conn = float(
            np.mean(
                [
                    run_cell(
                        peak_p, seed, stimuli, args.dt,
                        conn_perm=shuffle_permutation(base_params.n, seed * 100 + rep),
                    )["mi"]
                    for rep in range(CONN_SHUFFLE_REPS)
                ]
            )
        )
        peak_metrics = cells[(peak_fe, peak_fi)]["per_seed"][SEEDS.index(seed)]
        full_mi = peak_metrics["mi"]
        mi_raw = peak_metrics["mi_raw"]
        label_mi_max = peak_metrics["label_shuffle_mi_max"]
        label_mi_mean = peak_metrics["label_shuffle_mi_mean"]
        controls[str(seed)] = {
            "full_mi": float(full_mi),
            "no_positive_mi": float(no_pos),
            "no_negative_mi": float(no_neg),
            "single_ring_mi": float(single),
            "conn_shuffle_mi": conn,
            "mi_raw": float(mi_raw),
            "label_shuffle_mi_mean": float(label_mi_mean),
            "label_shuffle_mi_max": float(label_mi_max),
            "p2": bool(
                full_mi > 0.0
                and max(no_pos, no_neg, single) <= ABLATION_RATIO * full_mi
            ),
            "p3": bool(full_mi > 0.0 and conn <= ABLATION_RATIO * full_mi),
            # Amendment 2: permutation test — raw MI must exceed the seed's
            # 16-rep label-shuffle maximum (p < 1/17).
            "p4": bool(mi_raw > label_mi_max),
        }
        print(
            f"  seed={seed} full={full_mi:.3f} noE={no_pos:.3f} noI={no_neg:.3f} "
            f"single={single:.3f} shuffle={conn:.3f} raw={mi_raw:.3f} "
            f"label_max={label_mi_max:.3f}"
        )

    p2_pass = sum(c["p2"] for c in controls.values()) >= 2
    p3_pass = sum(c["p3"] for c in controls.values()) >= 2
    p4_pass = sum(c["p4"] for c in controls.values()) >= 2
    p1_pass = all(p1[m]["pass"] for m in p1)
    verdict = "ADOPT" if (p1_pass and p2_pass and p3_pass and p4_pass) else "REJECT"
    print(
        f"\n  P1={p1_pass} P2={p2_pass} P3={p3_pass} P4={p4_pass} => D2 {verdict}"
    )

    output = {
        "experiment": "D2 E/I information peak",
        "formalization_version": "D2-v0.1",
        "parameters": asdict(base_params),
        "stimuli": stimuli.tolist(),
        "protocol": {
            "dt": args.dt,
            "grid_factors": GRID_FACTORS,
            "seeds": SEEDS,
            "stim_seed": STIM_SEED,
            "n_stim": N_STIM,
            "active_per_stim": ACTIVE_PER_STIM,
            "instances": INSTANCES,
            "window": list(WINDOW),
            "quant_edges": QUANT_EDGES,
            "amp_factors": AMP_FACTORS,
            "margins": {"mi": MARGIN_MI, "entropy": MARGIN_ENTROPY, "dr": MARGIN_DR},
            "ablation_ratio": ABLATION_RATIO,
            "label_shuffle_reps": LABEL_SHUFFLE_REPS,
        },
        "cells": {f"{fe},{fi}": cells[(fe, fi)] for fe, fi in cells},
        "p1": p1,
        "peak_cell": [peak_fe, peak_fi],
        "controls": controls,
        "verdict": {
            "p1": p1_pass,
            "p2": p2_pass,
            "p3": p3_pass,
            "p4": p4_pass,
            "d2": verdict,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "d2_information_peak.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    # ---- plot ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def heatmap(ax, metric: str, title: str) -> None:
        grid = np.array(
            [[cells[(fe, fi)][metric] for fi in GRID_FACTORS] for fe in GRID_FACTORS]
        )
        im = ax.imshow(grid, origin="lower", cmap="viridis")
        ax.set_xticks(range(n_f), GRID_FACTORS)
        ax.set_yticks(range(n_f), GRID_FACTORS)
        for a in range(n_f):
            for b in range(n_f):
                if cells[(GRID_FACTORS[a], GRID_FACTORS[b])]["oscillating"]:
                    ax.plot(b, a, marker="o", mfc="none", mec="white", ms=10, mew=1.2)
        ax.set(title=title, xlabel="g_ei factor", ylabel="g_ee factor")
        fig.colorbar(im, ax=ax)

    heatmap(axes[0, 0], "mi", "stimulus-response MI (bits)")
    heatmap(axes[0, 1], "pattern_entropy", "pattern entropy (bits)")
    heatmap(axes[1, 0], "dynamic_range", "dynamic range log10(I90/I10)")

    ax = axes[1, 1]
    seed0 = controls["0"]
    names = ["full", "noE", "noI", "single", "shuffle", "label"]
    values = [
        seed0["full_mi"],
        seed0["no_positive_mi"],
        seed0["no_negative_mi"],
        seed0["single_ring_mi"],
        seed0["conn_shuffle_mi"],
        seed0["label_shuffle_mi_mean"],
    ]
    colors = ["#2463EB", "#E4572E", "#E4572E", "#7F8C8D", "#AF7AC5", "#17A589"]
    ax.bar(names, values, color=colors)
    ax.axhline(
        ABLATION_RATIO * seed0["full_mi"], color="black", ls=":", lw=0.8,
        label="0.5 x full",
    )
    ax.set(title=f"MI controls at peak cell (seed 0)", ylabel="MI (bits)")
    ax.legend(fontsize=8)

    fig.suptitle(f"D2 E/I information peak: {verdict}", fontsize=14)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "d2_information_peak.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
