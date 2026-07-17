"""E1: combine D2's carrier rule with D3's 5% near-critical margin.

Pre-registered in dev/LOG.md (2026-07-17, "E1 预注册 — D2 载体规则 ×
D3 近临界安全裕量").  D2 and D3 remain closed; this is a new engineering
validation on held-out dynamics seeds and held-out stimulus banks.

The hybrid policy is frozen before this run:

* always retain positive feedback;
* remove E<-I negative feedback only for the g_ee x0.8 row, retain it for
  g_ee >= x1.0;
* multiply all four feedback gains by 0.95, leaving inputs, thresholds and
  time constants unchanged.

It is compared with the original complete E/I system at nominal scale 1.00
and with a margin-only complete E/I system.  All three are evaluated under
global feedback-gain drift {-5%, 0, +5%}.  The no-positive control is measured
at zero drift.  The MI estimator and response protocol are inherited from D2.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict, replace

import numpy as np

from c2_verify import Params
from d2_information_peak import (
    ACTIVE_PER_STIM,
    INSTANCES,
    LABEL_SHUFFLE_REPS,
    N_STIM,
    QUANT_EDGES,
    TOTAL_TIME,
    WINDOW,
    mutual_information,
    pattern_entropy,
    quantize,
    response_feature,
    simulate_pattern,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
D2_JSON = os.path.join(RESULTS_DIR, "d2_information_peak.json")

HELDOUT_SEEDS = [3, 4, 5]
STIM_BANK_SEEDS = [20260718, 20260719]
GAIN_DRIFTS = [-0.05, 0.0, 0.05]
MARGIN_SCALE = 0.95
INFORMATIVE_MIN = 0.5
UTILITY_MIN_FRACTION = 0.80
UTILITY_MEDIAN_TOL = 0.10
ROBUST_MEDIAN_GAIN = 0.10
CARRIER_RATIO = 0.50
CARRIER_MIN_FRACTION = 0.90


def make_unique_stimuli(p: Params, seed: int) -> np.ndarray:
    """Create one held-out bank of eight unique 3-of-8 patterns."""
    combos = np.array(list(itertools.combinations(range(p.n), ACTIVE_PER_STIM)))
    rng = np.random.default_rng(seed)
    chosen = combos[rng.permutation(len(combos))[:N_STIM]]
    stimuli = np.zeros((N_STIM, p.n), dtype=np.float64)
    for idx, active in enumerate(chosen):
        stimuli[idx, active] = p.cue_amplitude
    return stimuli


def heldout_instance_seed(
    dynamics_seed: int, bank_seed: int, stim: int, instance: int
) -> int:
    return (
        dynamics_seed * 10_000_000
        + (bank_seed % 10_000) * 1_000
        + stim * 100
        + instance
    )


def run_information(
    p: Params,
    dynamics_seed: int,
    bank_seed: int,
    stimuli: np.ndarray,
    dt: float,
) -> dict:
    """D2's response/quantization/MI pipeline without its unrelated DR sweep."""
    features = np.empty((N_STIM * INSTANCES, p.n), dtype=np.float64)
    labels = np.repeat(np.arange(N_STIM), INSTANCES)
    for stim in range(N_STIM):
        for instance in range(INSTANCES):
            traj = simulate_pattern(
                p,
                stimuli[stim],
                heldout_instance_seed(dynamics_seed, bank_seed, stim, instance),
                TOTAL_TIME,
                dt,
            )
            features[stim * INSTANCES + instance] = response_feature(traj)

    bits = quantize(features)
    mi_raw = mutual_information(labels, bits)
    null_rng = np.random.default_rng(dynamics_seed * 1_000_000 + bank_seed)
    nulls = [
        mutual_information(labels[null_rng.permutation(len(labels))], bits)
        for _ in range(LABEL_SHUFFLE_REPS)
    ]
    mi_null_mean = float(np.mean(nulls))
    mean_patterns = features.reshape(N_STIM, INSTANCES, p.n).mean(axis=1)
    return {
        "mi": float(mi_raw - mi_null_mean),
        "mi_raw": float(mi_raw),
        "label_shuffle_mi_mean": mi_null_mean,
        "label_shuffle_mi_max": float(np.max(nulls)),
        "pattern_entropy": float(pattern_entropy(mean_patterns)),
        "response_mean": float(features.mean()),
        "response_std": float(features.std()),
    }


def condition_params(
    base: Params,
    fe: float,
    fi: float,
    condition: str,
    drift: float,
) -> Params:
    """Build one frozen E1 condition from the original D2 grid cell."""
    if condition == "exact_full":
        nominal_scale = 1.0
        remove_negative = False
        remove_positive = False
    elif condition == "margin_full":
        nominal_scale = MARGIN_SCALE
        remove_negative = False
        remove_positive = False
    elif condition == "hybrid":
        nominal_scale = MARGIN_SCALE
        remove_negative = fe < 1.0
        remove_positive = False
    elif condition == "no_positive":
        nominal_scale = MARGIN_SCALE
        remove_negative = fe < 1.0
        remove_positive = True
    else:
        raise ValueError(f"unknown condition: {condition}")

    scale = nominal_scale * (1.0 + drift)
    g_ee = 0.0 if remove_positive else base.g_ee * fe * scale
    g_ei = 0.0 if remove_negative else base.g_ei * fi * scale
    return replace(
        base,
        g_ee=g_ee,
        g_ei=g_ei,
        g_ie=base.g_ie * scale,
        g_ii=base.g_ii * scale,
    )


def load_frozen_cells() -> list[tuple[float, float]]:
    with open(D2_JSON, "r", encoding="utf-8") as handle:
        d2 = json.load(handle)
    cells = []
    for key, cell in d2["cells"].items():
        if float(cell["mi"]) >= INFORMATIVE_MIN:
            fe, fi = (float(part) for part in key.split(","))
            cells.append((fe, fi))
    return sorted(cells)


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.04, help="RK4 step")
    args = parser.parse_args()

    base = Params()
    cells = load_frozen_cells()
    if len(cells) != 10:
        raise RuntimeError(f"pre-registration expected 10 frozen cells, got {len(cells)}")
    banks = {seed: make_unique_stimuli(base, seed) for seed in STIM_BANK_SEEDS}
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== E1: D2 carrier rule x D3 5% margin ===")
    print(
        f"cells={len(cells)}, heldout_seeds={HELDOUT_SEEDS}, "
        f"stim_banks={STIM_BANK_SEEDS}, drifts={GAIN_DRIFTS}"
    )

    cases = []
    total = len(cells) * len(HELDOUT_SEEDS) * len(STIM_BANK_SEEDS)
    case_index = 0
    for fe, fi in cells:
        for dynamics_seed in HELDOUT_SEEDS:
            for bank_seed in STIM_BANK_SEEDS:
                case_index += 1
                results: dict[str, dict[str, dict]] = {
                    "exact_full": {},
                    "margin_full": {},
                    "hybrid": {},
                    "no_positive": {},
                }
                for condition in ("exact_full", "margin_full", "hybrid"):
                    for drift in GAIN_DRIFTS:
                        p = condition_params(base, fe, fi, condition, drift)
                        results[condition][f"{drift:+.2f}"] = run_information(
                            p, dynamics_seed, bank_seed, banks[bank_seed], args.dt
                        )
                p_no_positive = condition_params(
                    base, fe, fi, "no_positive", drift=0.0
                )
                results["no_positive"]["+0.00"] = run_information(
                    p_no_positive,
                    dynamics_seed,
                    bank_seed,
                    banks[bank_seed],
                    args.dt,
                )
                cases.append(
                    {
                        "fe": fe,
                        "fi": fi,
                        "dynamics_seed": dynamics_seed,
                        "stim_bank_seed": bank_seed,
                        "results": results,
                    }
                )
                exact = results["exact_full"]["+0.00"]["mi"]
                margin = results["margin_full"]["+0.00"]["mi"]
                hybrid = results["hybrid"]["+0.00"]["mi"]
                print(
                    f"  [{case_index:02d}/{total}] ({fe},{fi}) s={dynamics_seed} "
                    f"bank={bank_seed}: exact={exact:.3f} margin={margin:.3f} "
                    f"hybrid={hybrid:.3f}"
                )

    nominal_exact = [c["results"]["exact_full"]["+0.00"]["mi"] for c in cases]
    nominal_margin = [c["results"]["margin_full"]["+0.00"]["mi"] for c in cases]
    nominal_hybrid = [c["results"]["hybrid"]["+0.00"]["mi"] for c in cases]
    nominal_no_positive = [
        c["results"]["no_positive"]["+0.00"]["mi"] for c in cases
    ]
    informative_count = sum(mi >= INFORMATIVE_MIN for mi in nominal_hybrid)
    informative_fraction = informative_count / len(cases)
    nominal_exact_median = median(nominal_exact)
    nominal_margin_median = median(nominal_margin)
    nominal_hybrid_median = median(nominal_hybrid)
    utility_pass = bool(
        informative_fraction >= UTILITY_MIN_FRACTION
        and nominal_hybrid_median >= nominal_exact_median - UTILITY_MEDIAN_TOL
    )

    exact_worst = [
        min(c["results"]["exact_full"][f"{d:+.2f}"]["mi"] for d in GAIN_DRIFTS)
        for c in cases
    ]
    margin_worst = [
        min(c["results"]["margin_full"][f"{d:+.2f}"]["mi"] for d in GAIN_DRIFTS)
        for c in cases
    ]
    hybrid_worst = [
        min(c["results"]["hybrid"][f"{d:+.2f}"]["mi"] for d in GAIN_DRIFTS)
        for c in cases
    ]
    exact_worst_median = median(exact_worst)
    margin_worst_median = median(margin_worst)
    hybrid_worst_median = median(hybrid_worst)
    exact_collapse_count = sum(mi < INFORMATIVE_MIN for mi in exact_worst)
    margin_collapse_count = sum(mi < INFORMATIVE_MIN for mi in margin_worst)
    hybrid_collapse_count = sum(mi < INFORMATIVE_MIN for mi in hybrid_worst)
    robustness_pass = bool(
        hybrid_worst_median >= exact_worst_median + ROBUST_MEDIAN_GAIN
        and hybrid_collapse_count <= exact_collapse_count
    )

    carrier_flags = [
        no_pos <= CARRIER_RATIO * hybrid
        for no_pos, hybrid in zip(nominal_no_positive, nominal_hybrid)
    ]
    carrier_count = sum(carrier_flags)
    carrier_fraction = carrier_count / len(cases)
    carrier_pass = bool(carrier_fraction >= CARRIER_MIN_FRACTION)

    if utility_pass and robustness_pass and carrier_pass:
        verdict = "ADOPT"
    elif utility_pass and carrier_pass:
        verdict = "MIXED"
    else:
        verdict = "REJECT"

    summary = {
        "n_cases": len(cases),
        "nominal": {
            "exact_full_median": nominal_exact_median,
            "margin_full_median": nominal_margin_median,
            "hybrid_median": nominal_hybrid_median,
            "hybrid_informative_count": informative_count,
            "hybrid_informative_fraction": informative_fraction,
        },
        "worst_case_over_drift": {
            "exact_full_median": exact_worst_median,
            "margin_full_median": margin_worst_median,
            "hybrid_median": hybrid_worst_median,
            "hybrid_minus_exact_median": hybrid_worst_median - exact_worst_median,
            "exact_full_collapse_count": exact_collapse_count,
            "margin_full_collapse_count": margin_collapse_count,
            "hybrid_collapse_count": hybrid_collapse_count,
        },
        "carrier_control": {
            "pass_count": carrier_count,
            "pass_fraction": carrier_fraction,
        },
        "criteria": {
            "utility": utility_pass,
            "robustness": robustness_pass,
            "carrier": carrier_pass,
        },
        "verdict": verdict,
    }
    print("\n=== pre-registered judge ===")
    print(
        f"  E1-U utility: informative={informative_count}/{len(cases)} "
        f"({informative_fraction:.1%}), median hybrid={nominal_hybrid_median:.3f}, "
        f"exact={nominal_exact_median:.3f} => {utility_pass}"
    )
    print(
        f"  E1-R robustness: worst median hybrid={hybrid_worst_median:.3f}, "
        f"exact={exact_worst_median:.3f}, delta={hybrid_worst_median - exact_worst_median:+.3f}; "
        f"collapse hybrid={hybrid_collapse_count}, exact={exact_collapse_count} "
        f"=> {robustness_pass}"
    )
    print(
        f"  E1-C carrier: {carrier_count}/{len(cases)} ({carrier_fraction:.1%}) "
        f"=> {carrier_pass}"
    )
    print(f"  E1 verdict: {verdict}")

    output = {
        "experiment": "E1 D2 carrier rule x D3 5% gain margin",
        "formalization_version": "E1-v0.1",
        "source_artifact": D2_JSON,
        "base_parameters": asdict(base),
        "frozen_cells": [list(cell) for cell in cells],
        "protocol": {
            "dt": args.dt,
            "heldout_seeds": HELDOUT_SEEDS,
            "stim_bank_seeds": STIM_BANK_SEEDS,
            "gain_drifts": GAIN_DRIFTS,
            "margin_scale": MARGIN_SCALE,
            "n_stim": N_STIM,
            "active_per_stim": ACTIVE_PER_STIM,
            "instances": INSTANCES,
            "window": list(WINDOW),
            "quant_edges": QUANT_EDGES,
            "label_shuffle_reps": LABEL_SHUFFLE_REPS,
            "thresholds": {
                "informative_min": INFORMATIVE_MIN,
                "utility_min_fraction": UTILITY_MIN_FRACTION,
                "utility_median_tolerance": UTILITY_MEDIAN_TOL,
                "robust_median_gain": ROBUST_MEDIAN_GAIN,
                "carrier_ratio": CARRIER_RATIO,
                "carrier_min_fraction": CARRIER_MIN_FRACTION,
            },
        },
        "stimulus_banks": {str(k): v.tolist() for k, v in banks.items()},
        "cases": cases,
        "summary": summary,
    }
    json_path = os.path.join(RESULTS_DIR, "e1_hybrid_margin.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].boxplot(
        [nominal_exact, nominal_margin, nominal_hybrid, nominal_no_positive],
        tick_labels=["exact", "margin", "hybrid", "no positive"],
        showmeans=True,
    )
    axes[0, 0].axhline(INFORMATIVE_MIN, color="black", ls=":", lw=0.8)
    axes[0, 0].set(title="nominal held-out MI", ylabel="MI_corr (bits)")

    axes[0, 1].scatter(exact_worst, hybrid_worst, alpha=0.7)
    lo = min(exact_worst + hybrid_worst)
    hi = max(exact_worst + hybrid_worst)
    axes[0, 1].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[0, 1].set(
        title="worst MI over gain drift",
        xlabel="exact full",
        ylabel="hybrid",
    )

    cell_labels = [f"{fe},{fi}" for fe, fi in cells]
    exact_by_cell = []
    margin_by_cell = []
    hybrid_by_cell = []
    for fe, fi in cells:
        selected = [c for c in cases if c["fe"] == fe and c["fi"] == fi]
        exact_by_cell.append(
            np.mean([c["results"]["exact_full"]["+0.00"]["mi"] for c in selected])
        )
        margin_by_cell.append(
            np.mean([c["results"]["margin_full"]["+0.00"]["mi"] for c in selected])
        )
        hybrid_by_cell.append(
            np.mean([c["results"]["hybrid"]["+0.00"]["mi"] for c in selected])
        )
    x = np.arange(len(cells))
    axes[1, 0].plot(x, exact_by_cell, "-o", label="exact")
    axes[1, 0].plot(x, margin_by_cell, "-o", label="margin")
    axes[1, 0].plot(x, hybrid_by_cell, "-o", label="hybrid")
    axes[1, 0].set_xticks(x, cell_labels, rotation=45, ha="right")
    axes[1, 0].set(title="nominal MI by frozen cell", ylabel="mean MI_corr")
    axes[1, 0].legend()

    drift_labels = [f"{d:+.0%}" for d in GAIN_DRIFTS]
    for condition, color in [
        ("exact_full", "#2463EB"),
        ("margin_full", "#AF7AC5"),
        ("hybrid", "#17A589"),
    ]:
        drift_medians = [
            median([c["results"][condition][f"{d:+.2f}"]["mi"] for c in cases])
            for d in GAIN_DRIFTS
        ]
        axes[1, 1].plot(drift_labels, drift_medians, "-o", label=condition, color=color)
    axes[1, 1].set(title="pooled median MI under gain drift", ylabel="MI_corr")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"E1 D2 carrier x D3 margin: {verdict}", fontsize=14)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "e1_hybrid_margin.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
