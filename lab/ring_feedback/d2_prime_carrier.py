"""D2': is the E/I gain structure (not oscillation, not ring topology) the carrier?

Pre-registered in dev/LOG.md (2026-07-17, "D2′ 预注册 — E/I 增益结构是刺激信息的载体").
D2 was REJECTED, but its control decomposition showed that AT THE PEAK CELL
stimulus information requires intact E/I (ablations -> MI 0) yet not the ring
topology (connectivity shuffle keeps MI).  D2' generalises that statement to
ALL informative cells and judges it.

Design (fixed by pre-registration, do not tune after seeing results):

* case set: every (cell, seed) of the D2 grid whose full bias-corrected MI
  >= 0.5 bit, taken from the D2 artifact JSON (no full-E/I re-simulation);
* identical frozen pipeline as D2 (grid, stimuli, window, quantization,
  permutation-corrected MI, instance seeds); per case measure no_positive,
  no_negative, single_ring and the connectivity shuffle (5 reps);
* N1 (gain structure necessary): in 100% of cases
  max(MI_noE, MI_noI, MI_single) <= 0.5 x MI_full;
* N2 (ring topology unnecessary): in >= 80% of cases
  E[MI_shuffle] >= 0.5 x MI_full;
* verdict ADOPT only if N1 and N2 both hold.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import numpy as np

from c2_verify import Params
from d2_information_peak import (
    CONN_SHUFFLE_REPS,
    SEEDS,
    make_stimuli,
    run_cell,
    shuffle_permutation,
    single_ring_mi,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
D2_JSON = os.path.join(RESULTS_DIR, "d2_information_peak.json")

INFORMATIVE_MIN = 0.5
RATIO = 0.5
N2_MIN_FRACTION = 0.8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.04, help="RK4 step")
    args = parser.parse_args()

    base_params = Params()
    stimuli = make_stimuli(base_params)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(D2_JSON, "r", encoding="utf-8") as handle:
        d2 = json.load(handle)

    print("=== D2': E/I gain structure as the information carrier ===")
    cases = []
    for key, cell in d2["cells"].items():
        fe, fi = (float(v) for v in key.split(","))
        for si, seed in enumerate(SEEDS):
            full_mi = cell["per_seed"][si]["mi"]
            if full_mi >= INFORMATIVE_MIN:
                cases.append(
                    {"fe": fe, "fi": fi, "seed": seed, "full_mi": float(full_mi)}
                )
    print(f"informative cases (MI_corr >= {INFORMATIVE_MIN}): {len(cases)}")

    for case in cases:
        p = replace(
            base_params,
            g_ee=base_params.g_ee * case["fe"],
            g_ei=base_params.g_ei * case["fi"],
        )
        seed = case["seed"]
        no_e = run_cell(p, seed, stimuli, args.dt, condition="no_positive")["mi"]
        no_i = run_cell(p, seed, stimuli, args.dt, condition="no_negative")["mi"]
        single = single_ring_mi(p, seed, stimuli, args.dt)
        shuffle = float(
            np.mean(
                [
                    run_cell(
                        p, seed, stimuli, args.dt,
                        conn_perm=shuffle_permutation(
                            base_params.n, seed * 100 + rep
                        ),
                    )["mi"]
                    for rep in range(CONN_SHUFFLE_REPS)
                ]
            )
        )
        case.update(
            {
                "no_positive_mi": float(no_e),
                "no_negative_mi": float(no_i),
                "single_ring_mi": float(single),
                "conn_shuffle_mi": shuffle,
                "n1": bool(max(no_e, no_i, single) <= RATIO * case["full_mi"]),
                "n2": bool(shuffle >= RATIO * case["full_mi"]),
            }
        )
        print(
            f"  ({case['fe']},{case['fi']}) seed={seed} full={case['full_mi']:.3f} "
            f"noE={no_e:.3f} noI={no_i:.3f} single={single:.3f} "
            f"shuffle={shuffle:.3f} N1={case['n1']} N2={case['n2']}"
        )

    n_cases = len(cases)
    n1_count = sum(int(c["n1"]) for c in cases)
    n2_count = sum(int(c["n2"]) for c in cases)
    n2_fraction = n2_count / n_cases if n_cases else 0.0
    n1_pass = bool(n_cases > 0 and n1_count == n_cases)
    n2_pass = bool(n_cases > 0 and n2_fraction >= N2_MIN_FRACTION)
    verdict = "ADOPT" if (n1_pass and n2_pass) else "REJECT"
    print(
        f"\n  N1: {n1_count}/{n_cases} (need all) => {'PASS' if n1_pass else 'FAIL'}"
    )
    print(
        f"  N2: {n2_count}/{n_cases} = {n2_fraction:.1%} (need >= {N2_MIN_FRACTION:.0%})"
        f" => {'PASS' if n2_pass else 'FAIL'}"
    )
    print(f"  D2' verdict: {verdict}")

    output = {
        "experiment": "D2' E/I gain structure as information carrier",
        "formalization_version": "D2prime-v0.1",
        "source_artifact": D2_JSON,
        "protocol": {
            "dt": args.dt,
            "informative_min": INFORMATIVE_MIN,
            "ratio": RATIO,
            "n2_min_fraction": N2_MIN_FRACTION,
            "conn_shuffle_reps": CONN_SHUFFLE_REPS,
        },
        "cases": cases,
        "summary": {
            "n_cases": n_cases,
            "n1_count": n1_count,
            "n2_count": n2_count,
            "n2_fraction": n2_fraction,
            "n1_pass": n1_pass,
            "n2_pass": n2_pass,
            "verdict": verdict,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "d2_prime_carrier.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    x = np.arange(n_cases)
    width = 0.18
    labels = [
        f"({c['fe']},{c['fi']}) s{c['seed']}" for c in cases
    ]
    ax.bar(x - 1.5 * width, [c["full_mi"] for c in cases], width, label="full", color="#2463EB")
    ax.bar(
        x - 0.5 * width,
        [max(c["no_positive_mi"], c["no_negative_mi"], c["single_ring_mi"]) for c in cases],
        width,
        label="max ablation",
        color="#E4572E",
    )
    ax.bar(x + 0.5 * width, [c["conn_shuffle_mi"] for c in cases], width, label="conn shuffle", color="#AF7AC5")
    ax.bar(x + 1.5 * width, [RATIO * c["full_mi"] for c in cases], width, label="0.5 x full", color="#7F8C8D")
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
    ax.set(title="D2' cases: full vs ablation vs shuffle", ylabel="MI_corr (bits)")
    ax.legend(fontsize=8)

    ax = axes[1]
    grid_n1 = {}
    grid_n2 = {}
    for c in cases:
        key = (c["fe"], c["fi"])
        grid_n1.setdefault(key, []).append(c["n1"])
        grid_n2.setdefault(key, []).append(c["n2"])
    factors = sorted({c["fe"] for c in cases} | {c["fi"] for c in cases})
    mat = np.full((len(factors), len(factors)), np.nan)
    for (fe, fi), vals in grid_n2.items():
        mat[factors.index(fe), factors.index(fi)] = np.mean(vals)
    im = ax.imshow(mat, origin="lower", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(factors)), factors)
    ax.set_yticks(range(len(factors)), factors)
    ax.set(title="N2 (shuffle keeps MI) fraction per cell", xlabel="g_ei factor", ylabel="g_ee factor")
    fig.colorbar(im, ax=ax)

    fig.suptitle(f"D2' carrier test: N1={n1_count}/{n_cases}, N2={n2_fraction:.0%} => {verdict}")
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "d2_prime_carrier.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
