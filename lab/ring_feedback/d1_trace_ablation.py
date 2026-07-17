"""C2'': does trace maintenance causally depend on the intact E/I loop?

Pre-registered in dev/LOG.md (2026-07-17, "C2″ 预注册 — 痕迹维持的 E/I 因果依赖").
D1 showed the delay-period cue trace is linearly decodable with acc ~ 1.0, but
decodability is an observation, not a cause: the trace might be maintained by
the intact oscillatory E/I dynamics, or any dynamics — including ablated
systems collapsing to a cue-independent fixed point — might retain it.  C2''
decides causally with pre-registered ablations.

Design (fixed by pre-registration, do not tune after seeing results):

* identical frozen pipeline as C2' (d1_phase_readout): same 64 trajectories
  per point (8 cue nodes x 8 jitter instances), decode window t in [12,42),
  same 40-dim per-node phase features, same ridge decoder (lambda=1.0), same
  stratified split (instances 0-3 train / 4-7 test);
* conditions: full, no_positive (g_ee=0), no_negative (g_ei=0), single_ring
  (A3 tanh ring as in c2_verify.py);
* the dominant frequency of the point's FULL E/I autonomous run is used for
  the phase features of all four conditions;
* pass (single point): S1 full acc >= 0.60; S2 every ablated condition
  acc <= 0.325 (chance 0.125 + 0.20); S3 full label-shuffle acc in
  [0.025, 0.225]; points whose full E/I fails the C1 oscillation check count
  as FAIL (no trace carrier);
* grid verdict over the C2 local grid (0.9/1.0/1.1 x g_ee,g_ei) x seeds 0/1/2:
  pass rate >= 2/3 adopt, <= 1/3 reject, otherwise mixed.
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
    ring_kernel,
    rk4_step,
    sigmoid,
    simulate,
)
from d1_phase_readout import (
    CHANCE,
    CUE_END,
    INSTANCES,
    LABEL_RANGE,
    RIDGE_LAMBDA,
    SHUFFLE_REPS,
    TOTAL_TIME,
    TRAIN_INSTANCES,
    WINDOW,
    accuracy,
    fit_ridge,
    instance_seed,
    one_hot,
    phase_features,
    windowed_e,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

CONDITIONS = ["full", "no_positive", "no_negative", "single_ring"]
S1_ACC = 0.60
S2_ABLATION_MAX = 0.325


def simulate_cue_condition(
    p: Params,
    condition: str,
    cue_node: int,
    seed: int,
    total_time: float,
    dt: float,
) -> Trajectory:
    """Cued trajectory under one of the four pre-registered conditions."""
    if condition == "single_ring":
        rng = np.random.default_rng(seed)
        state = 0.01 * rng.standard_normal(p.n)

        def derivative_single(x: np.ndarray, t: float) -> np.ndarray:
            u = np.zeros(p.n, dtype=np.float64)
            if t < CUE_END:
                u[cue_node] = p.cue_amplitude
            return -x + np.tanh(p.single_ring_gain * ring_kernel(x, p.rho_e) + u)

        steps = int(round(total_time / dt))
        values = np.empty((steps, p.n), dtype=np.float64)
        times = (np.arange(steps, dtype=np.float64) + 1.0) * dt
        for step in range(steps):
            state = rk4_step(state, step * dt, dt, derivative_single)
            values[step] = state
        e = 0.5 * (values + 1.0)
        return Trajectory(times, e, np.zeros_like(e))

    q = p
    if condition == "no_positive":
        q = replace(q, g_ee=0.0)
    elif condition == "no_negative":
        q = replace(q, g_ei=0.0)
    elif condition != "full":
        raise ValueError(f"unknown condition: {condition}")

    rng = np.random.default_rng(seed)
    state = np.concatenate(
        [0.02 + 0.01 * rng.random(q.n), 0.02 + 0.01 * rng.random(q.n)]
    )

    def derivative(x: np.ndarray, t: float) -> np.ndarray:
        e, i = x[: q.n], x[q.n :]
        ke = ring_kernel(e, q.rho_e)
        ki = ring_kernel(i, q.rho_i, reverse=True)
        u = np.zeros(q.n, dtype=np.float64)
        if t < CUE_END:
            u[cue_node] = q.cue_amplitude
        de = (-e + sigmoid(q.g_ee * ke - q.g_ei * ki + u - q.theta_e)) / q.tau_e
        di = (-i + sigmoid(q.g_ie * ke - q.g_ii * ki - q.theta_i)) / q.tau_i
        return np.concatenate([de, di])

    steps = int(round(total_time / dt))
    states = np.empty((steps, 2 * q.n), dtype=np.float64)
    times = (np.arange(steps, dtype=np.float64) + 1.0) * dt
    for step in range(steps):
        state = rk4_step(state, step * dt, dt, derivative)
        states[step] = state
    return Trajectory(times, states[:, : q.n], states[:, q.n :])


def condition_accuracy(
    p: Params, condition: str, seed: int, freq: float, dt: float
) -> tuple[float, Trajectory]:
    """Decode accuracy of the frozen C2' pipeline under one condition."""
    rows, labels = [], []
    example: Trajectory | None = None
    t_win_ref: np.ndarray | None = None
    for node in range(p.n):
        for inst in range(INSTANCES):
            traj = simulate_cue_condition(
                p, condition, node, instance_seed(seed, node, inst), TOTAL_TIME, dt
            )
            e_win, t_win = windowed_e(traj, WINDOW[0], WINDOW[1])
            rows.append(phase_features(e_win, t_win, freq))
            labels.append(node)
            t_win_ref = t_win
            if node == 0 and inst == 0:
                example = traj
    X = np.array(rows)
    y = np.array(labels, dtype=np.int64)
    train = np.tile(np.arange(INSTANCES), p.n) < TRAIN_INSTANCES
    model = fit_ridge(X[train], one_hot(y[train], p.n), RIDGE_LAMBDA)
    return accuracy(model, X[~train], y[~train]), example


def evaluate_point(p: Params, seed: int, dt: float) -> dict:
    """Run the pre-registered C2'' protocol at one parameter point."""
    autonomous = simulate(p, "full", "autonomous", seed, total_time=60.0, dt=dt)
    osc = oscillation_metrics(autonomous)
    point: dict = {"seed": seed, "oscillation": osc}
    if not osc["pass"]:
        point.update({"point_pass": False, "reason": "no_oscillation"})
        return point

    freq = float(osc["dominant_frequency"])
    accs: dict[str, float] = {}
    examples: dict[str, Trajectory] = {}
    for condition in CONDITIONS:
        acc, example = condition_accuracy(p, condition, seed, freq, dt)
        accs[condition] = acc
        examples[condition] = example

    # label-shuffle on the full condition (leakage sanity), 4 reps
    rng = np.random.default_rng(seed * 77 + 13)
    rows, labels = [], []
    for node in range(p.n):
        for inst in range(INSTANCES):
            traj = simulate_cue_condition(
                p, "full", node, instance_seed(seed, node, inst), TOTAL_TIME, dt
            )
            e_win, t_win = windowed_e(traj, WINDOW[0], WINDOW[1])
            rows.append(phase_features(e_win, t_win, freq))
            labels.append(node)
    X = np.array(rows)
    y = np.array(labels, dtype=np.int64)
    train = np.tile(np.arange(INSTANCES), p.n) < TRAIN_INSTANCES
    acc_label = float(
        np.mean(
            [
                accuracy(
                    fit_ridge(
                        X[train],
                        one_hot(y[train][rng.permutation(int(train.sum()))], p.n),
                        RIDGE_LAMBDA,
                    ),
                    X[~train],
                    y[~train],
                )
                for _ in range(SHUFFLE_REPS)
            ]
        )
    )

    s1 = bool(accs["full"] >= S1_ACC)
    s2 = bool(all(accs[c] <= S2_ABLATION_MAX for c in CONDITIONS[1:]))
    s3 = bool(LABEL_RANGE[0] <= acc_label <= LABEL_RANGE[1])
    point.update(
        {
            "dominant_frequency": freq,
            "acc_by_condition": accs,
            "acc_label_shuffle": acc_label,
            "s1_trace_decodable": s1,
            "s2_ablation_destroys": s2,
            "s3_no_leakage": s3,
            "point_pass": bool(s1 and s2 and s3),
        }
    )
    if seed == 0:
        point["_plot_examples"] = examples
    return point


def parse_csv(text: str, cast: Callable) -> list:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def plot_results(base: dict, grid: dict, p: Params, output_path: str) -> None:
    import matplotlib.pyplot as plt

    examples = base.get("_plot_examples", {})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, condition, title in [
        (axes[0, 0], "full", "full E/I (cue node 0, inst 0)"),
        (axes[0, 1], "no_negative", "no negative feedback"),
    ]:
        traj = examples.get(condition)
        if traj is not None:
            mask = traj.time >= WINDOW[0]
            im = ax.imshow(
                traj.e[mask].T,
                aspect="auto",
                origin="lower",
                extent=[traj.time[mask][0], traj.time[mask][-1], -0.5, p.n - 0.5],
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, label="E")
        ax.set(title=title, xlabel="time", ylabel="node")

    ax = axes[1, 0]
    accs = base["acc_by_condition"]
    values = [accs[c] for c in CONDITIONS] + [base["acc_label_shuffle"]]
    names = CONDITIONS + ["label-shuf"]
    colors = ["#2463EB", "#E4572E", "#E4572E", "#7F8C8D", "#17A589"]
    ax.bar(names, values, color=colors)
    ax.axhline(CHANCE, color="black", ls="--", lw=0.8, label="chance")
    ax.axhline(S1_ACC, color="#2463EB", ls=":", lw=0.8, label="S1 threshold")
    ax.axhline(S2_ABLATION_MAX, color="#E4572E", ls=":", lw=0.8, label="S2 threshold")
    ax.set_ylim(0, 1.0)
    ax.set(title="Decode accuracy by condition (base point)", ylabel="test accuracy")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1, 1]
    factors = grid["factors"]
    pass_map = np.zeros((len(factors), len(factors)))
    for run in grid["runs"]:
        fi = factors.index(run["positive_factor"])
        gi = factors.index(run["negative_factor"])
        pass_map[fi, gi] += float(run["point_pass"])
    pass_map /= max(1, len(grid["seeds"]))
    im = ax.imshow(pass_map, origin="lower", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(factors)), factors)
    ax.set_yticks(range(len(factors)), factors)
    ax.set(title="C2'' pass rate per grid cell", xlabel="g_ei factor", ylabel="g_ee factor")
    fig.colorbar(im, ax=ax, label="pass rate")

    fig.suptitle(
        f"C2'': base={'PASS' if base['point_pass'] else 'FAIL'}, "
        f"grid={grid['pass_count']}/{grid['num_runs']} ({grid['verdict']})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base point seed")
    parser.add_argument("--seeds", default="0,1,2", help="grid seeds")
    parser.add_argument("--grid", default="0.9,1.0,1.1", help="local gain factors")
    parser.add_argument("--dt", type=float, default=0.04, help="RK4 step")
    args = parser.parse_args()

    seeds = parse_csv(args.seeds, int)
    factors = parse_csv(args.grid, float)
    base_params = Params()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== C2'': trace maintenance depends on intact E/I? ===")
    print(f"base seed={args.seed}, dt={args.dt}, grid={factors}, seeds={seeds}")

    base = evaluate_point(base_params, args.seed, args.dt)
    if "acc_by_condition" in base:
        print("\nBase point")
        print(f"  dominant frequency : {base['dominant_frequency']:.4f}")
        for condition in CONDITIONS:
            print(f"  {condition:12s} acc={base['acc_by_condition'][condition]:.4f}")
        print(f"  label-shuffle acc={base['acc_label_shuffle']:.4f}")
        for key in ["s1_trace_decodable", "s2_ablation_destroys", "s3_no_leakage"]:
            print(f"  {key:22s} {'PASS' if base[key] else 'FAIL'}")
    else:
        print(f"\nBase point: no oscillation ({base['reason']})")
    print(f"  base verdict: {'PASS' if base['point_pass'] else 'FAIL'}")

    print("\nRunning pre-registered local grid...")
    runs = []
    for seed in seeds:
        for positive_factor in factors:
            for negative_factor in factors:
                p = replace(
                    base_params,
                    g_ee=base_params.g_ee * positive_factor,
                    g_ei=base_params.g_ei * negative_factor,
                )
                point = evaluate_point(p, seed, args.dt)
                point.pop("_plot_examples", None)
                point.update(
                    {
                        "positive_factor": positive_factor,
                        "negative_factor": negative_factor,
                    }
                )
                runs.append(point)
    pass_count = sum(int(run["point_pass"]) for run in runs)
    num_runs = len(runs)
    pass_rate = pass_count / num_runs
    if pass_rate >= 2.0 / 3.0:
        verdict = "ADOPT"
    elif pass_rate <= 1.0 / 3.0:
        verdict = "REJECT"
    else:
        verdict = "MIXED"
    grid = {
        "factors": factors,
        "seeds": seeds,
        "num_runs": num_runs,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "verdict": verdict,
        "runs": runs,
    }
    print(f"  C2'' grid pass: {pass_count}/{num_runs} ({pass_rate:.1%}) => {verdict}")

    plot_examples = base.pop("_plot_examples", None)
    output = {
        "experiment": "C2'' trace maintenance under E/I ablation",
        "formalization_version": "C2pp-v0.1",
        "parameters": asdict(base_params),
        "protocol": {
            "dt": args.dt,
            "cue": "node k in {0..7}, amplitude 5, t in [0,2)",
            "instances_per_node": INSTANCES,
            "total_time": TOTAL_TIME,
            "decode_window": list(WINDOW),
            "conditions": CONDITIONS,
            "ridge_lambda": RIDGE_LAMBDA,
            "shuffle_reps": SHUFFLE_REPS,
            "thresholds": {
                "s1_acc": S1_ACC,
                "s2_ablation_max": S2_ABLATION_MAX,
                "label_range": list(LABEL_RANGE),
                "chance": CHANCE,
            },
        },
        "base": base,
        "grid": grid,
    }
    json_path = os.path.join(RESULTS_DIR, "d1_trace_ablation.json")
    png_path = os.path.join(RESULTS_DIR, "d1_trace_ablation.png")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    if plot_examples is not None:
        base["_plot_examples"] = plot_examples
    plot_results(base, grid, base_params, png_path)
    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
