"""D1 / C2': is the cue encoded in the phase/relative timing of the delay trajectory?

Pre-registered in dev/LOG.md (2026-07-17, "D1 预注册 — C2′ phase/polychronous 读出").
C2 showed the fixed-position readout cannot recover the cue (0/27).  C2' asks
whether the cue position is instead encoded in the relative phase / temporal
structure of the delay-period trajectory of the full E/I ring.

Design (fixed by pre-registration, do not tune after seeing results):

* full E/I ring, same Params() and integrator as c2_verify.py;
* stimuli: cue node k in {0..7}, amplitude 5, t in [0,2), then free run to
  total_time=44; 8 instances per node differing only in initial-state jitter;
* decode window t in [12,42); fixed-position readout window t in [40,42);
* phase features per node: [mean, std, amplitude, sin(phi), cos(phi)] at the
  point's autonomous dominant frequency; fixed features: mean softmax
  attention over the late window (C2 readout);
* decoder: closed-form ridge regression (lambda=1.0) to one-hot, argmax;
  stratified split instances 0-3 train / 4-7 test;
* negative controls (4 reps each): time-shuffle, per-node delay-shuffle,
  label-shuffle;
* pass (single point): R1 phase acc >= 0.60 and >= fixed acc + 0.30;
  R2/R3 shuffle acc loses at least half the above-chance advantage;
  R4 label-shuffle acc in [0.025, 0.225]; points failing the C1 oscillation
  check count as FAIL (no phase carrier);
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
    attention,
    oscillation_metrics,
    ring_kernel,
    rk4_step,
    sigmoid,
    simulate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Pre-registered constants.
INSTANCES = 8
TRAIN_INSTANCES = 4
TOTAL_TIME = 44.0
CUE_END = 2.0
WINDOW = (12.0, 42.0)
FIXED_WINDOW = (40.0, 42.0)
RIDGE_LAMBDA = 1.0
SHUFFLE_REPS = 4
CHANCE = 1.0 / 8.0
R1_ACC = 0.60
R1_MARGIN = 0.30
SHUFFLE_FACTOR = 0.5
LABEL_RANGE = (0.025, 0.225)


def instance_seed(seed: int, node: int, instance: int) -> int:
    """Deterministic per-trajectory jitter seed."""
    return seed * 1000 + node * 100 + instance


def simulate_cue(
    p: Params, cue_node: int, seed: int, total_time: float, dt: float
) -> Trajectory:
    """Full E/I ring with the cue delivered to an arbitrary node."""
    rng = np.random.default_rng(seed)
    state = np.concatenate(
        [0.02 + 0.01 * rng.random(p.n), 0.02 + 0.01 * rng.random(p.n)]
    )

    def derivative(x: np.ndarray, t: float) -> np.ndarray:
        e, i = x[: p.n], x[p.n :]
        ke = ring_kernel(e, p.rho_e)
        ki = ring_kernel(i, p.rho_i, reverse=True)
        u = np.zeros(p.n, dtype=np.float64)
        if t < CUE_END:
            u[cue_node] = p.cue_amplitude
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


def windowed_e(traj: Trajectory, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (traj.time >= start) & (traj.time < end)
    if not np.any(mask):
        raise ValueError(f"empty decode window [{start}, {end})")
    return traj.e[mask], traj.time[mask]


def phase_features(e_win: np.ndarray, t_win: np.ndarray, freq: float) -> np.ndarray:
    """Per-node [mean, std, amplitude, sin(phi), cos(phi)] at the dominant frequency."""
    t = t_win - t_win[0]
    ref = 2.0 * np.pi * freq * t
    cos_ref = np.cos(ref)[:, None]
    sin_ref = np.sin(ref)[:, None]
    centered = e_win - e_win.mean(axis=0, keepdims=True)
    a = 2.0 * (centered * cos_ref).mean(axis=0)
    b = 2.0 * (centered * sin_ref).mean(axis=0)
    amplitude = np.hypot(a, b)
    phi = np.arctan2(b, a)
    return np.concatenate(
        [e_win.mean(axis=0), e_win.std(axis=0), amplitude, np.sin(phi), np.cos(phi)]
    )


def fixed_features(traj: Trajectory, p: Params) -> np.ndarray:
    """C2-style readout: mean softmax attention over the late fixed window."""
    att = attention(traj, p)
    mask = (traj.time >= FIXED_WINDOW[0]) & (traj.time < FIXED_WINDOW[1])
    if not np.any(mask):
        raise ValueError("empty fixed readout window")
    return att[mask].mean(axis=0)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    # Features are O(1); a column varying by <1e-6 across samples is numerical
    # noise (e.g. ablated fixed-point conditions), not signal — treat as
    # constant so standardization cannot blow it up to overflow.
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd
    Zb = np.hstack([Z, np.ones((len(Z), 1))])
    reg = lam * np.eye(Zb.shape[1])
    reg[-1, -1] = 0.0  # bias is not penalized
    # SVD-based lstsq: same ridge solution as solve() but stable when the
    # feature matrix is near-collinear (rotational symmetry makes many
    # columns nearly identical). umath_linalg emits spurious divide/overflow
    # warnings on such systems even though the inputs are finite and the
    # minimum-norm solution is valid (verified); silence them locally.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        W, *_ = np.linalg.lstsq(Zb.T @ Zb + reg, Zb.T @ Y, rcond=None)
    return mu, sd, W


def predict_ridge(model: tuple[np.ndarray, np.ndarray, np.ndarray], X: np.ndarray) -> np.ndarray:
    mu, sd, W = model
    Zb = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = Zb @ W
    return np.argmax(scores, axis=1)


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    return float((predict_ridge(model, X) == y).mean())


def build_dataset(
    p: Params, seed: int, freq: float, dt: float
) -> dict[str, np.ndarray | list[np.ndarray] | tuple[np.ndarray, np.ndarray] | Trajectory]:
    """Simulate the pre-registered 8 nodes x 8 instances and compute features."""
    phase_rows, fixed_rows, labels, e_windows = [], [], [], []
    example_traj: Trajectory | None = None
    t_win_ref: np.ndarray | None = None
    for node in range(p.n):
        for inst in range(INSTANCES):
            traj = simulate_cue(p, node, instance_seed(seed, node, inst), TOTAL_TIME, dt)
            e_win, t_win = windowed_e(traj, WINDOW[0], WINDOW[1])
            phase_rows.append(phase_features(e_win, t_win, freq))
            fixed_rows.append(fixed_features(traj, p))
            labels.append(node)
            e_windows.append(e_win)
            t_win_ref = t_win
            if node == 0 and inst == 0:
                example_traj = traj
    y = np.array(labels, dtype=np.int64)
    train = (y >= 0) & (np.tile(np.arange(INSTANCES), p.n) < TRAIN_INSTANCES)
    return {
        "phase": np.array(phase_rows),
        "fixed": np.array(fixed_rows),
        "y": y,
        "train": train,
        "e_windows": e_windows,
        "t_win": t_win_ref,
        "example": example_traj,
    }


def shuffled_phase_accuracy(
    e_windows: list[np.ndarray],
    t_win: np.ndarray,
    freq: float,
    y: np.ndarray,
    train: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> float:
    """Accuracy after destroying temporal structure inside the decode window."""
    rows = []
    for e_win in e_windows:
        if mode == "time":
            e2 = e_win[rng.permutation(len(e_win))]
        elif mode == "delay":
            shifts = rng.integers(0, len(e_win), size=e_win.shape[1])
            e2 = np.stack(
                [np.roll(e_win[:, j], shifts[j]) for j in range(e_win.shape[1])], axis=1
            )
        else:
            raise ValueError(f"unknown shuffle mode: {mode}")
        rows.append(phase_features(e2, t_win, freq))
    X = np.array(rows)
    model = fit_ridge(X[train], one_hot(y[train], 8), RIDGE_LAMBDA)
    return accuracy(model, X[~train], y[~train])


def evaluate_point(p: Params, seed: int, dt: float) -> dict:
    """Run the full pre-registered C2' protocol at one parameter point."""
    autonomous = simulate(p, "full", "autonomous", seed, total_time=60.0, dt=dt)
    osc = oscillation_metrics(autonomous)
    point: dict = {"seed": seed, "oscillation": osc}
    if not osc["pass"]:
        point.update(
            {
                "point_pass": False,
                "reason": "no_oscillation",
            }
        )
        return point

    freq = float(osc["dominant_frequency"])
    data = build_dataset(p, seed, freq, dt)
    Xp, Xf, y, train = data["phase"], data["fixed"], data["y"], data["train"]
    Y = one_hot(y, 8)

    phase_model = fit_ridge(Xp[train], Y[train], RIDGE_LAMBDA)
    acc_phase = accuracy(phase_model, Xp[~train], y[~train])
    fixed_model = fit_ridge(Xf[train], Y[train], RIDGE_LAMBDA)
    acc_fixed = accuracy(fixed_model, Xf[~train], y[~train])

    rng = np.random.default_rng(seed * 77 + 13)
    acc_time = float(
        np.mean(
            [
                shuffled_phase_accuracy(
                    data["e_windows"], data["t_win"], freq, y, train, "time", rng
                )
                for _ in range(SHUFFLE_REPS)
            ]
        )
    )
    acc_delay = float(
        np.mean(
            [
                shuffled_phase_accuracy(
                    data["e_windows"], data["t_win"], freq, y, train, "delay", rng
                )
                for _ in range(SHUFFLE_REPS)
            ]
        )
    )
    acc_label = float(
        np.mean(
            [
                accuracy(
                    fit_ridge(
                        Xp[train],
                        one_hot(y[train][rng.permutation(int(train.sum()))], 8),
                        RIDGE_LAMBDA,
                    ),
                    Xp[~train],
                    y[~train],
                )
                for _ in range(SHUFFLE_REPS)
            ]
        )
    )

    r1 = bool(acc_phase >= R1_ACC and acc_phase >= acc_fixed + R1_MARGIN)
    r2 = bool((acc_time - CHANCE) <= SHUFFLE_FACTOR * (acc_phase - CHANCE))
    r3 = bool((acc_delay - CHANCE) <= SHUFFLE_FACTOR * (acc_phase - CHANCE))
    r4 = bool(LABEL_RANGE[0] <= acc_label <= LABEL_RANGE[1])
    point.update(
        {
            "dominant_frequency": freq,
            "acc_phase": acc_phase,
            "acc_fixed": acc_fixed,
            "acc_time_shuffle": acc_time,
            "acc_delay_shuffle": acc_delay,
            "acc_label_shuffle": acc_label,
            "r1_decodable": r1,
            "r2_time_structure": r2,
            "r3_relative_phase": r3,
            "r4_no_leakage": r4,
            "point_pass": bool(r1 and r2 and r3 and r4),
        }
    )
    if seed == 0:
        point["_plot_data"] = data
    return point


def parse_csv(text: str, cast: Callable) -> list:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def plot_results(base: dict, grid: dict, p: Params, output_path: str) -> None:
    import matplotlib.pyplot as plt

    data = base.get("_plot_data")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    if data is not None:
        traj: Trajectory = data["example"]
        mask = traj.time >= WINDOW[0]
        im = ax.imshow(
            traj.e[mask].T,
            aspect="auto",
            origin="lower",
            extent=[traj.time[mask][0], traj.time[mask][-1], -0.5, p.n - 0.5],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, label="E")
    ax.set(title="Delay trajectory (cue node 0, instance 0)", xlabel="time", ylabel="node")

    ax = axes[0, 1]
    if data is not None:
        Xp, y, train = data["phase"], data["y"], data["train"]
        # node-0 (sin(phi), cos(phi)) for every trajectory, colored by cue class
        for node_cls in range(p.n):
            sel = y == node_cls
            marker = "o" if node_cls < 4 else "s"
            ax.scatter(
                Xp[sel & train][:, 3 * p.n],
                Xp[sel & train][:, 4 * p.n],
                marker=marker,
                alpha=0.7,
                label=f"cue {node_cls} train",
            )
            ax.scatter(
                Xp[sel & ~train][:, 3 * p.n],
                Xp[sel & ~train][:, 4 * p.n],
                marker=marker,
                edgecolors="black",
                facecolors="none",
                alpha=0.9,
            )
        ax.legend(fontsize=7, ncol=2)
    ax.set(title="Node-0 phase features (open = test)", xlabel="sin(phi)", ylabel="cos(phi)")

    ax = axes[1, 0]
    names = ["phase", "fixed", "time-shuf", "delay-shuf", "label-shuf"]
    values = [
        base["acc_phase"],
        base["acc_fixed"],
        base["acc_time_shuffle"],
        base["acc_delay_shuffle"],
        base["acc_label_shuffle"],
    ]
    colors = ["#2463EB", "#7F8C8D", "#E4572E", "#E4572E", "#17A589"]
    ax.bar(names, values, color=colors)
    ax.axhline(CHANCE, color="black", ls="--", lw=0.8, label="chance")
    ax.axhline(R1_ACC, color="#2463EB", ls=":", lw=0.8, label="R1 threshold")
    ax.set_ylim(0, 1.0)
    ax.set(title="Decoder accuracy (base point)", ylabel="test accuracy")
    ax.legend(fontsize=8)

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
    ax.set(title="C2' pass rate per grid cell", xlabel="g_ei factor", ylabel="g_ee factor")
    fig.colorbar(im, ax=ax, label="pass rate")

    fig.suptitle(
        f"D1/C2': base={'PASS' if base['point_pass'] else 'FAIL'}, "
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

    print("=== D1 / C2': phase/polychronous readout judge experiment ===")
    print(f"base seed={args.seed}, dt={args.dt}, grid={factors}, seeds={seeds}")

    base = evaluate_point(base_params, args.seed, args.dt)
    if "acc_phase" in base:
        print("\nBase point")
        print(f"  dominant frequency : {base['dominant_frequency']:.4f}")
        print(f"  phase decoder      : {base['acc_phase']:.4f}")
        print(f"  fixed readout      : {base['acc_fixed']:.4f}")
        print(f"  time-shuffle       : {base['acc_time_shuffle']:.4f}")
        print(f"  delay-shuffle      : {base['acc_delay_shuffle']:.4f}")
        print(f"  label-shuffle      : {base['acc_label_shuffle']:.4f}")
        for key in ["r1_decodable", "r2_time_structure", "r3_relative_phase", "r4_no_leakage"]:
            print(f"  {key:18s} {'PASS' if base[key] else 'FAIL'}")
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
                point.pop("_plot_data", None)
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
    print(f"  C2' grid pass: {pass_count}/{num_runs} ({pass_rate:.1%}) => {verdict}")

    plot_data = base.pop("_plot_data", None)
    output = {
        "experiment": "D1 / C2' phase-polychronous readout",
        "formalization_version": "C2prime-v0.1",
        "parameters": asdict(base_params),
        "protocol": {
            "dt": args.dt,
            "cue": "node k in {0..7}, amplitude 5, t in [0,2)",
            "instances_per_node": INSTANCES,
            "total_time": TOTAL_TIME,
            "decode_window": list(WINDOW),
            "fixed_window": list(FIXED_WINDOW),
            "ridge_lambda": RIDGE_LAMBDA,
            "shuffle_reps": SHUFFLE_REPS,
            "thresholds": {
                "r1_acc": R1_ACC,
                "r1_margin": R1_MARGIN,
                "shuffle_factor": SHUFFLE_FACTOR,
                "label_range": list(LABEL_RANGE),
                "chance": CHANCE,
            },
        },
        "base": base,
        "grid": grid,
    }
    json_path = os.path.join(RESULTS_DIR, "d1_phase_readout.json")
    png_path = os.path.join(RESULTS_DIR, "d1_phase_readout.png")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    if plot_data is not None:
        base["_plot_data"] = plot_data
    plot_results(base, grid, base_params, png_path)
    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
