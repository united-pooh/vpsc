"""C2 judge experiment: are positive-memory and negative-gating roles separable?

The C-line model is a Wilson--Cowan excitatory/inhibitory (E/I) dual ring.
It tests two claims separately:

* C1: the complete E/I system has a bounded non-fixed oscillation that is lost
  when either feedback path is ablated;
* C2: positive feedback is specifically responsible for delayed memory, while
  negative feedback is specifically responsible for distractor gating.

The functional claim is tested with crossed interventions, not only by asking
whether the full system performs better.  A local 3 x 3 gain grid and multiple
seeds are reported so a single tuned point cannot establish either claim.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


@dataclass(frozen=True)
class Params:
    n: int = 8
    g_ee: float = 7.75
    g_ei: float = 6.70
    g_ie: float = 10.0
    g_ii: float = 6.30
    theta_e: float = 2.50
    theta_i: float = 5.75
    tau_e: float = 1.0
    tau_i: float = 5.80
    rho_e: float = 0.15
    rho_i: float = 0.15
    cue_amplitude: float = 5.0
    kappa: float = 6.0
    lambda_i: float = 1.0
    single_ring_gain: float = 2.0


@dataclass
class Trajectory:
    time: np.ndarray
    e: np.ndarray
    i: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def ring_kernel(x: np.ndarray, rho: float, reverse: bool = False) -> np.ndarray:
    """Local-plus-neighbor cyclic propagation; roll closes the directed ring."""
    shift = -1 if reverse else 1
    return (1.0 - rho) * x + rho * np.roll(x, shift)


def external_drive(t: float, p: Params, protocol: str) -> np.ndarray:
    """Pre-registered pulse--delay--distractor inputs."""
    u = np.zeros(p.n, dtype=np.float64)
    if protocol in {"autonomous", "memory", "gate"} and t < 2.0:
        u[0] = p.cue_amplitude
    if protocol == "gate" and 6.0 <= t < 8.0:
        u[p.n // 2] = p.cue_amplitude
    return u


def rk4_step(
    state: np.ndarray,
    t: float,
    dt: float,
    derivative: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    k1 = derivative(state, t)
    k2 = derivative(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivative(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivative(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_ei(
    p: Params,
    condition: str,
    protocol: str,
    seed: int,
    total_time: float,
    dt: float,
) -> Trajectory:
    """Integrate the full or ablated E/I ring with matched inputs and initials."""
    if condition not in {"full", "no_positive", "no_negative"}:
        raise ValueError(f"unknown E/I condition: {condition}")

    q = p
    if condition == "no_positive":
        q = replace(q, g_ee=0.0)
    elif condition == "no_negative":
        q = replace(q, g_ei=0.0)

    rng = np.random.default_rng(seed)
    state = np.concatenate(
        [0.02 + 0.01 * rng.random(q.n), 0.02 + 0.01 * rng.random(q.n)]
    )

    def derivative(x: np.ndarray, t: float) -> np.ndarray:
        e, i = x[: q.n], x[q.n :]
        ke = ring_kernel(e, q.rho_e)
        ki = ring_kernel(i, q.rho_i, reverse=True)
        u = external_drive(t, q, protocol)
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


def simulate_single_ring(
    p: Params,
    protocol: str,
    seed: int,
    total_time: float,
    dt: float,
) -> Trajectory:
    """A3-style single monotone tanh ring, retained as the prior baseline."""
    rng = np.random.default_rng(seed)
    state = 0.01 * rng.standard_normal(p.n)

    def derivative(x: np.ndarray, t: float) -> np.ndarray:
        u = external_drive(t, p, protocol)
        return -x + np.tanh(p.single_ring_gain * ring_kernel(x, p.rho_e) + u)

    steps = int(round(total_time / dt))
    values = np.empty((steps, p.n), dtype=np.float64)
    times = (np.arange(steps, dtype=np.float64) + 1.0) * dt
    for step in range(steps):
        state = rk4_step(state, step * dt, dt, derivative)
        values[step] = state

    # Map tanh state to [0, 1] only so it can share the attention readout.
    e = 0.5 * (values + 1.0)
    return Trajectory(times, e, np.zeros_like(e))


def simulate(
    p: Params,
    condition: str,
    protocol: str,
    seed: int,
    total_time: float,
    dt: float,
) -> Trajectory:
    if condition == "single_ring":
        return simulate_single_ring(p, protocol, seed, total_time, dt)
    return simulate_ei(p, condition, protocol, seed, total_time, dt)


def attention(traj: Trajectory, p: Params) -> np.ndarray:
    logits = p.kappa * (traj.e - p.lambda_i * traj.i)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def oscillation_metrics(
    traj: Trajectory,
    std_threshold: float = 0.02,
    purity_threshold: float = 0.50,
) -> dict[str, float | bool]:
    """Measure late oscillation; spectral purity includes peak leakage neighbors."""
    late = traj.e[len(traj.e) // 2 :]
    late_std = float(late.std(axis=0).mean())
    centered = late - late.mean(axis=0, keepdims=True)
    window = np.hanning(len(centered))[:, None]
    power = (np.abs(np.fft.rfft(centered * window, axis=0)) ** 2).sum(axis=1)
    power[0] = 0.0
    peak = int(np.argmax(power))
    lo, hi = max(1, peak - 1), min(len(power), peak + 2)
    total_power = float(power[1:].sum())
    spectral_purity = float(power[lo:hi].sum() / max(total_power, 1e-15))
    dt = float(traj.time[1] - traj.time[0])
    dominant_frequency = float(peak / (len(centered) * dt))
    all_state = np.concatenate([traj.e, traj.i], axis=1)
    bounded = bool(
        np.isfinite(all_state).all()
        and all_state.min() >= -1e-6
        and all_state.max() <= 1.0 + 1e-6
    )
    passed = bool(
        bounded and late_std > std_threshold and spectral_purity > purity_threshold
    )
    return {
        "late_std": late_std,
        "spectral_purity": spectral_purity,
        "dominant_frequency": dominant_frequency,
        "state_min": float(all_state.min()),
        "state_max": float(all_state.max()),
        "bounded": bounded,
        "pass": passed,
    }


def interval_mean(values: np.ndarray, time: np.ndarray, start: float, end: float) -> float:
    mask = (time >= start) & (time < end)
    if not np.any(mask):
        raise ValueError(f"empty readout interval [{start}, {end})")
    return float(values[mask].mean())


def functional_metrics(
    memory: dict[str, Trajectory],
    gate: dict[str, Trajectory],
    p: Params,
) -> tuple[dict[str, dict[str, float]], dict[str, float | bool]]:
    """Compute C2's memory/gating crossed-intervention matrix."""
    scores: dict[str, dict[str, float]] = {}
    for condition in memory:
        memory_attention = attention(memory[condition], p)
        gate_attention = attention(gate[condition], p)
        m_score = interval_mean(memory_attention[:, 0], memory[condition].time, 10.0, 12.0)
        target = interval_mean(gate_attention[:, 0], gate[condition].time, 8.0, 10.0)
        distractor = interval_mean(
            gate_attention[:, p.n // 2], gate[condition].time, 8.0, 10.0
        )
        scores[condition] = {
            "memory_target_attention": m_score,
            "gate_target_attention": target,
            "gate_distractor_attention": distractor,
            "gate_advantage": target - distractor,
        }

    full = scores["full"]
    no_e = scores["no_positive"]
    no_i = scores["no_negative"]
    delta_e_m = full["memory_target_attention"] - no_e["memory_target_attention"]
    delta_i_m = full["memory_target_attention"] - no_i["memory_target_attention"]
    delta_i_g = full["gate_advantage"] - no_i["gate_advantage"]
    delta_e_g = full["gate_advantage"] - no_e["gate_advantage"]
    passed = bool(
        delta_e_m > 0.10
        and delta_i_g > 0.10
        and delta_e_m > 2.0 * abs(delta_i_m)
        and delta_i_g > 2.0 * abs(delta_e_g)
    )
    verdict: dict[str, float | bool] = {
        "delta_positive_memory": delta_e_m,
        "delta_negative_memory_cross": delta_i_m,
        "delta_negative_gating": delta_i_g,
        "delta_positive_gating_cross": delta_e_g,
        "primary_memory_threshold_pass": bool(delta_e_m > 0.10),
        "primary_gating_threshold_pass": bool(delta_i_g > 0.10),
        "memory_diagonal_dominance_pass": bool(delta_e_m > 2.0 * abs(delta_i_m)),
        "gating_diagonal_dominance_pass": bool(delta_i_g > 2.0 * abs(delta_e_g)),
        "pass": passed,
    }
    return scores, verdict


def evaluate_point(
    p: Params,
    seed: int,
    dt: float,
    include_single: bool = False,
) -> tuple[dict, dict[str, dict[str, Trajectory]]]:
    conditions = ["full", "no_positive", "no_negative"]
    if include_single:
        conditions.append("single_ring")

    trajectories: dict[str, dict[str, Trajectory]] = {
        "autonomous": {},
        "memory": {},
        "gate": {},
    }
    for condition in conditions:
        trajectories["autonomous"][condition] = simulate(
            p, condition, "autonomous", seed, total_time=60.0, dt=dt
        )
        trajectories["memory"][condition] = simulate(
            p, condition, "memory", seed, total_time=14.0, dt=dt
        )
        trajectories["gate"][condition] = simulate(
            p, condition, "gate", seed, total_time=14.0, dt=dt
        )

    oscillation = {
        condition: oscillation_metrics(traj)
        for condition, traj in trajectories["autonomous"].items()
    }
    c1_pass = bool(
        oscillation["full"]["pass"]
        and not oscillation["no_positive"]["pass"]
        and not oscillation["no_negative"]["pass"]
    )
    scores, c2 = functional_metrics(trajectories["memory"], trajectories["gate"], p)
    result = {
        "seed": seed,
        "oscillation": oscillation,
        "c1_pass": c1_pass,
        "functional_scores": scores,
        "c2": c2,
    }
    return result, trajectories


def parse_csv(text: str, cast: Callable) -> list:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def robustness_grid(
    base: Params,
    factors: list[float],
    seeds: list[int],
    dt: float,
) -> dict:
    runs = []
    for seed in seeds:
        for positive_factor in factors:
            for negative_factor in factors:
                p = replace(
                    base,
                    g_ee=base.g_ee * positive_factor,
                    g_ei=base.g_ei * negative_factor,
                )
                result, _ = evaluate_point(p, seed, dt, include_single=False)
                runs.append(
                    {
                        "seed": seed,
                        "positive_factor": positive_factor,
                        "negative_factor": negative_factor,
                        "full_late_std": result["oscillation"]["full"]["late_std"],
                        "full_spectral_purity": result["oscillation"]["full"]["spectral_purity"],
                        "c1_pass": result["c1_pass"],
                        "c2_pass": result["c2"]["pass"],
                        "c2_deltas": {
                            key: value
                            for key, value in result["c2"].items()
                            if key.startswith("delta_")
                        },
                    }
                )
    count = len(runs)
    c1_count = sum(int(run["c1_pass"]) for run in runs)
    c2_count = sum(int(run["c2_pass"]) for run in runs)
    return {
        "factors": factors,
        "seeds": seeds,
        "num_runs": count,
        "c1_pass_count": c1_count,
        "c1_pass_rate": c1_count / count,
        "c2_pass_count": c2_count,
        "c2_pass_rate": c2_count / count,
        "runs": runs,
    }


def plot_results(
    base: dict,
    trajectories: dict[str, dict[str, Trajectory]],
    p: Params,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "full": "#2463EB",
        "no_positive": "#E4572E",
        "no_negative": "#17A589",
        "single_ring": "#7F8C8D",
    }
    labels = {
        "full": "full E/I",
        "no_positive": "no positive",
        "no_negative": "no negative",
        "single_ring": "A3 single ring",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for condition, traj in trajectories["autonomous"].items():
        ax.plot(traj.time, traj.e[:, 0], color=colors[condition], lw=1.4, label=labels[condition])
    ax.axvspan(0, 2, color="gold", alpha=0.18, label="cue")
    ax.set(title="Autonomous test: target E state", xlabel="time", ylabel="E[0]")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    full = trajectories["autonomous"]["full"]
    half = len(full.time) // 2
    ax.plot(full.e[half:, 0], full.i[half:, 0], color=colors["full"], lw=1.2)
    ax.set(title="Full E/I late phase portrait", xlabel="E[0]", ylabel="I[0]")

    ax = axes[1, 0]
    for condition, traj in trajectories["memory"].items():
        ax.plot(
            traj.time,
            attention(traj, p)[:, 0],
            color=colors[condition],
            lw=1.4,
            label=labels[condition],
        )
    ax.axvspan(10, 12, color="black", alpha=0.08, label="memory readout")
    ax.axhline(1.0 / p.n, color="black", ls="--", lw=0.8)
    ax.set(title="Memory assay: target attention", xlabel="time", ylabel="a[target]")

    ax = axes[1, 1]
    conditions = ["full", "no_positive", "no_negative"]
    memory = [base["functional_scores"][c]["memory_target_attention"] for c in conditions]
    gating = [base["functional_scores"][c]["gate_advantage"] for c in conditions]
    x = np.arange(len(conditions))
    width = 0.36
    ax.bar(x - width / 2, memory, width, label="memory M", color="#5DADE2")
    ax.bar(x + width / 2, gating, width, label="gating G", color="#AF7AC5")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [labels[c] for c in conditions], rotation=10)
    ax.set(title="Crossed interventions", ylabel="score")
    ax.legend()

    fig.suptitle(
        f"C2: C1={'PASS' if base['c1_pass'] else 'FAIL'}, "
        f"C2={'PASS' if base['c2']['pass'] else 'FAIL'}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="base run seed")
    parser.add_argument("--seeds", default="0,1,2", help="robustness seeds")
    parser.add_argument("--grid", default="0.9,1.0,1.1", help="local gain factors")
    parser.add_argument("--dt", type=float, default=0.04, help="RK4 step")
    args = parser.parse_args()

    seeds = parse_csv(args.seeds, int)
    factors = parse_csv(args.grid, float)
    base_params = Params()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== C-line v0.1: E/I dual-ring C2 judge experiment ===")
    print(f"base seed={args.seed}, dt={args.dt}, grid={factors}, seeds={seeds}")
    base, trajectories = evaluate_point(base_params, args.seed, args.dt, include_single=True)

    print("\nC1 autonomous dynamics")
    for condition, metrics in base["oscillation"].items():
        print(
            f"  {condition:12s} late_std={metrics['late_std']:.4f} "
            f"purity={metrics['spectral_purity']:.4f} "
            f"freq={metrics['dominant_frequency']:.4f} "
            f"=> {'PASS' if metrics['pass'] else 'FAIL'}"
        )
    print(f"  C1 verdict: {'PASS' if base['c1_pass'] else 'FAIL'}")

    print("\nC2 crossed interventions")
    for condition, scores in base["functional_scores"].items():
        print(
            f"  {condition:12s} M={scores['memory_target_attention']:+.4f} "
            f"G={scores['gate_advantage']:+.4f}"
        )
    for key, value in base["c2"].items():
        if key.startswith("delta_"):
            print(f"  {key:31s} {value:+.4f}")
    print(f"  C2 verdict: {'PASS' if base['c2']['pass'] else 'FAIL'}")

    print("\nRunning pre-registered local robustness grid...")
    robustness = robustness_grid(base_params, factors, seeds, args.dt)
    print(
        f"  C1 robust pass: {robustness['c1_pass_count']}/{robustness['num_runs']} "
        f"({robustness['c1_pass_rate']:.1%})"
    )
    print(
        f"  C2 robust pass: {robustness['c2_pass_count']}/{robustness['num_runs']} "
        f"({robustness['c2_pass_rate']:.1%})"
    )

    output = {
        "experiment": "C2 positive-memory / negative-gating separability",
        "formalization_version": "C-v0.1",
        "parameters": asdict(base_params),
        "protocol": {
            "dt": args.dt,
            "cue": "target node 0, amplitude 5, t in [0,2)",
            "memory_readout": "mean target attention, t in [10,12)",
            "distractor": "node N/2, amplitude 5, t in [6,8)",
            "gating_readout": "mean target minus distractor attention, t in [8,10)",
            "oscillation_thresholds": {"late_std": 0.02, "spectral_purity": 0.50},
            "c2_thresholds": {
                "primary_effect": 0.10,
                "diagonal_dominance_ratio": 2.0,
            },
        },
        "base": base,
        "robustness": robustness,
    }
    json_path = os.path.join(RESULTS_DIR, "c2_verify.json")
    png_path = os.path.join(RESULTS_DIR, "c2_verify.png")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    plot_results(base, trajectories, base_params, png_path)
    print(f"\n[written {json_path}]")
    print(f"[written {png_path}]")


if __name__ == "__main__":
    main()
