"""A3 nonlinear: can tanh mean-field nodes on a directed cycle produce sustained
oscillatory trajectories (limit cycles) that carry temporal structure — the
property the linear semigroup (a3_verify.py) failed to show?

A3 linear version was REFUTED (see dev/LOG.md): linear e^{t(P-I)} always
collapses to the stationary distribution; the cycle trajectory energy decayed to
1.2% by t=30; no persistent modes. The log flagged that "true non-convergence
needs nonlinearity (tanh saturation) or gap=0". This script tests the nonlinear
case — A3's last chance.

Nonlinear dynamics: x_{t+1} = (1 - leak) x_t + leak * tanh(W_rec x_t + b)
- W_rec: directed-cycle recurrent weights (asymmetric => complex spectrum)
- tanh saturation: can turn the spiral-in fixed point of the linear part into a
  stable limit cycle (sustained oscillation), because saturation bounds |x|
  while the linear rotation keeps spinning.

Predictions (nonlinear A3):
  (Q1) A directed cycle with tanh nodes produces a SUSTAINED oscillation
       (trajectory energy does NOT collapse; stays bounded away from 0).
       Compare: linear cycle collapses (a3_verify P4).
  (Q2) DECISIVE: a linear readout on the nonlinear trajectory fits a temporal
       task (periodic signal prediction) WITHOUT a time index, and significantly
       better than a DAG/chain with tanh (matched size).
  (Q3) The oscillation frequency is set by the cycle length / weight scale
       (predictable from the linearized spectrum).
  (Q4) Failure: if the nonlinear trajectory still collapses to a fixed point
       (no limit cycle), A3 nonlinear is also refuted.

Honesty: a limit cycle here is a property of the autonomous dynamics (no input
after t=0). Real temporal tasks need input-driven dynamics; this test isolates
whether the cycle+nonlinearity gives sustained structure at all.
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def cycle_W(N: int, scale: float = 1.0) -> torch.Tensor:
    """Directed-cycle recurrent weight matrix: W[i, (i+1)%N] = scale, else small noise.
    Asymmetric => complex eigenvalue spectrum."""
    W = torch.zeros(N, N)
    for i in range(N):
        W[i, (i + 1) % N] = scale
    return W


def chain_W(N: int, scale: float = 1.0) -> torch.Tensor:
    """Chain (DAG) recurrent weights: W[i, i+1] = scale, last node self-loop.
    Upper-triangular-ish; a control with NO cycle."""
    W = torch.zeros(N, N)
    for i in range(N - 1):
        W[i, i + 1] = scale
    W[N - 1, N - 1] = scale * 0.5  # sink, no return
    return W


def simulate(W: torch.Tensor, x0: torch.Tensor, T: int, leak: float = 0.1,
             bias: float = 0.0) -> torch.Tensor:
    """DISCRETE x_{t+1} = (1-leak) x_t + leak * tanh(W x_t + bias). Returns [T, N].
    Found to collapse to saturated fixed points (see log). Kept for comparison."""
    x = x0.clone()
    traj = []
    for _ in range(T):
        x = (1.0 - leak) * x + leak * torch.tanh(x @ W + bias)
        traj.append(x)
    return torch.stack(traj, dim=0)  # [T, N]


def simulate_ode(W: torch.Tensor, x0: torch.Tensor, T: int, dt: float = 0.05,
                 total_time: float = 30.0) -> torch.Tensor:
    """CONTINUOUS-TIME dx/dt = -x + tanh(W x), integrated by RK4. T samples over
    [0, total_time]. Continuous time + directed cycle + tanh saturation is the
    standard setting for limit cycles: the linear part (-I + W) has complex
    eigenvalues (rotation) when W is asymmetric with appropriate gain; tanh
    bounds the amplitude, turning a spiral-out into a stable limit cycle.

    This replaces the discrete iterate, which collapsed to saturated fixed
    points because the leak averaged out the rotation."""
    n_steps = int(total_time / dt)
    sample_every = max(1, n_steps // T)
    x = x0.clone()
    traj = []

    def f(x):
        return -x + torch.tanh(x @ W)

    for i in range(n_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % sample_every == 0:
            traj.append(x.clone())
    return torch.stack(traj, dim=0)  # [~T, N]


def trajectory_energy(traj: torch.Tensor) -> torch.Tensor:
    return (traj ** 2).sum(-1)  # [T]


# ----------------------------- Q1: sustained oscillation? -----------------------------

def verify_q1(N=16, T=200, seed=0):
    print("=== Q1: does tanh+directed-cycle sustain oscillation (no collapse)? ===")
    g = torch.Generator().manual_seed(seed)
    results = {}
    for label, Wfn in [("cycle", cycle_W), ("chain", chain_W)]:
        # scale must push Re(eig(-I+W))>0 for Hopf; for directed cycle of N=16,
        # largest |eig(W)|=scale, cos(2pi/16)=0.924, so scale>1.08 needed.
        for scale in [1.0, 1.5, 2.0, 3.0, 5.0]:
            W = Wfn(N, scale=scale)
            x0 = torch.randn(N, generator=g) * 0.5
            traj = simulate_ode(W, x0, T)
            E = trajectory_energy(traj)
            e0, e_end = float(E[0]), float(E[-1])
            # sustained if end energy is a meaningful fraction of start AND not collapsed
            ratio = e_end / (e0 + 1e-9)
            # also check oscillation: std of energy over last half
            late_E = E[T // 2:]
            osc = float(late_E.std().item())
            results[f"{label}_s{scale}"] = {"e0": e0, "e_end": e_end, "ratio": ratio, "late_std": osc}
            print(f"  {label} scale={scale}: E(0)={e0:.2f} E({T})={e_end:.2f} ratio={ratio:.3f} late_std={osc:.3f}")
    # pass if some cycle config has ratio>0.1 AND late_std>0.01 (sustained oscillation)
    cycle_sustained = any(results[f"cycle_s{s}"]["ratio"] > 0.1 and results[f"cycle_s{s}"]["late_std"] > 0.01
                          for s in [1.0, 1.5, 2.0, 3.0, 5.0])
    chain_sustained = any(results[f"chain_s{s}"]["ratio"] > 0.1 and results[f"chain_s{s}"]["late_std"] > 0.01
                          for s in [1.0, 1.5, 2.0, 3.0, 5.0])
    print(f"  cycle sustained? {cycle_sustained}   chain sustained? {chain_sustained}")
    q1_pass = cycle_sustained and not chain_sustained
    print(f"  Q1 verdict: {'PASS' if q1_pass else 'FAIL'}  (cycle oscillates, chain collapses)")
    return {"results": results, "cycle_sustained": cycle_sustained,
            "chain_sustained": chain_sustained, "q1_pass": q1_pass}


# ----------------------------- Q2: decisive temporal-task fit -----------------------------

def verify_q2(N=16, T=120, seed=0):
    print("\n=== Q2 (DECISIVE): temporal-task fit from nonlinear trajectory ===")
    g = torch.Generator().manual_seed(seed)
    # target: periodic signal (sum of two sines) — must be recovered from x(t) only
    ts = torch.linspace(0, 12.0, T)
    target = torch.sin(2 * math.pi * ts / 4.0) + 0.5 * torch.sin(2 * math.pi * ts / 2.5)

    def fit(Wfn, label, scale=2.0):
        torch.manual_seed(seed + 1)
        W = Wfn(N, scale=scale)
        x0 = torch.randn(N, generator=g) * 0.5
        traj = simulate_ode(W, x0, T)  # [T, N]
        # ridge readout, 70/30 train/test split (test = extrapolation in time)
        Ttr = int(0.7 * T)
        Xtr, ytr = traj[:Ttr], target[:Ttr]
        Xte, yte = traj[Ttr:], target[Ttr:]
        lam = 1e-3
        w = torch.linalg.solve(Xtr.t() @ Xtr + lam * torch.eye(N), Xtr.t() @ ytr)
        pred_te = Xte @ w
        mse_te = float(((pred_te - yte) ** 2).mean().item())
        ss = float(((yte - yte.mean()) ** 2).mean().item())
        r2 = 1.0 - mse_te / (ss + 1e-9)
        print(f"  {label}: test MSE={mse_te:.4f}  test R^2={r2:.4f}")
        return mse_te, r2

    # const baseline
    Ttr = int(0.7 * T)
    const_mse = float(((target[Ttr:] - target[:Ttr].mean()) ** 2).mean().item())
    print(f"  const-baseline: test MSE={const_mse:.4f}")

    best_cycle_r2 = -1e9
    for s in [1.5, 2.0, 3.0, 4.0]:
        _, r2 = fit(lambda N, scale=s: cycle_W(N, scale), f"cycle s={s}", scale=s)
        best_cycle_r2 = max(best_cycle_r2, r2)
    _, r2_chain = fit(chain_W, "chain  ")

    q2_pass = best_cycle_r2 > 0.5 and best_cycle_r2 > r2_chain + 0.2
    print(f"\n  best cycle R^2={best_cycle_r2:.4f}  chain R^2={r2_chain:.4f}")
    print(f"  Q2 verdict: {'PASS' if q2_pass else 'FAIL'}  (cycle R^2>0.5 AND >> chain)")
    return {"best_cycle_r2": best_cycle_r2, "chain_r2": r2_chain,
            "const_mse": const_mse, "q2_pass": q2_pass}


# ----------------------------- Q3: oscillation frequency vs cycle length -----------------------------

def verify_q3(T=300, seed=0):
    print("\n=== Q3: oscillation frequency set by cycle length? ===")
    g = torch.Generator().manual_seed(seed)
    rows = []
    for N in [8, 12, 16, 24, 32]:
        W = cycle_W(N, scale=3.0)
        x0 = torch.randn(N, generator=g) * 0.5
        traj = simulate_ode(W, x0, T)  # [T, N]
        # dominant frequency from a representative unit's trajectory (FFT)
        sig = traj[:, 0].numpy()
        # power spectrum
        fft = torch.fft.rfft(torch.tensor(sig))
        power = fft.abs() ** 2
        power[0] = 0  # drop DC
        peak_bin = int(power.argmax().item())
        freq = peak_bin / T  # cycles per sample
        rows.append((N, freq))
        print(f"  N={N}: dominant freq={freq:.4f} (cycles/sample), peak_bin={peak_bin}")
    # longer cycle => lower freq? (period ~ N)
    Ns = [r[0] for r in rows]
    fs = [r[1] for r in rows]
    def _corr(a, b):
        n = len(a); ma, mb = sum(a)/n, sum(b)/n
        num = sum((x-ma)*(y-mb) for x,y in zip(a,b))
        den = (sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**0.5
        return num/den if den>0 else 0
    corr = _corr(Ns, fs)
    q3_pass = corr < -0.3  # N up => freq down
    print(f"  corr(N, freq) = {corr:.3f}  (expect negative: longer cycle => lower freq)")
    print(f"  Q3 verdict: {'PASS' if q3_pass else 'FAIL/inconclusive'}")
    return {"rows": rows, "corr": corr, "q3_pass": q3_pass}


# ----------------------------- Q4: failure check -----------------------------

def verify_q4(N=16, T=500, seed=0):
    print("\n=== Q4 (failure check): does the nonlinear cycle avoid fixed-point collapse? ===")
    g = torch.Generator().manual_seed(seed)
    W = cycle_W(N, scale=3.0)
    x0 = torch.randn(N, generator=g) * 0.5
    traj = simulate_ode(W, x0, T)
    E = trajectory_energy(traj)
    # check last 100 steps: is it still moving (oscillating) or frozen at fixed point?
    late = E[-100:]
    late_std = float(late.std().item())
    final_change = float((E[-1] - E[-50]).abs().item())
    print(f"  late energy std (last 100) = {late_std:.4f}")
    print(f"  |E(-1) - E(-50)| = {final_change:.4f}")
    q4_pass = late_std > 0.01  # still oscillating, not frozen
    print(f"  Q4 verdict: {'PASS' if q4_pass else 'FAIL'}  "
          f"({'sustained oscillation' if q4_pass else 'collapsed to fixed point -> A3 nonlinear refuted'})")
    return {"late_std": late_std, "final_change": final_change, "q4_pass": q4_pass}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    r1 = verify_q1(seed=args.seed)
    r2 = verify_q2(seed=args.seed)
    r3 = verify_q3(seed=args.seed)
    r4 = verify_q4(seed=args.seed)

    print("\n=== SUMMARY (A3 nonlinear) ===")
    print(f"  Q1 (cycle sustains oscillation)     : {'PASS' if r1['q1_pass'] else 'FAIL'}")
    print(f"  Q2 (DECISIVE temporal-task fit)     : {'PASS' if r2['q2_pass'] else 'FAIL'}")
    print(f"  Q3 (freq vs cycle length)           : {'PASS' if r3['q3_pass'] else 'FAIL'}")
    print(f"  Q4 (no fixed-point collapse)        : {'PASS' if r4['q4_pass'] else 'FAIL'}")

    out = {"q1": r1, "q2": r2, "q3": r3, "q4": r4}
    out_path = os.path.join(RESULTS_DIR, "a3_nonlinear.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[written {out_path}]")

    try:
        import matplotlib.pyplot as plt
        # plot a sustained trajectory if Q1 passed
        W = cycle_W(16, scale=3.0)
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn(16, generator=g) * 0.5
        traj = simulate_ode(W, x0, 200)
        E = trajectory_energy(traj).numpy()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(E, label="cycle+tanh energy")
        ax[0].set_xlabel("t"); ax[0].set_ylabel("trajectory energy")
        ax[0].set_title(f"Q1: sustained oscillation?\n{'PASS' if r1['q1_pass'] else 'FAIL'}")
        ax[0].legend()
        ax[1].plot(traj[:, 0].numpy(), label="unit 0")
        ax[1].plot(traj[:, 4].numpy(), label="unit 4")
        ax[1].set_xlabel("t"); ax[1].set_ylabel("activation")
        ax[1].set_title("sample unit trajectories (cycle+tanh)")
        ax[1].legend()
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "a3_nonlinear.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
