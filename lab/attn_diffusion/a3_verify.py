"""A3 prototype: do cycles make propagation non-terminating, giving emergent
temporal structure without an explicit time axis?

Proposition (A3, see dev/LOG.md): on a cyclic graph, the continuous state
trajectory x(t) = exp(t*(P-I)) x(0) carries temporal structure because cycles
have complex conjugate eigenvalue pairs (oscillatory, slow-decaying modes),
whereas DAGs/trees have real eigenvalues (monotone decay, fast collapse to
stationary). Time is not an external axis here — it is the propagation duration,
and the trajectory itself is the temporal signal.

NOTE on honesty (from the log): linear semigroups ALWAYS converge to the
stationary distribution pi eventually (verified in a2_verify P3). A3's "temporal
structure" is the LONG TRANSIENT before convergence, made long by a small
spectral gap. True non-convergence needs nonlinearity (future work) or gap=0
(which degenerates mixing). This script tests the linear case first as the
cleanest cycle-vs-DAG discriminator.

Predictions:
  (P1) Cycle eigenvalues are complex (oscillatory); DAG eigenvalues are real.
       Cycle trajectory autocorrelation decays slowly; DAG fast.
  (P2) DECISIVE: without an explicit time index, a readout on x(t) fits a
       temporal task (periodic signal prediction) BETTER on a cycle than on a
       DAG of matched size.
  (P3) Smaller spectral gap => longer fittable temporal horizon (monotone).
  (P4) Failure check: if the cycle trajectory collapses to a single exponential
       (no persistent modes), A3 is refuted.
"""

import argparse
import json
import math
import os
import sys
from typing import List

import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def cycle_P(N: int, smooth: float = 1.0, directed: bool = True) -> torch.Tensor:
    """Row-stochastic cycle transition. directed=True gives a directed ring
    (i -> i+1 only), whose transition matrix is a cyclic permutation-ish matrix
    with COMPLEX eigenvalues e^{2 pi i k / N} — the oscillatory modes A3 needs.
    directed=False gives an undirected (symmetric) ring, whose P is symmetric
    with REAL eigenvalues — this was the bug in the first run: an undirected
    symmetric ring cannot produce complex eigenvalues."""
    A = torch.zeros(N, N)
    if directed:
        for i in range(N):
            A[i, (i + 1) % N] = 1.0
    else:
        for i in range(N):
            A[i, (i + 1) % N] = 1.0
            A[i, (i - 1) % N] = 1.0
    return Fnn.softmax(A * smooth, dim=-1)


def dag_P(N: int) -> torch.Tensor:
    """Row-stochastic DAG: edges only i -> j for j > i (upper triangular).
    Matching the cycle's sparsity (2 out-edges per node where possible)."""
    A = torch.zeros(N, N)
    for i in range(N):
        # forward edges only (acyclic)
        js = []
        if i + 1 < N:
            js.append(i + 1)
        if i + 2 < N:
            js.append(i + 2)
        if not js:  # sink node
            A[i, i] = 1.0
        else:
            for j in js:
                A[i, j] = 1.0
    return Fnn.softmax(A * 5.0, dim=-1)


def semigroup(P: torch.Tensor, t: float) -> torch.Tensor:
    N = P.shape[0]
    return torch.linalg.matrix_exp(t * (P - torch.eye(N, dtype=P.dtype, device=P.device)))


def spectral_gap(P: torch.Tensor) -> float:
    evals = torch.linalg.eigvals(P).real
    absv = evals.abs().sort(descending=True).values
    return float((1.0 - absv[1]).item())


def trajectory(P: torch.Tensor, x0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """x(t) for each t in ts. Returns [T, N, d]."""
    N = P.shape[0]
    I = torch.eye(N, dtype=P.dtype, device=P.device)
    Q = P - I
    traj = []
    for t in ts:
        St = torch.linalg.matrix_exp(t * Q)
        traj.append(St @ x0)  # [N, d]
    return torch.stack(traj, dim=0)  # [T, N, d]


# ----------------------------- P1: spectrum + autocorrelation -----------------------------

def verify_p1(N=24, seed=0):
    print("=== P1: cycle vs DAG spectrum and trajectory autocorrelation ===")
    g = torch.Generator().manual_seed(seed)
    Pc = cycle_P(N, smooth=2.0)
    Pd = dag_P(N)

    evals_c = torch.linalg.eigvals(Pc)
    evals_d = torch.linalg.eigvals(Pd)
    imag_c = float(evals_c.imag.abs().max().item())
    imag_d = float(evals_d.imag.abs().max().item())
    gc = spectral_gap(Pc)
    gd = spectral_gap(Pd)
    print(f"  cycle: max|Im(eigenval)| = {imag_c:.4f}  spectral_gap = {gc:.4f}")
    print(f"  DAG:   max|Im(eigenval)| = {imag_d:.4f}  spectral_gap = {gd:.4f}")

    # trajectory autocorrelation: how fast does x(t) decorrelate from x(0)?
    x0 = torch.randn(N, 4, generator=g)
    ts = torch.linspace(0, 20, 100)
    traj_c = trajectory(Pc, x0, ts)  # [T, N, d]
    traj_d = trajectory(Pd, x0, ts)
    # cosine sim of flattened state vs x0, over t
    def autocorr(traj, x0):
        x0f = x0.flatten()
        x0n = x0f / (x0f.norm() + 1e-9)
        sims = []
        for t in range(traj.shape[0]):
            xf = traj[t].flatten()
            sims.append(float((xf @ x0n / (xf.norm() + 1e-9)).item()))
        return sims
    ac_c = autocorr(traj_c, x0)
    ac_d = autocorr(traj_d, x0)
    # half-life: first t where |sim| < 0.5
    def halflife(sims, ts):
        for s, t in zip(sims, ts):
            if abs(s) < 0.5:
                return float(t)
        return float("inf")
    hc = halflife(ac_c, ts)
    hd = halflife(ac_d, ts)
    print(f"  autocorr half-life: cycle={hc:.2f}  DAG={hd:.2f}  (cycle should be longer)")
    p1_pass = imag_c > 0.1 and imag_d < 0.05 and hc > hd
    print(f"  P1 verdict: {'PASS' if p1_pass else 'FAIL'}  (cycle complex+slow-decorr, DAG real+fast)")
    return {"imag_c": imag_c, "imag_d": imag_d, "gap_c": gc, "gap_d": gd,
            "half_c": hc, "half_d": hd, "p1_pass": p1_pass,
            "ac_c": ac_c, "ac_d": ac_d, "ts": ts.tolist()}


# ----------------------------- P2: decisive temporal-task fit -----------------------------

def verify_p2(N=16, T=40, seed=0):
    print("\n=== P2 (DECISIVE): temporal-task fit without explicit time index ===")
    g = torch.Generator().manual_seed(seed)
    # Target temporal signal: a periodic function of time, scalar per timestep.
    # The readout must recover it from x(t) ALONE — no time index is fed.
    ts = torch.linspace(0.0, 8.0, T)
    target = torch.sin(2 * math.pi * ts / 4.0)  # period 4 in t-units

    # random initial state; trajectory is the ONLY input to the readout
    x0 = torch.randn(N, 1, generator=g)

    def fit_on_P(P, label):
        torch.manual_seed(seed + 1)
        traj = trajectory(P, x0, ts).squeeze(-1)  # [T, N]
        # linear readout: target_t = w . x(t). closed form ridge regression.
        # train on first 70%, test on last 30% (extrapolation in time).
        Ttr = int(0.7 * T)
        Xtr, ytr = traj[:Ttr], target[:Ttr]
        Xte, yte = traj[Ttr:], target[Ttr:]
        # ridge: w = (X^T X + lambda I)^-1 X^T y
        lam = 1e-2
        w = torch.linalg.solve(Xtr.t() @ Xtr + lam * torch.eye(N), Xtr.t() @ ytr)
        pred_tr = Xtr @ w
        pred_te = Xte @ w
        mse_tr = float(((pred_tr - ytr) ** 2).mean().item())
        mse_te = float(((pred_te - yte) ** 2).mean().item())
        # R^2 on test
        ss_tot = float(((yte - yte.mean()) ** 2).mean().item())
        r2_te = 1.0 - mse_te / (ss_tot + 1e-9)
        print(f"  {label}: train MSE={mse_tr:.4f}  test MSE={mse_te:.4f}  test R^2={r2_te:.4f}")
        return mse_te, r2_te

    Pc = cycle_P(N, smooth=2.0)
    Pd = dag_P(N)
    mse_c, r2_c = fit_on_P(Pc, "cycle")
    mse_d, r2_d = fit_on_P(Pd, "DAG  ")

    # baseline: constant predictor (mean of train target)
    const_mse = float(((target[T:] - target[:T].mean()) ** 2).mean() if False else
                      ((target[int(0.7*T):] - target[:int(0.7*T)].mean())**2).mean().item())
    print(f"  const-baseline test MSE = {const_mse:.4f}")
    p2_pass = r2_c > 0.5 and r2_c > r2_d + 0.2
    print(f"  P2 verdict: {'PASS' if p2_pass else 'FAIL'}  "
          f"(cycle R^2>0.5 AND >> DAG; cycle={r2_c:.3f} vs DAG={r2_d:.3f})")
    return {"mse_c": mse_c, "mse_d": mse_d, "r2_c": r2_c, "r2_d": r2_d,
            "const_mse": const_mse, "p2_pass": p2_pass}


# ----------------------------- P3: spectral gap vs temporal horizon -----------------------------

def verify_p3(N=16, seed=0):
    print("\n=== P3: smaller spectral gap => longer fittable temporal horizon ===")
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(N, 1, generator=g)

    # vary cycle "smoothness" (peakedness) to vary spectral gap
    smooths = [0.5, 1.0, 2.0, 4.0, 8.0]
    rows = []
    for sm in smooths:
        Pc = cycle_P(N, smooth=sm)
        gap = spectral_gap(Pc)
        # temporal horizon: how far in t can a ridge readout still predict sin
        # with R^2 > 0.5? extend T until it breaks.
        best_T = 0
        for T in [10, 20, 40, 80, 160]:
            ts = torch.linspace(0.0, 8.0 * (T / 40.0), T)
            target = torch.sin(2 * math.pi * ts / 4.0)
            traj = trajectory(Pc, x0, ts).squeeze(-1)
            Ttr = int(0.7 * T)
            Xtr, ytr = traj[:Ttr], target[:Ttr]
            Xte, yte = traj[Ttr:], target[Ttr:]
            lam = 1e-2
            try:
                w = torch.linalg.solve(Xtr.t() @ Xtr + lam * torch.eye(N), Xtr.t() @ ytr)
                pred = Xte @ w
                mse = float(((pred - yte) ** 2).mean().item())
                ss = float(((yte - yte.mean()) ** 2).mean().item())
                r2 = 1.0 - mse / (ss + 1e-9)
                if r2 > 0.5:
                    best_T = T
            except Exception:
                pass
        rows.append((sm, gap, best_T))
        print(f"  smooth={sm:4.1f}  gap={gap:.4f}  horizon(R^2>0.5)={best_T}")
    # smaller gap => longer horizon?
    gaps = [r[1] for r in rows]
    hors = [r[2] for r in rows]
    # rank correlation: gap up => horizon down (manual, py3.9 has no statistics.correlation)
    def _corr(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den > 0 else 0.0
    corr = _corr(gaps, hors)
    p3_pass = corr < -0.3
    print(f"  corr(gap, horizon) = {corr:.3f}  (expect negative: smaller gap -> longer horizon)")
    print(f"  P3 verdict: {'PASS' if p3_pass else 'FAIL/inconclusive'}")
    return {"rows": rows, "corr": corr, "p3_pass": p3_pass}


# ----------------------------- P4: persistent-mode failure check -----------------------------

def verify_p4(N=24, seed=0):
    print("\n=== P4 (failure check): does the cycle trajectory have persistent modes? ===")
    Pc = cycle_P(N, smooth=2.0)
    evals = torch.linalg.eigvals(Pc)
    # persistent modes = eigenvalues with |lambda| close to 1 AND nonzero imaginary
    absv = evals.abs()
    imag = evals.imag.abs()
    persistent = ((absv > 0.9) & (imag > 0.05)).sum().item()
    print(f"  cycle eigenvalues with |lambda|>0.9 AND |Im|>0.05: {persistent}")
    # trajectory energy over time: does it stay high (persistent) or collapse?
    x0 = torch.randn(N, 1)
    ts = torch.linspace(0, 30, 60)
    traj = trajectory(Pc, x0, ts).squeeze(-1)
    energy = (traj ** 2).sum(-1)  # [T]
    e0 = float(energy[0].item())
    e_end = float(energy[-1].item())
    print(f"  trajectory energy: t=0 -> {e0:.3f}, t=30 -> {e_end:.3f}  (ratio {e_end/e0:.3f})")
    p4_pass = persistent >= 2 and e_end / e0 > 0.1
    print(f"  P4 verdict: {'PASS' if p4_pass else 'FAIL'}  "
          f"(persistent oscillatory modes present; FAIL => trajectory collapses, A3 refuted)")
    return {"persistent_modes": int(persistent), "e_ratio": e_end / e0, "p4_pass": p4_pass}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    r1 = verify_p1(seed=args.seed)
    r2 = verify_p2(seed=args.seed)
    r3 = verify_p3(seed=args.seed)
    r4 = verify_p4(seed=args.seed)

    print("\n=== SUMMARY ===")
    print(f"  P1 (cycle complex+slow-decorr)      : {'PASS' if r1['p1_pass'] else 'FAIL'}")
    print(f"  P2 (DECISIVE temporal-task fit)     : {'PASS' if r2['p2_pass'] else 'FAIL'}")
    print(f"  P3 (gap vs horizon monotone)        : {'PASS' if r3['p3_pass'] else 'FAIL'}")
    print(f"  P4 (persistent modes, not collapse) : {'PASS' if r4['p4_pass'] else 'FAIL'}")

    out = {"p1": {k: v for k, v in r1.items() if k != "ac_c" and k != "ac_d" and k != "ts"},
           "p2": r2, "p3": r3, "p4": r4}
    out_path = os.path.join(RESULTS_DIR, "a3_verify.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[written {out_path}]")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(r1["ts"], r1["ac_c"], "-o", ms=3, label="cycle")
        ax[0].plot(r1["ts"], r1["ac_d"], "-o", ms=3, label="DAG")
        ax[0].axhline(0.5, color="gray", ls=":", lw=0.5)
        ax[0].axhline(-0.5, color="gray", ls=":", lw=0.5)
        ax[0].set_xlabel("t"); ax[0].set_ylabel("autocorr(x(t), x(0))")
        ax[0].set_title(f"P1: trajectory decorrelation\n{'PASS' if r1['p1_pass'] else 'FAIL'}")
        ax[0].legend()
        # P3: gap vs horizon
        gaps = [r[1] for r in r3["rows"]]
        hors = [r[2] for r in r3["rows"]]
        ax[1].plot(gaps, hors, "-o", ms=6)
        ax[1].set_xlabel("spectral gap"); ax[1].set_ylabel("temporal horizon (R^2>0.5)")
        ax[1].set_title(f"P3: gap vs horizon (corr={r3['corr']:.2f})\n{'PASS' if r3['p3_pass'] else 'FAIL'}")
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "a3_verify.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
