"""A2 prototype: verify the semigroup reformulation of softmax attention.

STATUS (2026-07-17): CORE EQUALITY REFUTED. See dev/LOG.md entry
"A2 核心等式被证伪 — 嵌入性问题". The proposition below is kept for record;
the numerical tests show P2 (the core equality) FAILS.

Proposition (A2, corrected — the original "softmax = heat-kernel small-t limit"
was falsified; small-t limit is the identity, not softmax):

    softmax attention P = exp(tau * (P - I))  evaluated at tau = 1

where P is the row-stochastic transition matrix built from QK:
    P_ij = exp(q_i . k_j / sqrt(d)) / sum_m exp(q_i . k_m / sqrt(d))

and exp(tau*(P-I)) is the continuous-time random-walk semigroup with generator
Q = P - I. tau is a continuous "propagation duration" parameter:
  tau -> 0  =>  identity (no mixing)
  tau = 1   =>  standard softmax attention   [REFUTED: exp(P-I) != P]
  tau -> inf=>  stationary distribution pi (over-smoothing)

REFUTATION: P2 fails — ||exp(P-I) - P||_max ~= 0.375-0.45. The generator of a
row-stochastic matrix is NOT (P-I); the true generator would be logm(P), but P is
generally NOT embeddable (logm(P) has large imaginary part, 2.3-8.7; even
symmetric P fails). This is the classical embedding problem for Markov chains
(Kingman 1962, Speakman 1967). softmax attention is a ONE-STEP discrete
transition that cannot be losslessly embedded in a continuous-time semigroup.

What SURVIVES (as approximation, not equality):
  (P1) exp(tau*(P-I)) -> I as tau -> 0.  [PASS]
  (P3) exp(tau*(P-I)) -> pi as tau -> inf. [PASS]
  (P4) tau-sweep accuracy peaks near tau~0.75-1 on a simple task (decisive test
       not fully executed — task too easy, no over-smoothing degradation). [INCONCLUSIVE]
  (P5) cyclic/low-spectral-gap graph slows tau->inf convergence. [PASS, speculative]
  (P2) exp(1*(P-I)) == P.  [FAIL — the core equality is false]
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


def softmax_attention(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Standard softmax attention. q,k: [N, d] -> P: [N, N] row-stochastic."""
    d = q.shape[-1]
    logits = q @ k.t() / math.sqrt(d)
    return Fnn.softmax(logits, dim=-1)


def semigroup(P: torch.Tensor, tau: float) -> torch.Tensor:
    """exp(tau * (P - I)), the continuous-time random-walk semigroup.
    P: [N, N] row-stochastic. tau: scalar (can be <0, =0, >0)."""
    N = P.shape[0]
    Q = P - torch.eye(N, device=P.device, dtype=P.dtype)
    return torch.linalg.matrix_exp(tau * Q)


def dirichlet_energy(P: torch.Tensor, x: torch.Tensor) -> float:
    """sum_ij P_ij (x_i - x_j)^2 ; measures how mixed x is under P. Identity -> 0."""
    N = x.shape[0]
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, d]
    return float((P * (diff ** 2).sum(-1)).sum().item())


def stationary_distribution(P: torch.Tensor) -> torch.Tensor:
    """Left leading eigenvector of P (row-stochastic): pi P = pi, sum=1.
    i.e. right eigenvector of P^T for eigenvalue 1."""
    evals, evecs = torch.linalg.eig(P.t())
    # pick eigenvalue closest to 1 (real part)
    idx = int((evals.real - 1.0).abs().argmin().item())
    pi = evecs[:, idx].real
    pi = pi.clamp(min=0)
    pi = pi / pi.sum()
    return pi


# ----------------------------- P1, P2, P3 -----------------------------

def verify_limits(N=32, d=8, seed=0):
    print("=== P1/P2/P3: limit behavior of the semigroup ===")
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(N, d, generator=g)
    k = torch.randn(N, d, generator=g)
    P = softmax_attention(q, k)
    x = torch.randn(N, d, generator=g)  # arbitrary signal

    # P1: tau -> 0 => I
    S_small = semigroup(P, tau=1e-6)
    I_err = float((S_small - torch.eye(N)).abs().max().item())
    E0 = dirichlet_energy(torch.eye(N), x)
    E_small = dirichlet_energy(S_small, x)
    print(f"  P1  tau=1e-6: ||S-I||_max = {I_err:.2e}  Dirichlet @tau~0 = {E_small:.4f} (identity -> 0)")
    p1_pass = I_err < 1e-3 and E_small < 0.05 * dirichlet_energy(P, x)

    # P2: tau = 1 => P (softmax)
    S1 = semigroup(P, tau=1.0)
    P2_err = float((S1 - P).abs().max().item())
    print(f"  P2  tau=1.0: ||exp(P-I) - softmax(P)||_max = {P2_err:.2e}")
    p2_pass = P2_err < 1e-4

    # P3: tau -> inf => pi (stationary)
    S_big = semigroup(P, tau=50.0)
    pi = stationary_distribution(P)
    # each row of S_big should approach pi
    row_max_dev = float((S_big - pi.unsqueeze(0)).abs().max().item())
    print(f"  P3  tau=50: max|row - pi| = {row_max_dev:.2e}  (pi = {pi[:5].tolist()}...)")
    p3_pass = row_max_dev < 1e-3

    # tau sweep: trace the spectrum of behavior
    print(f"  tau sweep (Dirichlet energy of a fixed signal):")
    for tau in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        S = semigroup(P, tau)
        print(f"    tau={tau:6.2f}  Dirichlet={dirichlet_energy(S, x):.4f}")

    return {"p1": p1_pass, "p2": p2_pass, "p3": p3_pass,
            "I_err": I_err, "P2_err": P2_err, "pi_dev": row_max_dev}


# ----------------------------- P4: decisive tau sweep on a task -----------------------------

class TinyAttnClassifier(torch.nn.Module):
    """Classify a bag of tokens [B, N, d] by attention pooling. The attention
    matrix is exp(tau*(P-I)) with P = softmax(QK), tau a fixed hyperparameter.
    A learnable per-token projection produces Q,K; the class head pools.
    """

    def __init__(self, d=16, n_classes=10, tau=1.0):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d)
        self.k_proj = torch.nn.Linear(d, d)
        self.v_proj = torch.nn.Linear(d, d)
        self.head = torch.nn.Linear(d, n_classes)
        self.tau = tau

    def forward(self, tokens):  # tokens: [B, N, d]
        B, N, d = tokens.shape
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        logits = q @ k.transpose(-1, -2) / math.sqrt(d)
        P = Fnn.softmax(logits, dim=-1)  # [B, N, N] row-stochastic
        if abs(self.tau - 1.0) < 1e-9:
            A = P  # standard softmax attention (P2: == exp(P-I) at tau=1)
        else:
            # batched matrix exp of tau*(P-I)
            I = torch.eye(N, device=P.device).unsqueeze(0)
            A = torch.linalg.matrix_exp(self.tau * (P - I))
        out = A @ v  # [B, N, d]
        pooled = out.mean(dim=1)
        return self.head(pooled)


def make_bag_dataset(n=2000, n_tokens=8, d=16, n_classes=10, seed=0):
    """Synthetic: each class has a characteristic token; classifier must pool the
    right signal. Forces non-trivial attention (positional mean pooling fails)."""
    g = torch.Generator().manual_seed(seed)
    # class prototypes
    proto = torch.randn(n_classes, d, generator=g)
    X = torch.randn(n, n_tokens, d, generator=g) * 0.3
    y = torch.randint(0, n_classes, (n,), generator=g)
    # inject one prototype token per sample at a random position
    for i in range(n):
        pos = torch.randint(0, n_tokens, (1,), generator=g).item()
        X[i, pos] = X[i, pos] + proto[y[i]]
    return X, y, proto


def train_eval_tau(tau, Xtr, ytr, Xte, yte, epochs=40, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    model = TinyAttnClassifier(d=Xtr.shape[-1], n_classes=int(ytr.max()) + 1, tau=tau)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = Fnn.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(-1)
        acc = (pred == yte).float().mean().item()
    return acc


def verify_p4(n=2000, n_tokens=8, d=16, epochs=40, seed=0):
    print("\n=== P4 (DECISIVE): tau-sweep accuracy on a bag-classification task ===")
    X, y, _ = make_bag_dataset(n=n, n_tokens=n_tokens, d=d, n_classes=10, seed=seed)
    ntr = int(0.8 * n)
    Xtr, ytr = X[:ntr], y[:ntr]
    Xte, yte = X[ntr:], y[ntr:]

    taus = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    accs = []
    for tau in taus:
        acc = train_eval_tau(tau, Xtr, ytr, Xte, yte, epochs=epochs, seed=seed)
        accs.append(acc)
        print(f"  tau={tau:5.2f}  test_acc={acc:.4f}")

    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    # decisive: must peak at an interior tau near 1, and degrade for large tau
    rises_before = any(accs[i] < accs[best_idx] for i in range(best_idx))
    falls_after = any(accs[i] < accs[best_idx] - 0.05 for i in range(best_idx, len(accs)))
    interior = rises_before and falls_after
    near_one = abs(taus[best_idx] - 1.0) <= 1.0
    print(f"\n  peak at tau*={taus[best_idx]:.2f} (acc={accs[best_idx]:.4f})")
    print(f"  interior peak? {interior}   rises-before={rises_before}   falls-after={falls_after}")
    print(f"  tau* near 1? {near_one}")
    verdict = "PASS" if (interior and near_one) else "FAIL/inconclusive"
    print(f"  P4 verdict: {verdict}  (interior peak near tau=1; monotone would refute A2)")
    return {"taus": taus, "accs": accs, "tau_star": taus[best_idx],
            "interior": interior, "near_one": near_one, "verdict": verdict}


# ----------------------------- P5: spectral gap / cycle -----------------------------

def verify_p5(N=32, d=8, seed=0):
    print("\n=== P5 (speculative): low spectral gap (cycle) slows tau->inf convergence ===")
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(N, d, generator=g)
    k = torch.randn(N, d, generator=g)
    P_dense = softmax_attention(q, k)

    # cycle-structured attention: strong only on ring neighbors
    A = torch.zeros(N, N)
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[i, (i - 1) % N] = 1.0
    P_cycle = Fnn.softmax(A * 5.0, dim=-1)  # near-deterministic ring walk

    def spec_gap(P):
        evals = torch.linalg.eigvals(P).real
        # second-largest |eigenvalue| (largest is 1)
        absv = evals.abs().sort(descending=True).values
        return float((1.0 - absv[1]).item())

    gd = spec_gap(P_dense)
    gc = spec_gap(P_cycle)
    print(f"  spectral gap: dense-attention P = {gd:.4f}   ring-cycle P = {gc:.4f}")

    # distance to stationary at a fixed large tau
    pi_d = stationary_distribution(P_dense)
    pi_c = stationary_distribution(P_cycle)
    for tau in [2.0, 10.0, 50.0]:
        Sd = semigroup(P_dense, tau)
        Sc = semigroup(P_cycle, tau)
        dd = float((Sd - pi_d.unsqueeze(0)).abs().max().item())
        dc = float((Sc - pi_c.unsqueeze(0)).abs().max().item())
        print(f"  tau={tau:5.1f}: max|row-pi|  dense={dd:.3e}  cycle={dc:.3e}")

    p5_pass = gc < gd  # cycle has smaller gap
    print(f"  cycle gap < dense gap? {p5_pass}  (smaller gap => slower mixing, the speculative 'time stretch')")
    return {"gap_dense": gd, "gap_cycle": gc, "p5_pass": p5_pass}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    r_limits = verify_limits(seed=args.seed)
    r_p4 = verify_p4(epochs=args.epochs, seed=args.seed)
    r_p5 = verify_p5(seed=args.seed)

    print("\n=== SUMMARY ===")
    print(f"  P1 (tau->0 = identity)         : {'PASS' if r_limits['p1'] else 'FAIL'}")
    print(f"  P2 (tau=1 = softmax)          : {'PASS' if r_limits['p2'] else 'FAIL'}")
    print(f"  P3 (tau->inf = stationary pi) : {'PASS' if r_limits['p3'] else 'FAIL'}")
    print(f"  P4 (decisive tau-sweep peak)  : {r_p4['verdict']}")
    print(f"  P5 (cycle slows mixing)       : {'PASS' if r_p5['p5_pass'] else 'FAIL'} (speculative)")

    out = {"limits": r_limits, "p4": r_p4, "p5": r_p5}
    out_path = os.path.join(RESULTS_DIR, "a2_verify.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[written {out_path}]")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(r_p4["taus"], r_p4["accs"], "-o", ms=5)
        ax.axvline(1.0, color="r", ls="--", label="tau=1 (standard softmax)")
        ax.set_xlabel("tau (propagation duration)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"A2 decisive test: accuracy vs tau\n{r_p4['verdict']}")
        ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "a2_verify.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
