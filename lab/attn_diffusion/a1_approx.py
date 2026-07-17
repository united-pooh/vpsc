"""A1 prototype: continuous heat-kernel / semigroup as an APPROXIMATE attention
operator with a learnable tau.

Context: A2's strict equality (softmax == exp(tau*(P-I)) at tau=1) was REFUTED
by the embedding problem (exp(P-I) != P, deviation ~0.375; logm(P) has large
imaginary part; row-stochastic matrices are generally not embeddable). A1 drops
the equality claim and asks the engineering question: is exp(tau*(P-I)) a useful
APPROXIMATE attention operator when tau is learnable?

Two honest framings:
  - exp(tau*(P-I)) is NOT softmax at any tau; it is a different operator that
    interpolates identity (tau->0) and stationary (tau->inf), passing NEAR but
    not through softmax.
  - The question is whether this approximate operator, with learnable tau per
    head/layer, can match or beat standard softmax attention on a real task.

Predictions:
  (R1) As a fixed operator, accuracy vs tau peaks somewhere (not monotone) —
       confirming tau is a meaningful knob even without the equality.
  (R2) DECISIVE: a learnable-tau exp(tau*(P-I)) attention matches or beats
       fixed-tau=1 (== standard softmax path) on a non-trivial task.
  (R3) On a harder task (more tokens, needs selective attention), large tau
       degrades (over-smoothing visible) — the property A2's P4 failed to show
       because its task was too easy.
  (R4) Failure: if learnable tau collapses to ~1 (no benefit) or accuracy is
       far below softmax, A1 is not useful as an attention replacement.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def semigroup_attn(P: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """exp(tau * (P - I)) for batched row-stochastic P [B,H,N,N] or [B,N,N].
    tau broadcastable to P. Approximate attention operator (NOT equal to P)."""
    *rest, N, _ = P.shape
    I = torch.eye(N, device=P.device, dtype=P.dtype).view(*([1] * len(rest)), N, N)
    return torch.linalg.matrix_exp(tau * (P - I))


class SemigroupAttnClassifier(torch.nn.Module):
    """Bag classifier. Attention = exp(tau*(P-I)) with P = softmax(QK).
    tau can be fixed (mode='fixed') or learnable per-head (mode='learn')."""

    def __init__(self, d=16, n_classes=10, n_heads=4, mode="fixed", tau_init=1.0):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d * n_heads)
        self.k_proj = torch.nn.Linear(d, d * n_heads)
        self.v_proj = torch.nn.Linear(d, d * n_heads)
        self.out = torch.nn.Linear(d * n_heads, d)
        self.head = torch.nn.Linear(d, n_classes)
        self.n_heads = n_heads
        self.mode = mode
        if mode == "learn":
            # learnable log-tau per head (init tau_init)
            self.log_tau = torch.nn.Parameter(torch.full((n_heads,), math.log(tau_init)))
        else:
            self.register_buffer("tau", torch.tensor(tau_init))

    def forward(self, tokens):  # [B, N, d]
        B, N, d = tokens.shape
        H = self.n_heads
        q = self.q_proj(tokens).view(B, N, H, d).transpose(1, 2)  # [B,H,N,d]
        k = self.k_proj(tokens).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(tokens).view(B, N, H, d).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(d)  # [B,H,N,N]
        P = Fnn.softmax(logits, dim=-1)
        if self.mode == "learn":
            tau = torch.exp(self.log_tau)  # [H]
            # per-head tau: [1,H,1,1]
            tau = tau.view(1, H, 1, 1)
            A = semigroup_attn(P, tau)
        else:
            A = semigroup_attn(P, self.tau)
        out = A @ v  # [B,H,N,d]
        out = out.transpose(1, 2).reshape(B, N, H * d)
        out = self.out(out)
        pooled = out.mean(dim=1)
        return self.head(pooled), (torch.exp(self.log_tau) if self.mode == "learn" else self.tau)


class SoftmaxClassifier(torch.nn.Module):
    """Baseline: standard softmax attention (tau=1 exact, no semigroup)."""

    def __init__(self, d=16, n_classes=10, n_heads=4):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d * n_heads)
        self.k_proj = torch.nn.Linear(d, d * n_heads)
        self.v_proj = torch.nn.Linear(d, d * n_heads)
        self.out = torch.nn.Linear(d * n_heads, d)
        self.head = torch.nn.Linear(d, n_classes)
        self.n_heads = n_heads

    def forward(self, tokens):
        B, N, d = tokens.shape
        H = self.n_heads
        q = self.q_proj(tokens).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(tokens).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(tokens).view(B, N, H, d).transpose(1, 2)
        A = Fnn.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), dim=-1)
        out = (A @ v).transpose(1, 2).reshape(B, N, H * d)
        return self.head(self.out(out).mean(1)), None


def make_hard_dataset(n=4000, n_tokens=32, d=8, n_classes=10, seed=0):
    """Hard task designed to make over-smoothing (large tau) hurt.

    Construction: every token carries a class prototype (its own random class),
    EXCEPT one 'cue' token whose class equals the sample label. The model must
    identify WHICH token is the cue (positioned randomly) and read its class.
    All tokens look similar (each is a prototype + noise), so there is no
    positional/feature shortcut — only attention can isolate the cue.

    Why this breaks at large tau: exp(tau*(P-I)) with large tau mixes all token
    representations toward the stationary distribution, washing out the
    cue-vs-distractor distinction. Small tau keeps representations local
    (selective). This is the over-smoothing regime A2's easy task failed to show.
    """
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(n_classes, d, generator=g)
    # each token's class drawn independently; cue token's class = sample label
    token_classes = torch.randint(0, n_classes, (n, n_tokens), generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    cue_pos = torch.randint(0, n_tokens, (n,), generator=g)
    for i in range(n):
        token_classes[i, cue_pos[i]] = y[i]
    # build tokens: prototype[token_class] + small noise
    X = proto[token_classes] + 0.5 * torch.randn(n, n_tokens, d, generator=g)
    return X, y


def train_eval(model, Xtr, ytr, Xte, yte, epochs=60, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits, _ = model(Xtr)
        Fnn.cross_entropy(logits, ytr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred, tau = model(Xte)
        acc = (pred.argmax(-1) == yte).float().mean().item()
    tau_val = None
    if tau is not None:
        tau_val = tau.detach().tolist() if hasattr(tau, "tolist") else float(tau)
    return acc, tau_val


def verify_r1_r3(Xtr, ytr, Xte, yte, d, seed=0):
    print("=== R1/R3: fixed-tau sweep (over-smoothing on hard task) ===")
    taus = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    accs = []
    for tau in taus:
        model = SemigroupAttnClassifier(d=d, mode="fixed", tau_init=tau)
        acc, _ = train_eval(model, Xtr, ytr, Xte, yte, seed=seed)
        accs.append(acc)
        print(f"  tau={tau:5.2f}  test_acc={acc:.4f}")
    best = max(range(len(accs)), key=lambda i: accs[i])
    rises = any(accs[i] < accs[best] - 0.02 for i in range(best))
    falls = any(accs[i] < accs[best] - 0.05 for i in range(best, len(accs)))
    interior = rises and falls
    print(f"  peak tau*={taus[best]:.2f} acc={accs[best]:.4f}  interior={interior} rises={rises} falls={falls}")
    print(f"  R1 (peak exists, non-monotone): {'PASS' if interior else 'FAIL'}")
    print(f"  R3 (over-smoothing at large tau): {'PASS' if falls else 'FAIL'}")
    return {"taus": taus, "accs": accs, "tau_star": taus[best], "interior": interior,
            "r1_pass": interior, "r3_pass": falls}


def verify_r2(Xtr, ytr, Xte, yte, d, seed=0):
    print("\n=== R2 (DECISIVE): learnable-tau semigroup vs standard softmax ===")
    # standard softmax baseline
    sm = SoftmaxClassifier(d=d)
    acc_sm, _ = train_eval(sm, Xtr, ytr, Xte, yte, seed=seed)
    print(f"  standard softmax (tau=1 exact):   acc={acc_sm:.4f}")

    # fixed tau=1 semigroup (the approximate operator, NOT equal to softmax)
    sg1 = SemigroupAttnClassifier(d=d, mode="fixed", tau_init=1.0)
    acc_sg1, _ = train_eval(sg1, Xtr, ytr, Xte, yte, seed=seed)
    print(f"  semigroup fixed tau=1 (approx):   acc={acc_sg1:.4f}  (gap vs softmax: {acc_sm-acc_sg1:+.4f})")

    # learnable tau semigroup
    sgl = SemigroupAttnClassifier(d=d, mode="learn", tau_init=1.0)
    acc_sgl, tau_learned = train_eval(sgl, Xtr, ytr, Xte, yte, seed=seed)
    print(f"  semigroup learnable tau:          acc={acc_sgl:.4f}  learned tau={[round(t,3) for t in tau_learned]}")

    r2_pass = acc_sgl >= acc_sg1 - 0.01 and acc_sgl >= acc_sm - 0.03
    print(f"  R2 verdict: {'PASS' if r2_pass else 'FAIL'}  "
          f"(learnable tau matches/beats fixed-tau=1 and stays within 3pp of softmax)")
    return {"acc_softmax": acc_sm, "acc_sg_fixed1": acc_sg1, "acc_sg_learn": acc_sgl,
            "tau_learned": tau_learned, "r2_pass": r2_pass}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    d = 8
    X, y = make_hard_dataset(n=4000, n_tokens=32, d=d, n_classes=10, seed=args.seed)
    ntr = int(0.8 * X.shape[0])
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    print(f"hard dataset: {X.shape}, train {ntr} / test {X.shape[0]-ntr}\n")

    r13 = verify_r1_r3(Xtr, ytr, Xte, yte, d, seed=args.seed)
    r2 = verify_r2(Xtr, ytr, Xte, yte, d, seed=args.seed)

    r4_pass = r2["acc_sg_learn"] < r2["acc_softmax"] - 0.10  # if >>10pp below softmax, A1 not useful
    print(f"\n=== SUMMARY (A1 approximate) ===")
    print(f"  R1 (tau peak exists)           : {'PASS' if r13['r1_pass'] else 'FAIL'}")
    print(f"  R2 (learnable tau competitive) : {'PASS' if r2['r2_pass'] else 'FAIL'}")
    print(f"  R3 (over-smoothing at large tau): {'PASS' if r13['r3_pass'] else 'FAIL'}")
    print(f"  R4 (not far below softmax)     : {'PASS' if not r4_pass else 'FAIL'} "
          f"(learnable {r2['acc_sg_learn']:.4f} vs softmax {r2['acc_softmax']:.4f})")

    out = {"r1_r3": r13, "r2": r2,
           "r4_pass": not r4_pass}
    out_path = os.path.join(RESULTS_DIR, "a1_approx.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[written {out_path}]")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(r13["taus"], r13["accs"], "-o", ms=5, label="semigroup fixed tau")
        ax.axhline(r2["acc_softmax"], color="r", ls="--", label=f"standard softmax ({r2['acc_softmax']:.3f})")
        ax.axhline(r2["acc_sg_learn"], color="g", ls="--", label=f"learnable tau ({r2['acc_sg_learn']:.3f})")
        ax.set_xlabel("tau"); ax.set_ylabel("test accuracy")
        ax.set_title("A1: approximate semigroup attention, tau sweep")
        ax.legend()
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "a1_approx.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
