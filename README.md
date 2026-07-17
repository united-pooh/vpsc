# VPSC — Variational Predictive Spiking Coding

A research prototype implementing a proposed SNN training method built on three
principles: the **free-energy principle** (target function), **mean-field /
statistical-physics annealing** (the surrogate-free differentiable forward), and
**game-theoretic potential-game structure** (multi-layer credit assignment).

This is the **B stage** (minimal runnable prototype) of a theory-first program.
The **C stage** (three theorems with proofs) lives in the design notes; this
code verifies the two load-bearing empirical predictions.

> Scope, as agreed: theory-first and publishable, accuracy not expected to lead.
> Task domain: temporal perception (DVS / speech) and control / decision.

---

## The method in one paragraph

A spike is reframed as a **posterior commitment** — the zero-temperature limit
of a mean-field magnetization `m = tanh(β·(u−θ))`, not a heuristic surrogate for
a step function. Training minimizes a **variational free energy**
`F = Σ_l F_l`, where each layer's `F_l = ½‖x_l − μ_l‖²/σ² + Ω(x_l)` is a local
prediction error plus an energy/sparsity cost. The task enters as a
**class-conditioned top-layer prior** (the label sets the top-down prediction),
not as a competing cross-entropy — so the generative objective is undistorted.

## The three theorems (C stage) and their status

| # | Claim | Proof status | Prototype status |
|---|-------|--------------|------------------|
| 1 | STDP is the zero-temperature limit of the free-energy synaptic update | Proven (finalized, `docs/theorem1.md`): square-energy threshold, two-τ window, envelope attributed to LIF dynamics | **PASS (shape)** — window fits $A\Delta e^{-\Delta/\tau}$, R²=0.82 (`deep_stdp.py`); sign is anti-Hebbian (open Q) |
| 2 | Layer-wise updates make F a potential game → F monotonically non-increasing, converging to Nash = local min | Holds under PC symmetric convention, round-robin/damped updates, layer strong convexity | **PASS — clean** (`toy_verify.py` P1, `shd_train.py`) |
| 3 | Recurrent mean-field layer has a Curie transition at `β_c = 1/ρ(J)`; susceptibility `χ` diverges there → Fisher info peaks | 3.1–3.3 strict; 3.4 (task-info peak) conditional on small-signal + label-via-field | **PASS — clean** (`toy_verify.py` P2) |
| 3′ | (deep extension) The criticality peak survives stacking recurrent layers: accuracy vs β peaks near `min_l 1/ρ(W_rec_l)` | open theoretically — this code is the first test | **PASS — clean** (`deep_critical.py` D2) |

---

## Deep-network extension (`recurrent.py`, `deep_critical.py`)

The feedforward VPSC layers have no within-layer feedback, so Theorem 3's
`β_c = 1/ρ(W)` was meaningful only on an isolated recurrent layer (`toy_verify`
P2). To test the criticality result *inside* a deep network, each layer here is a
**recurrent mean-field layer** whose state is the fixed point of
`m = tanh(β(W_rec·m + I))`, with `β_c = 1/ρ(W_rec)` per layer. The network
critical β is `min_l 1/ρ(W_rec_l)` (the layer that goes critical first).

Two deep-network predictions tested:

- **D1 (Theorem 2 + recurrence):** under the pure generative F (now including
  the Ising interaction `−½ x^T Ws x` + mean-field entropy), F is still
  monotonically non-increasing at fixed β. **PASS** — F 1064→558, Spearman −1.0,
  with `ρ(W_rec)` held at the cap (0.90).
- **D2 (Theorem 3 deep):** for a trained deep recurrent net (frozen weights),
  accuracy vs β peaks near the network `β_c`. **PASS — clean:** peak at β*=0.80
  vs predicted `β_c = 0.81` (ratio 0.99), interior peak (0.39 vs 0.24 tails).

**Honest caveats from this stage (real, not bugs):**

1. **D1 needs a hard spectral cap.** The Ising interaction `−½ x^T W_rec x` is
   unbounded below; pure-F training drives `ρ(W_rec) → ∞` and F → −∞ (a
   degenerate minimum, observed in the first run: F=−7173, ρ≈11). The mean-field
   entropy does *not* prevent this — it vanishes at saturation. A soft penalty
   is also insufficient (out-gained by the interaction). A **hard spectral
   projection** (`project_spectral`, rescale `ρ(Ws) ≤ rho_max` after each step)
   is required to keep the fixed point well-defined and F bounded. The trained
   ρ sits exactly at the cap — a boundary solution — so D1 means "monotone-F
   holds in the well-defined regime," not "the unconstrained objective is
   healthy." This is a genuine open problem for the theory: constraining the
   variational family to `ρ < 1/β` is natural (it is exactly the regime where
   the mean-field fixed point exists), but formalizing it as part of the free
   energy rather than an external projection is unfinished.
2. **D2 is the substantive win.** The criticality peak survives stacking and
   aligns with `min_l 1/ρ(W_rec_l)` to within 1%. This is the first evidence
   that Theorem 3 extends from a single isolated recurrent layer to a deep
   recurrent network driving a task readout.
3. **Accuracy is low (~0.39, chance 0.25).** Consistent with the theory-first,
   accuracy-second scope. The point of D2 is the *shape* of the accuracy-vs-β
   curve and its alignment with β_c, not the absolute accuracy.

---

## Theorem 1 verification (`docs/theorem1.md`, `deep_stdp.py`)

The finalized proof (with the three verification revisions absorbed) is in
`docs/theorem1.md`. The empirical test injects a controlled pre-post pulse pair
at lag $\Delta$ into a trained recurrent layer and measures the free-energy
synaptic gradient.

- **Shape — PASS.** The pre-before-post gradient fits $A\,\Delta\,e^{-\Delta/\tau}$
  with $\tau\approx 4$, $R^2\approx 0.82$, interior peak at $\Delta\approx 5$,
  exponential decay beyond. The non-trivial prediction (a window that *rises then
  falls*, from the $\Delta$ factor and the $e^{-\Delta/\tau}$ factor) is confirmed.
- **The envelope comes from LIF leaky dynamics, not $\beta\to\infty$ (Revision C,
  empirically supported).** The window appears only when the layer uses leaky
  integration (`leak<1.0`) AND keeps its state graph-connected across timesteps
  (`detach_state=False`). Full relaxation leaves no PSP trace; detached state
  blocks temporal credit assignment. Both are exactly what §4 of the proof predicts.
- **Sign is anti-Hebbian — open question.** The gradient is positive for
  pre-before-post, so descent *depresses* — opposite to standard STDP. Minimizing
  prediction error undoes the input correlation. Recovering Hebbian STDP needs a
  sign convention tied to evidence maximization vs. recognition-error minimization;
  flagged as open in `docs/theorem1.md` §6.

---

## Repository

```
vpsc/
  neurons.py        Mean-field LIF: m = tanh(β(u−θ)), hard spike, spectral_radius
  network.py        Feedforward VPSCNet: bottom-up recognition + top-down prediction, F, class prior
  recurrent.py      Recurrent mean-field layers (W_rec feedback) — deep-network extension
  free_energy.py    free_energy_loss (P1/Thm2), ce_loss (aux), BetaAnnealer
experiments/
  toy_verify.py     self-contained verification of Thm 2 + Thm 3 (runs in seconds)
  shd_train.py      VPSC on Spiking Heidelberg Digits (or synthetic fallback)
  deep_critical.py  deep recurrent net: D1 (F monotone + recurrence), D2 (Thm 3 deep)
  deep_stdp.py      Theorem 1: STDP window shape on the trained deep recurrent net
docs/
  theorem1.md       finalized Theorem 1 proof (with A/B/C revisions + empirical result)
results/            written by the experiments (curves + PNG)
```

## Run

```bash
pip install -r requirements.txt
python experiments/toy_verify.py --epochs 120      # core Thm 2 + Thm 3 verification
python experiments/deep_critical.py --epochs 100   # deep-network extension (Thm 2+3 deep)
python experiments/deep_stdp.py --epochs 80        # Theorem 1 STDP window
python experiments/shd_train.py --synthetic --epochs 20   # no download needed
# real SHD (downloads ~70MB HDF5 on first run):
python experiments/shd_train.py --epochs 30 --batch 64
```

## Expected results

`toy_verify.py`:
- **P1 (Theorem 2):** total free energy F falls from ~220 to ~20, Spearman(epoch,F) ≈ −0.99. **PASS.**
- **P2 (Theorem 3):** on a recurrent mean-field layer with `ρ(J)=1`, susceptibility `χ` rises smoothly as `β→1⁻` then jumps ~5× at `β_c=1`. The largest jump sits at `β≈1.01`. **PASS.**

`shd_train.py` (synthetic or real):
- F monotonically decreases (Spearman ≈ −0.99). **Theorem 2 PASS.**
- Test accuracy ≈ chance. This is expected and honest — see finding (4) below.

---

## Findings surfaced by the prototype (these are real, not bugs)

1. **Theorem 2's monotonicity needs the PURE generative objective.** Adding a
   cross-entropy term (F + λ·CE) breaks it: CE's gradient flows through the top
   layer's state and distorts the generative model, so F *rises*. The class label
   must enter as the top-layer prior, not as a competing loss. (First prototype
   attempt failed this way; fixed by switching to a class-conditioned prior.)

2. **Theorem 2's monotonicity needs FIXED β.** β-annealing changes the objective
   every step, so per-step monotonicity is not guaranteed; as β rises, tanh
   saturates and prediction errors grow, making F increase even though the model
   is "improving." Annealing is a practical technique, not part of the theorem.
   Verify Theorem 2 at fixed β; use annealing only for practical runs.

3. **Theorem 3 applies to RECURRENT mean-field layers.** The Curie transition at
   `β_c = 1/ρ(J)` comes from the self-consistent feedback `m = tanh(β(Jm+h))`.
   A feedforward VPSC layer has no `Jm` feedback, so `β_c = 1/ρ(W)` is *not*
   meaningful there. Testing Theorem 3 faithfully requires the recurrent layer
   (as `toy_verify.py` P2 does). Extending the criticality result to deep
   feedforward VPSC networks is an open theoretical question.

4. **The pure-F objective is not discriminative.** With only the generative free
   energy, accuracy sits near chance — the class prior tracks the mean top-state
   per class but nothing forces class-separable representations (a learnable
   prior even admits a collapse degeneracy; we fix it to distinct orthogonal
   targets). Practical classification needs either the discriminative readout
   (`ce_loss`, used for P2 network training) or a stronger PC inference loop.
   This is the expected cost of a theory-first, accuracy-second method.

## Next steps

- Theorem 1 spike-timing experiment: verify the zero-τ STDP window match on a
  single recurrent layer's learned synapses.
- Full PC inference (top-down feedback during evaluation, per-hypothesis) instead
  of the feedforward nearest-prior proxy — expected to lift accuracy off chance.
- Add the recurrent `Jm` term to VPSC layers so Theorem 3's `β_c = 1/ρ(W)`
  becomes meaningful inside the deep network, not only on the isolated layer.
- Real SHD / DVS128 numbers once a discriminative path is in place.
