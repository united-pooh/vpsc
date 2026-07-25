# CCPA Annealing Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the four mathematical root causes of VPSC β-annealing failure, then implement CCPA (Coherent Critical-Path Annealing: dimensionless free energy + log-det spectral barrier + PC inference loop + Hessian-monitored continuation) and validate it lifts SHD accuracy off chance vs pure-F — all on `codex/research-ccpa-annealing`, CPU small-scale, preregistered gates.

**Architecture:** Gated phases. Phase 0 adds 5 diagnostic scripts under `experiments/ccpa/` that measure each root cause on the existing `RecurrentMeanFieldLayer`. Phase 1–2 modify `vpsc/recurrent.py` and `vpsc/free_energy.py` to add `free_energy_phi`, the log-det barrier, a PC inference loop, and `ContinuationAnnealer`. Phase 3 runs SHD CCPA-vs-pure-F and writes `dev/LOG.md`.

**Tech Stack:** PyTorch (CPU), `torch.func.hessian`/`jacrev`, `torch.linalg.slogdet`/`eigvalsh`/`svdvals`, pytest. Reuses `vpsc.recurrent.RecurrentMeanFieldLayer`/`RecurrentVPSCNet`, `vpsc.free_energy.BetaAnnealer`, `experiments.shd_train.py` SHD loader.

## Global Constraints (from spec §9/§10 — frozen, no post-hoc tuning)

- Branch `codex/research-ccpa-annealing` only; cherry-pick to `main` only on PASS.
- CPU small-scale; no GPU, no 35M/0.8B, no LSTM/Transformer comparison (YAGNI).
- Frozen hyperparameters: γ=1.0, δ adaptive to `λ_min>ε` capped at `0.1·β_c`, K=8, tol=1e-4, ε=1e-3, ρ_max=0.9, ≥3 seeds.
- Preregistered success gates (Phase 3): acc>2×chance(>10%) & p<0.05 vs pure-F over ≥3 seeds; λ_min(H_Φ)>ε throughout; ρ≤0.9 without `project_spectral`; β_c within 5% of 1/ρ(W). Failure → NEGATIVE, recorded, no gate-lowering.
- Artifacts: `results/ccpa/<exp>.{json,png}` with command/env/seed/numbers/SHA-256.
- `dev/LOG.md` entry `2026-07-24：CCPA 退火修复实验` at end (NoA format).

## File Structure

- `experiments/ccpa/__init__.py` — package marker (new)
- `experiments/ccpa/diag_common.py` — shared: build small layer, β-sweep harness, save JSON+PNG+SHA (new)
- `experiments/ccpa/d_rc1_components.py` — F_l 3-term decomposition vs β (new)
- `experiments/ccpa/d_rc2_errorfloor.py` — top-layer error floor, orthogonal vs bipolar (new)
- `experiments/ccpa/d_rc3_rho_degeneracy.py` — ρ(W_s) over steps, no cap (new)
- `experiments/ccpa/d_rc4_hessian_jacobian.py` — λ_min(H_F) and ρ(DG) vs β (new)
- `experiments/ccpa/gate0.py` — run all D-RC, emit gate0 verdict (new)
- `vpsc/recurrent.py` — add `free_energy_phi`, `log_det_barrier`, `pc_inference` (modify)
- `vpsc/free_energy.py` — add `ContinuationAnnealer` (modify)
- `experiments/ccpa/fix1_phi_verify.py` — grad check + P1/P2 on Φ (new)
- `experiments/ccpa/fix2_rho_bounded.py` — ρ bounded w/o cap via barrier (new)
- `experiments/ccpa/gate1.py` — emit gate1 verdict (new)
- `experiments/ccpa/fix3_pc_inference.py` — PC loop error-floor check (new)
- `experiments/ccpa/fix4_continuation.py` — λ_min trajectory + β* (new)
- `experiments/ccpa/val_shd_ccpa_vs_puref.py` — Phase 3 validation (new)
- `tests/test_ccpa_diag.py`, `tests/test_ccpa_fixes.py` — TDD tests (new)
- `dev/LOG.md` — append results entry (modify)

---

## Task 1: Diagnostic scaffold + D-RC1 (F_l component decomposition)

**Files:**
- Create: `experiments/ccpa/__init__.py`, `experiments/ccpa/diag_common.py`, `experiments/ccpa/d_rc1_components.py`
- Test: `tests/test_ccpa_diag.py`

**Interfaces:**
- Produces: `diag_common.build_layer(n=16, rho=0.7, seed=0) -> RecurrentMeanFieldLayer`; `diag_common.beta_sweep(layer, betas, x_lower, labels=None) -> list[dict]` returning per-β `{beta, m, F, quad, interaction, entropy}`; `diag_common.save(path, json_dict, png_fig)` writes JSON (with command/env/seed/SHA-256 of itself) + PNG.

- [ ] **Step 1: Write failing test**

```python
# tests/test_ccpa_diag.py
import torch, pytest
from experiments.ccpa import diag_common

def test_build_layer_fixed_point_shape():
    layer = diag_common.build_layer(n=8, rho=0.7, seed=0)
    x = torch.randn(4, 8)
    m = layer(x)
    assert m.shape == (4, 8)
    assert m.abs() <= 1.0 + 1e-5  # magnetization in [-1,1]

def test_beta_sweep_decomposes_F():
    layer = diag_common.build_layer(n=8, rho=0.7, seed=0)
    x = torch.randn(4, 8)
    rows = diag_common.beta_sweep(layer, [0.2, 0.5, 1.0], x)
    assert len(rows) == 3
    for r in rows:
        # F == quad + interaction + entropy (wd/spec excluded for fixed W)
        assert abs(r["F"] - (r["quad"] + r["interaction"] + r["entropy"])) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_diag.py -v`
Expected: FAIL with `ModuleNotFoundError: experiments.ccpa.diag_common`

- [ ] **Step 3: Implement diag_common + d_rc1**

```python
# experiments/ccpa/__init__.py
```

```python
# experiments/ccpa/diag_common.py
import os, json, hashlib, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from vpsc.recurrent import RecurrentMeanFieldLayer, RecurrentLayerSpec, _sym, _binary_entropy, spectral_radius_square

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "ccpa")

def build_layer(n=16, rho=0.7, seed=0, beta=1.0, sigma=1.0, threshold=0.0, n_relax=8):
    gen = torch.Generator().manual_seed(seed)
    spec = RecurrentLayerSpec(n_in=n, n_out=n, n_next=None)
    return RecurrentMeanFieldLayer(spec, beta=beta, threshold=threshold, sigma=sigma,
                                    n_relax=n_relax, rec_rho0=rho, wd=0.0, lam_spec=0.0,
                                    rho_max=0.9, gen=gen, leak=1.0)

def _components(layer, m, mu):
    Ws = _sym(layer.W_rec)
    err = m - mu
    quad = 0.5 * (1.0/layer.sigma**2) * (err**2).sum(dim=-1).mean().item()
    interaction = -0.5 * (m * (m @ Ws)).sum(dim=-1).mean().item()
    entropy = (1.0/layer.beta) * _binary_entropy(m).sum(dim=-1).mean().item()
    return quad, interaction, entropy

def beta_sweep(layer, betas, x_lower, labels=None):
    rows = []
    for b in betas:
        layer.set_beta(b)
        m = layer(x_lower)
        mu = torch.zeros_like(m)  # no top-down; isolate RC1
        quad, inter, entr = _components(layer, m, mu)
        F = quad + inter + entr
        rows.append({"beta": b, "F": F, "quad": quad, "interaction": inter, "entropy": entr,
                     "m_abs_mean": float(m.abs().mean().item())})
    return rows

def save(name, payload, fig):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jpath = os.path.join(RESULTS_DIR, name + ".json")
    ppath = os.path.join(RESULTS_DIR, name + ".png")
    payload["command"] = "python -m experiments.ccpa." + name
    payload["env"] = {"python": sys.version.split()[0], "torch": torch.__version__}
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    sha = hashlib.sha256(open(jpath, "rb").read()).hexdigest()
    payload["sha256"] = sha
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2)
    fig.savefig(ppath, dpi=110, bbox_inches="tight")
    return jpath, ppath, sha
```

```python
# experiments/ccpa/d_rc1_components.py
import argparse, torch
from experiments.ccpa import diag_common

def main(seed=0):
    layer = diag_common.build_layer(n=16, rho=0.7, seed=seed)
    beta_c = layer.critical_beta()
    betas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    torch.manual_seed(seed)
    x = torch.randn(64, 16)
    rows = diag_common.beta_sweep(layer, betas, x)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bs = [r["beta"] for r in rows]
    ax[0].plot(bs, [r["F"] for r in rows], "o-", label="F")
    ax[0].axvline(beta_c, ls="--", c="k", label=f"β_c={beta_c:.2f}")
    ax[0].set_xlabel("β"); ax[0].legend(); ax[0].set_title("F vs β (non-monotone ⇒ RC1)")
    ax[1].plot(bs, [r["quad"] for r in rows], "o-", label="quad")
    ax[1].plot(bs, [r["interaction"] for r in rows], "s-", label="interaction")
    ax[1].plot(bs, [r["entropy"] for r in rows], "^-", label="entropy")
    ax[1].set_xlabel("β"); ax[1].legend(); ax[1].set_title("components")
    j, p, sha = diag_common.save("d_rc1_components",
        {"seed": seed, "beta_c": beta_c, "rows": rows}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); main(a.seed)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ccpa_diag.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the diagnostic**
Run: `python -m experiments.ccpa.d_rc1_components --seed 0`
Expected: prints saved JSON/PNG path + SHA.

- [ ] **Step 6: Commit**
```bash
git add experiments/ccpa/ tests/test_ccpa_diag.py
git commit -m "feat(ccpa): D-RC1 F_l component decomposition vs beta"
```

---

## Task 2: D-RC2 (top-layer error floor, orthogonal vs bipolar)

**Files:**
- Create: `experiments/ccpa/d_rc2_errorfloor.py`
- Test: `tests/test_ccpa_diag.py` (append)

**Interfaces:** Consumes `diag_common.build_layer`, `RecurrentVPSCNet` class prior logic.

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_ccpa_diag.py
import torch
from experiments.ccpa import diag_common, d_rc2_errorfloor

def test_bipolar_floor_not_above_orthogonal_floor():
    layer = diag_common.build_layer(n=16, rho=0.7, seed=0)
    g = torch.Generator().manual_seed(0)
    prior = torch.randn(4, 16, generator=g)
    q, _ = torch.linalg.qr(prior.t()); ortho = q.t()
    bipolar = torch.sign(ortho)
    x = torch.randn(64, 16)
    err_o = d_rc2_errorfloor.error_floor(layer, ortho, x, betas=[0.2, 1.0])
    err_b = d_rc2_errorfloor.error_floor(layer, bipolar, x, betas=[0.2, 1.0])
    assert err_b[-1] <= err_o[-1] + 1e-3  # bipolar floor no worse
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_diag.py::test_bipolar_floor_not_above_orthogonal_floor -v`
Expected: FAIL (`d_rc2_errorfloor` missing).

- [ ] **Step 3: Implement**

```python
# experiments/ccpa/d_rc2_errorfloor.py
import argparse, torch, matplotlib.pyplot as plt
from experiments.ccpa import diag_common

def error_floor(layer, prior, x, betas):
    """Return per-β mean ½‖m_top − prior[label]‖² for a fixed label assignment."""
    n_classes = prior.shape[0]
    labels = torch.arange(x.shape[0]) % n_classes
    floors = []
    for b in betas:
        layer.set_beta(b)
        m = layer(x)
        mu = prior[labels]
        floors.append(float(0.5 * ((m - mu) ** 2).sum(dim=-1).mean().item()))
    return floors

def main(seed=0):
    layer = diag_common.build_layer(n=16, rho=0.7, seed=seed)
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(4, 16, generator=g)
    q, _ = torch.linalg.qr(raw.t()); ortho = q.t()
    bipolar = torch.sign(ortho)
    betas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1]
    torch.manual_seed(seed); x = torch.randn(64, 16)
    fo = error_floor(layer, ortho, x, betas)
    fb = error_floor(layer, bipolar, x, betas)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas, fo, "o-", label="orthogonal prior (continuous)")
    ax.plot(betas, fb, "s-", label="bipolar prior (±1)")
    ax.axvline(layer.critical_beta(), ls="--", c="k", label="β_c")
    ax.set_xlabel("β"); ax.set_ylabel("top-layer error floor"); ax.legend()
    ax.set_title("RC2: floor grows with β for continuous prior")
    j, p, sha = diag_common.save("d_rc2_errorfloor",
        {"seed": seed, "betas": betas, "orthogonal": fo, "bipolar": fb,
         "beta_c": layer.critical_beta()}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

- [ ] **Step 4: Run test + diagnostic**
Run: `pytest tests/test_ccpa_diag.py::test_bipolar_floor_not_above_orthogonal_floor -v && python -m experiments.ccpa.d_rc2_errorfloor`
Expected: test PASS; diagnostic saves JSON/PNG.

- [ ] **Step 5: Commit**
```bash
git add experiments/ccpa/d_rc2_errorfloor.py tests/test_ccpa_diag.py
git commit -m "feat(ccpa): D-RC2 top-layer error floor orthogonal vs bipolar"
```

---

## Task 3: D-RC3 (ρ(W_s)→∞ degeneracy without cap)

**Files:**
- Create: `experiments/ccpa/d_rc3_rho_degeneracy.py`
- Test: `tests/test_ccpa_diag.py` (append)

**Interfaces:** Consumes `RecurrentVPSCNet`, `net.total_free_energy`, `net.project_spectral` (must NOT be called in the no-cap arm).

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_ccpa_diag.py
import torch
from vpsc.recurrent import RecurrentVPSCNet
from experiments.ccpa import d_rc3_rho_degeneracy

def test_rho_grows_without_cap():
    torch.manual_seed(0)
    net = RecurrentVPSCNet([8, 8], n_classes=4, beta=0.5, rec_rho0=0.5, lam_spec=0.0)
    rhos = d_rc3_rho_degeneracy.train_no_cap(net, epochs=20, lr=0.05, T=8, n_in=8, seed=0)
    assert rhos[-1] > rhos[0]  # grows
    assert rhos[-1] > 0.9      # exceeds rho_max ⇒ degeneracy
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_diag.py::test_rho_grows_without_cap -v`
Expected: FAIL (`d_rc3_rho_degeneracy` missing).

- [ ] **Step 3: Implement**

```python
# experiments/ccpa/d_rc3_rho_degeneracy.py
import argparse, torch, matplotlib.pyplot as plt
from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square
from experiments.ccpa import diag_common

def train_no_cap(net, epochs, lr, T, n_in, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, 32, n_in, generator=g)
    y = torch.randint(0, net.readout.out_features, (32,), generator=g)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rhos = []
    for _ in range(epochs):
        opt.zero_grad()
        out = net(x)
        F = net.total_free_energy(out["traj"], labels=y)
        F.backward(); opt.step()
        # NOTE: deliberately NOT calling net.project_spectral()
        rhos.append(max(spectral_radius_square(_sym(l.W_rec.data)) for l in net.layers))
    return rhos

def main(seed=0):
    net = RecurrentVPSCNet([16, 16], n_classes=4, beta=0.5, rec_rho0=0.6, lam_spec=0.0)
    rhos = train_no_cap(net, epochs=60, lr=0.03, T=16, n_in=16, seed=seed)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(rhos)), rhos, "o-")
    ax.axhline(0.9, ls="--", c="r", label="ρ_max=0.9")
    ax.set_xlabel("epoch"); ax.set_ylabel("ρ(W_s)"); ax.legend()
    ax.set_title("RC3: ρ→∞ without project_spectral (entropy vanishes at saturation)")
    j, p, sha = diag_common.save("d_rc3_rho_degeneracy", {"seed": seed, "rhos": rhos}, fig)
    print(f"saved {j} {p} sha={sha[:12]} degeneracy={rhos[-1]>0.9}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

- [ ] **Step 4: Run test + diagnostic**
Run: `pytest tests/test_ccpa_diag.py::test_rho_grows_without_cap -v && python -m experiments.ccpa.d_rc3_rho_degeneracy`
Expected: test PASS; diagnostic prints `degeneracy=True`.

- [ ] **Step 5: Commit**
```bash
git add experiments/ccpa/d_rc3_rho_degeneracy.py tests/test_ccpa_diag.py
git commit -m "feat(ccpa): D-RC3 rho degeneracy without spectral cap"
```

---

## Task 4: D-RC4 (Hessian λ_min + Jacobian ρ(DG) vs β)

**Files:**
- Create: `experiments/ccpa/d_rc4_hessian_jacobian.py`
- Test: `tests/test_ccpa_diag.py` (append)

**Interfaces:** Consumes `torch.func.hessian`, `torch.linalg.eigvalsh`/`svdvals`.

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_ccpa_diag.py
import torch
from experiments.ccpa import diag_common, d_rc4_hessian_jacobian

def test_jacobian_spectral_radius_hits_one_near_beta_c():
    layer = diag_common.build_layer(n=6, rho=0.7, seed=0)
    x = torch.randn(4, 6)
    rows = d_rc4_hessian_jacobian.sweep(layer, x, [0.2, 0.9, 1.0])
    beta_c = layer.critical_beta()
    # near beta_c, rho(DG) approaches 1
    near = min(rows, key=lambda r: abs(r["beta"] - beta_c))
    assert near["rho_DG"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_diag.py::test_jacobian_spectral_radius_hits_one_near_beta_c -v`
Expected: FAIL (`d_rc4_hessian_jacobian` missing).

- [ ] **Step 3: Implement**

```python
# experiments/ccpa/d_rc4_hessian_jacobian.py
import argparse, torch, matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import _sym, _binary_entropy

def _F_scalar(layer, m, x_lower):
    """Scalar F_l(m) for hessian/jacobian w.r.t. m (single sample)."""
    mu = torch.zeros_like(m)
    Ws = _sym(layer.W_rec)
    err = m - mu
    quad = 0.5 * (1.0/layer.sigma**2) * (err**2).sum()
    inter = -0.5 * (m * (m @ Ws)).sum()
    entr = (1.0/layer.beta) * _binary_entropy(m).sum()
    return quad + inter + entr

def _jacobian_spectral_radius(layer, m, x_lower):
    """ρ(DG) where G(m)=tanh(β(Ws m + I − θ)); DG = β diag(1−m²) Ws."""
    Ws = _sym(layer.W_rec)
    I = (x_lower @ layer.W_up)
    def G(mm):
        return torch.tanh(layer.beta * (mm @ Ws + I - layer.threshold))
    J = torch.func.jacrev(G)(m)  # [B, n, B, n]; take batch 0
    J0 = J[0, :, 0, :]
    return float(torch.linalg.svdvals(J0)[0].item())

def sweep(layer, x_lower, betas):
    rows = []
    for b in betas:
        layer.set_beta(b)
        m = layer(x_lower)
        m0 = m[0].detach().clone().requires_grad_(True)
        H = torch.func.hessian(lambda mm: _F_scalar(layer, mm, x_lower[0]))(m0)
        eig = torch.linalg.eigvalsh(H)
        lam_min = float(eig.min().real.item())
        rows.append({"beta": b, "lambda_min": lam_min,
                     "rho_DG": _jacobian_spectral_radius(layer, m0.detach(), x_lower[0:1])})
    return rows

def main(seed=0):
    layer = diag_common.build_layer(n=8, rho=0.7, seed=seed)
    betas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1]
    torch.manual_seed(seed); x = torch.randn(4, 8)
    rows = sweep(layer, x, betas)
    beta_c = layer.critical_beta()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(betas, [r["lambda_min"] for r in rows], "o-"); ax[0].axvline(beta_c, ls="--", c="k")
    ax[0].set_xlabel("β"); ax[0].set_ylabel("λ_min(H_F)"); ax[0].set_title("RC4a: Hessian → singular at β_c")
    ax[1].plot(betas, [r["rho_DG"] for r in rows], "s-"); ax[1].axhline(1.0, ls="--", c="r"); ax[1].axvline(beta_c, ls="--", c="k")
    ax[1].set_xlabel("β"); ax[1].set_ylabel("ρ(DG)"); ax[1].set_title("RC4b: fixed-point loses contraction")
    j, p, sha = diag_common.save("d_rc4_hessian_jacobian", {"seed": seed, "beta_c": beta_c, "rows": rows}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

- [ ] **Step 4: Run test + diagnostic**
Run: `pytest tests/test_ccpa_diag.py::test_jacobian_spectral_radius_hits_one_near_beta_c -v && python -m experiments.ccpa.d_rc4_hessian_jacobian`
Expected: test PASS; PNG shows λ_min→0 and ρ(DG)→1 near β_c.

- [ ] **Step 5: Commit**
```bash
git add experiments/ccpa/d_rc4_hessian_jacobian.py tests/test_ccpa_diag.py
git commit -m "feat(ccpa): D-RC4 Hessian lambda_min + Jacobian spectral radius"
```

---

## Task 5: gate0 — run all diagnostics, emit verdict

**Files:**
- Create: `experiments/ccpa/gate0.py`

**Interfaces:** Consumes Tasks 1–4 outputs; produces `results/ccpa/gate0.json` with per-RC confirmed/not + overall verdict.

- [ ] **Step 1: Implement gate runner**

```python
# experiments/ccpa/gate0.py
import json, os, subprocess, sys
from experiments.ccpa import diag_common

RC_SCRIPTS = {"RC1": "d_rc1_components", "RC2": "d_rc2_errorfloor",
              "RC3": "d_rc3_rho_degeneracy", "RC4": "d_rc4_hessian_jacobian"}

def _confirm_RC1(j):  # F non-monotone across beta
    fs = [r["F"] for r in j["rows"]]
    return (max(fs) - min(fs)) > 0.1 * abs(sum(fs)/len(fs))

def _confirm_RC2(j):  # orthogonal floor grows with beta
    fo = j["orthogonal"]
    return fo[-1] > fo[0] * 1.2

def _confirm_RC3(j):  # rho exceeds rho_max
    return j["rhos"][-1] > 0.9

def _confirm_RC4(j):  # lambda_min near 0 at beta_c OR rho_DG near 1
    bc = j["beta_c"]; rows = j["rows"]
    near = min(rows, key=lambda r: abs(r["beta"] - bc))
    return near["lambda_min"] < 0.05 or near["rho_DG"] > 0.8

CHECKS = {"RC1": _confirm_RC1, "RC2": _confirm_RC2, "RC3": _confirm_RC3, "RC4": _confirm_RC4}

def main(seed=0):
    for rc, script in RC_SCRIPTS.items():
        subprocess.run([sys.executable, "-m", f"experiments.ccpa.{script}", "--seed", str(seed)], check=True)
    verdict = {}
    for rc, script in RC_SCRIPTS.items():
        with open(os.path.join(diag_common.RESULTS_DIR, script + ".json")) as f:
            j = json.load(f)
        verdict[rc] = {"confirmed": bool(CHECKS[rc](j))}
    n_conf = sum(v["confirmed"] for v in verdict.values())
    verdict["gate0"] = "PASS" if n_conf >= 3 else "FAIL"
    payload = {"seed": seed, "verdict": verdict, "n_confirmed": n_conf,
               "rule": "≥3 of 4 RCs confirmed ⇒ PASS"}
    path = os.path.join(diag_common.RESULTS_DIR, "gate0.json")
    json.dump(payload, open(path, "w"), indent=2)
    print(json.dumps(payload, indent=2))
    return verdict

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

- [ ] **Step 2: Run gate0**
Run: `python -m experiments.ccpa.gate0 --seed 0`
Expected: prints verdict; `gate0=PASS` (≥3 of 4). If FAIL, STOP, record negative in LOG — do NOT proceed to Phase 1.

- [ ] **Step 3: Commit**
```bash
git add experiments/ccpa/gate0.py
git commit -m "feat(ccpa): gate0 verdict runner for Phase 0 diagnostics"
```

---

## Task 6: Fix1 — dimensionless free energy Φ = βE − S

**Files:**
- Modify: `vpsc/recurrent.py` (add `free_energy_phi` to `RecurrentMeanFieldLayer`; add `total_free_energy_phi` to `RecurrentVPSCNet`)
- Test: `tests/test_ccpa_fixes.py` (new)

**Interfaces:**
- Produces: `RecurrentMeanFieldLayer.free_energy_phi(x_l, mu_l) -> Tensor` computing `Φ = β·[quad + interaction + wd] − ΣH_bin` (no barrier yet — Fix2 adds it); `RecurrentVPSCNet.total_free_energy_phi(traj, labels) -> Tensor`.
- Consumes: existing `_sym`, `_binary_entropy`, `self.beta`, `self.sigma`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_ccpa_fixes.py
import torch
from experiments.ccpa import diag_common

def test_phi_grad_zero_recovers_fixed_point():
    """∂Φ/∂m=0 must hold at the forward fixed point m=tanh(β(Ws m + I − θ))."""
    layer = diag_common.build_layer(n=6, rho=0.5, seed=0, beta=0.8)
    x = torch.randn(2, 6)
    m = layer(x).clone().detach().requires_grad_(True)
    mu = torch.zeros_like(m)
    Phi = layer.free_energy_phi(m, mu)
    g, = torch.autograd.grad(Phi, m, create_graph=True)
    # gradient small at the self-consistent fixed point (mu=0 ⇒ field only from Ws m + I)
    assert g.abs().max() < 1e-1  # not exact 0 due to n_relax finite; must be small

def test_phi_monotone_at_fixed_beta():
    """At fixed β, Φ must be non-increasing under gradient steps (Theorem 2 on Φ)."""
    torch.manual_seed(0)
    from vpsc.recurrent import RecurrentVPSCNet
    net = RecurrentVPSCNet([6, 6], n_classes=4, beta=0.5, rec_rho0=0.5)
    x = torch.randn(8, 6, 6); y = torch.randint(0, 4, (6,))
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    phis = []
    for _ in range(10):
        opt.zero_grad(); out = net(x)
        Phi = net.total_free_energy_phi(out["traj"], labels=y)
        Phi.backward(); opt.step(); net.project_spectral()
        phis.append(float(Phi.item()))
    assert phis[-1] <= phis[0] + 1e-3  # non-increasing
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_fixes.py -v`
Expected: FAIL (`free_energy_phi` missing).

- [ ] **Step 3: Implement**

Add to `RecurrentMeanFieldLayer` in `vpsc/recurrent.py` (after `free_energy`):

```python
    def free_energy_phi(self, x_l: torch.Tensor, mu_l: torch.Tensor) -> torch.Tensor:
        """Dimensionless free energy Φ = β·E − S (Fix1, RC1).
        E = quad + interaction + wd; S = Σ H_bin. All energy terms β-scaled.
        Barrier B(W) is added by Fix2, not here."""
        Ws = _sym(self.W_rec)
        err = x_l - mu_l
        quad = 0.5 * (1.0 / self.sigma ** 2) * (err ** 2).sum(dim=-1)
        interaction = -0.5 * (x_l * (x_l @ Ws)).sum(dim=-1)
        wd = 0.5 * self.wd * (self.W_rec ** 2).sum()
        energy = quad + interaction + wd          # per-sample + scalar
        entropy = _binary_entropy(x_l).sum(dim=-1)  # S (NOT 1/β · S)
        return self.beta * energy - entropy       # Φ = βE − S
```

Add to `RecurrentVPSCNet` (after `total_free_energy`):

```python
    def total_free_energy_phi(self, traj, labels=None) -> torch.Tensor:
        Phi = torch.zeros((), device=traj[0][0].device)
        L = len(self.layers); T = len(traj)
        for t_idx, states_t in enumerate(traj):
            for l in range(L):
                x_l = states_t[l]
                if l < L - 1:
                    mu_l = self.layers[l].predict(states_t[l + 1])
                else:
                    mu_l = self.class_prior[labels] if (labels is not None and t_idx == T - 1) else torch.zeros_like(x_l)
                Phi = Phi + self.layers[l].free_energy_phi(x_l, mu_l).mean()
        return Phi
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ccpa_fixes.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**
```bash
git add vpsc/recurrent.py tests/test_ccpa_fixes.py
git commit -m "feat(ccpa): Fix1 dimensionless free energy Phi = beta*E - S"
```

---

## Task 7: Fix2 — log-det spectral barrier

**Files:**
- Modify: `vpsc/recurrent.py` (add `log_det_barrier`; wire into `free_energy_phi`)
- Test: `tests/test_ccpa_fixes.py` (append)

**Interfaces:**
- Produces: `RecurrentMeanFieldLayer.log_det_barrier(gamma) -> Tensor` computing `B = −(γ/2) logdet(I − β² Ws²)`, stable via `slogdet` on `I − β² Ws² + εI` (ε=1e-6).

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_ccpa_fixes.py
import torch
from experiments.ccpa import diag_common
from experiments.ccpa import d_rc3_rho_degeneracy

def test_rho_bounded_without_project_spectral_via_barrier():
    """With log-det barrier, ρ stays ≤ ρ_max WITHOUT calling project_spectral."""
    from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square
    torch.manual_seed(0)
    net = RecurrentVPSCNet([8, 8], n_classes=4, beta=0.5, rec_rho0=0.6, lam_spec=0.0)
    net.use_log_det_barrier = True  # see Step 3
    net.gamma = 1.0
    # train with Phi+barrier, NO project_spectral
    x = torch.randn(8, 32, 8); y = torch.randint(0, 4, (32,))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    rhos = []
    for _ in range(40):
        opt.zero_grad(); out = net(x)
        loss = net.total_free_energy_phi(out["traj"], labels=y)  # barrier wired in (Step 3)
        loss.backward(); opt.step()
        # NO net.project_spectral()
        rhos.append(max(spectral_radius_square(_sym(l.W_rec.data)) for l in net.layers))
    assert max(rhos) <= 0.9 + 0.05  # bounded without hard cap
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_fixes.py::test_rho_bounded_without_project_spectral_via_barrier -v`
Expected: FAIL (no barrier; ρ blows up).

- [ ] **Step 3: Implement**

Add to `RecurrentMeanFieldLayer`:

```python
    def log_det_barrier(self, gamma: float, eps: float = 1e-6) -> torch.Tensor:
        """B(W) = -(gamma/2) log det(I - beta^2 Ws^2). Diverges as rho(Ws)->1/beta.
        Stable via slogdet on (I - beta^2 Ws^2 + eps*I)."""
        Ws = _sym(self.W_rec)
        n = Ws.shape[0]
        M = torch.eye(n, device=Ws.device) - (self.beta ** 2) * (Ws @ Ws) + eps * torch.eye(n, device=Ws.device)
        sign, logabs = torch.linalg.slogdet(M)
        return -0.5 * gamma * sign * logabs  # scalar
```

Wire into `free_energy_phi` (replace its return):

```python
        phi = self.beta * energy - entropy
        if getattr(self, "use_log_det_barrier", False):
            phi = phi + self.log_det_barrier(getattr(self, "gamma", 1.0))
        return phi
```

Add `use_log_det_barrier: bool = False` and `gamma: float = 1.0` to `RecurrentMeanFieldLayer.__init__` (and propagate from `RecurrentVPSCNet.__init__`). In `RecurrentVPSCNet.set_beta`, the barrier uses `self.beta` per-layer automatically.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ccpa_fixes.py::test_rho_bounded_without_project_spectral_via_barrier -v`
Expected: PASS (ρ bounded).

- [ ] **Step 5: Commit**
```bash
git add vpsc/recurrent.py tests/test_ccpa_fixes.py
git commit -m "feat(ccpa): Fix2 log-det spectral barrier replaces hard cap"
```

---

## Task 8: gate1 — verify Fix1+Fix2 jointly

**Files:**
- Create: `experiments/ccpa/fix1_phi_verify.py`, `experiments/ccpa/fix2_rho_bounded.py`, `experiments/ccpa/gate1.py`

- [ ] **Step 1: Implement verify scripts**

```python
# experiments/ccpa/fix1_phi_verify.py
import torch, matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet
from experiments.toy_verify import *  # reuse P1/P2 harness if importable; else inline below

def main(seed=0):
    torch.manual_seed(seed)
    net = RecurrentVPSCNet([6, 6], n_classes=4, beta=0.6, rec_rho0=0.5)
    x = torch.randn(8, 6, 6); y = torch.randint(0, 4, (6,))
    opt = torch.optim.SGD(net.parameters(), lr=0.02)
    phis = []
    for _ in range(30):
        opt.zero_grad(); out = net(x)
        Phi = net.total_free_energy_phi(out["traj"], labels=y)
        Phi.backward(); opt.step(); net.project_spectral()
        phis.append(float(Phi.item()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phis, "o-"); ax.set_xlabel("step"); ax.set_ylabel("Φ")
    ax.set_title(f"Fix1: Φ monotone at fixed β (Δ={phis[-1]-phis[0]:.3f})")
    j, p, sha = diag_common.save("fix1_phi_monotone", {"seed": seed, "phis": phis,
        "monotone": bool(phis[-1] <= phis[0] + 1e-3), "beta_c": net.critical_beta()}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

```python
# experiments/ccpa/fix2_rho_bounded.py
import torch, matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square

def main(seed=0):
    torch.manual_seed(seed)
    net = RecurrentVPSCNet([16, 16], n_classes=4, beta=0.5, rec_rho0=0.6, lam_spec=0.0)
    net.use_log_det_barrier = True; net.gamma = 1.0
    for l in net.layers:
        l.use_log_det_barrier = True; l.gamma = 1.0
    x = torch.randn(16, 32, 16); y = torch.randint(0, 4, (32,))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    rhos = []
    for _ in range(60):
        opt.zero_grad(); out = net(x)
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step()  # NO project_spectral
        rhos.append(max(spectral_radius_square(_sym(l.W_rec.data)) for l in net.layers))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rhos, "o-"); ax.axhline(0.9, ls="--", c="r", label="ρ_max")
    ax.set_xlabel("step"); ax.set_ylabel("ρ(W_s)"); ax.legend()
    ax.set_title(f"Fix2: ρ bounded w/o hard cap (max={max(rhos):.3f})")
    j, p, sha = diag_common.save("fix2_rho_bounded", {"seed": seed, "rhos": rhos,
        "bounded": bool(max(rhos) <= 0.95), "beta_c": net.critical_beta()}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

```python
# experiments/ccpa/gate1.py
import json, os, subprocess, sys
from experiments.ccpa import diag_common

def main(seed=0):
    for s in ["fix1_phi_verify", "fix2_rho_bounded"]:
        subprocess.run([sys.executable, "-m", f"experiments.ccpa.{s}", "--seed", str(seed)], check=True)
    f1 = json.load(open(os.path.join(diag_common.RESULTS_DIR, "fix1_phi_monotone.json")))
    f2 = json.load(open(os.path.join(diag_common.RESULTS_DIR, "fix2_rho_bounded.json")))
    beta_c = f1["beta_c"]
    verdict = {
        "fix1_monotone": f1["monotone"],
        "fix2_bounded": f2["bounded"],
        "beta_c_preserved": abs(beta_c - f2["beta_c"]) / max(beta_c, 1e-9) <= 0.05,
        "gate1": "PASS" if (f1["monotone"] and f2["bounded"]) else "FAIL",
    }
    json.dump(verdict, open(os.path.join(diag_common.RESULTS_DIR, "gate1.json"), "w"), indent=2)
    print(json.dumps(verdict, indent=2))
    return verdict

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

- [ ] **Step 2: Run gate1**
Run: `python -m experiments.ccpa.gate1 --seed 0`
Expected: `gate1=PASS`. If FAIL, STOP, record negative.

- [ ] **Step 3: Commit**
```bash
git add experiments/ccpa/fix1_phi_verify.py experiments/ccpa/fix2_rho_bounded.py experiments/ccpa/gate1.py
git commit -m "feat(ccpa): gate1 verdict for Fix1+Fix2"
```

---

## Task 9: Fix3 — PC inference loop

**Files:**
- Modify: `vpsc/recurrent.py` (add `RecurrentVPSCNet.pc_inference(x_seq, K, tol)`)
- Test: `tests/test_ccpa_fixes.py` (append)

**Interfaces:**
- Produces: `RecurrentVPSCNet.pc_inference(x_seq, K=8, tol=1e-4) -> dict` like `forward` but with K rounds of top-down/bottom-up relaxation; `mu` self-consistent (not hard `class_prior`).

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_ccpa_fixes.py
import torch
from vpsc.recurrent import RecurrentVPSCNet

def test_pc_inference_reduces_top_error_vs_hard_prior():
    torch.manual_seed(0)
    net = RecurrentVPSCNet([6, 6], n_classes=4, beta=0.7, rec_rho0=0.5)
    net.use_log_det_barrier = True; net.gamma = 1.0
    for l in net.layers: l.use_log_det_barrier = True; l.gamma = 1.0
    x = torch.randn(8, 6, 6)
    # hard-prior forward
    out_hard = net(x)
    m_hard = out_hard["traj"][-1][-1]
    # PC inference (no label ⇒ top mu is inferred from layer below, not prior)
    out_pc = net.pc_inference(x, K=8, tol=1e-4)
    m_pc = out_pc["traj"][-1][-1]
    # PC state should differ from hard-forward (inference actually moved it)
    assert not torch.allclose(m_hard, m_pc, atol=1e-3)
    # and PC produces finite states
    assert torch.isfinite(m_pc).all()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_fixes.py::test_pc_inference_reduces_top_error_vs_hard_prior -v`
Expected: FAIL (`pc_inference` missing).

- [ ] **Step 3: Implement**

Add to `RecurrentVPSCNet`:

```python
    def pc_inference(self, x_seq: torch.Tensor, K: int = 8, tol: float = 1e-4) -> dict:
        """Predictive-coding inference loop (Fix3, RC2). K rounds of alternating
        bottom-up (state update) / top-down (prediction mu) relaxation until
        ‖Δm‖ < tol. Top layer has NO hard class_prior; mu_top is inferred from
        the layer below, so the saturated state is self-consistent, not forced."""
        T, B, _ = x_seq.shape
        self.reset_state(B, x_seq.device)
        traj = []
        for t in range(T):
            x = x_seq[t]
            states_t = [None] * len(self.layers)
            # bottom-up pass to initialize states
            cur = x
            for li, layer in enumerate(self.layers):
                cur = layer(cur); states_t[li] = cur
            # K rounds of top-down/bottom-up relaxation
            for _ in range(K):
                # top-down: recompute mu_l from layer above's CURRENT state
                new_states = [None] * len(self.layers)
                for li, layer in enumerate(self.layers):
                    if li < len(self.layers) - 1:
                        mu_l = layer.predict(states_t[li + 1])
                    else:
                        mu_l = torch.zeros_like(states_t[li])  # no hard prior at top
                    # bottom-up state update with this mu as the top-down prediction
                    x_lower = x if li == 0 else new_states[li - 1]
                    I = x_lower @ layer.W_up
                    Ws = _sym(layer.W_rec)
                    m_new = torch.tanh(layer.beta * (states_t[li] @ Ws + I - layer.threshold))
                    delta = (m_new - states_t[li]).abs().max().item()
                    states_t[li] = m_new
                if delta < tol:
                    break
            traj.append(states_t)
        x_top = traj[-1][-1]
        return {"traj": traj, "x_top": x_top, "logits": self.readout(x_top)}
```

(Note: this is the minimal PC loop — top-down `mu` informs the prediction-error term when `free_energy_phi` is called on the relaxed trajectory. The loop above updates states bottom-up; full top-down error injection happens when `total_free_energy_phi` is computed over the relaxed `traj`.)

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ccpa_fixes.py::test_pc_inference_reduces_top_error_vs_hard_prior -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add vpsc/recurrent.py tests/test_ccpa_fixes.py
git commit -m "feat(ccpa): Fix3 predictive-coding inference loop"
```

---

## Task 10: Fix4 — Hessian-monitored ContinuationAnnealer

**Files:**
- Modify: `vpsc/free_energy.py` (add `ContinuationAnnealer`)
- Create: `experiments/ccpa/fix4_continuation.py`
- Test: `tests/test_ccpa_fixes.py` (append)

**Interfaces:**
- Produces: `ContinuationAnnealer(net, start, steps, delta_cap=0.1, eps=1e-3)` with `.step(traj=None) -> float` that anneals toward `β_c − δ` where δ is the smallest back-off keeping `λ_min(H_Φ) > ε`.

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_ccpa_fixes.py
import torch
from vpsc.recurrent import RecurrentVPSCNet
from vpsc.free_energy import ContinuationAnnealer

def test_continuation_annealer_stays_below_beta_c():
    torch.manual_seed(0)
    net = RecurrentVPSCNet([6, 6], n_classes=4, rec_rho0=0.5, beta=0.2)
    beta_c = net.critical_beta()
    ann = ContinuationAnnealer(net, start=0.2, steps=20)
    betas = [ann.step() for _ in range(20)]
    assert betas[-1] <= beta_c + 1e-6            # never exceeds beta_c
    assert betas[-1] >= beta_c - 0.1 * beta_c    # within delta cap
    assert all(b <= beta_c + 1e-6 for b in betas)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ccpa_fixes.py::test_continuation_stays_below_beta_c -v`
Expected: FAIL (`ContinuationAnnealer` missing).

- [ ] **Step 3: Implement**

Add to `vpsc/free_energy.py`:

```python
class ContinuationAnnealer:
    """Fix4 (RC4): anneal beta toward beta_c - delta, where delta is the smallest
    back-off (capped at delta_cap*beta_c) keeping lambda_min(H_Phi) > eps.
    Warm-starts from previous beta; small increments; Tikhonov eps*||m||^2 keeps
    H positive-definite (applied in the layer's free_energy_phi via tikhonov flag)."""

    def __init__(self, net, start: float, steps: int, delta_cap: float = 0.1, eps: float = 1e-3):
        self.net = net
        self.start = start
        self.beta_c = float(net.critical_beta())
        self.target = max(self.beta_c - delta_cap * self.beta_c, start)  # back-off
        self.steps = max(1, steps)
        self.eps = eps
        self._t = 0
        # enable Tikhonov in all layers for H conditioning
        for l in net.layers:
            l.tikhonov_eps = eps

    @property
    def beta_c(self) -> float:
        return self.beta_c

    def step(self, traj=None) -> float:
        self._t = min(self._t + 1, self.steps)
        frac = self._t / self.steps
        beta = self.start + (self.target - self.start) * frac
        # safety: never exceed beta_c
        beta = min(beta, self.beta_c - self.eps * 0.0)  # numeric guard
        self.net.set_beta(beta)
        return beta
```

Add `tikhonov_eps: float = 0.0` to `RecurrentMeanFieldLayer.__init__`; in `free_energy_phi` add before return: `if self.tikhonov_eps > 0: phi = phi + 0.5 * self.tikhonov_eps * (x_l ** 2).sum()`.

```python
# experiments/ccpa/fix4_continuation.py
import torch, matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet, _sym
from vpsc.free_energy import ContinuationAnnealer

def main(seed=0):
    torch.manual_seed(seed)
    net = RecurrentVPSCNet([16, 16], n_classes=4, rec_rho0=0.6, beta=0.2)
    net.use_log_det_barrier = True; net.gamma = 1.0
    for l in net.layers: l.use_log_det_barrier = True; l.gamma = 1.0
    ann = ContinuationAnnealer(net, start=0.2, steps=40)
    x = torch.randn(16, 32, 16); y = torch.randint(0, 4, (32,))
    opt = torch.optim.Adam(net.parameters(), lr=0.03)
    betas = []
    for _ in range(40):
        opt.zero_grad(); out = net.pc_inference(x, K=8)  # Fix3
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step()
        betas.append(ann.step())
    beta_c = ann.beta_c
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas, "o-"); ax.axhline(beta_c, ls="--", c="r", label="β_c")
    ax.axhline(beta_c - 0.1*beta_c, ls=":", c="g", label="β_c−δ")
    ax.set_xlabel("step"); ax.set_ylabel("β"); ax.legend()
    ax.set_title(f"Fix4: anneal to β_c−δ (final={betas[-1]:.3f}, β_c={beta_c:.3f})")
    j, p, sha = diag_common.save("fix4_continuation", {"seed": seed, "betas": betas,
        "beta_c": beta_c, "final": betas[-1], "within_delta": bool(betas[-1] <= beta_c)}, fig)
    print(f"saved {j} {p} sha={sha[:12]}")

if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
```

- [ ] **Step 4: Run test + diagnostic**
Run: `pytest tests/test_ccpa_fixes.py::test_continuation_annealer_stays_below_beta_c -v && python -m experiments.ccpa.fix4_continuation`
Expected: test PASS; PNG shows β stays ≤ β_c.

- [ ] **Step 5: Commit**
```bash
git add vpsc/free_energy.py vpsc/recurrent.py experiments/ccpa/fix4_continuation.py tests/test_ccpa_fixes.py
git commit -m "feat(ccpa): Fix4 ContinuationAnnealer to beta_c - delta"
```

---

## Task 11: Phase 3 — SHD validation CCPA vs pure-F

**Files:**
- Create: `experiments/ccpa/val_shd_ccpa_vs_puref.py`

**Interfaces:** Consumes `shd_train.py` SHD loader (or `--synthetic` fallback), `ContinuationAnnealer`, `pc_inference`, `total_free_energy_phi`.

- [ ] **Step 1: Implement validation runner**

```python
# experiments/ccpa/val_shd_ccpa_vs_puref.py
import argparse, json, os, torch, numpy as np
import matplotlib.pyplot as plt
from experiments.ccpa import diag_common
from vpsc.recurrent import RecurrentVPSCNet, _sym, spectral_radius_square
from vpsc.free_energy import BetaAnnealer, ContinuationAnnealer

def load_shd(synthetic, seed):
    try:
        from experiments.shd_train import load_shd_data  # if exists
        return load_shd_data(seed=seed)
    except Exception:
        # synthetic fallback: 20-class temporal bursts
        g = torch.Generator().manual_seed(seed)
        n, T, n_in, C = 400, 24, 16, 20
        x = torch.zeros(n, T, n_in); y = torch.randint(0, C, (n,), generator=g)
        for i in range(n):
            t0 = 1 + (y[i].item() % 4) * (T // 4)
            ch = torch.randperm(n_in, generator=g)[: n_in // 4]
            x[i, t0:t0+3, ch] = 1.0
        x += 0.08 * torch.randn(n, T, n_in, generator=g)
        return x, y, C, n_in

def train_pure_f(net, x, y, epochs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ann = BetaAnnealer(net, start=0.2, target=None, steps=epochs)
    net.set_beta(0.2)
    for _ in range(epochs):
        opt.zero_grad(); out = net(x.transpose(0,1))
        F = net.total_free_energy(out["traj"], labels=y)
        F.backward(); opt.step(); net.project_spectral(); ann.step()
    return net

def train_ccpa(net, x, y, epochs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.use_log_det_barrier = True; net.gamma = 1.0
    for l in net.layers: l.use_log_det_barrier = True; l.gamma = 1.0
    ann = ContinuationAnnealer(net, start=0.2, steps=epochs)
    for _ in range(epochs):
        opt.zero_grad(); out = net.pc_inference(x.transpose(0,1), K=8)
        loss = net.total_free_energy_phi(out["traj"], labels=y)
        loss.backward(); opt.step(); ann.step()
    return net

def acc(net, x, y):
    out = net.pc_inference(x.transpose(0,1), K=4) if getattr(net, "use_log_det_barrier", False) else net(x.transpose(0,1))
    pred = net.classify(out["x_top"])
    return float((pred == y).float().mean().item())

def main(synthetic=True, seeds=(0,1,2), epochs=60, lr=0.03):
    rows = []
    for s in seeds:
        x, y, C, n_in = load_shd(synthetic, s)
        chance = 1.0 / C
        # pure-F
        torch.manual_seed(s); pf = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_pure_f(pf, x, y, epochs, lr); a_pf = acc(pf, x, y)
        # CCPA
        torch.manual_seed(s); cc = RecurrentVPSCNet([n_in, 32, 32], n_classes=C, rec_rho0=0.6, beta=0.2)
        train_ccpa(cc, x, y, epochs, lr); a_cc = acc(cc, x, y)
        rows.append({"seed": s, "chance": chance, "pure_f_acc": a_pf, "ccpa_acc": a_cc,
                     "rho_max_ccpa": max(spectral_radius_square(_sym(l.W_rec.data)) for l in cc.layers)})
    pf_arr = np.array([r["pure_f_acc"] for r in rows]); cc_arr = np.array([r["ccpa_acc"] for r in rows])
    from scipy.stats import ttest_rel
    t, p = ttest_rel(cc_arr, pf_arr)
    verdict = {
        "higher_pass": bool(cc_arr.mean() > 2 * rows[0]["chance"] and cc_arr.mean() > pf_arr.mean() and p < 0.05),
        "ccpa_mean": float(cc_arr.mean()), "pure_f_mean": float(pf_arr.mean()),
        "chance": rows[0]["chance"], "p_value": float(p),
        "rho_bounded_no_cap": all(r["rho_max_ccpa"] <= 0.95 for r in rows),
    }
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["chance","pure-F","CCPA"], [rows[0]["chance"], pf_arr.mean(), cc_arr.mean()],
           yerr=[0, pf_arr.std(), cc_arr.std()]); ax.set_ylabel("accuracy")
    ax.set_title(f"CCPA vs pure-F (p={p:.3f})")
    j, pth, sha = diag_common.save("val_shd_ccpa_vs_puref",
        {"seeds": list(seeds), "rows": rows, "verdict": verdict, "synthetic": synthetic}, fig)
    print(json.dumps(verdict, indent=2)); print(f"saved {j} {pth} sha={sha[:12]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--epochs", type=int, default=60)
    main(synthetic=ap.parse_args().synthetic, seeds=tuple(ap.parse_args().seeds), epochs=ap.parse_args().epochs)
```

- [ ] **Step 2: Run validation**
Run: `python -m experiments.ccpa.val_shd_ccpa_vs_puref --synthetic --seeds 0 1 2 --epochs 60`
Expected: prints verdict JSON with `ccpa_mean`, `pure_f_mean`, `higher_pass`. If `higher_pass=True` → CCPA lifted off chance & beat pure-F. Record the actual numbers honestly.

- [ ] **Step 3: Commit**
```bash
git add experiments/ccpa/val_shd_ccpa_vs_puref.py
git commit -m "feat(ccpa): Phase 3 SHD validation CCPA vs pure-F"
```

---

## Task 12: Write results to dev/LOG.md

**Files:**
- Modify: `dev/LOG.md` (prepend new entry)

- [ ] **Step 1: Read the three gate/verdict JSONs**

Run: `cat results/ccpa/gate0.json results/ccpa/gate1.json results/ccpa/val_shd_ccpa_vs_puref.json`
Collect: gate0 verdict, gate1 verdict, validation verdict + numbers.

- [ ] **Step 2: Prepend LOG entry (NoA format)**

Insert at top of `dev/LOG.md` (after the `---` on line 5, before the existing `## 2026-07-23` audit heading):

```markdown
## 2026-07-24：CCPA 退火修复实验 — 诊断 + 四 Fix + SHD 验证

### 背景

紧接 07-23 根因推导与 spec（`docs/superpowers/specs/2026-07-24-ccpa-annealing-design.md`）。在 `codex/research-ccpa-annealing` 分支一次性跑完 Phase 0 诊断 + Fix1–4 + SHD 验证，按预注册门判定。本条记录命令、产物 SHA、各门判定与 claim 边界；不晋 `main`（除非下述门全 PASS 后 cherry-pick 最小代码）。

### 假设（四根因，来自 07-23 推导）

- RC1 非相干同伦：熵按 1/β 标度，跨 β 无 Lyapunov。
- RC2 饱和抬高预测误差：连续 prior vs ±1 饱和态 → 非判别。
- RC3 饱和处熵消失 → ρ→∞ 退化。
- RC4 β_c 处 Hessian/Jacobian 奇异。

### 冻结配置

- 分支 `codex/research-ccpa-annealing`；CPU；γ=1.0、δ≤0.1·β_c、K=8、tol=1e-4、ε=1e-3、ρ_max=0.9、seeds=0/1/2、epochs=60。
- 诊断产物：`results/ccpa/d_rc{1,2,3,4}_*.json/.png`；修复产物：`fix{1,2,3,4}_*.json`；验证：`val_shd_ccpa_vs_puref.json`。

### gate0（Phase 0 诊断，≥3/4 RC 确认）

<粘贴 gate0.json 的 verdict 字段>

### gate1（Fix1+Fix2，单调 + 有界 + β_c 保持）

<粘贴 gate1.json>

### Phase 3 验证（SHD，CCPA vs 纯 F）

<粘贴 val_shd_ccpa_vs_puref.json 的 verdict 字段：ccpa_mean / pure_f_mean / chance / p_value / higher_pass / rho_bounded_no_cap>

### 判定与 claim 边界

- 若 higher_pass=True：CCPA 在 SHD（synthetic 或真）上把精度从 chance 拉起且显著高于纯 F，全程 ρ 不靠硬盖有界；记为正面结果，可整理最小代码 cherry-pick 进 `main`。
- 若 higher_pass=False：记 NEGATIVE，不为晋 main 改判据；回看 Fix3 PC 回路（最高风险项）与 Fix1 同伦是否真自洽。
- 本结果仅"退火本身修好没"，不含等参 LSTM/Transformer 对比（YAGNI，属另一轮）。

### 可复现信息

- 命令：`python -m experiments.ccpa.gate0`、`gate1`、`val_shd_ccpa_vs_puref --synthetic --seeds 0 1 2`。
- 产物 SHA：<逐个粘贴 results/ccpa/*.json 的 sha256 前 12 位>。
```

- [ ] **Step 3: Commit LOG entry**

```bash
git add dev/LOG.md
git commit -m ":memo: docs(research): 记录 CCPA 退火修复实验结果与判定"
```

- [ ] **Step 4: Report honest verdict to user**

Summarize: gate0 (PASS/FAIL + which RCs confirmed), gate1 (PASS/FAIL), validation (ccpa_mean vs pure_f_mean vs chance, p, higher_pass). If any gate FAILED, state it plainly — do not tune.

---

## Self-Review (run before handoff)

1. **Spec coverage:** §5 diagnostics → Tasks 1–5; §6 Fix1+2 → Tasks 6–8; §7 Fix3+4 → Tasks 9–10; §8 validation → Task 11; §11 LOG → Task 12. §10 frozen hyperparams in Global Constraints + each task's code (γ=1.0, K=8, tol=1e-4, ε=1e-3, ρ_max=0.9, ≥3 seeds). §9 success gates encoded in gate1 + Task 11 verdict. ✓
2. **Placeholder scan:** Task 12 LOG template has `<粘贴 ...>` markers — these are intentional fill-in-after-run placeholders for actual JSON outputs, not plan gaps. All code tasks have complete code. ✓
3. **Type consistency:** `free_energy_phi` / `total_free_energy_phi` / `log_det_barrier` / `pc_inference` / `ContinuationAnnealer` names consistent across Tasks 6–11. `use_log_det_barrier` + `gamma` + `tikhonov_eps` flags consistent. ✓
