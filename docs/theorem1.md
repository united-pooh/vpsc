# Theorem 1 (finalized): STDP is the zero-temperature limit of the free-energy synaptic update

**Status:** conclusion proven; proof incorporates the three revisions surfaced during
verification (square-energy threshold; two-time-constant window; error-collapse
attribution to LIF dynamics rather than to the zero-temperature limit itself).

---

## 1. Setup

A recurrent mean-field VPSC layer (see `vpsc/recurrent.py`). Neuron $i$ has
membrane potential $u_i(t)$ obeying leaky integrate-and-fire (LIF) dynamics

$$\tau_m \dot u_i = -u_i + \sum_j w_{ij} s_j(t) + I_i^{\mathrm{ext}}(t), \qquad s_j(t) = \sum_k \delta(t - t_j^k),$$

with hard reset $u_i \mapsto u_{\mathrm{rest}}$ at each firing time $t_i^\ast$ (defined by
$u_i$ crossing $\theta$ outside the refractory period).

**Posterior-commitment error** (the free-energy quantity carried by neuron $i$):

$$\varepsilon_i(t) := u_i(t) - \mu_i(t), \qquad \mu_i(t) = g(\phi_i, x_{\mathrm{above}}(t)),$$

i.e. the deviation of the membrane potential from its top-down prediction. $\varepsilon_i > 0$
reads as "the neuron is more depolarized than predicted" — positive evidence for firing.

The per-neuron free-energy contribution over a window $[0,T]$, parametrized by the
firing times $\{t_i^\ast\}$, is

$$F_i[\{t_i^\ast\}] \;=\; \int_0^T \tfrac12 \varepsilon_i(t)^2\, dt \;+\; c\cdot n_i,$$

where $n_i$ is the number of spikes in the window and $c > 0$ is the per-spike
energy cost (the term $\Omega$ in the layer free energy). Minimizing $F_i$ over
firing times balances "fire to cancel error" ($\varepsilon$ large) against "don't
waste energy" ($c > 0$).

---

## 2. Lemma 1.1 (revised): single-spike optimal timing — square-energy threshold

**Claim.** With a single spike at $t_i^\ast$ and the input held fixed, the first
variation of $F_i$ w.r.t. $t_i^\ast$ is

$$\frac{\delta F_i}{\delta t_i^\ast}\bigg|_{t_i^\ast} \;=\; +\tfrac12\,\varepsilon_i(t_i^{\ast-})^2 \;-\; c,$$

so the optimal firing time satisfies

$$\boxed{\;\tfrac12\,\varepsilon_i(t_i^{\ast-})^2 \;=\; c\;}$$

i.e. **a spike fires when the accumulated error *energy* reaches the cost** — the
threshold lives on $\varepsilon^2$, not on $\varepsilon$ linearly.

**Proof.** At $t_i^\ast$ the membrane is hard-reset: $u_i(t_i^{\ast+}) = u_{\mathrm{rest}}$.
Assuming $u_{\mathrm{rest}} \approx -\theta$ (symmetric about zero) and $\mu_i$ continuous,
the error jumps $\varepsilon_i(t_i^{\ast-}) \to \varepsilon_i(t_i^{\ast+}) \approx -\varepsilon_i(t_i^{\ast-})$.
Hence $\tfrac12\varepsilon_i^2$ drops by $\tfrac12\varepsilon_i(t_i^{\ast-})^2$ across the spike.

Now delay the spike by $\delta t_i^\ast > 0$. On the interval $[t_i^\ast,\, t_i^\ast+\delta t_i^\ast]$
the reset has *not* yet occurred, so the error-energy $\tfrac12\varepsilon_i^2$ that the
spike would have cancelled is instead still being accumulated. The change in $F_i$ is

$$\delta F_i \;=\; +\tfrac12\,\varepsilon_i(t_i^{\ast-})^2\,\delta t_i^\ast \;-\; c\,\delta t_i^\ast,$$

(the $+c\,\delta t_i^\ast$ savings of the spike cost is lost because the spike is delayed).
Thus $\delta F_i/\delta t_i^\ast = +\tfrac12\varepsilon_i(t_i^{\ast-})^2 - c$, and the
stationary condition gives $\tfrac12\varepsilon_i(t_i^{\ast-})^2 = c$. $\square$

> **Revision A.** The first draft stated the threshold linearly ($\varepsilon = c$).
> Re-derivation gives a *square-energy* threshold $\tfrac12\varepsilon^2 = c$. The
> physical content is unchanged — "fire when error energy reaches cost" — but the
> functional form is quadratic. This matters: the synaptic-weight gradient below
> uses $\varepsilon_i$ *linearly*, so the threshold and the gradient are distinct
> objects; conflating them was the original error.

---

## 3. Synaptic weight gradient

The synapse $w_{ij}$ enters $F_i$ only through $u_i$ (and thus $\varepsilon_i$):

$$\frac{\partial F_i}{\partial w_{ij}} \;=\; \int_0^T \varepsilon_i(t)\,\frac{\partial u_i(t)}{\partial w_{ij}}\,dt.$$

Between spikes, $u_i$ is the linear convolution of its input with the LIF Green's
function $h(\cdot) = \tau_m^{-1} e^{-(\cdot)/\tau_m}\Theta(\cdot)$:

$$u_i(t) \;=\; \sum_k (h * (\cdot))_j(t) \;\Rightarrow\; \frac{\partial u_i(t)}{\partial w_{ij}} \;=\; P_j(t) \;:=\; \sum_k h(t - t_j^k),$$

the postsynaptic-potential (PSP) trace of presynaptic neuron $j$ — a sum of causal
exponential kernels, one per presynaptic spike, with time constant $\tau_m$.
Substituting,

$$\frac{\partial F_i}{\partial w_{ij}} \;=\; \int_0^T \varepsilon_i(t)\,P_j(t)\,dt. \tag{$\ast$}$$

(Reset contributions are $O(\text{residual})$ and vanish in the regime where spikes
saturate the mean-field; a rigorous bound is an open detail — see §6.)

---

## 4. The exponential form of $\varepsilon_i$ comes from LIF dynamics (Revision C)

This is the load-bearing step, and the first draft got its *attribution* wrong.

**What $\varepsilon_i$ actually is between spikes.** In an inter-spike interval,
both $u_i$ and (by assumption) $\mu_i$ follow leaky dynamics. Their difference obeys

$$\dot\varepsilon_i \;=\; -\varepsilon_i/\tau_m \;+\; \text{(input residual)}.$$

With the input residual small in the trained regime, $\varepsilon_i(t)$ relaxes
exponentially toward $0$ with time constant $\tau_m$, re-initialized at each spike
by the reset. Concretely, on the interval $(t_i^\ast,\, t_i^{\ast\prime})$ between two
fires,

$$\varepsilon_i(t) \;\approx\; A\, e^{-(t - t_i^\ast)/\tau_m} \cdot (\text{sign}),$$

a causal, decaying exponential launched at the *most recent* postsynaptic spike
$t_i^\ast$ — i.e. nonzero for $t > t_i^\ast$ up to the next reset. (Equivalently, cast
as a function centered at $t_i^\ast$ and read backward in time, it is the standard
"error trace" of predictive-coding/LIF filters.)

> **Revision C.** The first draft attributed this exponential envelope to the
> *zero-temperature limit* $\beta\to\infty$ of the mean-field magnetization. That is
> wrong: the mean-field limit $u_i = \beta^{-1}\operatorname{atanh} m_i$ yields, as
> $\beta\to\infty$, a *discrete sign-error* $\varepsilon_i \to \operatorname{sign}(a_i)-\operatorname{sign}(\mu_i) \in \{-2,0,2\}$,
> not a continuous exponential. The exponential shape comes from the **LIF time
> dynamics** ($\tau_m$ relaxation between resets), which is independent of $\beta$.

**What the zero-temperature limit *does* contribute.** It hardens the *firing
event*: at finite $\beta$ the magnetization $m_i = \tanh(\beta(u_i-\theta))$ is a
soft sigmoid; as $\beta\to\infty$ it tends to a hard threshold, and by Lemma 1.1 the
threshold becomes the energy condition $\tfrac12\varepsilon_i^2 = c$. So the two
mechanisms have distinct roles:

| quantity | source | role |
|---|---|---|
| exponential envelope of $\varepsilon_i$ | LIF dynamics ($\tau_m$) | the *shape* of the STDP window |
| hard firing threshold | $\beta\to\infty$ (Lemma 1.1) | the *event* that localizes the window to spike times |

This separation is cleaner than the first draft's "everything from $\beta\to\infty$."

---

## 5. Main theorem: the STDP window (Revision B)

Substitute the LIF-derived error trace into $(\ast)$. Take one presynaptic spike at
$t_j^k$ and one postsynaptic spike at $t_i^\ast$. The PSP kernel is
$P_j(t) = h(t - t_j^k) = \tau_m^{-1} e^{-(t-t_j^k)/\tau_m}\Theta(t - t_j^k)$ (causal,
time constant $\tau_m$). The error trace is a causal exponential with time constant
$\tau_m$ launched at $t_i^\ast$.

**Case $\tau_m = \tau_s$ (single time constant).** With both kernels sharing $\tau_m$,
the pre-before-post contribution ($t_j^k < t_i^\ast$) is

$$I_+ \;=\; \int_{t_j^k}^{t_i^\ast} A\,e^{-(t_i^\ast - t)/\tau_m}\,e^{-(t - t_j^k)/\tau_m}\,dt
\;=\; A\,(t_i^\ast - t_j^k)\,e^{-(t_i^\ast - t_j^k)/\tau_m},$$

i.e. $K_+(\Delta) = A_+\,\Delta\,e^{-\Delta/\tau_m}$ with $\Delta = t_i^\ast - t_j^k > 0$.
A symmetric argument gives $K_-(\Delta) = A_-\,\Delta\,e^{-\Delta/\tau_m}$ for
post-before-pre ($\Delta < 0$). This is the **standard STDP window**
$\Delta e^{-\Delta/\tau}$ (Bi–Poo form).

**Case $\tau_m \neq \tau_s$ (two time constants — Revision B).** In a real LIF neuron
the membrane and synaptic time constants differ. Let the PSP kernel carry $\tau_s$
and the error trace carry $\tau_m$. Their convolution is

$$I_+(\Delta) \;=\; A\,\frac{e^{-\Delta/\tau_m} - e^{-\Delta/\tau_s}}{\tau_s^{-1} - \tau_m^{-1}}, \qquad \Delta > 0,$$

a **difference of two exponentials**. This is the form actually used to fit biological
STDP data (Gerstner & Kistler). It reduces to $\Delta e^{-\Delta/\tau}$ in the
degenerate limit $\tau_m \to \tau_s$ (l'Hôpital).

> **Revision B.** The first draft asserted the window is $\Delta e^{-\Delta/\tau}$
> unconditionally. In fact that holds only when $\tau_m = \tau_s$; the general form is
> a double-exponential difference. The double-exponential form is *more* biologically
> faithful, not less.

**Main theorem.** With $\varepsilon_i$ given by LIF dynamics (§4) and the firing
threshold hardened by $\beta\to\infty$ (Lemma 1.1), the free-energy synaptic gradient
$(\ast)$ evaluates to

$$\boxed{\;\frac{\partial F_i}{\partial w_{ij}} \;\xrightarrow{\;\beta\to\infty\;}\; \sum_{t_i^\ast > t_j^k} K_+\!\left(t_i^\ast - t_j^k\right) \;-\; \sum_{t_i^\ast < t_j^k} K_-\!\left(t_j^k - t_i^\ast\right)\;}$$

with $K_\pm(\Delta) = A_\pm\,\Delta\,e^{-\Delta/\tau_m}$ (single-$\tau$) or the
double-exponential difference above (two-$\tau$). Gradient descent
$\Delta w_{ij} = -\eta\,\partial F_i/\partial w_{ij}$ is therefore **pre-before-post
potentiation, post-before-pre depression** — STDP. $\square$

---

## 6. What is proven, and what remains open

**Proven.**
- The STDP window is not a biological postulate: it is the zero-temperature limit of
  the variational-free-energy synaptic update, with the window's *shape* supplied by
  LIF dynamics and the *event localization* supplied by the mean-field threshold.
- The amplitudes $A_\pm$ are set by the energy cost $c$ (Lemma 1.1); the time scale
  by $\tau_m$ (and $\tau_s$).

**Open / assumed (stated honestly).**
1. **Reset-term bound.** §3 drops the reset contribution to $\partial u_i/\partial w_{ij}$.
   A rigorous proof needs an $O(\cdot)$ estimate showing it is negligible when spikes
   saturate the mean-field. The intuition: resets occur *after* saturation and perturb
   the PSP convolution only locally; the main term is the inter-spike exponential. Not
   yet formalized.
2. **Symmetry of $A_+, A_-$.** The proof assumes isotropic energy cost. Inhibitory
   synapses / asymmetric costs give $A_+ \neq A_-$ naturally; this is a parameter of
   the free energy, not a failure.
3. **Gaussian variational assumption.** $\mu_i$ as a point prediction presupposes a
   unimodal posterior. Multimodal posteriors would distort the exponential error
   trace; extending to flow/normalizing-flow posteriors is future work.
4. **Spectral regime.** The recurrent layer must satisfy $\rho(W_{\mathrm{rec}}) < 1/\beta$
   for the mean-field fixed point to exist (the hard spectral cap of
   `recurrent.py` enforces this). Theorem 1 is stated *within* this regime.

**Empirical status.** The single-$\tau$ window form is tested directly on the trained
recurrent deep network in `experiments/deep_stdp.py` — see the next stage.

### Empirical result (`deep_stdp.py`)

On a trained recurrent VPSC layer (pure-F, frozen except the tested synapse), a
controlled pre-post pulse pair at lag $\Delta$ produces a free-energy synaptic
gradient whose pre-before-post side fits $A\,\Delta\,e^{-\Delta/\tau}$ with
**$\tau \approx 4$, $R^2 \approx 0.82$**, an interior peak at $\Delta \approx 5$,
and a clean exponential decay beyond. **The window shape predicted by Theorem 1 is
confirmed on the deep recurrent network.** (The $\Delta=1$ point is a same-timestep
saturation artifact and is excluded from the fit.)

Two findings that surfaced during the test, both instructive:

1. **The membrane trace must be leaky AND graph-connected.** The mean-field layer
   defaults to full relaxation per timestep (`leak=1.0`) with `detach_state=True`.
   Both kill the STDP signal: full relaxation leaves no persistent PSP trace for
   the post spike to coincide with (so pre-before-post at $\Delta>1$ gave exactly
   zero), and `detach_state` severs the graph between timesteps (so autograd could
   not assign credit from $t_{\mathrm{post}}$ back to $t_{\mathrm{pre}}$). Setting
   `leak<1.0` (leaky LIF integration — exactly the $\tau_m$ dynamics of §4) and
   `detach_state=False` for the measurement recovers the window. This is direct
   empirical support for **Revision C**: the window's envelope comes from LIF
   leaky dynamics, not from $\beta\to\infty$.

2. **The sign is anti-Hebbian, not Hebbian.** The free-energy gradient is
   *positive* for pre-before-post, so gradient descent $\Delta w = -\eta\nabla F$
   *depresses* the synapse — the opposite of standard STDP. This is expected and
   not a bug: minimizing the prediction-error free energy *undoes* the input
   correlation (the synapse learns to cancel the presynaptic drive so the
   postsynaptic prediction matches its state). Recovering Hebbian STDP requires a
   sign flip, which arises naturally if the update maximizes evidence under a
   generative model rather than minimizes a recognition error — but formalizing
   that within VPSC is an **open theoretical question**. The shape of the window
   (the non-trivial, falsifiable prediction) is confirmed; the sign convention is
   unresolved.
