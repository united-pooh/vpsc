# VPSC on MNIST — comparison vs ANN (MLP) and CNN

A direct, honest comparison requested by the user: how does the spiking,
surrogate-free VPSC net stack up against a vanilla MLP and a small CNN on MNIST?

## Honest framing (read first)

MNIST is a **static image** task. VPSC is a **temporal spiking** net — its whole
point is to exploit time (LIF membrane dynamics, spike timing, recurrence).
Feeding it static MNIST via rate coding (T-step Bernoulli sampling at pixel
intensity) is a **stress test on чужой территории**, not its home turf.

- **CNN wins by design**: weight sharing + translation equivariance are
  architecturally matched to spatial images. It is expected to top the table.
- **VPSC is not trying to beat CNN here.** The point is to quantify the gap and
  the compute cost — the honest number the user asked for. VPSC's advantages
  (low-latency event processing, energy efficiency on neuromorphic hardware,
  the theoretical properties of `docs/theorem1.md`) simply do not show up on a
  static, rate-coded, GPU/CPU-simulated benchmark.

So: **a gap on MNIST is not a refutation of VPSC.** It is the expected result of
running a temporal-event model on a static task.

## What the comparison controls

- Same MNIST train/test split, same batch size, same epoch budget.
- VPSC and MLP have comparable parameter counts (~230k vs ~218k). CNN is smaller
  (~57k) but uses conv weight sharing.
- All trained with Adam + cross-entropy. (VPSC's readout uses CE; the theory
  concerns inference-time β, so the training rule is immaterial to Theorem 3.)

## Run

```bash
python lab/mnist/run_all.py --epochs 8 --T 10
# outputs: lab/mnist/results/mnist_compare.{json,png}
```

`--T` is the number of rate-coding timesteps for VPSC (more T = more temporal
samples = better accuracy but slower).

## Expected interpretation

| model | role | expected |
|---|---|---|
| CNN | spatial-image specialist | highest acc |
| MLP | fair-parameter ANN baseline | mid |
| VPSC | temporal spiking net, off-domain | lowest acc, but competitive |

The honest takeaway: on a static task, a spiking net is competitive-but-behind.
On its actual home turf — event-based temporal data (DVS, SHD speech, control) —
the comparison would be different, and that is where VPSC's design pays off
(see `experiments/shd_train.py` and the deep recurrent tests).

## Files

```
lab/mnist/
  data.py          MNIST loader + rate coding
  vpsc_mnist.py    VPSC training/eval on rate-coded MNIST
  baselines.py     MLP + CNN
  run_all.py       runs all three, prints table, writes json+png
  results/         mnist_compare.{json,png}
```
