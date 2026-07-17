"""Run VPSC vs ANN (MLP) vs CNN on MNIST and print a comparison table.

Honest framing: MNIST is a static image task. CNNs are architecturally matched to
spatial images (weight sharing, translation equivariance), so they win by design.
VPSC is a temporal spiking net — feeding it static MNIST via rate coding is a
stress test, not its home turf. The point is to quantify the gap, not to claim a
win. See the README's "honest framing" note.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lab.mnist import vpsc_mnist, baselines  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--T", type=int, default=10, help="VPSC timesteps for rate coding")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, epochs={args.epochs}, VPSC T={args.T}\n")

    results = {}

    print("=== MLP (ANN) ===")
    results["MLP"] = baselines.run_mlp(epochs=args.epochs, device=device)
    print()

    print("=== CNN ===")
    results["CNN"] = baselines.run_cnn(epochs=args.epochs, device=device)
    print()

    print("=== VPSC (spiking, rate-coded) ===")
    results["VPSC"] = vpsc_mnist.run(epochs=args.epochs, T=args.T, device=device, seed=args.seed)
    print()

    # ---- comparison table ----
    print("=" * 60)
    print(f"{'model':<8} {'test_acc':>10} {'params':>10} {'train_time_s':>14}")
    print("-" * 60)
    for name in ["MLP", "CNN", "VPSC"]:
        r = results[name]
        print(f"{name:<8} {r['test_acc']:>10.4f} {r['params']:>10d} {r['time']:>14.1f}")
    print("=" * 60)

    # gaps vs CNN
    cnn_acc = results["CNN"]["test_acc"]
    for name in ["MLP", "VPSC"]:
        gap = cnn_acc - results[name]["test_acc"]
        print(f"  {name} vs CNN gap: -{gap*100:.2f} pp")

    out = {
        "config": {"epochs": args.epochs, "T": args.T, "seed": args.seed, "device": device},
        "results": {
            name: {"test_acc": r["test_acc"], "params": r["params"], "time_s": r["time"]}
            for name, r in results.items()
        },
    }
    out_path = os.path.join(RESULTS_DIR, "mnist_compare.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[results written to {out_path}]")

    # ---- plot ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        names = ["MLP", "CNN", "VPSC"]
        accs = [results[n]["test_acc"] for n in names]
        params = [results[n]["params"] for n in names]
        times = [results[n]["time"] for n in names]
        ax[0].bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52"])
        ax[0].set_ylabel("test accuracy")
        ax[0].set_title("MNIST test accuracy")
        ax[0].set_ylim(0, 1)
        for i, a in enumerate(accs):
            ax[0].text(i, a + 0.01, f"{a:.3f}", ha="center")
        ax[1].bar(names, times, color=["#4C72B0", "#55A868", "#C44E52"])
        ax[1].set_ylabel("training time (s)")
        ax[1].set_title("Training time (CPU)")
        for i, t in enumerate(times):
            ax[1].text(i, t + 1, f"{t:.0f}s", ha="center")
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "mnist_compare.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot written to {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
