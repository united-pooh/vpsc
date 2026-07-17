"""Benchmark: params, train time, inference time for MLP / CNN / VPSC on MNIST.

All three on the same hardware (CPU), same data, same batch size. Inference time
is measured on the full 10k test set, averaged over a few repeats after warmup.

VPSC is reported two ways:
  - per-image-step: the wall-clock cost of one forward pass (T timesteps of the
    spiking net). This is the fair comparison to a single ANN forward.
  - note: VPSC's inference naturally processes T timesteps; ANN/CNN process one.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vpsc import VPSCConfig, VPSCNet  # noqa: E402
from lab.mnist.data import get_mnist, rate_code  # noqa: E402
from lab.mnist.baselines import MLP, CNN  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def train_one(net, train_loader, epochs, lr, device, kind, T=10, seed=0):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.to(device)
    gen = torch.Generator(device=device).manual_seed(seed) if kind == "vpsc" else None
    t0 = time.time()
    for _ in range(epochs):
        net.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if kind == "vpsc":
                x_seq = rate_code(images, T, gen)
            else:
                x_seq = images
            opt.zero_grad()
            out = net(x_seq)
            logits = out["logits"] if kind == "vpsc" else out
            loss = Fnn.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
    return time.time() - t0


@torch.no_grad()
def infer_time(net, test_loader, device, kind, T=10, seed=1, repeats=3):
    net.eval()
    gen = torch.Generator(device=device).manual_seed(seed) if kind == "vpsc" else None
    # warmup
    for images, _ in test_loader:
        images = images.to(device)
        x = rate_code(images, T, gen) if kind == "vpsc" else images
        net(x)
        break
    times = []
    n_samples = 0
    for _r in range(repeats):
        t0 = time.time()
        n_samples = 0
        for images, _ in test_loader:
            images = images.to(device)
            x = rate_code(images, T, gen) if kind == "vpsc" else images
            net(x)
            n_samples += images.size(0)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    return avg, n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, epochs={args.epochs}, T={args.T}, batch={args.batch_size}\n")

    train_loader, test_loader = get_mnist(batch_size=args.batch_size)

    models = [
        ("MLP",  MLP(), "mlp"),
        ("CNN",  CNN(), "cnn"),
        ("VPSC", VPSCConfig(sizes=[784, 256, 64], n_classes=10, beta=1.0, tau_m=10.0,
                            threshold=0.5, sigma=1.0, sparsity=1e-4), "vpsc"),
    ]

    results = {}
    for name, net, kind in models:
        if kind == "vpsc":
            net = VPSCNet(net)
        print(f"=== {name} ===")
        n_params = count_params(net)
        train_t = train_one(net, train_loader, args.epochs, 1e-3, device, kind, args.T)
        inf_t, n = infer_time(net, test_loader, device, kind, args.T)
        ms_per_img = inf_t / n * 1000
        throughput = n / inf_t
        print(f"  params: {n_params}")
        print(f"  train time ({args.epochs} epochs): {train_t:.2f}s")
        print(f"  inference time (full test set, avg 3): {inf_t:.3f}s")
        print(f"  per-image: {ms_per_img:.3f} ms  |  throughput: {throughput:.0f} img/s\n")
        results[name] = {
            "params": n_params,
            "train_time_s": train_t,
            "infer_time_s": inf_t,
            "ms_per_image": ms_per_img,
            "throughput_img_s": throughput,
            "n_test": n,
        }

    # ---- comparison table ----
    print("=" * 78)
    print(f"{'model':<6} {'params':>9} {'train_s':>9} {'infer_s':>9} {'ms/img':>8} {'img/s':>9}")
    print("-" * 78)
    for name in ["MLP", "CNN", "VPSC"]:
        r = results[name]
        print(f"{name:<6} {r['params']:>9d} {r['train_time_s']:>9.2f} "
              f"{r['infer_time_s']:>9.3f} {r['ms_per_image']:>8.3f} {r['throughput_img_s']:>9.0f}")
    print("=" * 78)

    # ranks
    print("\nRanks (smaller/shorter/faster = better):")
    by_params = sorted(results, key=lambda n: results[n]["params"])
    by_train = sorted(results, key=lambda n: results[n]["train_time_s"])
    by_inf = sorted(results, key=lambda n: results[n]["ms_per_image"])
    print(f"  params  (fewest→most): {' < '.join(by_params)}")
    print(f"  train   (short→long) : {' < '.join(by_train)}")
    print(f"  infer   (fast→slow)  : {' < '.join(by_inf)}")

    out = {"config": {"epochs": args.epochs, "T": args.T, "device": device}, "results": results}
    with open(os.path.join(RESULTS_DIR, "benchmark.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[written {os.path.join(RESULTS_DIR, 'benchmark.json')}]")


if __name__ == "__main__":
    main()
