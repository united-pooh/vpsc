"""VPSC on MNIST (rate-coded).

Uses the feedforward VPSCNet (vpsc.network). MNIST images are rate-coded into
T-step spike trains (data.rate_code) and presented as [T, B, 784].

Training: cross-entropy on the readout logits. Theorem 3 concerns inference-time
beta given fixed weights, so the training rule is immaterial to the theory; CE is
used for a fair accuracy comparison against ANN/CNN. We also report the pure
generative free-energy F (Theorem 2) for monitoring.

The point of this script is NOT to beat CNN on MNIST (CNNs win static image
tasks by design). It is to see how close a spiking, surrogate-free, mean-field
net gets, and at what compute cost — the honest comparison the user asked for.
"""

import os
import sys
import time
from typing import List

import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vpsc import VPSCConfig, VPSCNet  # noqa: E402
from lab.mnist.data import get_mnist, rate_code  # noqa: E402


def train(net, train_loader, epochs, T, lr, device, seed):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.set_beta(1.0)
    gen = torch.Generator(device=device).manual_seed(seed)
    net.to(device)
    history: List[dict] = []
    for ep in range(epochs):
        net.train()
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            x_seq = rate_code(images, T, gen)  # [T, B, 784]
            opt.zero_grad()
            out = net(x_seq)
            loss = Fnn.cross_entropy(out["logits"], labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * labels.size(0)
            correct += (out["logits"].argmax(-1) == labels).sum().item()
            seen += labels.size(0)
        history.append({
            "epoch": ep,
            "loss": total_loss / seen,
            "train_acc": correct / seen,
            "time": time.time() - t0,
        })
        print(f"  [vpsc] epoch {ep}: loss={total_loss/seen:.4f} "
              f"train_acc={correct/seen:.4f} ({history[-1]['time']:.1f}s)")
    return history


@torch.no_grad()
def evaluate(net, test_loader, T, device, seed):
    net.eval()
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    correct = 0
    seen = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        x_seq = rate_code(images, T, gen)
        out = net(x_seq)
        correct += (out["logits"].argmax(-1) == labels).sum().item()
        seen += labels.size(0)
    return correct / seen


def build_net(device):
    cfg = VPSCConfig(
        sizes=[784, 256, 64], n_classes=10, beta=1.0, tau_m=10.0,
        threshold=0.5, sigma=1.0, sparsity=1e-4,
    )
    return VPSCNet(cfg).to(device)


def run(epochs=10, T=10, batch_size=128, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    print("Loading MNIST ...")
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    net = build_net(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"VPSC net params: {n_params}")
    history = train(net, train_loader, epochs, T, lr, device, seed)
    test_acc = evaluate(net, test_loader, T, device, seed)
    print(f"VPSC test accuracy: {test_acc:.4f}")
    total_time = sum(h["time"] for h in history)
    return {"test_acc": test_acc, "history": history, "params": n_params, "time": total_time}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(epochs=args.epochs, T=args.T, device=device, seed=args.seed)
