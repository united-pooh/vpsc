"""Fixation experiment: continuously present the SAME image, like a human staring.

Motivation (user): rate-coding an image into T=10 steps and discarding it is
unlike perception — a human staring at a picture keeps receiving it, and the
neural response evolves. Here we hold the image fixed and drive the VPSC net for
many timesteps, recording the readout at EACH step, to see how accuracy and
confidence evolve over prolonged fixation.

Two sub-modes:
  - 'static':   the exact same input vector every step (deterministic drive).
  - 'poisson':  the image intensity is the firing RATE; each step re-samples
                Bernoulli spikes (photoreceptor noise), but the underlying image
                is constant. This is biologically more plausible.

Outputs: accuracy-vs-timestep curve, mean confidence-vs-timestep curve, and the
"first-correct time" distribution (how many ms of fixation until the answer
stabilizes on the right class).
"""

import argparse
import json
import os
import sys
from typing import List

import torch
import torch.nn.functional as Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vpsc import VPSCConfig, VPSCNet  # noqa: E402
from lab.mnist.data import get_mnist  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_net(device):
    cfg = VPSCConfig(sizes=[784, 256, 64], n_classes=10, beta=1.0, tau_m=10.0,
                    threshold=0.5, sigma=1.0, sparsity=1e-4)
    return VPSCNet(cfg).to(device)


def train(net, train_loader, epochs, lr, device, mode, T_train):
    """Train with the SAME drive mode used at fixation eval, so train/eval
    distributions match. T_train timesteps per sample."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.set_beta(1.0)
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(epochs):
        net.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            flat = images.view(images.size(0), -1)  # [B, 784]
            if mode == "static":
                x_seq = flat.unsqueeze(0).expand(T_train, -1, -1)  # [T,B,784]
            else:  # poisson, re-sampled per step
                rnd = (torch.rand(T_train, *flat.shape, device=device) if device != "cpu"
                       else torch.rand(T_train, *flat.shape, generator=gen))
                x_seq = (rnd < flat.unsqueeze(0)).to(flat.dtype)
            opt.zero_grad()
            out = net(x_seq)
            loss = Fnn.cross_entropy(out["logits"], labels)
            loss.backward()
            opt.step()


@torch.no_grad()
def fixation_eval(net, test_loader, T_max, mode, device, seed):
    """Hold each image fixed for T_max steps; the net's membrane state is carried
    across steps (integrated), and we read out the classification at EVERY step.
    This is the genuine 'prolonged fixation' test: does integrating over time
    improve accuracy / confidence?"""
    net.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    acc_t = torch.zeros(T_max)
    conf_t = torch.zeros(T_max)
    n = 0
    first_correct: List[int] = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)
        flat = images.view(B, -1)
        if mode == "static":
            x_seq = flat.unsqueeze(0).expand(T_max, -1, -1)
        else:
            rnd = (torch.rand(T_max, *flat.shape, device=device) if device != "cpu"
                   else torch.rand(T_max, *flat.shape, generator=gen))
            x_seq = (rnd < flat.unsqueeze(0)).to(flat.dtype)

        # Single forward over the whole fixation window: state is integrated
        # across timesteps (LIF membrane accumulates), readout at every step.
        out = net(x_seq, return_all_logits=True)
        all_logits = out["all_logits"]  # [T, B, C]
        probs = Fnn.softmax(all_logits, dim=-1)
        conf, pred = probs.max(dim=-1)   # [T, B]

        correct = (pred == labels.unsqueeze(0)).to(torch.float64)  # [T, B]
        acc_t += correct.sum(dim=1).cpu()
        conf_t += conf.sum(dim=1).cpu()
        n += B

        for b in range(B):
            ch = correct[:, b]
            wrong = (ch == 0).nonzero(as_tuple=True)[0]
            last_wrong = int(wrong[-1].item()) if wrong.numel() > 0 else -1
            if ch[-1] > 0:
                first_correct.append(last_wrong + 1)
            else:
                first_correct.append(-1)

    acc_t /= n
    conf_t /= n
    return acc_t.tolist(), conf_t.tolist(), first_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--T_max", type=int, default=60, help="fixation timesteps")
    ap.add_argument("--mode", choices=["static", "poisson"], default="poisson")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"Loading MNIST ... (device={device})")
    train_loader, test_loader = get_mnist(batch_size=256)
    net = build_net(device)
    print(f"Training VPSC {args.epochs} epochs (mode={args.mode}, T_train=12) ...")
    train(net, train_loader, args.epochs, 1e-3, device, args.mode, T_train=12)

    print(f"Fixation eval: T_max={args.T_max}, mode={args.mode} ...")
    acc_t, conf_t, first_correct = fixation_eval(net, test_loader, args.T_max, args.mode, device, args.seed)

    print(f"\nacc at t=0:   {acc_t[0]:.4f}")
    print(f"acc at t={args.T_max//2}: {acc_t[args.T_max//2]:.4f}")
    print(f"acc at t={args.T_max-1}: {acc_t[args.T_max-1]:.4f}")
    print(f"max acc over fixation: {max(acc_t):.4f} at t={acc_t.index(max(acc_t))}")
    fc = [x for x in first_correct if x >= 0]
    if fc:
        import statistics
        print(f"stable-correct fixation time: mean={statistics.mean(fc):.1f} steps, "
              f"median={statistics.median(fc):.1f}")
    never = sum(1 for x in first_correct if x < 0)
    print(f"never stable-correct: {never}/{len(first_correct)}")

    out = {"mode": args.mode, "T_max": args.T_max,
           "acc_t": acc_t, "conf_t": conf_t,
           "first_correct": first_correct,
           "max_acc": max(acc_t), "max_acc_t": acc_t.index(max(acc_t))}
    out_path = os.path.join(RESULTS_DIR, f"fixation_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[written {out_path}]")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(acc_t, "-o", ms=3)
        ax[0].set_xlabel("fixation timestep")
        ax[0].set_ylabel("test accuracy")
        ax[0].set_title(f"Accuracy over fixation ({args.mode})\n"
                        f"peak {max(acc_t):.4f} @ t={acc_t.index(max(acc_t))}")
        ax[0].axhline(0.9795, color="r", ls="--", lw=0.8, label="MLP baseline 0.9795")
        ax[0].legend()
        ax[1].plot(conf_t, "-o", ms=3, color="green")
        ax[1].set_xlabel("fixation timestep")
        ax[1].set_ylabel("mean confidence")
        ax[1].set_title(f"Confidence over fixation ({args.mode})")
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"fixation_{args.mode}.png")
        fig.savefig(plot_path, dpi=110)
        print(f"[plot {plot_path}]")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
