"""ANN (MLP) and CNN baselines on MNIST.

Same data, same training budget as VPSC. The MLP matches VPSC's parameter scale
roughly; the CNN is the standard small convnet expected to win on static images.
"""

import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from lab.mnist.data import get_mnist


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = Fnn.relu(self.fc1(x))
        x = Fnn.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = Fnn.max_pool2d(Fnn.relu(self.conv1(x)), 2)
        x = Fnn.max_pool2d(Fnn.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = Fnn.relu(self.fc1(x))
        return self.fc2(x)


def _train(net, train_loader, epochs, lr, device):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
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
            opt.zero_grad()
            logits = net(images)
            loss = Fnn.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            seen += labels.size(0)
        history.append({
            "epoch": ep, "loss": total_loss / seen,
            "train_acc": correct / seen, "time": time.time() - t0,
        })
        name = net.__class__.__name__
        print(f"  [{name}] epoch {ep}: loss={total_loss/seen:.4f} "
              f"train_acc={correct/seen:.4f} ({history[-1]['time']:.1f}s)")
    return history


@torch.no_grad()
def _eval(net, test_loader, device):
    net.eval()
    correct = 0
    seen = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        correct += (logits.argmax(-1) == labels).sum().item()
        seen += labels.size(0)
    return correct / seen


def run_mlp(epochs=10, lr=1e-3, batch_size=128, device="cpu"):
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    net = MLP()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"MLP params: {n_params}")
    history = _train(net, train_loader, epochs, lr, device)
    acc = _eval(net, test_loader, device)
    print(f"MLP test accuracy: {acc:.4f}")
    return {"test_acc": acc, "history": history, "params": n_params,
            "time": sum(h["time"] for h in history)}


def run_cnn(epochs=10, lr=1e-3, batch_size=128, device="cpu"):
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    net = CNN()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"CNN params: {n_params}")
    history = _train(net, train_loader, epochs, lr, device)
    acc = _eval(net, test_loader, device)
    print(f"CNN test accuracy: {acc:.4f}")
    return {"test_acc": acc, "history": history, "params": n_params,
            "time": sum(h["time"] for h in history)}
