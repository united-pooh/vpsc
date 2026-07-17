"""MNIST data + rate coding for VPSC.

VPSC is a temporal spiking net; MNIST is static. We rate-code each image into a
T-step sequence by Bernoulli sampling at the pixel intensity (a cheap, deterministic-
enough proxy for Poisson rate coding). Output shape [T, B, 784].

ANN/CNN baselines consume the static image [B, 1, 28, 28] directly.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_mnist(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    os.makedirs(DATA_DIR, exist_ok=True)
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def rate_code(images: torch.Tensor, T: int, generator: torch.Generator) -> torch.Tensor:
    """images: [B, 1, 28, 28] in [0,1]. Returns [T, B, 784] Bernoulli spikes
    at rate = pixel intensity. A fresh sample per epoch keeps it stochastic."""
    B = images.shape[0]
    flat = images.view(B, -1)  # [B, 784]
    # [T, B, 784]
    spikes = (torch.rand(T, B, flat.shape[1], generator=generator) < flat.unsqueeze(0)).to(flat.dtype)
    return spikes


def images_flat(images: torch.Tensor) -> torch.Tensor:
    """[B,1,28,28] -> [B, 784] for the MLP baseline."""
    return images.view(images.shape[0], -1)
