"""VPSC — Variational Predictive Spiking Coding.

Research prototype verifying two predictions of the theory:
  * Theorem 2: total free energy F is monotonically non-increasing over training.
  * Theorem 3: task accuracy vs beta peaks near beta_c ~ 1 / spectral_radius(W).
"""

from .neurons import LIFConfig, MeanFieldLIF, spectral_radius
from .network import VPSCConfig, VPSCNet, VPSCLayer, LayerSpec
from .free_energy import BetaAnnealer, VPSCObjective, free_energy_loss, ce_loss

__all__ = [
    "LIFConfig",
    "MeanFieldLIF",
    "spectral_radius",
    "VPSCConfig",
    "VPSCNet",
    "VPSCLayer",
    "LayerSpec",
    "BetaAnnealer",
    "VPSCObjective",
    "free_energy_loss",
    "ce_loss",
]
