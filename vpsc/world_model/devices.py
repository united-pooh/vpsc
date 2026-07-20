"""Adaptive device selection for the SG28 scaling-direction experiments.

Policy (user-confirmed 2026-07-20): ``cuda`` if available, else ``mps`` if
available, else ``cpu``.  This mirrors the main codebase's CUDA/HIP convention
on the ROCm box while still using the Apple-Silicon GPU on macOS.  The frozen
``e3_sg26c`` harness is deliberately CUDA-only and is NOT changed; only the new
SG28 experiments use this helper.
"""

from __future__ import annotations

import torch


def choose_device(requested: str = "auto") -> torch.device:
    """Resolve a device per the cuda→mps→cpu adaptive policy.

    ``requested`` may be ``"auto"`` (apply the policy), or an explicit
    ``"cuda"``/``"mps"``/``"cpu"``.  Explicit requests that are unavailable
    raise ``RuntimeError`` so a silent CPU fallback never masks a missing GPU.
    """
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("requested MPS but MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    # auto: cuda -> mps -> cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index or 0} ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def synchronize(device: torch.device) -> None:
    """Best-effort device synchronize for fair timing; no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


__all__ = ["choose_device", "device_label", "synchronize"]
