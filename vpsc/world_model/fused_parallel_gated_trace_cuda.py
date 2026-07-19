"""Lazy CUDA binding for SG25F block-parallel affine trace scans."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "vpsc/cuda"
SOURCES = (
    SOURCE_DIR / "sg25f_parallel_gated_trace.cpp",
    SOURCE_DIR / "sg25f_parallel_gated_trace_kernel.cu",
)
_EXTENSION: Any = None
_LOAD_AUDIT: Optional[Dict[str, Any]] = None


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCES:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_extension(*, verbose: bool = False) -> Tuple[Any, Dict[str, Any]]:
    global _EXTENSION, _LOAD_AUDIT
    if _EXTENSION is not None and _LOAD_AUDIT is not None:
        return _EXTENSION, dict(_LOAD_AUDIT)
    from torch.utils.cpp_extension import load

    fingerprint = _source_fingerprint()
    name = f"vpsc_sg25f_{fingerprint[:12]}"
    default_build = Path("/root/autodl-tmp/torch_extensions/sg25f")
    build_directory = Path(
        os.environ.get("VPSC_SG25F_BUILD_DIR", str(default_build))
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    executable_directory = Path(sys.executable).parent
    if (executable_directory / "ninja").is_file():
        os.environ["PATH"] = (
            str(executable_directory)
            + os.pathsep
            + os.environ.get("PATH", "")
        )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    started = time.perf_counter_ns()
    _EXTENSION = load(
        name=name,
        sources=[str(path) for path in SOURCES],
        extra_cflags=("-O3",),
        extra_cuda_cflags=("-O3", "-lineinfo"),
        build_directory=str(build_directory),
        verbose=verbose,
        with_cuda=True,
    )
    elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
    _LOAD_AUDIT = {
        "extension_name": name,
        "source_sha256": fingerprint,
        "sources": tuple(str(path) for path in SOURCES),
        "build_directory": str(build_directory),
        "load_compile_seconds": elapsed_seconds,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    return _EXTENSION, dict(_LOAD_AUDIT)


class _FusedParallelGatedTrace(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        packed_drives: torch.Tensor,
        query_indices: torch.Tensor,
        decays: torch.Tensor,
        initial_e: torch.Tensor,
        initial_i: torch.Tensor,
        spike_threshold: float,
        surrogate_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not packed_drives.is_cuda or packed_drives.dtype != torch.float32:
            raise TypeError("SG25F parallel trace requires CUDA float32 drives")
        if packed_drives.ndim != 4 or packed_drives.shape[1] != 4:
            raise ValueError("packed drives must be [batch,4,state,time]")
        if query_indices.ndim != 2 or query_indices.dtype != torch.long:
            raise TypeError("SG25F query_indices must be int64 [batch,query]")
        extension, _ = load_extension()
        raw, final_e, final_i, previous, writes = extension.forward(
            packed_drives.contiguous(),
            query_indices.contiguous(),
            decays.contiguous(),
            initial_e.contiguous(),
            initial_i.contiguous(),
            float(spike_threshold),
            float(surrogate_scale),
        )
        ctx.save_for_backward(
            packed_drives, query_indices, decays, previous, writes, raw
        )
        ctx.spike_threshold = float(spike_threshold)
        ctx.surrogate_scale = float(surrogate_scale)
        return raw, final_e, final_i

    @staticmethod
    def backward(
        ctx: Any,
        grad_raw: Optional[torch.Tensor],
        grad_final_e: Optional[torch.Tensor],
        grad_final_i: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
    ]:
        packed_drives, query_indices, decays, previous, writes, raw = (
            ctx.saved_tensors
        )
        if grad_raw is None:
            grad_raw = torch.zeros_like(raw)
        if grad_final_e is None:
            grad_final_e = torch.zeros_like(packed_drives[:, 0, :, 0])
        if grad_final_i is None:
            grad_final_i = torch.zeros_like(packed_drives[:, 0, :, 0])
        extension, _ = load_extension()
        grad_packed, grad_decays, grad_initial_e, grad_initial_i = (
            extension.backward(
                grad_raw.contiguous(),
                grad_final_e.contiguous(),
                grad_final_i.contiguous(),
                packed_drives.contiguous(),
                query_indices.contiguous(),
                decays.contiguous(),
                previous.contiguous(),
                writes.contiguous(),
                raw.contiguous(),
                ctx.spike_threshold,
                ctx.surrogate_scale,
            )
        )
        return (
            grad_packed,
            None,
            grad_decays,
            grad_initial_e,
            grad_initial_i,
            None,
            None,
        )


def fused_parallel_gated_trace(
    drives: torch.Tensor,
    query_indices: torch.Tensor,
    decays: torch.Tensor,
    initial_e: torch.Tensor,
    initial_i: torch.Tensor,
    *,
    spike_threshold: float,
    surrogate_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if drives.ndim != 3:
        raise ValueError("drives must be [batch,time,4*state]")
    state_dim = int(decays.shape[1])
    if int(drives.shape[2]) != 4 * state_dim:
        raise ValueError("drive width mismatch")
    packed = (
        drives.view(drives.shape[0], drives.shape[1], 4, state_dim)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    return _FusedParallelGatedTrace.apply(
        packed,
        query_indices,
        decays,
        initial_e,
        initial_i,
        spike_threshold,
        surrogate_scale,
    )


def extension_audit() -> Optional[Dict[str, Any]]:
    return None if _LOAD_AUDIT is None else dict(_LOAD_AUDIT)
