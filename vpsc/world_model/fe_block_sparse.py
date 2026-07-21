"""FE-1: Free-Energy-Gated Block-Sparse Temporal SNN.

Block-level hard routing over E time-scale experts (distinct decay bands, like
d4), with **lazy decay** for inactive experts and a cheap free-energy predictor
q_φ as the routing signal. Two F definitions (user-requested, both tested):

  FE-1b (F-inertial): F̂_e = ‖z_t − d_e z_{t-1}‖²  (inertia-continuation, no W_down)
  FE-1a (F-PC):       F̂_e = ‖x − W_down x_upper‖²/σ²  (reconstructive PC F, Theorem 2)

Key property enabling real compute savings: an inactive expert's trace evolves
as z_{t+Δ} = d^Δ z_t (no writes), so it is NOT recomputed each block — on
re-activation it is recovered by one elementwise scalar mul d^Δ. Within an
active block the affine scan runs dense (O(log B) depth, fused-kernel
compatible). Routing is per-block (B∈{32,64}), not per-token, to avoid
scatter/gather overhead.

Training: dense warm-up (all experts run, hard-routed output via straight-
through) supervises q_φ with the true per-block F; soft→hard anneal. Inference:
only the selected expert(s) run — real conditional compute.

This module implements FE-1b (F-inertial) as the base; FE-1a (F-PC) subclasses
to add W_down and override the F definition. Both share the block-router and
lazy-decay machinery.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
from torch import Tensor

from .cores import CoreOutput, E3GatedTraceScanCore, E3ScanState, TemporalCore
from .scaling_variants import _MoEGatedTraceCore  # reuse expert construction


class _FEPredictor(nn.Module):
    """Cheap q_φ: block statistics → per-expert predicted free energy F̂.

    Inputs per block: pooled input stats (mean, std, change-norm) + previous
    block's min F̂ (router state). Output: F̂_e for each expert (lower = better
    fit = route here).
    """

    def __init__(self, n_stats: int, n_experts: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats + 1, hidden), nn.GELU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, stats: Tensor, prev_min_f: Tensor) -> Tensor:
        # stats: [B, n_stats], prev_min_f: [B, 1] → F̂ [B, n_experts]
        return self.net(torch.cat([stats, prev_min_f], dim=-1))


class BlockSparseFECore(TemporalCore):
    """FE-1b: block-sparse temporal SNN, F-inertial routing, lazy decay.

    - E experts, distinct decay bands (short/mid/long), each an E3 gated-trace core.
    - Block size B: routing decision per block, not per token.
    - q_φ predicts F̂_e per block; argmin selects active expert(s).
    - Inactive experts: state lazily decayed as d^(Δ·B) on re-activation (no scan).
    - Training: dense (all experts scan) + straight-through hard route; q_φ
      supervised by true inertial F. Inference: only selected expert scans.
    """

    def __init__(self, input_dim: int, hidden_dim: int, *,
                 n_experts: int = 3, state_dim: Optional[int] = None,
                 block_size: int = 32, topk: int = 1, fused: bool = False,
                 fe_threshold: float = 0.0, f_supervise: bool = True):
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        self.n_experts = int(n_experts)
        self.block_size = int(block_size)
        self.topk = int(topk)
        self.fused = bool(fused)
        self.f_supervise = bool(f_supervise)
        self.fe_threshold = float(fe_threshold)  # ΔF̂ below this → run top-2
        # build experts with distinct decay bands (like d4)
        import torch as _t
        lo = _t.linspace(0.50, 0.80, n_experts).tolist()
        hi = _t.linspace(0.80, 0.99, n_experts).tolist()
        if fused:
            kw = dict(execution_mode="scan", scan_math_mode="cuda_fused",
                      eligibility_backward_mode="reverse_adjoint")
        else:
            kw = dict(execution_mode="scan", scan_math_mode="hillis_steele")
        self.experts = nn.ModuleList([
            E3GatedTraceScanCore(input_dim, hidden_dim, state_dim=state_dim,
                                 min_decay=float(lo[e]), max_decay=float(hi[e]),
                                 min_initial_decay=float(lo[e]) + 0.02,
                                 max_initial_decay=float(hi[e]) - 0.02, **kw)
            for e in range(n_experts)
        ])
        self.output_norm = nn.LayerNorm(4 * (state_dim or hidden_dim))
        self.output_projection = nn.Linear(4 * (state_dim or hidden_dim), hidden_dim)
        # q_φ: block stats = [mean, std, change-norm] (3) → F̂ per expert
        self.router = _FEPredictor(n_stats=3, n_experts=n_experts)
        # per-expert decay buffer (for lazy decay); filled at first forward
        self._last_usage: Optional[Tensor] = None
        self._last_router_f: Optional[Tensor] = None  # for TB logging
        self._true_f: Optional[Tensor] = None

    # -- free energy definition (FE-1b: inertial), log-domain for stability --
    def block_free_energy(self, z_block: Tensor, expert_idx: int) -> Tensor:
        """Per-sample **log** inertial F for a block's trace.

        log F = logsumexp(2 log|z_t − d z_{t-1}|)  — numerically stable: stays
        in log-domain, gradient is softmax weights (bounded ≤1, no explosion).
        argmin log F = argmin F, so routing is unchanged. q_φ predicts log F̂
        directly; supervision uses log F (never exp back). Replaces the square-
        sum (which overflowed → NaN) the way log-add replaces product-overflow.
        z_block: [B_bt, B, hidden]. Returns [B_bt].
        """
        d = self._expert_decay(expert_idx)  # [state]
        diff = z_block[:, 1:] - d * z_block[:, :-1]  # [B_bt, B-1, hidden]
        # log|diff| + log|diff| = 2 log|diff|; clamp log input away from 0
        log_sq = 2.0 * torch.log(diff.abs().clamp_min(1e-8))  # [B_bt, B-1, hidden]
        # logsumexp over (B-1, hidden) → [B_bt]
        return torch.logsumexp(log_sq.reshape(log_sq.shape[0], -1), dim=-1)

    def _expert_decay(self, e: int) -> Tensor:
        """Get the per-state decay vector for expert e (E+I averaged)."""
        dl = self.experts[e].decay_logits  # [2, state]
        return torch.sigmoid(dl).mean(dim=0)

    def _expert_decay_ei(self, e: int) -> Tuple[Tensor, Tensor]:
        """Per-state (E, I) decay vectors for expert e (for lazy decay)."""
        dl = torch.sigmoid(self.experts[e].decay_logits)  # [2, state]
        return dl[0], dl[1]

    def _block_stats(self, x: Tensor) -> Tensor:
        """Pool [B, T, D] into per-block stats [n_blocks, B_bt, 3]."""
        B, T, D = x.shape
        bs = self.block_size
        nblk = T // bs
        xb = x[:, :nblk * bs].reshape(B, nblk, bs, D)
        mean = xb.mean(dim=[2, 3])                      # [B, nblk]
        std = xb.std(dim=[2, 3])                        # [B, nblk]
        diff = torch.zeros_like(mean)
        diff[:, 1:] = ((xb[:, 1:] - xb[:, :-1]) ** 2).mean(dim=[2, 3]).sqrt()
        return torch.stack([mean, std, diff], dim=-1)   # [B, nblk, 3]

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        return tuple(exp.initial_state(batch_size, device=device, dtype=dtype)
                     for exp in self.experts)

    def forward(self, x: Tensor, state=None, *, detach_state: bool = False,
                sparse_inference: bool = False, streaming: bool = False) -> CoreOutput:
        """x: [B, T, input_dim]. Block-routed forward.

        sparse_inference=False (training/warm-up): all experts scan, hard-route
          output via straight-through; q_φ supervised by true F.
        sparse_inference=True (eval): only selected expert(s) scan per block;
          inactive experts lazily decayed.
        streaming=False (default): each block re-initializes expert state to zero
          (no cross-block carry). The E3 [0,1] trace invariant holds only for
          z_0=0; cross-block carry with non-zero initial state can push traces
          >1 and trip E3's _validate_state. Streaming=True carries state across
          blocks (with clamp) — enables true lazy decay but may hit the bound
          check on long sequences; use with care.
        """
        B, T, _ = x.shape
        bs = self.block_size
        nblk = T // bs
        if state is None:
            state = self.initial_state(B, device=x.device, dtype=x.dtype)

        stats = self._block_stats(x)  # [B, nblk, 3]
        prev_min_f = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        f_preds: List[Tensor] = []
        out_seq = torch.zeros(B, T, self.output_dim, device=x.device, dtype=x.dtype)
        usage = torch.zeros(self.n_experts, device=x.device)
        true_fs: List[Tensor] = []
        cur_states = list(state)

        for k in range(nblk):
            x_blk = x[:, k*bs:(k+1)*bs]
            st_k = stats[:, k]
            f_hat = self.router(st_k, prev_min_f)
            f_preds.append(f_hat)
            if self.topk >= self.n_experts:
                sel = torch.ones(B, self.n_experts, device=x.device)
            else:
                _, idx = f_hat.topk(self.topk, dim=-1, largest=False)
                sel = torch.zeros_like(f_hat).scatter_(-1, idx, 1.0)
                if self.n_experts > 1 and self.topk == 1:
                    fs = f_hat.sort(dim=-1).values
                    delta = fs[:, 1] - fs[:, 0]
                    ambig = (delta < self.fe_threshold).float().unsqueeze(-1)
                    _, idx2 = f_hat.topk(2, dim=-1, largest=False)
                    sel2 = torch.zeros_like(f_hat).scatter_(-1, idx2[:, :2], 1.0)
                    sel = sel * (1 - ambig) + sel2 * ambig

            block_outs: List[Tensor] = []
            block_traces: List[Tensor] = []
            new_states = []
            for e in range(self.n_experts):
                if sparse_inference and not sel[:, e].any():
                    # lazy decay: skip scan, decay state by d_e^bs (per E/I)
                    if streaming:
                        d_e, d_i = self._expert_decay_ei(e)
                        fe, fi = d_e ** bs, d_i ** bs
                        es = cur_states[e].layers[0]
                        ne = E3ScanState(layers=(type(es)(
                            excitatory=(fe * es.excitatory).clamp(0.0, 1.0),
                            inhibitory=(fi * es.inhibitory).clamp(0.0, 1.0)),))
                    else:
                        ne = self.experts[e].initial_state(B, device=x.device, dtype=x.dtype)
                    new_states.append(ne)
                    block_outs.append(None)
                    block_traces.append(None)
                    continue
                # block state: zero (non-streaming) or carried+clamped (streaming)
                if streaming:
                    blk_state = cur_states[e]
                else:
                    blk_state = None  # E3 re-initializes to zero
                out = self.experts[e](x_blk, blk_state, detach_state=detach_state)
                if streaming:
                    es = out.state.layers[0]
                    clamped = E3ScanState(layers=(type(es)(
                        excitatory=es.excitatory.clamp(0.0, 1.0),
                        inhibitory=es.inhibitory.clamp(0.0, 1.0)),))
                    new_states.append(clamped)
                else:
                    new_states.append(out.state)
                block_outs.append(out.sequence)
                block_traces.append(out.sequence)
            cur_states = new_states

            # compute true F (inertial) per expert for supervision
            if self.f_supervise:
                blk_fs = []
                for e in range(self.n_experts):
                    if block_traces[e] is None:
                        # lazy/inactive expert: large finite log-F (not inf, to
                        # keep log-domain supervision finite & differentiable)
                        blk_fs.append(torch.full((B,), 50.0, device=x.device))
                    else:
                        blk_fs.append(self.block_free_energy(block_traces[e], e))
                true_fs.append(torch.stack(blk_fs, dim=-1))  # [B, n_experts]

            # combine: hard route (straight-through for training)
            stacked = torch.stack([o if o is not None else
                                   torch.zeros(B, bs, self.output_dim, device=x.device)
                                   for o in block_outs], dim=-1)  # [B, bs, out, E]
            # use sel for this block: [B, E] → expand
            w = sel.unsqueeze(1).unsqueeze(2)  # [B,1,1,E]
            if not sparse_inference:
                # straight-through: forward hard, backward soft (softmax of -F̂)
                soft = torch.softmax(-f_hat, dim=-1)
                w = w.detach() + soft.unsqueeze(1).unsqueeze(2) - soft.unsqueeze(1).unsqueeze(2).detach()
            combined = (stacked * w).sum(dim=-1)  # [B, bs, out]
            out_seq[:, k*bs:(k+1)*bs] = combined
            usage += sel.sum(dim=0)
            prev_min_f = f_hat.min(dim=-1, keepdim=True).values.detach()

        usage = usage / (B * nblk)
        self._last_usage = usage
        self._last_router_f = torch.stack(f_preds, dim=1)  # [B, nblk, E]
        if true_fs:
            self._true_f = torch.stack(true_fs, dim=1)  # [B, nblk, E]

        next_state = tuple(cur_states)
        return CoreOutput(sequence=out_seq, state=next_state)


class BlockSparseFEPCCore(BlockSparseFECore):
    """FE-1a: F-PC routing — reconstructive free energy with W_down.

    Adds a top-down generative weight W_down so F_block = ‖x − W_down x_upper‖²/σ²
    (the VPSCNet.layer_free_energy structure, reattached to the gated-trace line).
    This is the Theorem-2 F; it also activates W_down (fixing the S1 capacity hole
    where W_down had grad=None in the byte-LM).
    """

    def __init__(self, input_dim, hidden_dim, *, n_experts=3, state_dim=None,
                 block_size=32, topk=1, fused=False, fe_threshold=0.0,
                 sigma: float = 1.0):
        super().__init__(input_dim, hidden_dim, n_experts=n_experts,
                         state_dim=state_dim, block_size=block_size, topk=topk,
                         fused=fused, fe_threshold=fe_threshold, f_supervise=True)
        d = hidden_dim  # F operates on out.sequence (post projection)
        # W_down: predict block state from pooled context (top-down generative)
        self.W_down = nn.Linear(d, d, bias=False)
        nn.init.kaiming_uniform_(self.W_down.weight, a=5 ** 0.5)
        self.sigma = float(sigma)

    def block_free_energy(self, z_block: Tensor, expert_idx: int) -> Tensor:
        """PC F (log-domain): logsumexp(2 log|z − W_down(pooled z)|) − 2 log σ.

        log-domain stable; argmin preserved; gradient bounded. See base class
        docstring for the log-F rationale.
        """
        pooled = z_block.mean(dim=1)  # [B_bt, hidden]
        mu = self.W_down(pooled).unsqueeze(1)  # [B_bt, 1, hidden]
        err = z_block - mu
        log_sq = 2.0 * torch.log(err.abs().clamp_min(1e-8))
        log_f = torch.logsumexp(log_sq.reshape(log_sq.shape[0], -1), dim=-1)
        return log_f - 2.0 * math.log(self.sigma)


__all__ = ["BlockSparseFECore", "BlockSparseFEPCCore"]
