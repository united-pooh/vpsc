"""SG28 scaling-direction model variants.

Six research directions (D1-D6) built on the main-line gated-trace SNN
(``E3GatedTraceScanCore``) and ``CausalLanguageModel``.  These are smoke-scale
mechanism implementations: they verify each direction's forward/backward and
give a 3-architecture NLL signal on a portable text task.  Formal TextWorld
evaluation (edit/room/sensitivity, SG27B Triton fused backend) is deferred to
the ROCm box; the code is portable via ``vpsc.world_model.devices.choose_device``.

Conventions:
* MoE wrappers (D1/D4/D6) use soft differentiable gating for smoke and report
  effective expert usage as a sparsity diagnostic.  Hard top-k with a
  straight-through estimator is the formal-path option (flag ``topk``).
* Core wrappers preserve the ``TemporalCore`` interface so they drop into
  ``CausalLanguageModel`` unchanged.
* MTP head variants (D3/D5) expose ``mtp_depth`` and ``mtp_logits(hidden)`` so
  the training harness can add a non-autoregressive block-MTP loss without
  overriding ``CausalLanguageModel.forward``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cores import (
    CoreOutput,
    E3GatedTraceScanCore,
    E3ScanState,
    TemporalCore,
)
from .lm import CausalLanguageModel


# ---------------------------------------------------------------------------
# D1 — spike-routed spatial mixture-of-experts (core wrapper)
# ---------------------------------------------------------------------------


class _MoEGatedTraceCore(TemporalCore[Tuple[E3ScanState, ...]]):
    """Shared base: E expert E3 gated-trace cores, per-timestep soft gate.

    Subclasses define ``_gate_logits(x)`` (the routing signal).  State is the
    tuple of per-expert E3 states.  All experts compute (dense) for smoke;
    ``topk>0`` enables hard top-k selection with a straight-through gradient.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        n_experts: int = 3,
        state_dim: Optional[int] = None,
        topk: int = 0,
        expert_kwargs: Optional[dict] = None,
        fused: bool = False,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=hidden_dim)
        self.n_experts = int(n_experts)
        self.topk = int(topk)
        self.fused = bool(fused)
        if self.fused:
            kw = dict(execution_mode="scan", scan_math_mode="cuda_fused",
                      eligibility_backward_mode="reverse_adjoint")
        else:
            kw = dict(execution_mode="scan", scan_math_mode="hillis_steele")
        if expert_kwargs:
            kw.update(expert_kwargs)
        self.experts = nn.ModuleList(
            [E3GatedTraceScanCore(input_dim, hidden_dim, state_dim=state_dim, **kw)
             for _ in range(self.n_experts)]
        )
        # Each subclass owns its router (spike / change / action signal) and
        # implements _gate_logits; no unused base router (which would leave a
        # grad-None parameter).
        self._last_usage: Optional[Tensor] = None  # diagnostic [n_experts]

    def _gate_logits(self, x: Tensor) -> Tensor:
        """[B, T, n_experts] routing logits from input [B, T, input_dim]."""
        raise NotImplementedError

    def _expert_forward(self, exp: E3GatedTraceScanCore, x: Tensor, st, detach_state: bool):
        """Per-expert forward; fused path uses the dense multi-query kernel."""
        if self.fused:
            T = x.shape[1]
            query = torch.arange(T, device=x.device, dtype=torch.long)
            return exp.forward_multi_query_eligibility(x, query, st, detach_state=detach_state)
        return exp(x, st, detach_state=detach_state)

    def _combine(self, gate_logits: Tensor, expert_sequences: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Soft (or top-k straight-through) combine of expert sequences.

        expert_sequences: list of E tensors [B, T, output_dim].
        Returns (combined [B, T, output_dim], usage [n_experts]).
        """
        stacked = torch.stack(expert_sequences, dim=-1)  # [B, T, out, E]
        if self.topk > 0 and self.topk < self.n_experts:
            topk_val, topk_idx = gate_logits.topk(self.topk, dim=-1)
            soft = torch.softmax(topk_val, dim=-1)          # [B,T,topk]
            mask = torch.zeros_like(gate_logits)
            mask.scatter_(-1, topk_idx, soft)               # [B,T,E] sparse weights
            # straight-through: forward = hard mask, backward = full softmax
            full = torch.softmax(gate_logits, dim=-1)
            weights = mask.detach() + full - full.detach()
        else:
            weights = torch.softmax(gate_logits, dim=-1)    # [B, T, E]
        combined = (stacked * weights.unsqueeze(2)).sum(dim=-1)  # [B, T, out]
        usage = weights.detach().mean(dim=(0, 1))           # [E]
        return combined, usage

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        return tuple(exp.initial_state(batch_size, device=device, dtype=dtype)
                     for exp in self.experts)

    def forward(self, x: Tensor, state=None, *, detach_state: bool = False):
        if state is None:
            state = self.initial_state(x.shape[0], device=x.device, dtype=x.dtype)
        gate_logits = self._gate_logits(x)  # [B, T, E]
        sequences: List[Tensor] = []
        next_states = []
        for e, exp in enumerate(self.experts):
            out = self._expert_forward(exp, x, state[e], detach_state)
            sequences.append(out.sequence)
            next_states.append(out.state)
        combined, usage = self._combine(gate_logits, sequences)
        self._last_usage = usage
        return CoreOutput(sequence=combined, state=tuple(next_states))


class SpikeRoutedMoECore(_MoEGatedTraceCore):
    """D1: route by a spike-activity signal derived from the input.

    The router input is the binary event rate (fraction of active input
    dimensions per timestep), making the gate a function of spike activity as
    the direction specifies.  A linear map turns that scalar-per-step activity
    into expert logits.
    """

    def __init__(self, input_dim, hidden_dim, *, n_experts=3, state_dim=None, topk=0,
                 expert_kwargs=None, spike_threshold: float = 0.0, fused: bool = False):
        super().__init__(input_dim, hidden_dim, n_experts=n_experts, state_dim=state_dim,
                         topk=topk, expert_kwargs=expert_kwargs, fused=fused)
        # Router on a 2-vector per timestep: [mean activity, std activity]
        self.activity_router = nn.Linear(2, self.n_experts)
        self.spike_threshold = float(spike_threshold)

    def _gate_logits(self, x: Tensor) -> Tensor:
        active = (x > self.spike_threshold).to(x.dtype)  # [B,T,D]
        mean_act = active.mean(dim=-1, keepdim=True)     # [B,T,1]
        std_act = active.std(dim=-1, keepdim=True)       # [B,T,1]
        feats = torch.cat([mean_act, std_act], dim=-1)   # [B,T,2]
        return self.activity_router(feats)


# ---------------------------------------------------------------------------
# D4 — temporal-timescale MoE (experts differ in decay, routed by change)
# ---------------------------------------------------------------------------


class TemporalMoEGatedTraceCore(_MoEGatedTraceCore):
    """D4: experts specialise by decay horizon (short/medium/long), routed by
    input-change magnitude (transition signal).

    Each expert e gets a distinct [min_decay, max_decay] band.  The router
    sees the per-timestep input-change norm, so transition-heavy steps favour
    short-decay experts and steady context favours long-decay ones.  Decay is
    per-state inside each expert's scan (the SG27B primitive), so this keeps
    the scan depth O(log T).
    """

    def __init__(self, input_dim, hidden_dim, *, n_experts=3, state_dim=None, topk=0,
                 decay_bands: Optional[List[Tuple[float, float]]] = None, fused: bool = False):
        if decay_bands is None:
            # spread decay bands across [0.5, 0.99]: short -> long
            lo = torch.linspace(0.50, 0.80, n_experts).tolist()
            hi = torch.linspace(0.80, 0.99, n_experts).tolist()
            decay_bands = [(float(lo[e]), float(hi[e])) for e in range(n_experts)]
        if len(decay_bands) != n_experts:
            raise ValueError("decay_bands must have n_experts entries")
        expert_kwargs_list = []
        for (dmin, dmax) in decay_bands:
            expert_kwargs_list.append(dict(
                min_decay=dmin, max_decay=dmax,
                min_initial_decay=max(dmin + 0.02, dmin), max_initial_decay=min(dmax - 0.02, dmax),
            ))
        # _MoEGatedTraceCore applies one expert_kwargs to all; build experts directly.
        super().__init__(input_dim, hidden_dim, n_experts=n_experts, state_dim=state_dim,
                         topk=topk, expert_kwargs=None, fused=fused)
        if fused:
            kw = dict(execution_mode="scan", scan_math_mode="cuda_fused",
                      eligibility_backward_mode="reverse_adjoint")
        else:
            kw = dict(execution_mode="scan", scan_math_mode="hillis_steele")
        self.experts = nn.ModuleList([
            E3GatedTraceScanCore(input_dim, hidden_dim, state_dim=state_dim, **kw, **expert_kwargs_list[e])
            for e in range(n_experts)
        ])
        # Router on [change_norm, level] per timestep
        self.change_router = nn.Linear(2, self.n_experts)

    def _gate_logits(self, x: Tensor) -> Tensor:
        # per-timestep input-change norm (transition signal)
        diff = torch.zeros_like(x)
        diff[:, 1:] = x[:, 1:] - x[:, :-1]
        change = diff.norm(dim=-1, keepdim=True)   # [B,T,1]
        level = x.norm(dim=-1, keepdim=True)       # [B,T,1]
        feats = torch.cat([change, level], dim=-1)  # [B,T,2]
        return self.change_router(feats)


# ---------------------------------------------------------------------------
# D3 — non-autoregressive block multi-token prediction (head variant)
# ---------------------------------------------------------------------------


class BlockMTPCausalLM(CausalLanguageModel):
    """D3: adds k-1 extra LM heads predicting tokens t+2..t+k from hidden[t].

    Non-autoregressive (no self-roll-in), fixing SG26C's exposure bias.  The
    main ``lm_head`` still predicts t+1.  The harness computes the combined
    loss via :meth:`mtp_loss`.
    """

    def __init__(self, vocab_size, core, *, mtp_depth: int = 4, dropout=0.0,
                 padding_idx=0, ignore_index=None, tie_weights=False, head_bias=True):
        super().__init__(vocab_size, core, dropout=dropout, padding_idx=padding_idx,
                         ignore_index=ignore_index, tie_weights=tie_weights, head_bias=head_bias)
        if mtp_depth < 2:
            raise ValueError("mtp_depth must be >= 2 (main head covers j=1)")
        self.mtp_depth = int(mtp_depth)
        # extra heads for j = 2..k
        self.mtp_heads = nn.ModuleList([
            nn.Linear(core.output_dim, vocab_size, bias=head_bias) for _ in range(mtp_depth - 1)
        ])

    def mtp_logits(self, hidden_states: Tensor) -> Tensor:
        """Return extra-head logits [k-1, B, T, vocab] for j=2..k."""
        return torch.stack([head(hidden_states) for head in self.mtp_heads], dim=0)

    def mtp_loss(self, hidden_states: Tensor, targets: Tensor) -> Tuple[Tensor, List[float]]:
        """Non-autoregressive block-MTP loss for j=2..k.

        ``targets`` is position-aligned (targets[:, t] = token at t).  Head j
        at position t predicts targets[:, t+j].  Returns (mean_loss, per_head_nll).
        """
        extra = self.mtp_logits(hidden_states)  # [k-1, B, T, V]
        T = targets.shape[1]
        total = hidden_states.new_zeros(())
        per_head: List[float] = []
        for idx, j in enumerate(range(2, self.mtp_depth + 1)):
            if T - j <= 0:
                continue
            logits_j = extra[idx][:, : T - j, :]      # head at t=0..T-j-1
            tgt_j = targets[:, j: T]                   # token at t+j
            loss_j = F.cross_entropy(logits_j.reshape(-1, logits_j.shape[-1]),
                                     tgt_j.reshape(-1), ignore_index=self.ignore_index)
            total = total + loss_j
            per_head.append(float(loss_j.detach()))
        n = max(1, len(per_head))
        return total / n, per_head


# ---------------------------------------------------------------------------
# D5 — semigroup exp(jQ) block decoder (head variant on D3)
# ---------------------------------------------------------------------------


class SemigroupBlockMTPCausalLM(BlockMTPCausalLM):
    """D5: constrain the k-1 extra heads to one generator Q.

    ``head_j(h) = lm_head(h @ expm((j-1) Q))`` for j=2..k, so a single
    [d,d] generator Q produces all extra heads (big saving when vocab >> d).
    ``matrix_exp`` is computed on CPU and moved back (MPS lacks the op; CUDA
    and CPU support it natively), preserving autograd.
    """

    def __init__(self, vocab_size, core, *, mtp_depth: int = 4, dropout=0.0,
                 padding_idx=0, ignore_index=None, tie_weights=False, head_bias=True):
        super().__init__(vocab_size, core, mtp_depth=mtp_depth, dropout=dropout,
                         padding_idx=padding_idx, ignore_index=ignore_index,
                         tie_weights=tie_weights, head_bias=head_bias)
        d = core.output_dim
        self.Q = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.Q.weight.mul_(0.01)  # small init: expm((j-1)Q) ≈ I
        # remove the free per-head linears from D3 (replaced by Q + shared lm_head)
        del self.mtp_heads
        self.mtp_heads = nn.ModuleList()  # keep attribute for param-count sanity

    def _expm(self, power: int) -> Tensor:
        w = (power * self.Q.weight).cpu()
        e = torch.linalg.matrix_exp(w)
        return e.to(self.Q.weight.device)

    def mtp_logits(self, hidden_states: Tensor) -> Tensor:
        heads = [self.lm_head(hidden_states @ self._expm(j - 1))
                 for j in range(2, self.mtp_depth + 1)]
        return torch.stack(heads, dim=0)

    def mtp_loss(self, hidden_states: Tensor, targets: Tensor) -> Tuple[Tensor, List[float]]:
        extra = self.mtp_logits(hidden_states)
        T = targets.shape[1]
        total = hidden_states.new_zeros(())
        per_head: List[float] = []
        for idx, j in enumerate(range(2, self.mtp_depth + 1)):
            if T - j <= 0:
                continue
            logits_j = extra[idx][:, : T - j, :]
            tgt_j = targets[:, j: T]
            loss_j = F.cross_entropy(logits_j.reshape(-1, logits_j.shape[-1]),
                                     tgt_j.reshape(-1), ignore_index=self.ignore_index)
            total = total + loss_j
            per_head.append(float(loss_j.detach()))
        n = max(1, len(per_head))
        return total / n, per_head


# ---------------------------------------------------------------------------
# D6 — action-routed experts (core wrapper + factorised head hook)
# ---------------------------------------------------------------------------


class ActionRoutedMoECore(_MoEGatedTraceCore):
    """D6: experts routed by an action signal embedded into the input stream.

    The harness prepends an action embedding to ``x`` (last dim) so the router
    can separate experts by action type (move/look/take/...).  This is the
    SG28A factorised-objective mechanism at smoke scale; the full room/exits
    factorised head needs TextWorld and is deferred.  ``last_action_usage``
    reports per-expert usage for diagnostics.
    """

    def __init__(self, input_dim, hidden_dim, *, n_experts=3, state_dim=None, topk=0,
                 expert_kwargs=None, n_actions: int = 4, action_embed_dim: int = 8,
                 fused: bool = False):
        # core.input_dim = token dim (for CausalLM embedding); experts see token+action.
        super().__init__(input_dim, hidden_dim, n_experts=n_experts,
                         state_dim=state_dim, topk=topk, expert_kwargs=expert_kwargs,
                         fused=fused)
        self.n_actions = int(n_actions)
        self.action_embed_dim = int(action_embed_dim)
        # rebuild experts to take the augmented (token+action) input, honoring fused
        if fused:
            kw = dict(execution_mode="scan", scan_math_mode="cuda_fused",
                      eligibility_backward_mode="reverse_adjoint")
        else:
            kw = dict(execution_mode="scan", scan_math_mode="hillis_steele")
        if expert_kwargs:
            kw.update(expert_kwargs)
        aug_dim = input_dim + action_embed_dim
        self.experts = nn.ModuleList([
            E3GatedTraceScanCore(aug_dim, hidden_dim, state_dim=state_dim, **kw)
            for _ in range(n_experts)
        ])
        self.action_embedding = nn.Embedding(n_actions, action_embed_dim)
        self.action_router = nn.Linear(action_embed_dim, self.n_experts)
        # Side channel: the harness sets this before CausalLanguageModel.forward
        # (which does not forward an `actions` kwarg).  None => action-free context.
        self._actions: Optional[Tensor] = None

    def set_actions(self, actions: Optional[Tensor]) -> None:
        """Set per-timestep action ids [B,T] for the next forward; None clears."""
        self._actions = actions

    def forward(self, x: Tensor, state=None, *, detach_state: bool = False):
        """x: [B,T,input_dim]; actions come from :meth:`set_actions`."""
        B, T, _ = x.shape
        actions = self._actions
        if actions is None:
            act_emb = x.new_zeros(B, T, self.action_embed_dim)
        else:
            act_emb = self.action_embedding(actions)  # [B,T,action_embed_dim]
        x_aug = torch.cat([x, act_emb], dim=-1)  # [B,T,input_dim+ae]
        # route by action embedding
        gate_logits = self.action_router(act_emb)  # [B,T,E]
        if state is None:
            state = self.initial_state(B, device=x.device, dtype=x.dtype)
        sequences: List[Tensor] = []
        next_states = []
        for e, exp in enumerate(self.experts):
            out = self._expert_forward(exp, x_aug, state[e], detach_state)
            sequences.append(out.sequence)
            next_states.append(out.state)
        combined, usage = self._combine(gate_logits, sequences)
        self._last_usage = usage
        return CoreOutput(sequence=combined, state=tuple(next_states))


__all__ = [
    "SpikeRoutedMoECore",
    "TemporalMoEGatedTraceCore",
    "BlockMTPCausalLM",
    "SemigroupBlockMTPCausalLM",
    "ActionRoutedMoECore",
]
