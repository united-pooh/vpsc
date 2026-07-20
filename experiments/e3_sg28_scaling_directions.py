#!/usr/bin/env python3
"""SG28: VPSC 规模化六方向（D1-D6）smoke 驱动。

在 main 分支的 gated-trace SNN（E3GatedTraceScanCore，portable scan）之上实现六个
规模化方向，与本机 LSTM/Transformer 基线在 tiny wikitext char-LM 上对比 NLL。

设备：cuda.is_available()→CUDA，elif MPS→MPS，else CPU（vpsc.world_model.devices）。
本机（macOS）落 MPS。TextWorld 主线任务（edit/room/sensitivity）需 ROCm + textworld 包，
本机不可跑；正式 SNN-vs-Transformer 评估留给 ROCm。本脚本只做 smoke：机制正确性 + tiny 3 架构 NLL。

方向：
  base  E3 SNN 基线
  d1    脉冲路由空间 MoE（SpikeRoutedMoECore）
  d2    D1 + 门控无 BPTT（路由器冻结，专家在固定门下训练）
  d3    非自回归块 MTP（BlockMTPCausalLM）
  d4    时间尺度 MoE（TemporalMoEGatedTraceCore，专家 decay 不同）
  d5    半群 exp(jQ) 块解码（SemigroupBlockMTPCausalLM）
  d6    动作路由专家（ActionRoutedMoECore，合成动作）
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vpsc.world_model.cores import (  # noqa: E402
    E3GatedTraceScanCore, StatefulLSTMCore, CausalTransformerCore,
)
from vpsc.world_model.lm import CausalLanguageModel  # noqa: E402
from vpsc.world_model.factory import FairLMConfig, build_model_suite  # noqa: E402
from vpsc.world_model.scaling_variants import (  # noqa: E402
    SpikeRoutedMoECore, TemporalMoEGatedTraceCore,
    BlockMTPCausalLM, SemigroupBlockMTPCausalLM, ActionRoutedMoECore,
)
from vpsc.world_model.devices import choose_device, device_label, synchronize  # noqa: E402
from vpsc.world_model import wikitext as wt  # noqa: E402


# ---------------------------------------------------------------------------
# 数据：tiny wikitext char-LM
# ---------------------------------------------------------------------------

@dataclass
class CharCorpus:
    chars: List[str]
    stoi: Dict[str, int]
    itos: Dict[int, str]
    train_ids: Tensor
    valid_ids: Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.chars)


def load_char_corpus(cache_dir: Path, train_chars: int, valid_chars: int,
                     seq_len: int) -> CharCorpus:
    paths = wt.prepare_wikitext2(cache_dir, download=True)
    train_text = paths.split("train").read_text(encoding="utf-8")[:train_chars]
    valid_text = paths.split("valid").read_text(encoding="utf-8")[:valid_chars]
    chars = sorted(set(train_text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    train_ids = torch.tensor([stoi[c] for c in train_text if c in stoi], dtype=torch.long)
    valid_ids = torch.tensor([stoi[c] for c in valid_text if c in stoi], dtype=torch.long)
    return CharCorpus(chars, stoi, itos, train_ids, valid_ids)


def make_sequences(ids: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
    """Non-overlapping next-token chunks. Returns (inputs, targets) [N, seq_len]."""
    n = (len(ids) - 1) // seq_len
    if n <= 0:
        raise ValueError("corpus too short for seq_len")
    inputs = ids[: n * seq_len].view(n, seq_len)
    targets = ids[1: n * seq_len + 1].view(n, seq_len)
    return inputs, targets


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------

D_MODEL = 32
STATE_DIM = 32
N_EXPERTS = 3
MTP_DEPTH = 4
N_ACTIONS = 4


@dataclass(frozen=True)
class ModelSpec:
    """Configurable model dimensions for parameter-matching and scale sweeps."""
    d_model: int = D_MODEL
    state_dim: int = STATE_DIM
    n_experts: int = N_EXPERTS
    mtp_depth: int = MTP_DEPTH
    n_actions: int = N_ACTIONS


class FusedSNNCausalLM(CausalLanguageModel):
    """SNN LM that routes through the fused CUDA gated-trace kernel.

    Overrides forward to call ``forward_multi_query_eligibility`` with dense
    query indices (every position queried) so the SG27B/SG25C fused kernel is
    used instead of the portable Hillis-Steele scan.  Only valid when the core
    is a bare ``E3GatedTraceScanCore`` built with
    ``scan_math_mode='cuda_fused', eligibility_backward_mode='reverse_adjoint'``.
    """

    def forward(self, input_ids, state=None, *, labels=None, targets=None,
                detach_state=False):
        T = input_ids.shape[1]
        embedded = self.input_dropout(self.embedding(input_ids))
        query = torch.arange(T, device=input_ids.device, dtype=torch.long)
        core_out = self.core.forward_multi_query_eligibility(
            embedded, query, state, detach_state=detach_state,
        )
        hidden = self.output_norm(self.output_dropout(core_out.sequence))
        logits = self.lm_head(hidden)
        loss = None
        target_count = None
        loss_mode = "none"
        if labels is not None:
            loss, target_count = self._loss(logits[:, :-1], labels[:, 1:])
            loss_mode = "causal_shift"
        elif targets is not None:
            loss, target_count = self._loss(logits, targets)
            loss_mode = "aligned"
        from vpsc.world_model.lm import CausalLMOutput
        return CausalLMOutput(logits=logits, state=core_out.state,
                              hidden_states=hidden, loss=loss,
                              target_count=target_count, loss_mode=loss_mode)


def build_snn_base(vocab: int, spec: ModelSpec, fused: bool = False) -> CausalLanguageModel:
    d = spec.d_model
    if fused:
        core = E3GatedTraceScanCore(
            d, d, state_dim=spec.state_dim,
            scan_math_mode="cuda_fused",
            eligibility_backward_mode="reverse_adjoint",
        )
        return FusedSNNCausalLM(vocab, core)
    return CausalLanguageModel(vocab, E3GatedTraceScanCore(d, d, state_dim=spec.state_dim))


def build_variant(variant: str, vocab: int, spec: ModelSpec, fused: bool = False) -> CausalLanguageModel:
    d = spec.d_model
    if variant == "base":
        return build_snn_base(vocab, spec, fused=fused)
    if variant == "d1":
        return CausalLanguageModel(vocab, SpikeRoutedMoECore(
            d, d, n_experts=spec.n_experts, state_dim=spec.state_dim, fused=fused))
    if variant == "d2":
        # same model as d1; router frozen in the training loop (no BPTT for gate)
        return CausalLanguageModel(vocab, SpikeRoutedMoECore(
            d, d, n_experts=spec.n_experts, state_dim=spec.state_dim, fused=fused))
    if variant == "d3":
        return BlockMTPCausalLM(vocab, E3GatedTraceScanCore(d, d, state_dim=spec.state_dim),
                                mtp_depth=spec.mtp_depth)
    if variant == "d4":
        return CausalLanguageModel(vocab, TemporalMoEGatedTraceCore(
            d, d, n_experts=spec.n_experts, state_dim=spec.state_dim, fused=fused))
    if variant == "d5":
        return SemigroupBlockMTPCausalLM(vocab, E3GatedTraceScanCore(d, d, state_dim=spec.state_dim),
                                         mtp_depth=spec.mtp_depth)
    if variant == "d6":
        return CausalLanguageModel(vocab, ActionRoutedMoECore(
            d, d, n_experts=spec.n_experts, state_dim=spec.state_dim,
            n_actions=spec.n_actions, fused=fused))
    raise ValueError(variant)


def build_ann_suite(vocab: int, spec: ModelSpec) -> Tuple[CausalLanguageModel, CausalLanguageModel]:
    suite = build_model_suite(FairLMConfig(vocab_size=vocab, d_model=spec.d_model,
                                           auto_match_parameters=False))
    return suite.lstm, suite.transformer


def count_params_of(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_state_dim(variant: str, vocab: int, target_params: int, spec: ModelSpec,
                    fused: bool = False, tol: float = 0.05) -> ModelSpec:
    """Binary-search state_dim so the SNN variant's param count ≈ target.

    For variants where state_dim does not change param count much (d3/d5 heads
    dominate), returns spec unchanged.  Reports the matched count.
    """
    best = spec
    best_diff = abs(count_params_of(build_variant(variant, vocab, spec, fused=fused)) - target_params)
    lo, hi = 8, 2048
    for _ in range(20):
        mid = (lo + hi) // 2
        cand = ModelSpec(spec.d_model, mid, spec.n_experts, spec.mtp_depth, spec.n_actions)
        n = count_params_of(build_variant(variant, vocab, cand, fused=fused))
        diff = n - target_params
        if abs(diff) < best_diff:
            best, best_diff = cand, abs(diff)
        if abs(diff) <= target_params * tol:
            return cand
        if diff < 0:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def is_mtp(model: CausalLanguageModel) -> bool:
    return isinstance(model, (BlockMTPCausalLM, SemigroupBlockMTPCausalLM))


def is_action(model: CausalLanguageModel) -> bool:
    return isinstance(model.core, ActionRoutedMoECore)


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------

def synthesize_actions(targets: Tensor) -> Tensor:
    """D6 has no TextWorld; derive a *label-independent* synthetic action per token.

    Must NOT depend on ``targets`` (that leaks the label — see SG28 smoke 2026-07-20
    where ``targets % N_ACTIONS`` gave a spurious 2.529 bpc).  Use the position index
    mod N_ACTIONS instead: a structural, label-free signal that lets the action-routed
    experts specialise by position without seeing the answer.
    """
    T = targets.shape[1]
    positions = torch.arange(T, device=targets.device)
    return positions.unsqueeze(0).expand_as(targets) % N_ACTIONS


def compute_loss(model: CausalLanguageModel, inputs: Tensor, targets: Tensor,
                 mtp_weight: float = 0.5) -> Tuple[Tensor, Dict[str, float]]:
    if is_action(model):
        model.core.set_actions(synthesize_actions(targets))
    out = model(inputs, targets=targets)  # aligned targets (next-token at each pos)
    loss = out.loss
    diag = {"ce": float(out.loss.detach())}
    if is_mtp(model):
        extra, per_head = model.mtp_loss(out.hidden_states, targets)
        loss = loss + mtp_weight * extra
        diag["mtp_extra"] = float(extra.detach())
        diag["mtp_per_head"] = per_head
    return loss, diag


def run_epoch(model, inputs_t, targets_t, batch_size, device, *,
              optimizer=None, grad_clip=1.0, mtp_weight=0.5, seed=0):
    training = optimizer is not None
    model.train(training)
    n = inputs_t.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if training else torch.arange(n)
    total_loss = 0.0
    total_tokens = 0
    usage_accum: Optional[Tensor] = None
    usage_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    t_start = time.perf_counter()
    from tqdm.auto import tqdm
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(range(0, n, batch_size), leave=False,
                    desc=f"{'train' if training else 'eval'} {getattr(model, 'core', model).__class__.__name__ if hasattr(model,'core') else model.__class__.__name__}")
        for start in pbar:
            sel = idx[start:start + batch_size]
            inp = inputs_t[sel].to(device)
            tgt = targets_t[sel].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, diag = compute_loss(model, inp, tgt, mtp_weight=mtp_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += float(diag["ce"]) * inp.numel()
            total_tokens += inp.numel()
            u = getattr(model.core, "_last_usage", None) if hasattr(model, "core") else None
            if u is not None:
                usage_accum = u.clone() if usage_accum is None else usage_accum + u
                usage_count += 1
            if total_tokens % (batch_size * 20) < batch_size:
                pbar.set_postfix(loss=f"{total_loss/max(1,total_tokens):.3f}")
    synchronize(device)
    elapsed = time.perf_counter() - t_start
    mean_ce = total_loss / max(1, total_tokens)
    usage = (usage_accum / usage_count).tolist() if (usage_accum is not None and usage_count) else None
    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MiB
    return {"ce": mean_ce, "nll": mean_ce, "bpc": mean_ce / math.log(2.0),
            "usage": usage, "elapsed_s": elapsed,
            "tokens_per_s": total_tokens / max(elapsed, 1e-9),
            "peak_mem_mib": peak_mem}


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"registered": total}


def train_model(name: str, model, train_inp, train_tgt, valid_inp, valid_tgt,
                device, epochs, batch_size, lr, freeze_router=False,
                tb_writer=None, tb_prefix: str = ""):
    if freeze_router:
        # D2: gate gets no BPTT — freeze all router-like params in the core
        for pn, p in model.core.named_parameters():
            if "router" in pn:
                p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    npar = count_params(model)
    history = []
    t0 = time.perf_counter()
    train_peak_mem = 0.0
    train_wall = 0.0
    train_tokens_total = 0
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_inp, train_tgt, batch_size, device,
                       optimizer=optimizer, seed=ep)
        va = run_epoch(model, valid_inp, valid_tgt, batch_size, device, seed=0)
        train_peak_mem = max(train_peak_mem, tr["peak_mem_mib"])
        train_wall += tr["elapsed_s"]
        train_tokens_total += train_inp.numel()
        history.append({"epoch": ep, "train_bpc": tr["bpc"], "valid_bpc": va["bpc"],
                        "train_tok_s": tr["tokens_per_s"], "train_peak_mem_mib": tr["peak_mem_mib"]})
        print(f"  {name} ep={ep}/{epochs} train_bpc={tr['bpc']:.3f} valid_bpc={va['bpc']:.3f}"
              f" tok/s={tr['tokens_per_s']:.0f}"
              + (f" mem={tr['peak_mem_mib']:.1f}MiB" if tr['peak_mem_mib'] else "")
              + (f" usage={[f'{u:.2f}' for u in tr['usage']]}" if tr['usage'] else ""))
        if tb_writer is not None:
            pfx = f"{tb_prefix}/" if tb_prefix else ""
            tb_writer.add_scalar(f"{pfx}train_bpc", tr["bpc"], ep)
            tb_writer.add_scalar(f"{pfx}valid_bpc", va["bpc"], ep)
            tb_writer.add_scalar(f"{pfx}train_tok_s", tr["tokens_per_s"], ep)
            if tr["peak_mem_mib"]:
                tb_writer.add_scalar(f"{pfx}peak_mem_mib", tr["peak_mem_mib"], ep)
            tb_writer.flush()
    elapsed = time.perf_counter() - t0
    final = run_epoch(model, valid_inp, valid_tgt, batch_size, device, seed=0)
    if tb_writer is not None:
        pfx = f"{tb_prefix}/" if tb_prefix else ""
        tb_writer.add_scalar(f"{pfx}final_valid_bpc", final["bpc"], epochs)
        tb_writer.add_scalar(f"{pfx}params", npar["registered"], 0)
        tb_writer.flush()
    return {"name": name, "params": npar, "valid_bpc": final["bpc"],
            "usage": final["usage"], "history": history, "elapsed_s": elapsed,
            "train_tokens_per_s": train_tokens_total / max(train_wall, 1e-9),
            "peak_mem_mib": train_peak_mem,
            "train_wall_s": train_wall}


# ---------------------------------------------------------------------------
# 主
# ---------------------------------------------------------------------------

@dataclass
class SmokeConfig:
    variant: str
    train_chars: int = 131072
    valid_chars: int = 32768
    seq_len: int = 64
    batch_size: int = 32
    epochs: int = 2
    lr: float = 1e-3
    seeds: Tuple[int, ...] = (0,)


def run_variant(variant: str, cfg: SmokeConfig, cache_dir: Path, device: torch.device,
                spec: ModelSpec, fused: bool = False, match_params: bool = False,
                ann_baselines: Optional[Dict[str, Dict]] = None,
                ann_spec: Optional[ModelSpec] = None) -> Dict[str, Any]:
    label = f"{variant}" + ("+fused" if fused else "")
    print(f"\n===== variant={label} device={device_label(device)} =====")
    corpus = load_char_corpus(cache_dir, cfg.train_chars, cfg.valid_chars, cfg.seq_len)
    train_inp, train_tgt = make_sequences(corpus.train_ids, cfg.seq_len)
    valid_inp, valid_tgt = make_sequences(corpus.valid_ids, cfg.seq_len)
    print(f"vocab={corpus.vocab_size} train_seq={train_inp.shape[0]} valid_seq={valid_inp.shape[0]}")

    # ANN baselines (built from ann_spec, trained once per spec, reused across variants)
    if ann_baselines is None:
        ann_baselines = {}
        ann_s = ann_spec if ann_spec is not None else spec
        for ann_name in ("lstm", "transformer"):
            runs = []
            for sd in cfg.seeds:
                torch.manual_seed(sd)
                m = build_ann_suite(corpus.vocab_size, ann_s)[0 if ann_name == "lstm" else 1].to(device)
                runs.append(train_model(f"{ann_name}", m, train_inp, train_tgt, valid_inp, valid_tgt,
                                        device, cfg.epochs, cfg.batch_size, cfg.lr))
            ann_baselines[ann_name] = _aggregate_seeds(runs)

    # SNN variant — optional param-match to ANN params
    snn_spec = spec
    target = ann_baselines["lstm"]["params"]["registered"]
    if match_params:
        snn_spec = match_state_dim(variant, corpus.vocab_size, target, spec, fused=fused)
        snn_params = count_params_of(build_variant(variant, corpus.vocab_size, snn_spec, fused=fused))
        print(f"  [match-params] target(ann)={target} snn_state_dim={snn_spec.state_dim} snn_params={snn_params}")
    snn_runs = []
    for sd in cfg.seeds:
        torch.manual_seed(sd)
        snn = build_variant(variant, corpus.vocab_size, snn_spec, fused=fused).to(device)
        snn_runs.append(train_model(label, snn, train_inp, train_tgt, valid_inp, valid_tgt,
                                    device, cfg.epochs, cfg.batch_size, cfg.lr,
                                    freeze_router=(variant == "d2")))
    results = {"snn_" + label: _aggregate_seeds(snn_runs),
               "lstm": ann_baselines["lstm"], "transformer": ann_baselines["transformer"]}
    return {"variant": label, "vocab": corpus.vocab_size,
            "train_seqs": int(train_inp.shape[0]), "valid_seqs": int(valid_inp.shape[0]),
            "spec": asdict(snn_spec), "fused": fused, "match_params": match_params,
            "results": results}


def _aggregate_seeds(runs: List[Dict]) -> Dict[str, Any]:
    """Mean±std over seeds for the key metrics; keep last run's history/usage for inspection."""
    if len(runs) == 1:
        return runs[0]
    bpcs = [r["valid_bpc"] for r in runs]
    tps = [r["train_tokens_per_s"] for r in runs]
    mems = [r["peak_mem_mib"] for r in runs]
    return {
        "name": runs[0]["name"], "params": runs[0]["params"],
        "valid_bpc": float(np.mean(bpcs)), "valid_bpc_std": float(np.std(bpcs)),
        "train_tokens_per_s": float(np.mean(tps)), "train_tokens_per_s_std": float(np.std(tps)),
        "peak_mem_mib": float(np.mean(mems)),
        "per_seed_bpc": bpcs, "usage": runs[-1]["usage"],
        "history": runs[-1]["history"], "elapsed_s": float(np.mean([r["elapsed_s"] for r in runs])),
        "train_wall_s": float(np.mean([r["train_wall_s"] for r in runs])),
    }


# ---------------------------------------------------------------------------
# scale-sweep（硬件瓶颈扩展扫描）
# ---------------------------------------------------------------------------

def run_scale_sweep(cfg: SmokeConfig, cache_dir: Path, device: torch.device,
                    widths: Tuple[int, ...], batches: Tuple[int, ...],
                    archs: Tuple[str, ...], fused: bool, out: Path) -> Dict[str, Any]:
    """Sweep (arch, d_model, batch); record BPC/tok/s/peak-mem/OOM up to T4 limit."""
    corpus = load_char_corpus(cache_dir, cfg.train_chars, cfg.valid_chars, cfg.seq_len)
    train_inp, train_tgt = make_sequences(corpus.train_ids, cfg.seq_len)
    valid_inp, valid_tgt = make_sequences(corpus.valid_ids, cfg.seq_len)
    rows = []
    oom_skip: Dict[str, set] = {a: set() for a in archs}  # arch -> set of widths already OOM
    for arch in archs:
        for w in widths:
            for bs in batches:
                # skip if this arch already OOM'd at this width (larger batch won't help)
                if w in oom_skip[arch]:
                    continue
                spec = ModelSpec(d_model=w, state_dim=w, n_experts=3, mtp_depth=4, n_actions=4)
                try:
                    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
                    torch.manual_seed(0)
                    if arch in ("lstm", "transformer"):
                        m = build_ann_suite(corpus.vocab_size, spec)[0 if arch == "lstm" else 1].to(device)
                        r = train_model(arch, m, train_inp, train_tgt, valid_inp, valid_tgt,
                                        device, cfg.epochs, bs, cfg.lr)
                    else:
                        m = build_variant(arch, corpus.vocab_size, spec, fused=fused and arch in ("base", "d1", "d4", "d6")).to(device)
                        r = train_model(arch, m, train_inp, train_tgt, valid_inp, valid_tgt,
                                        device, cfg.epochs, bs, cfg.lr)
                    peak = r["peak_mem_mib"]
                    print(f"  {arch:11s} d={w:4d} bs={bs:3d}: bpc={r['valid_bpc']:.3f} "
                          f"tok/s={r['train_tokens_per_s']:.0f} mem={peak:.0f}MiB params={r['params']['registered']}")
                    rows.append({"arch": arch, "d_model": w, "batch": bs,
                                 "valid_bpc": r["valid_bpc"], "tok_s": r["train_tokens_per_s"],
                                 "peak_mem_mib": peak, "params": r["params"]["registered"],
                                 "oom": False})
                    if device.type == "cuda" and peak >= 0.9 * 15 * 1024:
                        oom_skip[arch].add(w)
                        print(f"    -> 触瓶颈 ({peak:.0f}MiB ≥ 13.5GB)，跳过更大 batch/width")
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  {arch:11s} d={w:4d} bs={bs:3d}: OOM")
                    rows.append({"arch": arch, "d_model": w, "batch": bs, "oom": True})
                    oom_skip[arch].add(w)
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  {arch:11s} d={w:4d} bs={bs:3d}: OOM (runtime)")
                        rows.append({"arch": arch, "d_model": w, "batch": bs, "oom": True})
                        oom_skip[arch].add(w)
                        torch.cuda.empty_cache() if device.type == "cuda" else None
                    else:
                        raise
    payload = {"experiment": "SG28 scale-sweep", "device": str(device),
               "widths": list(widths), "batches": list(batches), "archs": list(archs),
               "fused": fused, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="all",
                    choices=("base", "d1", "d2", "d3", "d4", "d5", "d6", "all"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-chars", type=int, default=131072)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--valid-chars", type=int, default=262144)
    ap.add_argument("--fused", action="store_true", help="SNN 用 fused CUDA gated-trace kernel")
    ap.add_argument("--match-params", action="store_true", help="二分 state_dim 使 SNN 参数 ≈ ANN")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--d-model", type=int, default=D_MODEL)
    ap.add_argument("--state-dim", type=int, default=STATE_DIM)
    ap.add_argument("--n-experts", type=int, default=N_EXPERTS)
    ap.add_argument("--mtp-depth", type=int, default=MTP_DEPTH)
    ap.add_argument("--n-actions", type=int, default=N_ACTIONS)
    ap.add_argument("--scale-sweep", action="store_true", help="硬件瓶颈扩展扫描")
    ap.add_argument("--sweep-widths", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    ap.add_argument("--sweep-batches", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--sweep-archs", nargs="+",
                    default=["base", "d1", "d4", "lstm", "transformer"])
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "e3_sg28_cache")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "e3_scan" / "e3_sg28_smoke.json")
    args = ap.parse_args()
    if not args.smoke and not args.formal:
        args.smoke = True
    device = choose_device(args.device)
    print(f"device: {device_label(device)}  mode={'smoke' if args.smoke else 'formal'}  "
          f"fused={args.fused} match={args.match_params} seeds={args.seeds}")

    spec = ModelSpec(d_model=args.d_model, state_dim=args.state_dim, n_experts=args.n_experts,
                     mtp_depth=args.mtp_depth, n_actions=args.n_actions)
    cfg = SmokeConfig(variant=args.variant, epochs=args.epochs,
                      train_chars=args.train_chars, valid_chars=args.valid_chars,
                      batch_size=args.batch_size, seeds=tuple(args.seeds))

    if args.scale_sweep:
        payload = run_scale_sweep(cfg, args.cache_dir, device,
                                  tuple(args.sweep_widths), tuple(args.sweep_batches),
                                  tuple(args.sweep_archs), args.fused, args.out)
        print(f"\n结果：{args.out}")
        _print_sweep_summary(payload)
        return

    variants = ("base", "d1", "d2", "d3", "d4", "d5", "d6") if args.variant == "all" else (args.variant,)
    all_results = []
    ann_baselines: Optional[Dict[str, Dict]] = None
    for v in variants:
        vr = run_variant(v, cfg, args.cache_dir, device, spec=spec, fused=args.fused,
                         match_params=args.match_params, ann_baselines=ann_baselines, ann_spec=spec)
        if ann_baselines is None:
            ann_baselines = {"lstm": vr["results"]["lstm"], "transformer": vr["results"]["transformer"]}
        all_results.append(vr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "SG28 scaling directions", "device": str(device),
               "config": asdict(cfg), "spec": asdict(spec), "variants": all_results}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果：{args.out}")
    print("\n=== 摘要 (valid BPC mean±std / tok/s / 显存) ===")
    for vr in all_results:
        v = vr["variant"]
        for name, r in vr["results"].items():
            u = (f" usage={[f'{x:.2f}' for x in r['usage']]}" if r['usage'] else "")
            mem = f" mem={r['peak_mem_mib']:.0f}MiB" if r.get('peak_mem_mib') else ""
            std = f"±{r['valid_bpc_std']:.3f}" if "valid_bpc_std" in r else ""
            print(f"  {v:9s} {name:14s}: bpc={r['valid_bpc']:.3f}{std} "
                  f"tok/s={r.get('train_tokens_per_s',0):.0f} params={r['params']['registered']}{mem}{u}")


def _print_sweep_summary(payload: Dict) -> None:
    rows = [r for r in payload["rows"] if not r.get("oom")]
    print("\n=== scale-sweep 各架构最大可行规模 ===")
    by_arch: Dict[str, List[Dict]] = {}
    for r in rows:
        by_arch.setdefault(r["arch"], []).append(r)
    for arch, rs in by_arch.items():
        rs.sort(key=lambda r: r["d_model"])
        biggest = rs[-1]
        print(f"  {arch:11s}: max d_model={biggest['d_model']} bs={biggest['batch']} "
              f"-> bpc={biggest['valid_bpc']:.3f} tok/s={biggest['tok_s']:.0f} "
              f"mem={biggest['peak_mem_mib']:.0f}MiB params={biggest['params']}")


if __name__ == "__main__":
    main()
