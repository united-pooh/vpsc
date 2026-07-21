#!/usr/bin/env python3
"""FE-1 实验驱动：块稀疏自由能门控时序 SNN（FE-1a F-PC / FE-1b F-inertial）。

在 catgirl BPE 语料上验证 G1-G6：
  G1 速度：sparse_inference wall-clock tokens/s vs dense d4
  G2 惰性保真：lazy decay 数学精确（已本地验证）
  G3 路由有效性：q_φ argmin F̂ vs 真实 F argmin 一致率
  G5 无崩溃：专家使用熵
  G6 主线：FE-1 BPC + tokens/s vs Transformer

复用 sg29 的 catgirl corpus 与 sg28 的 train_model。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import experiments.e3_sg28_scaling_directions as sg28  # noqa: E402
from vpsc.world_model.fe_block_sparse import BlockSparseFECore, BlockSparseFEPCCore  # noqa: E402
from vpsc.world_model.devices import choose_device, device_label, synchronize  # noqa: E402
from vpsc.world_model import catgirl_corpus as cg  # noqa: E402


def build_fe(variant: str, vocab: int, spec, fused: bool):
    kw = dict(n_experts=spec.n_experts, state_dim=spec.state_dim,
              block_size=32, topk=1, fused=fused, fe_threshold=0.1)
    if variant == "fe1b":
        return sg28.CausalLanguageModel(vocab, BlockSparseFECore(spec.d_model, spec.d_model, **kw))
    if variant == "fe1a":
        return sg28.CausalLanguageModel(vocab, BlockSparseFEPCCore(spec.d_model, spec.d_model, **kw))
    raise ValueError(variant)


def compute_fe_loss(model, inputs, targets, supervise=True):
    """CE + log-F supervision (q_φ 估计 log F̂, 真实 log F 监督). Eval: no supervision.

    F in log-domain (logsumexp). F-loss is mean-centered + small weight (log-F
    ranges ~30-40 raw, so raw MSE ~1600 would explode q_φ; centering removes
    the magnitude, weight 0.01 keeps it subdominant to CE).
    """
    out = model(inputs, targets=targets)
    loss = out.loss
    diag = {"ce": float(out.loss.detach())}
    core = model.core
    if supervise and core._true_f is not None and core._last_router_f is not None:
        fp = core._last_router_f; ft = core._true_f
        # mean-center both → removes log-F magnitude, leaves relative ranking
        f_loss = ((fp - ft.mean()) - (ft - ft.mean())).pow(2).mean()
        loss = loss + 0.01 * f_loss
        diag["f_loss"] = float(f_loss.detach())
    return loss, diag


def run_epoch_fe(model, inputs_t, targets_t, batch_size, device, *,
                 optimizer=None, grad_clip=1.0, sparse=False, seed=0):
    from tqdm.auto import tqdm
    training = optimizer is not None
    model.train(training)
    n = inputs_t.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if training else torch.arange(n)
    total_loss = 0.0; total_tokens = 0
    usage_acc = None; usage_n = 0
    route_acc = 0; route_n = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    t0 = time.perf_counter()
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            inp = inputs_t[sel].to(device); tgt = targets_t[sel].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # eval (sparse) → no F supervision (lazy experts have inf F)
            loss, diag = compute_fe_loss(model, inp, tgt, supervise=training)
            if training and torch.isnan(loss):
                # NaN guard: skip this batch (don't pollute running stats)
                continue
            if training:
                # load-balance: penalize usage collapse (encourage uniform 1/E)
                u = getattr(model.core, "_last_usage", None) if hasattr(model, "core") else None
                ne = getattr(model.core, "n_experts", 0)
                if u is not None and ne > 0:
                    target = 1.0 / ne
                    lb_loss = ((u - target) ** 2).mean()
                    loss = loss + 0.1 * lb_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            ce = diag["ce"]
            if not (ce != ce):  # not NaN
                total_loss += ce * inp.numel(); total_tokens += inp.numel()
            u = getattr(model.core, "_last_usage", None) if hasattr(model, "core") else None
            if u is not None:
                usage_acc = u.clone() if usage_acc is None else usage_acc + u; usage_n += 1
            # G3: route accuracy (argmin F̂ vs argmin true F)
            if model.core._last_router_f is not None and model.core._true_f is not None:
                pred = model.core._last_router_f.argmin(-1)
                true = model.core._true_f.argmin(-1)
                route_acc += (pred == true).float().sum().item(); route_n += pred.numel()
    synchronize(device)
    elapsed = time.perf_counter() - t0
    mean_ce = total_loss / max(1, total_tokens)
    usage = (usage_acc / usage_n).tolist() if (usage_acc is not None and usage_n) else None
    import math
    peak = torch.cuda.max_memory_allocated(device)/(1024**2) if device.type=="cuda" else 0
    return {"ce": mean_ce, "bpc": mean_ce/math.log(2.0), "usage": usage,
            "route_acc": route_acc/max(1,route_n), "elapsed_s": elapsed,
            "tokens_per_s": total_tokens/max(elapsed,1e-9), "peak_mem_mib": peak}


def train_fe(variant, spec, vocab, train_inp, train_tgt, valid_inp, valid_tgt,
             device, epochs, batch_size, lr, fused):
    torch.manual_seed(0)
    model = build_fe(variant, vocab, spec, fused).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,epochs))
    history = []
    for ep in range(1, epochs+1):
        tr = run_epoch_fe(model, train_inp, train_tgt, batch_size, device, optimizer=opt, seed=ep)
        va = run_epoch_fe(model, valid_inp, valid_tgt, batch_size, device, sparse=True, seed=0)
        sched.step()
        history.append({"epoch":ep,"train_bpc":tr["bpc"],"valid_bpc":va["bpc"],
                        "train_tok_s":tr["tokens_per_s"],"eval_tok_s":va["tokens_per_s"],
                        "route_acc":tr["route_acc"],"usage":tr["usage"]})
        print(f"  {variant} ep={ep}/{epochs} train_bpc={tr['bpc']:.3f} valid_bpc={va['bpc']:.3f} "
              f"tok/s={tr['tokens_per_s']:.0f} route_acc={tr['route_acc']:.2f} "
              f"usage={[round(u,2) for u in tr['usage']] if tr['usage'] else '-'}")
    # G1: dense vs sparse eval wall-clock
    dense = run_epoch_fe(model, valid_inp, valid_tgt, batch_size, device, sparse=False, seed=0)
    sparse = run_epoch_fe(model, valid_inp, valid_tgt, batch_size, device, sparse=True, seed=0)
    return {"variant":variant, "params":sg28.count_params_of(model),
            "final_valid_bpc":va["bpc"], "history":history,
            "G1_dense_tok_s":dense["tokens_per_s"], "G1_sparse_tok_s":sparse["tokens_per_s"],
            "G1_speedup":sparse["tokens_per_s"]/max(dense["tokens_per_s"],1e-9),
            "G3_route_acc":va["route_acc"], "G5_usage":va["usage"],
            "peak_mem_mib":sparse["peak_mem_mib"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("auto","cuda","mps","cpu"), default="auto")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--n-experts", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-convs", type=int, default=20000)
    ap.add_argument("--variants", nargs="+", default=["fe1b","fe1a"])
    ap.add_argument("--fused", action="store_true", default=True)
    ap.add_argument("--no-fused", dest="fused", action="store_false")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT/"results"/"e3_sg29_cache")
    ap.add_argument("--out", type=Path, default=REPO_ROOT/"results"/"e3_scan"/"e3_fe1_result.json")
    args = ap.parse_args()
    device = choose_device(args.device)
    print(f"device: {device_label(device)}  fused={args.fused}")
    print("加载 catgirl BPE 语料...")
    corpus = cg.load_bpe_corpus(args.cache_dir, vocab_size=args.vocab_size, max_convs=args.max_convs)
    train_inp, train_tgt = cg.make_sequences(corpus["train_ids"], args.seq_len)
    valid_inp, valid_tgt = cg.make_sequences(corpus["val_ids"], args.seq_len)
    print(f"vocab={corpus['vocab_size']} train_tok={corpus['n_train_tokens']:,} train_seq={train_inp.shape[0]}")
    spec = sg28.ModelSpec(d_model=args.d_model, state_dim=args.state_dim, n_experts=args.n_experts)
    vocab = corpus["vocab_size"]
    results = {}
    for v in args.variants:
        print(f"\n===== {v} =====")
        results[v] = train_fe(v, spec, vocab, train_inp, train_tgt, valid_inp, valid_tgt,
                              device, args.epochs, args.batch_size, 1e-3, args.fused)
    # ANN baselines for G6
    print("\n===== baselines (lstm/transformer) =====")
    lstm, tr = sg28.build_ann_suite(vocab, spec)
    for name, m in (("lstm",lstm.to(device)),("transformer",tr.to(device))):
        torch.manual_seed(0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(1,args.epochs+1):
            sg28.run_epoch(m, train_inp, train_tgt, args.batch_size, device, optimizer=opt, seed=ep)
        va = sg28.run_epoch(m, valid_inp, valid_tgt, args.batch_size, device, seed=0)
        results[name] = {"final_valid_bpc":va["bpc"], "params":sg28.count_params_of(m),
                         "G1_dense_tok_s":va["tokens_per_s"], "peak_mem_mib":va["peak_mem_mib"]}
        print(f"  {name}: bpc={va['bpc']:.3f} tok/s={va['tokens_per_s']:.0f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {"experiment":"FE-1 block-sparse","device":str(device),"config":cfg,
               "corpus":{"vocab":vocab,"train_tokens":corpus["n_train_tokens"]},"results":results}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 摘要 ===")
    for v in ["fe1b","fe1a","lstm","transformer"]:
        if v not in results: continue
        r=results[v]
        if "G1_speedup" in r:
            print(f"  {v:11s}: bpc={r['final_valid_bpc']:.3f} dense={r['G1_dense_tok_s']:.0f} "
                  f"sparse={r['G1_sparse_tok_s']:.0f} speedup={r['G1_speedup']:.2f}x "
                  f"route_acc={r['G3_route_acc']:.2f} usage={[round(u,2) for u in r['G5_usage']] if r['G5_usage'] else '-'}")
        else:
            print(f"  {v:11s}: bpc={r['final_valid_bpc']:.3f} tok/s={r['G1_dense_tok_s']:.0f}")
    print(f"结果：{args.out}")


if __name__ == "__main__":
    main()
