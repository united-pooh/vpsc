#!/usr/bin/env python3
"""SG30：d4 等参复跑 pilot — 把 LSTM/Transformer 的 d_model 二分搜索到 d4 的参数量，
在等参下比 d4 vs LSTM vs Transformer（SG29 是不等参，d4=2.1x LSTM）。

诚实门槛：d4 须在等参下胜 LSTM（不只胜 Transformer）才能称"胜最强 ANN 基线"。
本 pilot：1 seed × 1 epoch × MPS，验证 setup（参数匹配 + 训练通 + BPC 合理），
非最终结论。全量 3 seeds × 3 ep 留 FE-2H CUDA。
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import experiments.e3_sg28_scaling_directions as sg28  # noqa: E402
from vpsc.world_model.devices import choose_device, device_label  # noqa: E402
from vpsc.world_model import catgirl_corpus as cg  # noqa: E402
from vpsc.world_model.factory import FairLMConfig, build_model_suite  # noqa: E402


def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_ann_d_model(arch: str, vocab: int, target: int, tol: float = 0.05,
                      num_heads: int = 4) -> Tuple[int, int]:
    """Binary-search d_model (multiple of num_heads) so arch's param count ≈ target."""
    lo, hi = 64, 1024
    best_d, best_diff = 128, float("inf")
    for _ in range(20):
        mid = ((lo + hi) // 2 // num_heads) * num_heads  # divisible by num_heads
        if mid < 4:
            mid = num_heads
        cfg = FairLMConfig(vocab_size=vocab, d_model=mid, num_heads=num_heads, auto_match_parameters=False)
        suite = build_model_suite(cfg)
        m = suite.lstm if arch == "lstm" else suite.transformer
        n = count(m)
        d = n - target
        if abs(d) < best_diff:
            best_d, best_diff = mid, abs(d)
        if abs(d) <= target * tol:
            return mid, n
        if d < 0:
            lo = mid + num_heads
        else:
            hi = mid - num_heads
    cfg = FairLMConfig(vocab_size=vocab, d_model=best_d, num_heads=num_heads, auto_match_parameters=False)
    suite = build_model_suite(cfg)
    m = suite.lstm if arch == "lstm" else suite.transformer
    return best_d, count(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    ap.add_argument("--d-model", type=int, default=128, help="d4 的 d_model")
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--n-experts", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-convs", type=int, default=2000, help="pilot 限对话数")
    ap.add_argument("--archs", nargs="+", default=["d4", "lstm", "transformer"])
    ap.add_argument("--fused", action="store_true", default=True)
    ap.add_argument("--no-fused", dest="fused", action="store_false")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "e3_sg29_cache")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "e3_scan" / "e3_sg30_d4_matched.json")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = choose_device(args.device)
    print(f"device: {device_label(device)}  fused={args.fused}  archs={args.archs}")

    print("加载猫娘 BPE 语料 (pilot, max_convs=%d)..." % args.max_convs)
    corpus = cg.load_bpe_corpus(args.cache_dir, vocab_size=args.vocab_size, max_convs=args.max_convs)
    print(f"vocab={corpus['vocab_size']} train_tokens={corpus['n_train_tokens']:,} val_tokens={corpus['n_val_tokens']:,}")
    train_inp, train_tgt = cg.make_sequences(corpus["train_ids"], args.seq_len)
    valid_inp, valid_tgt = cg.make_sequences(corpus["val_ids"], args.seq_len)
    print(f"seq_len={args.seq_len} train_seqs={train_inp.shape[0]} valid_seqs={valid_inp.shape[0]}")

    spec = sg28.ModelSpec(d_model=args.d_model, state_dim=args.state_dim,
                          n_experts=args.n_experts, mtp_depth=4, n_actions=4)
    vocab = corpus["vocab_size"]

    # target = d4 param count
    d4 = sg28.build_variant("d4", vocab, spec, fused=args.fused)
    target = count(d4)
    print(f"\n[d4] d_model={spec.d_model} state={spec.state_dim} experts={spec.n_experts} -> params={target:,} (TARGET)")

    # match LSTM / Transformer d_model to target
    matched = {}
    for arch in ("lstm", "transformer"):
        if arch not in args.archs:
            continue
        dm, actual = match_ann_d_model(arch, vocab, target)
        matched[arch] = (dm, actual)
        print(f"[{arch}] matched d_model={dm} -> params={actual:,} (target {target:,}, diff {actual-target:+,})")

    results = {a: [] for a in args.archs}
    for arch in args.archs:
        for seed in args.seeds:
            torch.manual_seed(seed)
            print(f"\n===== {arch} seed={seed} =====")
            if arch == "d4":
                m = sg28.build_variant("d4", vocab, spec, fused=args.fused).to(device)
            else:
                dm, _ = matched[arch]
                cfg = FairLMConfig(vocab_size=vocab, d_model=dm, num_heads=4, auto_match_parameters=False)
                suite = build_model_suite(cfg)
                m = (suite.lstm if arch == "lstm" else suite.transformer).to(device)
            t0 = time.perf_counter()
            r = sg28.train_model(arch, m, train_inp, train_tgt, valid_inp, valid_tgt,
                                 device, args.epochs, args.batch_size, args.lr)
            r["seed"] = seed; r["wall_s"] = time.perf_counter() - t0; r["params"] = count(m)
            print(f"  -> valid_bpc={r['valid_bpc']:.4f} tok/s={r['train_tokens_per_s']:.0f} "
                  f"wall={r['wall_s']:.1f}s params={r['params']:,}")
            results[arch].append(r)
            del m
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {a: {"bpc_mean": float(np.mean([r["valid_bpc"] for r in results[a]])),
                   "params": results[a][0]["params"]} for a in args.archs if results[a]}
    import json
    payload = {"experiment": "SG30 d4 matched-params pilot", "device": str(device),
               "target_params": target, "matched": {k: {"d_model": v[0], "params": v[1]} for k, v in matched.items()},
               "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "summary": summary, "raw": {a: [{k: v for k, v in r.items() if k in ("valid_bpc", "train_tokens_per_s", "params", "wall_s", "seed")} for r in results[a]] for a in args.archs}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 摘要 (pilot, 1ep) ===")
    for a, s in summary.items():
        print(f"  {a:11s}: bpc={s['bpc_mean']:.4f} params={s['params']:,}")
    print(f"\n结果：{args.out}")


if __name__ == "__main__":
    main()
