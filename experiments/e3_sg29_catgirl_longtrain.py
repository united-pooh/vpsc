#!/usr/bin/env python3
"""SG29：猫娘 110k BPE 语料大规模长训练——SNN vs Transformer 决定性实验。

数据：``cyberlangke/Nana-catgirl-dataset-110k``（110k 中文猫娘对话），BPE 分词
（vocab 可配，默认 8192），拼连续文本做 next-token LM。经 hf-mirror 下载。

目的：scale-sweep 发现大规模+短训练下 SNN 超 Transformer，但 Transformer 欠训。
本条用更大 BPE 语料 + 长训练 + 多 seed，判 Transformer 充分训练后 SNN 是否仍超。
tensorboard 监控 + tqdm 进度（run_epoch 内）。

复用 sg28 的 build_variant/build_ann_suite/train_model（已加 tb_writer）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import experiments.e3_sg28_scaling_directions as sg28  # noqa: E402
from vpsc.world_model.devices import choose_device, device_label  # noqa: E402
from vpsc.world_model import catgirl_corpus as cg  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def run_arch(arch: str, vocab: int, spec, train_inp, train_tgt, valid_inp, valid_tgt,
             device, epochs, batch_size, lr, seed: int, fused: bool, tb_dir: Optional[Path]):
    """Train one (arch, seed). arch ∈ {base, d4, lstm, transformer}."""
    torch.manual_seed(seed)
    if arch in ("lstm", "transformer"):
        m = sg28.build_ann_suite(vocab, spec)[0 if arch == "lstm" else 1].to(device)
    else:
        m = sg28.build_variant(arch, vocab, spec, fused=fused).to(device)
    writer = None
    if tb_dir is not None and SummaryWriter is not None:
        writer = SummaryWriter(str(tb_dir / f"{arch}" / f"seed{seed}"))
    label = f"{arch}" + ("+fused" if fused and arch != "lstm" and arch != "transformer" else "")
    r = sg28.train_model(label, m, train_inp, train_tgt, valid_inp, valid_tgt,
                         device, epochs, batch_size, lr, tb_writer=writer, tb_prefix=label)
    if writer is not None:
        writer.flush(); writer.close()
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--state-dim", type=int, default=128)
    ap.add_argument("--n-experts", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-convs", type=int, default=None, help="限制对话数（调试）")
    ap.add_argument("--archs", nargs="+", default=["base", "d4", "lstm", "transformer"])
    ap.add_argument("--fused", action="store_true", default=True)
    ap.add_argument("--no-fused", dest="fused", action="store_false")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "results" / "e3_sg29_cache")
    ap.add_argument("--tb-dir", type=Path, default=REPO_ROOT / "results" / "e3_sg29_tb")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "e3_scan" / "e3_sg29_catgirl_longtrain.json")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = choose_device(args.device)
    print(f"device: {device_label(device)}  fused={args.fused}  archs={args.archs}  seeds={args.seeds}")

    print("加载猫娘 BPE 语料...")
    corpus = cg.load_bpe_corpus(args.cache_dir, vocab_size=args.vocab_size,
                                max_convs=args.max_convs)
    print(f"vocab={corpus['vocab_size']} train_tokens={corpus['n_train_tokens']:,} "
          f"val_tokens={corpus['n_val_tokens']:,} train_convs={corpus['n_train_convs']}")
    train_inp, train_tgt = cg.make_sequences(corpus["train_ids"], args.seq_len)
    valid_inp, valid_tgt = cg.make_sequences(corpus["val_ids"], args.seq_len)
    print(f"seq_len={args.seq_len} train_seqs={train_inp.shape[0]} valid_seqs={valid_inp.shape[0]}")

    spec = sg28.ModelSpec(d_model=args.d_model, state_dim=args.state_dim,
                          n_experts=args.n_experts, mtp_depth=4, n_actions=4)
    vocab = corpus["vocab_size"]
    args.tb_dir.mkdir(parents=True, exist_ok=True)

    # 每个 arch × seed 训练
    results: Dict[str, List[Dict]] = {a: [] for a in args.archs}
    for arch in args.archs:
        for seed in args.seeds:
            print(f"\n===== {arch} seed={seed} =====")
            t0 = time.perf_counter()
            r = run_arch(arch, vocab, spec, train_inp, train_tgt, valid_inp, valid_tgt,
                         device, args.epochs, args.batch_size, args.lr, seed, args.fused, args.tb_dir)
            r["seed"] = seed
            r["wall_s"] = time.perf_counter() - t0
            print(f"  -> valid_bpc={r['valid_bpc']:.4f} tok/s={r['train_tokens_per_s']:.0f} "
                  f"mem={r['peak_mem_mib']:.0f}MiB wall={r['wall_s']:.1f}s params={r['params']['registered']}")
            results[arch].append(r)

    # 聚合 mean±std
    summary = {}
    for arch, runs in results.items():
        if not runs:
            continue
        bpcs = [r["valid_bpc"] for r in runs]
        tps = [r["train_tokens_per_s"] for r in runs]
        mems = [r["peak_mem_mib"] for r in runs]
        summary[arch] = {
            "valid_bpc_mean": float(np.mean(bpcs)), "valid_bpc_std": float(np.std(bpcs)),
            "tok_s_mean": float(np.mean(tps)), "tok_s_std": float(np.std(tps)),
            "mem_mib_mean": float(np.mean(mems)),
            "params": runs[0]["params"]["registered"],
            "per_seed_bpc": bpcs, "per_seed_tok_s": tps,
        }

    payload = {"experiment": "SG29 catgirl BPE long-train", "device": str(device),
               "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "corpus": {k: (str(v) if isinstance(v, Path) else v) for k, v in corpus.items()
                          if k not in ("train_ids", "val_ids")},
               "spec": {"d_model": spec.d_model, "state_dim": spec.state_dim,
                        "n_experts": spec.n_experts},
               "summary": summary, "raw_results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 摘要 (valid BPC mean±std / tok/s / 显存) ===")
    for arch in args.archs:
        s = summary.get(arch)
        if not s:
            continue
        print(f"  {arch:11s}: bpc={s['valid_bpc_mean']:.4f}±{s['valid_bpc_std']:.4f} "
              f"tok/s={s['tok_s_mean']:.0f} mem={s['mem_mib_mean']:.0f}MiB params={s['params']}")
    print(f"\n结果：{args.out}")
    print(f"tensorboard：{args.tb_dir}  (tensorboard --logdir {args.tb_dir})")


if __name__ == "__main__":
    main()
