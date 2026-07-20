"""猫娘 110k BPE 语料管线（SG29 大规模长训练用）。

数据集：``cyberlangke/Nana-catgirl-dataset-110k``（110,216 条中文猫娘对话，
system="你是猫娘奈奈。"）。经 ``HF_ENDPOINT=https://hf-mirror.com`` 下载（服务器直连
huggingface.co 被墙）。

把每条对话的 messages 拼成连续文本（role 内容用换行分隔），对话间用空行分隔；
在该文本上训练 BPE tokenizer（vocab 可配，默认 8192）；tokenize 成训练/验证张量。
tokenizer 与 tokenized 张量缓存到 cache_dir，重跑秒级加载。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _conv_to_text(messages: List[dict]) -> str:
    """把一轮对话的 messages 拼成连续文本。role 行 + content 行，换行分隔。"""
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def load_catgirl_texts(cache_dir: Path, max_convs: Optional[int] = None,
                       redownload: bool = False) -> Tuple[List[str], dict]:
    """返回 (对话文本列表, 元信息)。经 hf-mirror 下载，缓存到 cache_dir。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts_path = cache_dir / "catgirl_texts.txt"
    meta_path = cache_dir / "catgirl_meta.json"
    if texts_path.exists() and not redownload:
        texts = texts_path.read_text(encoding="utf-8").split("\n\n")
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if max_convs is not None:
            texts = texts[:max_convs]
        return texts, meta
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("cyberlangke/Nana-catgirl-dataset-110k", split="train")
    texts = [_conv_to_text(r["messages"]) for r in ds if r.get("messages")]
    if max_convs is not None:
        texts = texts[:max_convs]
    texts_path.write_text("\n\n".join(texts), encoding="utf-8")
    import json
    meta = {"n_convs": len(texts), "source": "cyberlangke/Nana-catgirl-dataset-110k",
            "total_chars": sum(len(t) for t in texts)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return texts, meta


def train_bpe(texts: List[str], cache_dir: Path, vocab_size: int = 8192) -> dict:
    """在 texts 上训练 BPE tokenizer，保存到 cache_dir。返回 tokenizer 与词表大小。"""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    tok_path = cache_dir / f"catgirl_bpe_{vocab_size}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if tok_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
        return {"tokenizer": tok, "vocab_size": tok.get_vocab_size(), "path": str(tok_path)}
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size,
                         special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tok.train_from_iterator(texts, trainer)
    tok.save(str(tok_path))
    return {"tokenizer": tok, "vocab_size": tok.get_vocab_size(), "path": str(tok_path)}


def tokenize_corpus(texts: List[str], tokenizer, cache_dir: Path,
                    split: str = "train") -> Tensor:
    """tokenize texts → 1D long tensor。缓存到 cache_dir/split_ids.pt。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids_path = cache_dir / f"catgirl_{split}_ids.pt"
    if ids_path.exists():
        return torch.load(ids_path, weights_only=True)
    enc = tokenizer.encode_batch(texts)
    all_ids: List[int] = []
    eos = tokenizer.token_to_id("[EOS]")
    for e in enc:
        all_ids.extend(e.ids)
        if eos is not None:
            all_ids.append(eos)
    t = torch.tensor(all_ids, dtype=torch.long)
    torch.save(t, ids_path)
    return t


def load_bpe_corpus(cache_dir: Path, vocab_size: int = 8192,
                    max_convs: Optional[int] = None,
                    val_frac: float = 0.05) -> dict:
    """端到端：下载→训BPE→tokenize→train/valid 切分。返回 {ids, vocab_size, meta}。"""
    texts, meta = load_catgirl_texts(cache_dir / "raw", max_convs=max_convs)
    bpe = train_bpe(texts, cache_dir / "bpe", vocab_size=vocab_size)
    tok = bpe["tokenizer"]
    n_val = max(1, int(len(texts) * val_frac))
    train_texts = texts[:-n_val]
    val_texts = texts[-n_val:]
    train_ids = tokenize_corpus(train_texts, tok, cache_dir / "tok", split="train")
    val_ids = tokenize_corpus(val_texts, tok, cache_dir / "tok", split="val")
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "vocab_size": bpe["vocab_size"],
        "tokenizer_path": bpe["path"],
        "n_train_convs": len(train_texts),
        "n_val_convs": len(val_texts),
        "n_train_tokens": int(train_ids.numel()),
        "n_val_tokens": int(val_ids.numel()),
        "source_meta": meta,
    }


def make_sequences(ids: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
    """Non-overlapping next-token chunks [N, seq_len] (inputs, targets shift-1)."""
    n = (len(ids) - 1) // seq_len
    if n <= 0:
        raise ValueError("corpus too short for seq_len")
    inputs = ids[: n * seq_len].view(n, seq_len)
    targets = ids[1: n * seq_len + 1].view(n, seq_len)
    return inputs, targets


__all__ = ["load_bpe_corpus", "make_sequences", "train_bpe", "load_catgirl_texts"]
