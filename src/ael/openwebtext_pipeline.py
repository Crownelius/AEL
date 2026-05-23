"""Stream + tokenize OpenWebText for fant3+AEL pretraining.

OpenWebText is the community reimplementation of OpenAI's WebText (the data
GPT-2 was trained on). HuggingFace: Skylion007/openwebtext.

Same shape as fineweb_pipeline.py — produces a uint16 numpy array drop-in
for the existing training loop.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def build_openwebtext_tokens(
    out_path: Path,
    tokenizer_path: Path,
    target_tokens: int = 100_000_000,
    hf_dataset: str = "Skylion007/openwebtext",
    split: str = "train",
    log_every: int = 5_000_000,
) -> dict:
    """Stream OpenWebText and tokenize until we hit target_tokens."""
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    eos_id = None
    try:
        eos_id = tok.token_to_id("<|endoftext|>") or tok.token_to_id("</s>")
    except Exception:
        pass
    if eos_id is None:
        eos_id = tok.get_vocab_size() - 1

    print(f"Loading streaming dataset {hf_dataset}/{split}...")
    ds = load_dataset(hf_dataset, split=split, streaming=True, trust_remote_code=True)

    ids: list[int] = []
    n_docs = 0
    t0 = time.time()
    last_logged = 0
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        enc = tok.encode(text)
        ids.extend(enc.ids)
        ids.append(eos_id)
        n_docs += 1
        if len(ids) - last_logged >= log_every:
            elapsed = time.time() - t0
            print(f"  docs={n_docs:>6}  tokens={len(ids):>10,}  "
                  f"rate={len(ids)/elapsed:.0f} tok/sec  elapsed={elapsed:.0f}s")
            last_logged = len(ids)
        if len(ids) >= target_tokens:
            break

    arr = np.array(ids[:target_tokens], dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return {
        "tokens": int(arr.size),
        "docs": n_docs,
        "elapsed": time.time() - t0,
        "out": str(out_path),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target-tokens", type=int, default=100_000_000)
    p.add_argument("--out", default="D:/AEL/data/openwebtext_tokens_100m.npy")
    args = p.parse_args()
    stats = build_openwebtext_tokens(
        out_path=Path(args.out),
        tokenizer_path=Path("D:/fant3/output/tokenizer/tokenizer_v2.json"),
        target_tokens=args.target_tokens,
    )
    print(f"\nDone: {stats}")
