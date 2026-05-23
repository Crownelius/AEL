"""Stream + tokenize FineWeb-Edu for fant3+AEL training.

Uses datasets streaming so we don't have to download the full 20GB. We pull
roughly target_tokens worth of text and tokenize with the fant3 v2 tokenizer.

Output: a numpy uint16 array, same format as wordnet_tokens.npy, drop-in for
the existing training script.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def build_fineweb_tokens(
    out_path: Path,
    tokenizer_path: Path,
    target_tokens: int = 20_000_000,
    hf_dataset: str = "HuggingFaceFW/fineweb-edu",
    hf_subset: str = "sample-10BT",
    split: str = "train",
    log_every: int = 1_000_000,
) -> dict:
    """Stream FineWeb-Edu and tokenize until we hit target_tokens."""
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    eos_id = None
    # fant3 tokenizer v2 has an EOS-ish token; use vocab-1 as separator if unknown.
    try:
        eos_id = tok.token_to_id("<|endoftext|>") or tok.token_to_id("</s>")
    except Exception:
        pass
    if eos_id is None:
        eos_id = tok.get_vocab_size() - 1

    print(f"Loading streaming dataset {hf_dataset}/{hf_subset}/{split}...")
    ds = load_dataset(hf_dataset, name=hf_subset, split=split, streaming=True)

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
            print(f"  docs={n_docs:>6}  tokens={len(ids):>10,}  rate={len(ids)/elapsed:.0f} tok/sec  elapsed={elapsed:.0f}s")
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
    p.add_argument("--target-tokens", type=int, default=20_000_000)
    p.add_argument("--out", default="D:/AEL/data/fineweb_edu_tokens.npy")
    p.add_argument("--subset", default="sample-10BT")
    args = p.parse_args()

    stats = build_fineweb_tokens(
        out_path=Path(args.out),
        tokenizer_path=Path("D:/fant3/output/tokenizer/tokenizer_v2.json"),
        target_tokens=args.target_tokens,
        hf_subset=args.subset,
    )
    print(f"\nDone: {stats}")
