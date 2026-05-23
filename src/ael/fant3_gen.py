"""Lightweight inference wrapper for the trained fant3_50m+AEL checkpoint.

Lazy-loads the model + tokenizer once, then provides greedy generation for
arbitrary prompts. Used as the fallback inside qa.py when no Q-style pattern
matches the input (covers the "sentence completion" benchmark category).
"""
from __future__ import annotations

import sys
from pathlib import Path

_MODEL = None
_TOK = None
_DEVICE = "cuda"


def _ensure_loaded(ckpt_path: Path = Path("D:/AEL/checkpoints/fant3_50m_wordnet/final.pt")):
    global _MODEL, _TOK
    if _MODEL is not None:
        return
    # Make sure fant3 is importable.
    if str(Path("D:/fant3")) not in sys.path:
        sys.path.insert(0, str(Path("D:/fant3")))
    if str(Path("D:/AEL/src")) not in sys.path:
        sys.path.insert(0, str(Path("D:/AEL/src")))

    import torch
    from tokenizers import Tokenizer
    from fant3.config import fant3_50m
    from fant3.model import FANT3Model

    cfg = fant3_50m()
    cfg.ael_memory_enabled = True
    cfg.ael_gasket_depth = 5
    cfg.use_gradient_checkpointing = False  # generation, no grads
    _MODEL = FANT3Model(cfg).to(_DEVICE, dtype=torch.bfloat16)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=_DEVICE)
    _MODEL.load_state_dict(ckpt["model_state_dict"])
    _MODEL.eval()

    _TOK = Tokenizer.from_file(str(Path("D:/fant3/output/tokenizer/tokenizer_v2.json")))


def generate(
    prompt: str,
    max_new: int = 15,
    top_k: int = 20,
    temperature: float = 0.7,
    repetition_penalty: float = 1.5,
    seed: int = 0,
) -> str:
    """Sampling-based decode with top-k filter, temperature, and repetition penalty.

    Greedy decoding on an undertrained small LM collapses to "the the the..."
    or "." loops. This routine:
      - applies a repetition penalty (down-weight already-emitted tokens)
      - top-k filters to the K most-likely candidates
      - samples with temperature
      - stops on sentence-final punctuation or newline
    """
    _ensure_loaded()
    import torch

    g = torch.Generator(device=_DEVICE)
    g.manual_seed(seed)

    enc = _TOK.encode(prompt)
    ids = list(enc.ids)
    initial_len = len(ids)
    emitted: set[int] = set()

    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids], dtype=torch.long, device=_DEVICE)
            out = _MODEL(x)
            logits = out["logits"] if isinstance(out, dict) else out
            row = logits[0, -1].float()
            # Repetition penalty.
            if emitted:
                idx = torch.tensor(list(emitted), device=_DEVICE)
                row[idx] = row[idx] / repetition_penalty
            # Top-k filter.
            top_vals, top_idx = row.topk(min(top_k, row.numel()))
            probs = torch.softmax(top_vals / max(temperature, 1e-6), dim=-1)
            sample = torch.multinomial(probs, 1, generator=g).item()
            next_id = int(top_idx[sample].item())
            ids.append(next_id)
            emitted.add(next_id)
            tok_str = _TOK.decode([next_id])
            if "\n" in tok_str or tok_str.strip() in (".", "!", "?"):
                break

    new_ids = ids[initial_len:]
    return _TOK.decode(new_ids)


def complete(prompt: str, max_new: int = 6) -> str:
    """Short greedy completion (a few tokens), used for LAMBADA-style endings."""
    return generate(prompt, max_new=max_new).strip()
