"""Train fant3_50m + AEL on the WordNet corpus.

Loads the tokenized WordNet corpus, samples random fixed-length chunks,
runs AdamW for N steps, saves a checkpoint and a loss curve.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Locate fant3 + AEL.
sys.path.insert(0, str(Path("D:/fant3")))
sys.path.insert(0, str(Path("D:/AEL/src")))

from fant3.config import fant3_50m
from fant3.model import FANT3Model


def sample_batch(tokens: np.ndarray, batch_size: int, seq_len: int, device: str, rng: np.random.Generator):
    """Random crops from the token stream."""
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[s : s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def run(
    steps: int = 500,
    batch: int = 1,
    seq: int = 512,
    lr: float = 5e-4,
    warmup: int = 50,
    log_every: int = 20,
    save_every: int = 250,
    seed: int = 0,
    ckpt_dir: Path = Path("D:/AEL/checkpoints/fant3_50m_wordnet"),
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print("Loading tokens...")
    tokens = np.load("D:/AEL/data/wordnet_tokens.npy")
    print(f"  {len(tokens):,} tokens")

    print("Building model...")
    cfg = fant3_50m()
    cfg.ael_memory_enabled = True
    cfg.ael_gasket_depth = 5
    cfg.use_gradient_checkpointing = True  # safe at 50m
    model = FANT3Model(cfg).to("cuda", dtype=torch.bfloat16)
    params = sum(p.numel() for p in model.parameters())
    print(f"  fant3_50m+AEL: {params/1e6:.2f}M params")

    # bnb 8-bit optimizer.
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    def get_lr(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        # Cosine to 10% of lr.
        progress = (step - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        cur_lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = sample_batch(tokens, batch, seq, "cuda", rng)
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            y.reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(loss.item()))

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            toks = batch * seq * (step + 1)
            tps = toks / elapsed
            mean_loss = sum(losses[-log_every:]) / log_every
            print(f"  step {step+1:5d}/{steps}  loss={mean_loss:.3f}  lr={cur_lr:.2e}  {tps:.0f} tok/s  elapsed={elapsed:.0f}s")

        if (step + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step+1:05d}.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "losses": losses,
                "cfg_dict": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__ if not isinstance(getattr(cfg, k), tuple)},
            }, ckpt_path)
            print(f"    saved {ckpt_path}")

    # Final save.
    ckpt_final = ckpt_dir / "final.pt"
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "losses": losses,
    }, ckpt_final)
    print(f"\nFinal checkpoint: {ckpt_final}")

    # Loss summary.
    initial = sum(losses[:20]) / max(20, 1)
    final = sum(losses[-20:]) / max(20, 1)
    print(f"  initial loss (first 20 steps): {initial:.3f}")
    print(f"  final loss (last 20 steps):    {final:.3f}")
    print(f"  reduction: {initial - final:.3f}  ({100*(initial-final)/initial:.1f}%)")

    return {"initial_loss": initial, "final_loss": final, "losses": losses}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq",   type=int, default=512)
    p.add_argument("--lr",    type=float, default=5e-4)
    args = p.parse_args()
    run(steps=args.steps, batch=args.batch, seq=args.seq, lr=args.lr)
