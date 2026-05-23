"""Topology Phase 1: General English pretrain on OpenWebText.

Fresh init (no WordNet warm-start). The first phase in our topology-aware
training plan — establish the embedding/dense layers' baseline for English.

Target: loss < 4.5 on OpenWebText holdout, coherent next-token completions
of common phrases.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("D:/fant3")))
sys.path.insert(0, str(Path("D:/AEL/src")))

from fant3.config import fant3_50m
from fant3.model import FANT3Model


def sample_batch(tokens, batch_size, seq_len, device, rng):
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[s : s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def run(
    steps: int = 5000,
    batch: int = 2,
    seq: int = 512,
    lr: float = 5e-4,                       # higher LR since we're starting fresh
    warmup: int = 200,
    log_every: int = 100,
    save_every: int = 1000,
    seed: int = 0,
    tokens_path: Path = Path("D:/AEL/data/openwebtext_tokens_100m.npy"),
    ckpt_dir: Path = Path("D:/AEL/checkpoints/p1_openwebtext"),
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"Loading OpenWebText tokens from {tokens_path}...")
    tokens = np.load(tokens_path)
    print(f"  {len(tokens):,} tokens")

    print("Building model (FRESH INIT — no warm-start)...")
    cfg = fant3_50m()
    cfg.ael_memory_enabled = True
    cfg.ael_gasket_depth = 5
    cfg.use_gradient_checkpointing = True
    model = FANT3Model(cfg).to("cuda", dtype=torch.bfloat16)
    params = sum(p.numel() for p in model.parameters())
    print(f"  fant3_50m+AEL: {params/1e6:.2f}M params")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.1)

    def get_lr(step):
        if step < warmup:
            return lr * (step + 1) / warmup
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
        out = model(x); logits = out["logits"] if isinstance(out, dict) else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            y.reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(loss.item()))
        # Periodic cache clear to fight Windows allocator fragmentation.
        if (step + 1) % 50 == 0:
            torch.cuda.empty_cache()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            tps = batch * seq * (step + 1) / elapsed
            mean_loss = sum(losses[-log_every:]) / log_every
            print(f"  step {step+1:5d}/{steps}  loss={mean_loss:.3f}  lr={cur_lr:.2e}  {tps:.0f} tok/s  elapsed={elapsed:.0f}s")

        if (step + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step+1:05d}.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "losses": losses,
            }, ckpt_path)

    final_path = ckpt_dir / "final.pt"
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "losses": losses,
    }, final_path)

    first20 = sum(losses[:20]) / 20
    last20 = sum(losses[-20:]) / 20
    print(f"\n  Initial loss (first 20): {first20:.3f}")
    print(f"  Final loss (last 20):    {last20:.3f}")
    print(f"  Reduction: {first20 - last20:.3f}")
    print(f"  Saved: {final_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq",   type=int, default=512)
    p.add_argument("--lr",    type=float, default=5e-4)
    args = p.parse_args()
    run(steps=args.steps, batch=args.batch, seq=args.seq, lr=args.lr)
