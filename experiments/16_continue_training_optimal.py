"""Continue training fant3_50m+AEL from step-500 checkpoint at throughput-
optimal config (batch=2 seq=512).
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
    resume_ckpt: Path = Path("D:/AEL/checkpoints/fant3_50m_wordnet/final.pt"),
    additional_steps: int = 1000,
    batch: int = 2,
    seq: int = 512,
    lr: float = 3e-4,         # smaller LR since we're continuing
    warmup: int = 50,
    log_every: int = 50,
    save_every: int = 250,
    seed: int = 1,
    ckpt_dir: Path = Path("D:/AEL/checkpoints/fant3_50m_wordnet"),
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"Loading checkpoint {resume_ckpt}...")
    ckpt = torch.load(resume_ckpt, weights_only=False, map_location="cuda")
    prev_step = ckpt.get("step", 0)
    print(f"  previous step: {prev_step}, previous losses len: {len(ckpt.get('losses', []))}")

    print("Loading tokens...")
    tokens = np.load("D:/AEL/data/wordnet_tokens.npy")

    print("Building model...")
    cfg = fant3_50m()
    cfg.ael_memory_enabled = True
    cfg.ael_gasket_depth = 5
    cfg.use_gradient_checkpointing = True
    model = FANT3Model(cfg).to("cuda", dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model_state_dict"])
    print("  weights loaded from checkpoint")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    def get_lr(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, additional_steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))

    losses = list(ckpt.get("losses", []))
    initial_loss_window = sum(losses[-20:]) / max(1, min(20, len(losses)))
    print(f"  loss at resume (last-20 mean): {initial_loss_window:.3f}")
    print(f"  starting {additional_steps} additional steps @ batch={batch} seq={seq}")
    t0 = time.time()
    model.train()
    for step in range(additional_steps):
        cur_lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = sample_batch(tokens, batch, seq, "cuda", rng)
        out = model(x); logits = out["logits"] if isinstance(out, dict) else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
        )
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(loss.item()))

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            tok_processed = batch * seq * (step + 1)
            tps = tok_processed / elapsed
            mean_loss = sum(losses[-log_every:]) / log_every
            print(f"  step {step+1:5d}/{additional_steps}  loss={mean_loss:.3f}  lr={cur_lr:.2e}  {tps:.0f} tok/s  elapsed={elapsed:.0f}s")

        if (step + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{prev_step + step + 1:05d}.pt"
            torch.save({
                "step": prev_step + step + 1,
                "model_state_dict": model.state_dict(),
                "losses": losses,
            }, ckpt_path)

    final_path = ckpt_dir / "final.pt"
    torch.save({
        "step": prev_step + additional_steps,
        "model_state_dict": model.state_dict(),
        "losses": losses,
    }, final_path)
    final_loss_window = sum(losses[-20:]) / 20
    print(f"\n  Final loss (last 20): {final_loss_window:.3f}")
    print(f"  Reduction from resume: {initial_loss_window - final_loss_window:.3f}")
    print(f"  Saved: {final_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq",   type=int, default=512)
    p.add_argument("--lr",    type=float, default=3e-4)
    args = p.parse_args()
    run(additional_steps=args.steps, batch=args.batch, seq=args.seq, lr=args.lr)
