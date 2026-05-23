"""Continue fant3_50m+AEL training on FineWeb-Edu.

Resumes from the WordNet-trained checkpoint (loss 4.21). If broader data
lifts the WordNet pattern-saturation, loss should drop noticeably and the
model's generations should diversify away from "X is a kind of Y".
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
    resume_ckpt: Path = Path("D:/AEL/checkpoints/fant3_50m_fineweb/final.pt"),
    additional_steps: int = 3000,
    batch: int = 2,
    seq: int = 512,
    lr: float = 1.5e-4,
    warmup: int = 100,
    log_every: int = 100,
    save_every: int = 500,
    seed: int = 2,
    tokens_path: Path = Path("D:/AEL/data/fineweb_edu_tokens_50m.npy"),
    ckpt_dir: Path = Path("D:/AEL/checkpoints/fant3_50m_fineweb"),
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"Loading FineWeb tokens from {tokens_path}...")
    tokens = np.load(tokens_path)
    print(f"  {len(tokens):,} tokens")

    print(f"Loading checkpoint {resume_ckpt}...")
    ckpt = torch.load(resume_ckpt, weights_only=False, map_location="cuda")
    prev_step = ckpt.get("step", 0)
    prev_losses = ckpt.get("losses", [])
    print(f"  previous step: {prev_step}, losses len: {len(prev_losses)}")
    if prev_losses:
        prev_last20 = sum(prev_losses[-20:]) / min(20, len(prev_losses))
        print(f"  loss at resume (last-20 mean): {prev_last20:.3f}")

    print("Building model...")
    cfg = fant3_50m()
    cfg.ael_memory_enabled = True
    cfg.ael_gasket_depth = 5
    cfg.use_gradient_checkpointing = True
    model = FANT3Model(cfg).to("cuda", dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model_state_dict"])
    print("  weights loaded from WordNet checkpoint")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

    def get_lr(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, additional_steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))

    losses = list(prev_losses)
    fineweb_losses = []   # track only the FineWeb portion separately
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  starting {additional_steps} FineWeb steps @ batch={batch} seq={seq}")
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
        l = float(loss.item())
        losses.append(l)
        fineweb_losses.append(l)

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            tps = batch * seq * (step + 1) / elapsed
            mean_loss = sum(fineweb_losses[-log_every:]) / log_every
            print(f"  step {step+1:5d}/{additional_steps}  loss={mean_loss:.3f}  lr={cur_lr:.2e}  {tps:.0f} tok/s  elapsed={elapsed:.0f}s")

        if (step + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{prev_step + step + 1:05d}.pt"
            torch.save({
                "step": prev_step + step + 1,
                "model_state_dict": model.state_dict(),
                "losses": losses,
                "fineweb_losses": fineweb_losses,
            }, ckpt_path)

    final_path = ckpt_dir / "final.pt"
    torch.save({
        "step": prev_step + additional_steps,
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "fineweb_losses": fineweb_losses,
    }, final_path)

    fw_first = sum(fineweb_losses[:20]) / max(1, min(20, len(fineweb_losses)))
    fw_last = sum(fineweb_losses[-20:]) / max(1, min(20, len(fineweb_losses)))
    print(f"\n  FineWeb portion: first-20 loss={fw_first:.3f}  last-20 loss={fw_last:.3f}")
    print(f"  Reduction during FineWeb: {fw_first - fw_last:.3f}")
    print(f"  Saved: {final_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq",   type=int, default=512)
    p.add_argument("--lr",    type=float, default=1.5e-4)
    args = p.parse_args()
    run(additional_steps=args.steps, batch=args.batch, seq=args.seq, lr=args.lr)
