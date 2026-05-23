"""P1 continuation with gradient accumulation.

Continues fant3_50m+AEL training from the 5000-step OpenWebText checkpoint
(loss 6.57). Adds gradient accumulation so the effective batch is >1 without
risking the Windows-fragmentation OOM that ~physical batch>=2 triggers
over long runs.

  effective_batch = physical_batch * accum_steps
  Each opt.step() now averages gradient over `accum_steps` micro-batches.
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
    resume_ckpt: Path = Path("D:/AEL/checkpoints/p1_openwebtext/final.pt"),
    opt_steps: int = 2500,
    accum_steps: int = 4,           # effective_batch = 1 * 4 = 4
    batch: int = 1,
    seq: int = 512,
    lr: float = 3e-4,
    warmup_opt: int = 100,          # in optimizer-steps (not micro-steps)
    log_every_opt: int = 50,        # log every N optimizer-steps
    save_every_opt: int = 500,
    seed: int = 1,
    tokens_path: Path = Path("D:/AEL/data/openwebtext_tokens_100m.npy"),
    ckpt_dir: Path = Path("D:/AEL/checkpoints/p1_openwebtext"),
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"Loading OpenWebText tokens from {tokens_path}...")
    tokens = np.load(tokens_path)
    print(f"  {len(tokens):,} tokens")

    print(f"Loading checkpoint {resume_ckpt}...")
    ckpt = torch.load(resume_ckpt, weights_only=False, map_location="cuda")
    prev_step = ckpt.get("step", 0)
    prev_losses = ckpt.get("losses", [])
    print(f"  previous opt steps: {prev_step}, losses len: {len(prev_losses)}")
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
    print("  weights loaded")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.1)

    def get_lr(opt_step):
        if opt_step < warmup_opt:
            return lr * (opt_step + 1) / warmup_opt
        progress = (opt_step - warmup_opt) / max(1, opt_steps - warmup_opt)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    losses = list(prev_losses)
    new_losses: list[float] = []
    eff_batch = batch * accum_steps
    print(f"  starting {opt_steps} opt-steps  (effective batch = {eff_batch}, accum={accum_steps})\n")
    t0 = time.time()
    model.train()

    for opt_step in range(opt_steps):
        cur_lr = get_lr(opt_step)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad()
        step_losses = []
        for micro in range(accum_steps):
            x, y = sample_batch(tokens, batch, seq, "cuda", rng)
            out = model(x); logits = out["logits"] if isinstance(out, dict) else out
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
            )
            # Scale loss so backward accumulates the AVERAGE gradient.
            (loss / accum_steps).backward()
            step_losses.append(float(loss.item()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        # Loss to log = average of the micro-batch losses.
        ml = sum(step_losses) / accum_steps
        new_losses.append(ml)
        losses.append(ml)

        if (opt_step + 1) % 25 == 0:
            torch.cuda.empty_cache()

        if (opt_step + 1) % log_every_opt == 0:
            elapsed = time.time() - t0
            tok_processed = eff_batch * seq * (opt_step + 1)
            tps = tok_processed / elapsed
            mean_loss = sum(new_losses[-log_every_opt:]) / log_every_opt
            print(f"  opt_step {opt_step+1:5d}/{opt_steps}  loss={mean_loss:.3f}  lr={cur_lr:.2e}  {tps:.0f} tok/s  elapsed={elapsed:.0f}s")

        if (opt_step + 1) % save_every_opt == 0:
            ckpt_path = ckpt_dir / f"gradaccum_step_{prev_step + opt_step + 1:05d}.pt"
            torch.save({
                "step": prev_step + opt_step + 1,
                "model_state_dict": model.state_dict(),
                "losses": losses,
                "effective_batch": eff_batch,
            }, ckpt_path)

    final_path = ckpt_dir / "final_gradaccum.pt"
    torch.save({
        "step": prev_step + opt_steps,
        "model_state_dict": model.state_dict(),
        "losses": losses,
        "effective_batch": eff_batch,
    }, final_path)

    first20 = sum(new_losses[:20]) / max(1, min(20, len(new_losses)))
    last20  = sum(new_losses[-20:]) / max(1, min(20, len(new_losses)))
    print(f"\n  gradaccum portion first-20: {first20:.3f}")
    print(f"  gradaccum portion last-20:  {last20:.3f}")
    print(f"  Reduction: {first20 - last20:.3f}")
    print(f"  Saved: {final_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--opt-steps",   type=int,   default=2500)
    p.add_argument("--accum-steps", type=int,   default=4)
    p.add_argument("--batch",       type=int,   default=1)
    p.add_argument("--seq",         type=int,   default=512)
    p.add_argument("--lr",          type=float, default=3e-4)
    args = p.parse_args()
    run(opt_steps=args.opt_steps, accum_steps=args.accum_steps, batch=args.batch, seq=args.seq, lr=args.lr)
