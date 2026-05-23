#!/usr/bin/env python3
"""Self-contained training entrypoint for RunPod / Kaggle / any fresh Linux box.

Single-file, idempotent, autoresumable. Designed so the *only* manual steps
on a fresh pod are:

    git clone https://github.com/Crownelius/AEL.git
    git clone https://github.com/Crownelius/fant3.git
    cd AEL
    python pod/pod_train.py                    # full default run
    python pod/pod_train.py --steps 50 --smoke # local dress-rehearsal

Everything else (deps, data, checkpoints, logging) is automatic.

Design constraints driven by the "no wasted RunPod credits" rule:
  - POSIX paths throughout
  - pip-installs missing deps at startup (one-shot, idempotent)
  - tokens cached to ./pod_data/{dataset}_{N}tok.npy -- never re-downloaded
  - all stdout AND stderr mirrored to ./pod_logs/{timestamp}.log
  - atomic checkpoint writes (tmpfile + rename)
  - checkpoint saved at startup, every N opt-steps, and at clean exit
  - autoresume: if checkpoint exists at out_dir/final.pt, load and continue
  - explicit non-zero exit code on any failure
  - optional --shutdown flag triggers `runpodctl stop pod $RUNPOD_POD_ID` on
    successful exit so we don't burn credits idling
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Phase 0 — paths + logging (must come before any heavy imports)
# ---------------------------------------------------------------------------

THIS = Path(__file__).resolve()
AEL_ROOT = THIS.parent.parent              # .../AEL/
FANT3_ROOT = AEL_ROOT.parent / "fant3"     # sibling repo

POD_DATA = AEL_ROOT / "pod_data"
POD_LOGS = AEL_ROOT / "pod_logs"
POD_CKPT = AEL_ROOT / "pod_ckpt"
for p in (POD_DATA, POD_LOGS, POD_CKPT):
    p.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Mirror writes to both the original stream and a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data); s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass


_LOG_FILE = POD_LOGS / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_LOG_FH = open(_LOG_FILE, "w", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.stdout, _LOG_FH)
sys.stderr = _Tee(sys.stderr, _LOG_FH)

print(f"[pod_train] log file: {_LOG_FILE}")
print(f"[pod_train] AEL_ROOT={AEL_ROOT}")
print(f"[pod_train] FANT3_ROOT={FANT3_ROOT}  (exists={FANT3_ROOT.exists()})")


# ---------------------------------------------------------------------------
# Phase 1 — dependency check / install (idempotent)
# ---------------------------------------------------------------------------

REQUIRED_DEPS = {
    # import_name : pip_name
    "torch":         "torch>=2.5.0",
    "numpy":         "numpy>=1.26",
    "datasets":      "datasets>=3.0",
    "tokenizers":    "tokenizers>=0.20",
    "transformers":  "transformers>=4.40",
    "bitsandbytes":  "bitsandbytes>=0.43",
}


def ensure_deps() -> None:
    missing = []
    for mod, pip_spec in REQUIRED_DEPS.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_spec)
    if missing:
        print(f"[pod_train] installing missing deps: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        )
    else:
        print("[pod_train] all deps present.")


# ---------------------------------------------------------------------------
# Phase 2 — fant3 sys.path wiring (only after deps are confirmed)
# ---------------------------------------------------------------------------

def setup_paths() -> None:
    if not FANT3_ROOT.exists():
        raise SystemExit(
            f"[pod_train] fant3 repo not found at {FANT3_ROOT}.\n"
            f"  fix: cd {AEL_ROOT.parent} && git clone https://github.com/Crownelius/fant3.git"
        )
    for p in (str(FANT3_ROOT), str(AEL_ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Phase 3 — token data: load if cached, stream from HF if not
# ---------------------------------------------------------------------------

def _find_existing_cache(dataset: str, target_tokens: int) -> Path | None:
    """Search known cache locations for a usable token file (size >= target)."""
    import numpy as np
    candidates: list[Path] = []
    safe_ds = dataset.replace("/", "_")
    legacy = AEL_ROOT / "data"
    # 1. pod_data canonical name (exact target match)
    canonical = POD_DATA / f"{safe_ds}__default__{target_tokens}.npy"
    if canonical.exists():
        return canonical
    # 2. legacy data/ directory: any .npy whose name hints at the same dataset
    if legacy.exists():
        hint = "openwebtext" if "openwebtext" in dataset.lower() else (
            "fineweb"   if "fineweb"   in dataset.lower() else (
            "wordnet"   if "wordnet"   in dataset.lower() else None))
        if hint:
            candidates.extend(legacy.glob(f"*{hint}*tokens*.npy"))
            candidates.extend(legacy.glob(f"*{hint}*.npy"))
    # 3. pod_data fuzzy matches
    candidates.extend(POD_DATA.glob(f"{safe_ds}*.npy"))
    # Pick the largest that is still >= target.
    best: Path | None = None
    best_size = -1
    for c in candidates:
        try:
            arr = np.load(c, mmap_mode="r")
            n = len(arr)
        except Exception:
            continue
        if n >= target_tokens and n > best_size:
            best, best_size = c, n
    return best


def ensure_tokens(
    dataset: str,
    subset: str | None,
    target_tokens: int,
    cache_dir: Path = POD_DATA,
    max_retries: int = 5,
):
    """Returns a numpy uint16 array of token IDs, cached to disk.

    Cache hit policy (in order):
      1. canonical pod_data cache  ->  use as-is
      2. legacy AEL/data file from prior sessions  ->  truncate to target_tokens
      3. stream from HuggingFace (with retry-with-backoff)
    """
    import numpy as np
    existing = _find_existing_cache(dataset, target_tokens)
    if existing is not None:
        arr = np.load(existing)
        if len(arr) > target_tokens:
            arr = arr[:target_tokens]
        print(f"[pod_train] tokens cache hit: {existing} ({len(arr):,} tokens, asked {target_tokens:,})")
        return arr

    print(f"[pod_train] tokens cache miss; streaming {dataset}/{subset} with retries...")
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok_path = FANT3_ROOT / "output" / "tokenizer" / "tokenizer_v2.json"
    if not tok_path.exists():
        raise SystemExit(f"[pod_train] tokenizer not found at {tok_path}")
    tok = Tokenizer.from_file(str(tok_path))
    try:
        eos_id = tok.token_to_id("<|endoftext|>") or tok.token_to_id("</s>")
    except Exception:
        eos_id = None
    if eos_id is None:
        eos_id = tok.get_vocab_size() - 1

    ids: list[int] = []
    n_docs = 0
    t0 = time.time()
    last_log = 0
    log_every = max(target_tokens // 20, 100_000)

    for attempt in range(max_retries):
        try:
            ds_kwargs = dict(split="train", streaming=True, trust_remote_code=True)
            if subset:
                ds = load_dataset(dataset, name=subset, **ds_kwargs)
            else:
                ds = load_dataset(dataset, **ds_kwargs)
            # Skip the docs we've already consumed (HF streaming has no resume —
            # we just iterate from scratch and drop until we reach our position).
            # This is wasteful if attempt > 0 but at least makes the run survive.
            seen = 0
            for ex in ds:
                if seen < n_docs:
                    seen += 1
                    continue
                text = ex.get("text") or ex.get("content") or ""
                if not text:
                    continue
                enc = tok.encode(text)
                ids.extend(enc.ids)
                ids.append(eos_id)
                n_docs += 1
                if len(ids) - last_log >= log_every:
                    print(f"  streamed {n_docs:>7} docs  {len(ids):>12,} tokens  "
                          f"({len(ids)/(time.time()-t0):.0f} tok/sec)")
                    last_log = len(ids)
                if len(ids) >= target_tokens:
                    break
            if len(ids) >= target_tokens:
                break
        except Exception as e:
            backoff = min(60, 2 ** attempt)
            print(f"[pod_train] HF stream error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {str(e)[:120]}")
            print(f"  retrying in {backoff}s, resuming at doc {n_docs}, tokens so far {len(ids):,}")
            time.sleep(backoff)

    if len(ids) < target_tokens:
        raise SystemExit(
            f"[pod_train] could not reach {target_tokens:,} tokens after {max_retries} retries "
            f"(only got {len(ids):,}). Aborting to avoid wasting GPU."
        )

    import numpy as np
    arr = np.array(ids[:target_tokens], dtype=np.uint16)
    safe_subset = (subset or "default").replace("/", "_")
    cache_path = cache_dir / f"{dataset.replace('/', '_')}__{safe_subset}__{target_tokens}.npy"
    # numpy.save auto-appends ".npy" if the path doesn't already end in it.
    # Use a file-handle to bypass that and write to the exact path we want.
    tmp = cache_path.parent / (cache_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.replace(cache_path)
    print(f"[pod_train] cached {len(arr):,} tokens -> {cache_path}")
    return arr


# ---------------------------------------------------------------------------
# Phase 4 — model build (fant3_50m + AEL memory)
# ---------------------------------------------------------------------------

def build_model(use_ael: bool = True, multi_gpu: bool = False):
    import torch
    from fant3.config import fant3_50m, fant3_10m
    from fant3.model import FANT3Model

    cfg = fant3_50m()
    if use_ael:
        cfg.ael_memory_enabled = True
        cfg.ael_gasket_depth = 5
    cfg.use_gradient_checkpointing = True

    # Precision choice: bf16 on Ampere+ (sm_80+, e.g. A100, RTX 30xx/40xx, A6000).
    # T4 / older Turing is sm_75 -- bf16 runs via fp32 emulation, much slower.
    # We pick bf16 by default; fp16 path is a future optimization.
    dtype = torch.bfloat16
    cc = torch.cuda.get_device_capability()
    if cc < (8, 0):
        print(f"[pod_train] WARN: GPU compute capability {cc} (<sm_80). "
              f"bf16 is emulated; throughput will be ~30-40% lower than ideal.")

    model = FANT3Model(cfg).to("cuda", dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    n_gpu = torch.cuda.device_count()
    print(f"[pod_train] model: fant3_50m{'+AEL' if use_ael else ''}  {n_params/1e6:.2f}M params  "
          f"({n_gpu} GPU{'s' if n_gpu != 1 else ''} visible)")

    if multi_gpu and n_gpu > 1:
        print(f"[pod_train] wrapping in DataParallel across {n_gpu} GPUs")
        model = torch.nn.DataParallel(model)
    return model, cfg, n_gpu


# ---------------------------------------------------------------------------
# Phase 5 — train loop with grad-accum, atomic checkpoints, fragmentation guard
# ---------------------------------------------------------------------------

def atomic_save(state: dict, path: Path) -> None:
    """Write to a temp file, then rename — checkpoint can never be half-written."""
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def sample_batch(tokens, batch_size, seq_len, device, rng):
    import numpy as np
    import torch
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[s : s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def train(
    tokens,
    opt_steps: int,
    accum_steps: int = 4,
    batch: int = 1,
    seq: int = 512,
    lr: float = 3e-4,
    warmup_opt: int = 100,
    weight_decay: float = 0.1,
    log_every: int = 25,
    save_every: int = 250,
    seed: int = 0,
    ckpt_dir: Path = POD_CKPT,
    use_ael: bool = True,
    resume: bool = True,
    multi_gpu: bool = False,
) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model, cfg, n_gpu = build_model(use_ael=use_ael, multi_gpu=multi_gpu)
    # When DataParallel is active, batches must split evenly across GPUs.
    # Bump physical batch to n_gpu if it was 1.
    if n_gpu > 1 and batch < n_gpu:
        print(f"[pod_train] bumping physical batch {batch} -> {n_gpu} so DataParallel can split")
        batch = n_gpu
    # Get the underlying model for state_dict ops (works for both DP-wrapped and bare).
    bare_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses: list[float] = []
    start_opt = 0
    final_ckpt = ckpt_dir / "final.pt"
    if resume and final_ckpt.exists():
        print(f"[pod_train] resuming from {final_ckpt}")
        ckpt = torch.load(final_ckpt, weights_only=False, map_location="cuda")
        bare_model.load_state_dict(ckpt["model_state_dict"])
        losses = list(ckpt.get("losses", []))
        start_opt = ckpt.get("step", 0)
        print(f"  resumed step={start_opt}  last-20 loss={sum(losses[-20:])/max(20,1):.3f}")
    else:
        print("[pod_train] no resume checkpoint; starting fresh")

    def get_lr(step):
        if step < warmup_opt:
            return lr * (step + 1) / warmup_opt
        progress = (step - warmup_opt) / max(1, opt_steps - warmup_opt)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))

    eff_batch = batch * accum_steps
    print(f"[pod_train] starting {opt_steps - start_opt} opt-steps "
          f"(physical batch={batch}, accum={accum_steps}, effective batch={eff_batch}, seq={seq})")
    t0 = time.time()
    model.train()
    nan_strikes = 0

    try:
        for opt_step in range(start_opt, opt_steps):
            cur_lr = get_lr(opt_step)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            opt.zero_grad()
            step_losses = []
            for _ in range(accum_steps):
                x, y = sample_batch(tokens, batch, seq, "cuda", rng)
                out = model(x); logits = out["logits"] if isinstance(out, dict) else out
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
                )
                if torch.isnan(loss):
                    nan_strikes += 1
                    if nan_strikes >= 3:
                        raise RuntimeError(f"NaN loss 3x in a row at step {opt_step}")
                    print(f"[pod_train] WARN: NaN loss at step {opt_step}, skipping micro-batch")
                    continue
                nan_strikes = 0
                (loss / accum_steps).backward()
                step_losses.append(float(loss.item()))

            if not step_losses:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            mean_loss = sum(step_losses) / len(step_losses)
            losses.append(mean_loss)

            if (opt_step + 1) % 25 == 0:
                torch.cuda.empty_cache()

            if (opt_step + 1) % log_every == 0:
                elapsed = time.time() - t0
                done = opt_step + 1 - start_opt
                tps = eff_batch * seq * done / elapsed
                window = min(log_every, len(losses))
                window_loss = sum(losses[-window:]) / window
                print(f"  opt_step {opt_step+1:6d}/{opt_steps}  "
                      f"loss={window_loss:.3f}  lr={cur_lr:.2e}  {tps:.0f} tok/s  "
                      f"elapsed={elapsed:.0f}s")

            if (opt_step + 1) % save_every == 0:
                atomic_save({
                    "step": opt_step + 1,
                    "model_state_dict": bare_model.state_dict(),
                    "losses": losses,
                    "effective_batch": eff_batch,
                }, final_ckpt)

    except KeyboardInterrupt:
        print("[pod_train] interrupted; saving emergency checkpoint")
        atomic_save({
            "step": opt_step + 1,
            "model_state_dict": bare_model.state_dict(),
            "losses": losses,
            "effective_batch": eff_batch,
        }, final_ckpt)
        raise
    except Exception:
        traceback.print_exc()
        try:
            atomic_save({
                "step": opt_step + 1,
                "model_state_dict": bare_model.state_dict(),
                "losses": losses,
                "effective_batch": eff_batch,
                "crashed": True,
            }, ckpt_dir / "crash.pt")
            print("[pod_train] crash checkpoint saved to crash.pt")
        except Exception:
            pass
        raise

    # Final clean save.
    atomic_save({
        "step": opt_steps,
        "model_state_dict": bare_model.state_dict(),
        "losses": losses,
        "effective_batch": eff_batch,
    }, final_ckpt)
    elapsed_total = time.time() - t0
    print(f"[pod_train] training complete: {opt_steps - start_opt} opt-steps in {elapsed_total:.0f}s")
    first20 = sum(losses[max(0, start_opt):max(0, start_opt) + 20]) / 20 if len(losses) >= 20 else float("nan")
    last20 = sum(losses[-20:]) / 20 if len(losses) >= 20 else float("nan")
    print(f"[pod_train] loss: first-20={first20:.3f}  last-20={last20:.3f}  reduction={first20-last20:.3f}")
    return {"final_loss": last20, "first_loss": first20, "elapsed": elapsed_total, "opt_steps": opt_steps}


# ---------------------------------------------------------------------------
# Phase 6 — optional shutdown (only on confirmed success)
# ---------------------------------------------------------------------------

def maybe_shutdown(shutdown: bool) -> None:
    if not shutdown:
        return
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("[pod_train] --shutdown set but RUNPOD_POD_ID env not present; skipping.")
        return
    if not shutil.which("runpodctl"):
        print("[pod_train] --shutdown set but runpodctl not on PATH; skipping.")
        return
    print(f"[pod_train] training succeeded; calling 'runpodctl stop pod {pod_id}' to halt billing")
    try:
        subprocess.run(["runpodctl", "stop", "pod", pod_id], check=True, timeout=30)
    except Exception as e:
        print(f"[pod_train] WARN: runpodctl stop failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps",        type=int,   default=5000,  help="total opt-steps")
    p.add_argument("--accum-steps",  type=int,   default=4)
    p.add_argument("--batch",        type=int,   default=1)
    p.add_argument("--seq",          type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--tokens",       type=int,   default=100_000_000, help="tokens to stream/cache")
    p.add_argument("--dataset",      type=str,   default="Skylion007/openwebtext")
    p.add_argument("--subset",       type=str,   default=None)
    p.add_argument("--no-ael",       action="store_true",  help="disable AEL memory module")
    p.add_argument("--multi-gpu",    action="store_true",  help="wrap model in DataParallel across all visible GPUs (Linux + multi-GPU only)")
    p.add_argument("--no-resume",    action="store_true",  help="ignore any existing checkpoint and start fresh")
    p.add_argument("--smoke",        action="store_true",  help="tiny dress-rehearsal run (~100 opt-steps, ~2M tokens)")
    p.add_argument("--shutdown",     action="store_true",  help="run 'runpodctl stop pod $RUNPOD_POD_ID' on success")
    args = p.parse_args()

    if args.smoke:
        args.steps = 100
        args.tokens = 2_000_000
        print("[pod_train] SMOKE mode: 100 opt-steps, 2M tokens")

    print(f"[pod_train] argv: {sys.argv}")
    print(f"[pod_train] args: {vars(args)}")

    try:
        ensure_deps()
        setup_paths()
        tokens = ensure_tokens(args.dataset, args.subset, args.tokens)
        result = train(
            tokens=tokens,
            opt_steps=args.steps,
            accum_steps=args.accum_steps,
            batch=args.batch,
            seq=args.seq,
            lr=args.lr,
            use_ael=not args.no_ael,
            resume=not args.no_resume,
            multi_gpu=args.multi_gpu,
        )
        # Dump result summary as JSON for downstream tooling.
        with open(POD_LOGS / "last_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"[pod_train] result: {result}")
        maybe_shutdown(args.shutdown)
        return 0
    except KeyboardInterrupt:
        print("[pod_train] interrupted by user")
        return 130
    except SystemExit as e:
        return int(e.code or 1)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
