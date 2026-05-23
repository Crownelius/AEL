# Pod runbook — RunPod / Kaggle / any fresh Linux box

This directory contains the self-contained training script we use when running
on rented GPU. The script `pod_train.py` is the ONLY thing you need to launch.

## Quick reference

```bash
# fresh pod / Linux box — exactly these 4 lines:
cd /workspace
git clone https://github.com/Crownelius/AEL.git
git clone https://github.com/Crownelius/fant3.git
cd AEL && python pod/pod_train.py --steps 25000 --tokens 200_000_000 --shutdown
```

That's it. Everything else (deps, data, checkpoints, logging, shutdown) is automatic.

## Common invocations

| Goal | Command |
|---|---|
| Smoke test (any box, ~3 min) | `python pod/pod_train.py --smoke --no-resume` |
| Local dress-rehearse | `python pod/pod_train.py --steps 150 --tokens 2_000_000` |
| Full RunPod run (auto-shutdown) | `python pod/pod_train.py --steps 25000 --tokens 200_000_000 --shutdown` |
| Disable AEL memory (control) | add `--no-ael` |
| Force fresh start | add `--no-resume` |
| Custom batch / accum / lr | `--batch 1 --accum-steps 8 --lr 3e-4` |

## What the script handles for you

1. **Dep install** — checks for torch / datasets / tokenizers / transformers / bitsandbytes and `pip install`s any missing.
2. **Token cache** — looks in `pod_data/`, then in legacy `data/`, then streams from HuggingFace. Streaming has 5-retry exponential backoff; resumes from the last good doc count.
3. **Model build** — fant3_50m + AEL memory, bf16, gradient-checkpointed, 8-bit AdamW.
4. **Atomic checkpoints** — every 250 opt-steps, written tmp + rename so a crash can't half-write `final.pt`. Crash also writes `pod_ckpt/crash.pt`.
5. **Resume** — on restart, automatically picks up `pod_ckpt/final.pt` and continues from the last opt-step.
6. **Fragmentation guard** — `torch.cuda.empty_cache()` every 25 opt-steps to prevent the Windows-style allocator-fragmentation OOM we saw on the RTX 3060.
7. **Full logging** — every line of stdout/stderr is mirrored to `pod_logs/{timestamp}.log`. No more lost training output to a `| tail`.
8. **Clean exit** — exit code 0 on success, 1 on error, 130 on Ctrl-C. JSON summary at `pod_logs/last_result.json`.
9. **Auto-shutdown** — `--shutdown` triggers `runpodctl stop pod $RUNPOD_POD_ID` if both the env var and the CLI are present. **Don't forget this on a paid pod.**

## Budget math for a $20 RunPod credit

| GPU | Hourly | $20 buys | Approx. tokens trained |
|---|---|---|---|
| RTX 4090 (24 GB) | ~$0.40 | 50 hr | ~3 B at our scale |
| A6000 (48 GB) | ~$0.50 | 40 hr | ~3 B (more VRAM headroom) |
| A100 40 GB | ~$1.20 | 16 hr | ~1.5 B |
| H100 | ~$3.50 | 5.7 hr | ~0.8 B |

Recommendation for our 50M model: **RTX 4090** is the sweet spot — cheapest per
hour, plenty of VRAM, and we don't need the A100's interconnect since we're
single-GPU. One ~24-hour run = $9-10, leaves cushion.

## Validation log (local dress-rehearsal)

```
SMOKE (100 opt-steps, 2M tokens, fresh init):
  loss 10.47 → 9.03 in 174s on RTX 3060, 1175 tok/sec, clean exit

RESUME (50 more opt-steps, same data):
  resumed at step=100 loss=9.03; trained to step=150 loss=7.75
  in 70s, 1457 tok/sec (cache hot), clean exit
```

If those numbers reproduce on a Linux pod with similar magnitude, the script
is working as intended.
