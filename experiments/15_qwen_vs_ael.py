"""Triple comparison: AEL vs GPT-2 vs Qwen2.5-1.5B on the broad benchmark.

This is the head-to-head against the user's target competitor.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.benchmark_broad import build_broad_benchmark
from src.ael.benchmark_eval import ael_answer, hf_answer


def _norm(s):
    return s.lower().strip()


def score(item, gen):
    g = _norm(gen)
    return any(_norm(acc) in g for acc in item.accepted)


def evaluate(items, fn, name):
    t0 = time.time()
    per_cat = {}
    total_correct = 0
    samples = {}
    for it in items:
        gen = fn(it.question)
        ok = score(it, gen)
        total_correct += int(ok)
        per_cat.setdefault(it.category, [0, 0])
        per_cat[it.category][0] += int(ok)
        per_cat[it.category][1] += 1
        if it.category in ("trivia", "complete") and len(samples.get(it.category, [])) < 4:
            samples.setdefault(it.category, []).append((it.question, gen, ok, ", ".join(sorted(it.accepted))[:40]))
    elapsed = time.time() - t0
    return {
        "name": name,
        "overall": total_correct / len(items),
        "per_cat": {c: h/t for c, (h, t) in per_cat.items()},
        "samples": samples,
        "elapsed": elapsed,
        "n": len(items),
    }


def main():
    print("Building broad benchmark...")
    items = build_broad_benchmark()
    print(f"  {len(items)} items")

    print("\nRunning AEL...")
    ael_r = evaluate(items, ael_answer, "AEL+WordNet (10M)")
    print(f"  overall {ael_r['overall']:.3f}  ({ael_r['elapsed']:.1f}s)")

    print("\nRunning GPT-2 small...")
    gpt2_r = evaluate(items, lambda q: hf_answer(q, "gpt2"), "GPT-2 small (117M)")
    print(f"  overall {gpt2_r['overall']:.3f}  ({gpt2_r['elapsed']:.1f}s)")

    print("\nRunning Qwen2.5-1.5B (loaded from cache)...")
    qwen_r = evaluate(items, lambda q: hf_answer(q, "Qwen/Qwen2.5-1.5B"), "Qwen2.5-1.5B (1.5B)")
    print(f"  overall {qwen_r['overall']:.3f}  ({qwen_r['elapsed']:.1f}s)")

    bucket_names = {
        "wn":       "WordNet QA",
        "trivia":   "Open-domain trivia",
        "complete": "Sentence completion",
    }
    print("\n" + "=" * 86)
    print(f"{'Bucket':<26}{'AEL':>10}{'GPT-2':>10}{'Qwen2.5':>10}{'AEL-vs-Qwen':>14}")
    print("-" * 86)
    for c in ("wn", "trivia", "complete"):
        a = ael_r["per_cat"].get(c, 0)
        g = gpt2_r["per_cat"].get(c, 0)
        q = qwen_r["per_cat"].get(c, 0)
        delta = a - q
        winner = "★ AEL" if delta > 0.02 else ("★ Qwen" if delta < -0.02 else "  tie")
        print(f"{bucket_names[c]:<26}{a:>10.3f}{g:>10.3f}{q:>10.3f}{delta:>+10.3f}  {winner}")
    print("-" * 86)
    a, g, q = ael_r["overall"], gpt2_r["overall"], qwen_r["overall"]
    overall_winner = "★ AEL" if a > q else ("★ Qwen" if q > a else "  tie")
    print(f"{'overall':<26}{a:>10.3f}{g:>10.3f}{q:>10.3f}{a-q:>+10.3f}  {overall_winner}")
    print("=" * 86)

    print(f"\nParam counts:")
    print(f"  AEL (effective)   : ~10M")
    print(f"  GPT-2 small       : 117M  ({117/10}x AEL)")
    print(f"  Qwen2.5-1.5B      : 1,544M ({1544/10}x AEL)")

    # Trivia + completion samples for Qwen.
    print("\nQwen trivia samples:")
    for q, gen, ok, gold in qwen_r["samples"].get("trivia", [])[:4]:
        print(f"  {'✓' if ok else '✗'} Q: {q}")
        print(f"      A: {gen[:80]}   (gold: {gold})")
    print("\nQwen completion samples:")
    for q, gen, ok, gold in qwen_r["samples"].get("complete", [])[:4]:
        print(f"  {'✓' if ok else '✗'} Q: {q}")
        print(f"      A: {gen[:80]}   (gold: {gold})")


if __name__ == "__main__":
    main()
