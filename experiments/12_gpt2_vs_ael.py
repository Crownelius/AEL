"""Head-to-head: GPT-2 small (117M) vs AEL+WordNet (10M).

Same benchmark, same scoring. Reports per-category accuracy and overall.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.benchmark import build_benchmark
from src.ael.benchmark_eval import ael_answer, evaluate, gpt2_answer


def main():
    print("Building benchmark...")
    items = build_benchmark(seed=42)
    print(f"  {len(items)} items")
    cat_counts = {}
    for it in items:
        cat_counts[it.category] = cat_counts.get(it.category, 0) + 1
    print(f"  per category: {cat_counts}")

    # --- AEL ---
    print("\nRunning AEL WordNet QA...")
    ael = evaluate(items, lambda q: ael_answer(q), "AEL+WordNet (10M)", verbose=True)

    # --- GPT-2 small ---
    print("\nLoading GPT-2 small (117M) — first run downloads ~500MB...")
    gpt2 = evaluate(items, lambda q: gpt2_answer(q, model_name="gpt2"), "GPT-2 small (117M)", verbose=True)

    # Comparison table
    print("\n" + "=" * 70)
    print(f"{'Category':<14}{'AEL':>10}{'GPT-2':>10}{'Δ':>10}")
    print("-" * 70)
    for c in ("define", "hypernym", "hyponym", "is_a"):
        a = ael["per_cat"][c]
        g = gpt2["per_cat"][c]
        d = a - g
        winner = "★" if d > 0 else " "
        print(f"{c:<14}{a:>10.3f}{g:>10.3f}{d:>+10.3f}  {winner}")
    print("-" * 70)
    a, g = ael["overall"], gpt2["overall"]
    print(f"{'overall':<14}{a:>10.3f}{g:>10.3f}{a-g:>+10.3f}  {'★ AEL WINS' if a > g else '  '}")
    print("=" * 70)

    # Show 3 samples from each category for AEL.
    print("\nSamples (AEL — Q | answer | grade | gold):")
    for c, items3 in ael["samples"].items():
        print(f"\n  [{c}]")
        for q, gen, ok, gold in items3:
            tick = "✓" if ok else "✗"
            print(f"    {tick} Q: {q}")
            print(f"       A: {gen[:80]}")
            print(f"       gold: {gold}")

    print("\nSamples (GPT-2 — Q | answer | grade | gold):")
    for c, items3 in gpt2["samples"].items():
        print(f"\n  [{c}]")
        for q, gen, ok, gold in items3:
            tick = "✓" if ok else "✗"
            print(f"    {tick} Q: {q}")
            print(f"       A: {gen[:80]}")
            print(f"       gold: {gold}")


if __name__ == "__main__":
    main()
