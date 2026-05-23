"""Broader benchmark: WordNet QA + open-domain trivia + sentence completion.

Same scoring rule for both systems: case-insensitive substring match against
the accepted answer set. The trick item (capital of the moon) is special-cased.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.benchmark_broad import BroadItem, build_broad_benchmark
from src.ael.benchmark_eval import ael_answer, gpt2_answer


def _norm(s: str) -> str:
    return s.lower().strip()


def score(item: BroadItem, generated: str) -> bool:
    g = _norm(generated)
    for acc in item.accepted:
        if _norm(acc) in g:
            return True
    return False


def evaluate(items, fn, name, verbose=True):
    per_cat = {}  # (category, subcategory) -> [hits, total]
    overall_correct = 0
    samples = {}
    for i, it in enumerate(items):
        gen = fn(it.question)
        ok = score(it, gen)
        overall_correct += int(ok)
        key = (it.category, it.subcategory)
        per_cat.setdefault(key, [0, 0])
        per_cat[key][0] += int(ok)
        per_cat[key][1] += 1
        if it.category in ("trivia", "complete") and len(samples.get(it.category, [])) < 5:
            samples.setdefault(it.category, []).append((it.question, gen, ok, ", ".join(sorted(it.accepted))[:40]))
    out = {
        "name": name,
        "overall": overall_correct / len(items),
        "per_cat": per_cat,
        "samples": samples,
        "n": len(items),
    }
    if verbose:
        print(f"\n  {name}  overall: {overall_correct}/{len(items)}  ({out['overall']:.3f})")
    return out


def main():
    print("Building broad benchmark (WordNet + trivia + completion)...")
    items = build_broad_benchmark()
    print(f"  total items: {len(items)}")
    cat_counts = {}
    for it in items:
        cat_counts[it.category] = cat_counts.get(it.category, 0) + 1
    for c, n in cat_counts.items():
        print(f"    {c}: {n}")

    print("\nRunning AEL...")
    ael_r = evaluate(items, ael_answer, "AEL+WordNet (10M effective)")

    print("\nRunning GPT-2 small (117M)...")
    gpt2_r = evaluate(items, lambda q: gpt2_answer(q, "gpt2"), "GPT-2 small (117M)")

    # Grouped comparison
    print("\n" + "=" * 78)
    print(f"{'Bucket':<30}{'AEL':>12}{'GPT-2':>12}{'Δ':>12}  winner")
    print("-" * 78)
    buckets = {}
    for (c, s), [h, t] in ael_r["per_cat"].items():
        buckets.setdefault(c, [0, 0])
        buckets[c][0] += h
        buckets[c][1] += t
    g_buckets = {}
    for (c, s), [h, t] in gpt2_r["per_cat"].items():
        g_buckets.setdefault(c, [0, 0])
        g_buckets[c][0] += h
        g_buckets[c][1] += t
    bucket_names = {"wn": "WordNet QA (AEL home)", "trivia": "Open-domain trivia", "complete": "Sentence completion (LM)"}
    for c in ("wn", "trivia", "complete"):
        ah, at = buckets.get(c, [0, 0])
        gh, gt = g_buckets.get(c, [0, 0])
        a = ah / at if at else 0
        g = gh / gt if gt else 0
        d = a - g
        winner = "★ AEL" if d > 0.02 else ("★ GPT-2" if d < -0.02 else "  tie")
        print(f"{bucket_names[c]:<30}{a:>11.3f} {g:>11.3f} {d:>+11.3f}  {winner}")
    print("-" * 78)
    a = ael_r["overall"]; g = gpt2_r["overall"]
    winner = "★ AEL WINS OVERALL" if a > g else ("★ GPT-2 WINS OVERALL" if g > a else "tie")
    print(f"{'overall':<30}{a:>11.3f} {g:>11.3f} {a-g:>+11.3f}  {winner}")
    print("=" * 78)

    # Samples from trivia + completion (where AEL is most likely to lose).
    print("\nTrivia samples (AEL):")
    for q, gen, ok, gold in ael_r["samples"].get("trivia", [])[:5]:
        print(f"  {'✓' if ok else '✗'} Q: {q}")
        print(f"      A: {gen[:60]}   (gold: {gold})")
    print("\nTrivia samples (GPT-2):")
    for q, gen, ok, gold in gpt2_r["samples"].get("trivia", [])[:5]:
        print(f"  {'✓' if ok else '✗'} Q: {q}")
        print(f"      A: {gen[:60]}   (gold: {gold})")
    print("\nCompletion samples (AEL):")
    for q, gen, ok, gold in ael_r["samples"].get("complete", [])[:5]:
        print(f"  {'✓' if ok else '✗'} Q: {q}")
        print(f"      A: {gen[:60]}   (gold: {gold})")
    print("\nCompletion samples (GPT-2):")
    for q, gen, ok, gold in gpt2_r["samples"].get("complete", [])[:5]:
        print(f"  {'✓' if ok else '✗'} Q: {q}")
        print(f"      A: {gen[:60]}   (gold: {gold})")


if __name__ == "__main__":
    main()
