"""Iteration 1 evaluation: WordNet on the AEL gasket.

Metrics:
  sibling@k    -- fraction of true WordNet siblings in top-k AEL retrieval
  hypernym@k   -- fraction of true ancestor chain in top-k
  random@k     -- baseline: random retrieval on the same vocabulary

We compare Euclidean and graph-hop retrieval.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.embed import embed_wordnet_on_gasket
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.retrieval import wn_knn_euclidean, wn_knn_graph
from src.ael.wordnet_data import load_noun_subset


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def run(
    max_depth_wn: int = 6,
    max_nodes_wn: int = 1000,
    max_depth_gasket: int = 7,
    k: int = 10,
    sample_size: int = 200,
    seed: int = 0,
) -> None:
    print(f"Loading WordNet subset (depth<={max_depth_wn}, max {max_nodes_wn} nodes)...")
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    print(f"  loaded {len(sub)} synsets")

    print(f"Building gasket to depth {max_depth_gasket}...")
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    print(f"  {len(g.circles)} circles")

    if len(g.circles) < len(sub):
        print(f"  WARN: not enough circles ({len(g.circles)}) for synsets ({len(sub)})")

    print("Embedding WordNet onto gasket...")
    emb = embed_wordnet_on_gasket(sub, g, root_circle=1)
    embedded = len(emb.wn_to_circle)
    print(f"  embedded {embedded}/{len(sub)} synsets "
          f"({embedded/len(sub):.1%})")

    # Pick query nodes: must be embedded, must have >=2 siblings, must have an ancestor.
    candidates = [
        n for n in emb.wn_to_circle
        if len(sub.siblings(n)) >= 2 and sub.nodes[n].parent is not None
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating on {len(queries)} query synsets, k={k}")

    sib_p_euc = sib_r_euc = hyp_p_euc = 0.0
    sib_p_grf = sib_r_grf = hyp_p_grf = 0.0
    sib_p_rnd = sib_r_rnd = hyp_p_rnd = 0.0

    all_embedded = list(emb.wn_to_circle.keys())

    for q in queries:
        true_sibs = set(sub.siblings(q)) & set(emb.wn_to_circle)
        true_hyps = set(sub.hypernyms(q)) & set(emb.wn_to_circle)
        # We measure retrieval of (siblings) and (any ancestor) separately.

        euc_retr = [name for name, _ in wn_knn_euclidean(g, emb, q, k)]
        grf_retr = [name for name, _ in wn_knn_graph(g, emb, q, k)]
        rnd_retr = rng.sample([n for n in all_embedded if n != q], min(k, len(all_embedded) - 1))

        sib_p_euc += precision_at_k(euc_retr, true_sibs, k)
        sib_r_euc += recall_at_k(euc_retr, true_sibs, k)
        hyp_p_euc += precision_at_k(euc_retr, true_hyps, k)

        sib_p_grf += precision_at_k(grf_retr, true_sibs, k)
        sib_r_grf += recall_at_k(grf_retr, true_sibs, k)
        hyp_p_grf += precision_at_k(grf_retr, true_hyps, k)

        sib_p_rnd += precision_at_k(rnd_retr, true_sibs, k)
        sib_r_rnd += recall_at_k(rnd_retr, true_sibs, k)
        hyp_p_rnd += precision_at_k(rnd_retr, true_hyps, k)

    n = len(queries)
    print("\n" + "=" * 70)
    print(f"Method        sibling-P@{k}  sibling-R@{k}  hypernym-P@{k}")
    print("-" * 70)
    print(f"AEL euclidean  {sib_p_euc/n:>10.4f}    {sib_r_euc/n:>10.4f}     {hyp_p_euc/n:>10.4f}")
    print(f"AEL graph-hop  {sib_p_grf/n:>10.4f}    {sib_r_grf/n:>10.4f}     {hyp_p_grf/n:>10.4f}")
    print(f"random         {sib_p_rnd/n:>10.4f}    {sib_r_rnd/n:>10.4f}     {hyp_p_rnd/n:>10.4f}")
    print("=" * 70)

    # Qualitative samples.
    print("\nQualitative samples (Euclidean, top-5):")
    for q in queries[:5]:
        retr = wn_knn_euclidean(g, emb, q, 5)
        true_sibs = set(sub.siblings(q))
        marked = [f"{name}{'*' if name in true_sibs else ''}" for name, _ in retr]
        parent = sub.nodes[q].parent or "<root>"
        print(f"  {q}  (parent: {parent})")
        print(f"    -> {marked}")
    print("  (* = true sibling)")


if __name__ == "__main__":
    run()
