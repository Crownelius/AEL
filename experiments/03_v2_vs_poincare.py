"""Iteration-2 evaluation: AEL v2 (region + curvature-budget) vs Poincaré.

Uses identical query set & vocab to experiment 02 for direct comparison.
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim.models.poincare import PoincareModel

from src.ael.embed import embed_wordnet_on_gasket
from src.ael.embed_v2 import embed_wordnet_on_gasket_v2
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.retrieval import euclidean_knn, graph_knn
from src.ael.wordnet_data import load_noun_subset


def precision_at_k(retr, rel, k):
    if not rel:
        return float("nan")
    top = retr[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in rel) / k


def recall_at_k(retr, rel, k):
    if not rel:
        return float("nan")
    return sum(1 for r in retr[:k] if r in rel) / len(rel)


def ael_retrieve(g, emb, q, k, method):
    qidx = emb.wn_to_circle[q]
    pool = set(emb.circle_to_wn.keys())
    if method == "euclidean":
        hits = euclidean_knn(g, qidx, k, restrict_to=pool)
    else:
        hits = graph_knn(g, qidx, k, restrict_to=pool)
    return [emb.circle_to_wn[j] for j, _ in hits]


def poincare_retrieve(model, q, k, vocab):
    if q not in model.kv:
        return []
    sims = model.kv.most_similar(q, topn=len(vocab))
    return [n for n, _ in sims if n in vocab][:k]


def run(
    max_depth_wn=6,
    max_nodes_wn=1000,
    max_depth_gasket=7,
    k=10,
    sample_size=200,
    poincare_dim=10,
    poincare_epochs=100,
    seed=0,
):
    print("Loading WordNet...")
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    print(f"  {len(sub)} synsets")

    print("Building gasket...")
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    print(f"  {len(g.circles)} circles")

    print("Embedding v1 (BFS first-available)...")
    emb_v1 = embed_wordnet_on_gasket(sub, g, root_circle=1)
    print(f"  v1: {len(emb_v1.wn_to_circle)} embedded")

    print("Embedding v2 (region + curvature-budget)...")
    emb_v2 = embed_wordnet_on_gasket_v2(sub, g)
    print(f"  v2: {len(emb_v2.wn_to_circle)} embedded")

    print("Training Poincaré baseline...")
    edges = [(n, node.parent) for n, node in sub.nodes.items() if node.parent]
    pmodel = PoincareModel(edges, size=poincare_dim, negative=10)
    pmodel.train(epochs=poincare_epochs, print_every=50)

    # Common vocab.
    vocab = (
        set(emb_v1.wn_to_circle)
        & set(emb_v2.wn_to_circle)
        & set(pmodel.kv.key_to_index)
    )
    candidates = [
        n for n in vocab
        if len(set(sub.siblings(n)) & vocab) >= 2 and sub.nodes[n].parent
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating {len(queries)} queries on vocab of {len(vocab)}, k={k}")

    methods = [
        ("AEL v1 euclidean", lambda q: ael_retrieve(g, emb_v1, q, k, "euclidean")),
        ("AEL v1 graph-hop", lambda q: ael_retrieve(g, emb_v1, q, k, "graph")),
        ("AEL v2 euclidean", lambda q: ael_retrieve(g, emb_v2, q, k, "euclidean")),
        ("AEL v2 graph-hop", lambda q: ael_retrieve(g, emb_v2, q, k, "graph")),
        ("Poincaré",         lambda q: poincare_retrieve(pmodel, q, k, vocab)),
        ("random",           None),
    ]

    totals = {name: {"sp": 0.0, "sr": 0.0, "hp": 0.0} for name, _ in methods}
    vocab_list = list(vocab)

    for q in queries:
        true_sibs = set(sub.siblings(q)) & vocab
        true_hyps = set(sub.hypernyms(q)) & vocab
        for name, fn in methods:
            if name == "random":
                retr = rng.sample([v for v in vocab_list if v != q], min(k, len(vocab_list) - 1))
            else:
                retr = fn(q)
            totals[name]["sp"] += precision_at_k(retr, true_sibs, k)
            totals[name]["sr"] += recall_at_k(retr, true_sibs, k)
            totals[name]["hp"] += precision_at_k(retr, true_hyps, k)

    n = len(queries)
    print("\n" + "=" * 78)
    print(f"Method              sibling-P@{k}  sibling-R@{k}  hypernym-P@{k}")
    print("-" * 78)
    for name, _ in methods:
        m = totals[name]
        print(f"{name:<18}  {m['sp']/n:>11.4f}    {m['sr']/n:>11.4f}     {m['hp']/n:>11.4f}")
    print("=" * 78)


if __name__ == "__main__":
    run()
