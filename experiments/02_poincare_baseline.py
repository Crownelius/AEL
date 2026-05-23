"""Poincaré baseline: train Nickel & Kiela embeddings on the same WordNet subset,
evaluate with identical metrics, compare to AEL.
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
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.retrieval import wn_knn_euclidean, wn_knn_graph
from src.ael.wordnet_data import load_noun_subset


def precision_at_k(retrieved, relevant, k):
    if not relevant:
        return float("nan")
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return float("nan")
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def build_poincare(sub, dim=10, epochs=50):
    """Train Poincaré embeddings on hypernym edges of the subset."""
    edges = []
    for name, node in sub.nodes.items():
        if node.parent is not None and node.parent in sub.nodes:
            # Convention: (hyponym, hypernym) so the model learns 'is-a'.
            edges.append((name, node.parent))
    print(f"  Poincaré training: {len(edges)} hypernym edges, dim={dim}, epochs={epochs}")
    model = PoincareModel(edges, size=dim, negative=10)
    model.train(epochs=epochs, print_every=50)
    return model


def poincare_knn(model, query, k, vocab):
    """Top-k by Poincaré distance in the trained model."""
    if query not in model.kv:
        return []
    # gensim's most_similar uses hyperbolic distance.
    sims = model.kv.most_similar(query, topn=len(vocab))
    return [name for name, _ in sims[:k] if name in vocab][:k]


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

    print("Embedding WordNet onto gasket...")
    emb = embed_wordnet_on_gasket(sub, g, root_circle=1)
    print(f"  {len(emb.wn_to_circle)} embedded")

    print("Training Poincaré baseline...")
    pmodel = build_poincare(sub, dim=poincare_dim, epochs=poincare_epochs)

    # Build query set common to both.
    candidates = [
        n for n in emb.wn_to_circle
        if n in pmodel.kv
        and len(sub.siblings(n)) >= 2
        and sub.nodes[n].parent is not None
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    queries = candidates[:sample_size]
    vocab = set(emb.wn_to_circle.keys()) & set(pmodel.kv.key_to_index)
    print(f"\nEvaluating {len(queries)} queries, vocab size {len(vocab)}, k={k}")

    metrics = {
        "AEL euclidean": {"sib_p": 0, "sib_r": 0, "hyp_p": 0},
        "AEL graph-hop": {"sib_p": 0, "sib_r": 0, "hyp_p": 0},
        "Poincaré":      {"sib_p": 0, "sib_r": 0, "hyp_p": 0},
        "random":        {"sib_p": 0, "sib_r": 0, "hyp_p": 0},
    }

    vocab_list = list(vocab)

    for q in queries:
        true_sibs = set(sub.siblings(q)) & vocab
        true_hyps = set(sub.hypernyms(q)) & vocab

        euc = [n for n, _ in wn_knn_euclidean(g, emb, q, k)]
        grf = [n for n, _ in wn_knn_graph(g, emb, q, k)]
        poi = poincare_knn(pmodel, q, k, vocab)
        rnd = rng.sample([n for n in vocab_list if n != q], min(k, len(vocab_list) - 1))

        for name, retr in [
            ("AEL euclidean", euc),
            ("AEL graph-hop", grf),
            ("Poincaré", poi),
            ("random", rnd),
        ]:
            metrics[name]["sib_p"] += precision_at_k(retr, true_sibs, k)
            metrics[name]["sib_r"] += recall_at_k(retr, true_sibs, k)
            metrics[name]["hyp_p"] += precision_at_k(retr, true_hyps, k)

    n = len(queries)
    print("\n" + "=" * 75)
    print(f"Method          sibling-P@{k}  sibling-R@{k}  hypernym-P@{k}")
    print("-" * 75)
    for name in ["AEL euclidean", "AEL graph-hop", "Poincaré", "random"]:
        m = metrics[name]
        print(f"{name:<14}  {m['sib_p']/n:>11.4f}    {m['sib_r']/n:>11.4f}     {m['hyp_p']/n:>11.4f}")
    print("=" * 75)


if __name__ == "__main__":
    run()
