"""Upper-half-space H^3 lift + cone attention vs Poincaré.

The corrected 3D embedding: each circle = (x, y, r) in UHS.
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim.models.poincare import PoincareModel

from src.ael.cone_uhs import ConeUHS, basis_axes_3d, multihead_score, six_axes_3d
from src.ael.embed_v2 import embed_wordnet_on_gasket_v2
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.retrieval import euclidean_knn
from src.ael.uhs import all_uhs_points, uhs_dist, uhs_log
from src.ael.wordnet_data import load_noun_subset


def p_at_k(retr, rel, k):
    if not rel: return float("nan")
    top = retr[:k]
    if not top: return 0.0
    return sum(1 for r in top if r in rel) / k


def r_at_k(retr, rel, k):
    if not rel: return float("nan")
    return sum(1 for r in retr[:k] if r in rel) / len(rel)


def uhs_dist_retrieve(points, emb, q, k):
    qidx = emb.wn_to_circle[q]
    apex = points[qidx]
    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q:
            continue
        scored.append((uhs_dist(apex, points[cidx]), synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def cone_uhs_omni(points, emb, q, k, aperture, sigma):
    qidx = emb.wn_to_circle[q]
    apex = points[qidx]
    cones = [ConeUHS(apex=apex, axis=a, aperture=aperture, sigma=sigma) for a in six_axes_3d()]
    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q: continue
        w = multihead_score(cones, points[cidx], reduce="max")
        scored.append((-w, synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def cone_uhs_to_parent(points, emb, sub, q, k, aperture, sigma):
    qidx = emb.wn_to_circle[q]
    apex = points[qidx]
    parent = sub.nodes[q].parent
    if parent is None or parent not in emb.wn_to_circle:
        return cone_uhs_omni(points, emb, q, k, aperture, sigma)
    target = points[emb.wn_to_circle[parent]]
    axis = uhs_log(apex, target)
    if float(np.linalg.norm(axis)) < 1e-9:
        return cone_uhs_omni(points, emb, q, k, aperture, sigma)
    cone = ConeUHS(apex=apex, axis=axis, aperture=aperture, sigma=sigma)
    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q: continue
        w = cone.weight(points[cidx])
        scored.append((-w, synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def estimate_sigma(points, indices, n=100):
    sample = random.sample(indices, min(len(indices), n))
    dists = [uhs_dist(points[sample[i]], points[sample[j]])
             for i in range(len(sample)) for j in range(i + 1, len(sample))]
    dists.sort()
    return dists[len(dists) // 2] if dists else 1.0


def run(max_depth_wn=6, max_nodes_wn=1000, max_depth_gasket=7,
        k=10, sample_size=150, seed=0):
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    emb = embed_wordnet_on_gasket_v2(sub, g)
    print(f"  {len(g.circles)} circles, {len(emb.wn_to_circle)} embedded")
    points = all_uhs_points(g)
    sigma = estimate_sigma(points, list(emb.circle_to_wn.keys()))
    print(f"  median UHS distance: {sigma:.4f}")

    print("Training Poincaré...")
    edges = [(n, node.parent) for n, node in sub.nodes.items() if node.parent]
    pmodel = PoincareModel(edges, size=10, negative=10)
    pmodel.train(epochs=100, print_every=50)

    vocab = set(emb.wn_to_circle) & set(pmodel.kv.key_to_index)
    candidates = [n for n in vocab
                  if len(set(sub.siblings(n)) & vocab) >= 2
                  and sub.nodes[n].parent in vocab]
    rng = random.Random(seed); rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating {len(queries)} queries, k={k}")

    methods = [
        ("AEL 2D euclidean",        lambda q: [emb.circle_to_wn[j] for j, _ in
                                                euclidean_knn(g, emb.wn_to_circle[q], k,
                                                              restrict_to=set(emb.circle_to_wn.keys()))]),
        ("AEL UHS distance",        lambda q: uhs_dist_retrieve(points, emb, q, k)),
        ("AEL UHS cone (omni 0.4)", lambda q: cone_uhs_omni(points, emb, q, k, 0.4, sigma)),
        ("AEL UHS cone (omni 1.5)", lambda q: cone_uhs_omni(points, emb, q, k, 1.5, sigma)),
        ("AEL UHS cone (->p 0.3)",  lambda q: cone_uhs_to_parent(points, emb, sub, q, k, 0.3, sigma * 2)),
        ("AEL UHS cone (->p 0.8)",  lambda q: cone_uhs_to_parent(points, emb, sub, q, k, 0.8, sigma * 2)),
        ("AEL UHS cone (->p 1.5)",  lambda q: cone_uhs_to_parent(points, emb, sub, q, k, 1.5, sigma * 2)),
        ("Poincaré (10d)",          lambda q: [n for n, _ in pmodel.kv.most_similar(q, topn=len(vocab))][:k]),
    ]
    totals = {name: {"sp": 0.0, "sr": 0.0, "hp": 0.0} for name, _ in methods}

    for q in queries:
        true_sibs = set(sub.siblings(q)) & vocab
        true_hyps = set(sub.hypernyms(q)) & vocab
        for name, fn in methods:
            retr = [r for r in fn(q) if r in vocab]
            totals[name]["sp"] += p_at_k(retr, true_sibs, k)
            totals[name]["sr"] += r_at_k(retr, true_sibs, k)
            totals[name]["hp"] += p_at_k(retr, true_hyps, k)

    n = len(queries)
    print("\n" + "=" * 78)
    print(f"Method                          sibling-P@{k}  sibling-R@{k}  hypernym-P@{k}")
    print("-" * 78)
    for name, _ in methods:
        m = totals[name]
        print(f"{name:<30}  {m['sp']/n:>11.4f}    {m['sr']/n:>11.4f}     {m['hp']/n:>11.4f}")
    print("=" * 78)


if __name__ == "__main__":
    run()
