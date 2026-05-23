"""Phase A finale: 3D cone-on-hyperboloid vs Poincaré.

Lift each gasket circle to its Descartes 4-vector, project to H^3, build a
cone at each query with axis = log_u(parent_circle). Same metrics as before.
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

from src.ael.cone_3d import Cone3D, default_basis_axes, make_multihead_3d, multihead_score
from src.ael.descartes_3d import HyperboloidPoint, hyper_dist, hyper_log, to_hyperboloid
from src.ael.embed_v2 import embed_wordnet_on_gasket_v2
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.lift import all_hyperboloid_points
from src.ael.retrieval import euclidean_knn
from src.ael.wordnet_data import load_noun_subset


def p_at_k(retr, rel, k):
    if not rel:
        return float("nan")
    top = retr[:k]
    if not top: return 0.0
    return sum(1 for r in top if r in rel) / k


def r_at_k(retr, rel, k):
    if not rel:
        return float("nan")
    return sum(1 for r in retr[:k] if r in rel) / len(rel)


def cone3d_axis_to_parent(
    points: list[HyperboloidPoint],
    emb,
    sub,
    q: str,
    k: int,
    aperture: float,
    sigma: float,
):
    """Retrieve top-k by 3D cone whose axis points from query toward parent."""
    qidx = emb.wn_to_circle[q]
    apex = points[qidx]

    parent_name = sub.nodes[q].parent
    if parent_name is None or parent_name not in emb.wn_to_circle:
        # No parent -> fall back to omnidirectional multi-axis cone.
        axes = default_basis_axes(apex)
        cones = make_multihead_3d(apex, axes, aperture=aperture, sigma=sigma)
    else:
        pidx = emb.wn_to_circle[parent_name]
        target = points[pidx]
        axis = hyper_log(apex, target)
        if float(np.dot(axis, axis)) < 1e-12:
            axes = default_basis_axes(apex)
            cones = make_multihead_3d(apex, axes, aperture=aperture, sigma=sigma)
        else:
            cones = [Cone3D(apex=apex, axis=axis, aperture=aperture, sigma=sigma)]

    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q:
            continue
        w = multihead_score(cones, points[cidx], reduce="max")
        scored.append((-w, synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def cone3d_omnidirectional(
    points: list[HyperboloidPoint],
    emb,
    q: str,
    k: int,
    aperture: float,
    sigma: float,
):
    """Multi-head 3D cone with three orthogonal axes -- no parent hint."""
    qidx = emb.wn_to_circle[q]
    apex = points[qidx]
    axes = default_basis_axes(apex)
    cones = make_multihead_3d(apex, axes, aperture=aperture, sigma=sigma)
    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q:
            continue
        w = multihead_score(cones, points[cidx], reduce="max")
        scored.append((-w, synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def hyper_dist_retrieve(points, emb, q, k):
    """Pure geodesic distance retrieval on the hyperboloid (no cone)."""
    qidx = emb.wn_to_circle[q]
    apex = points[qidx]
    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q:
            continue
        d = hyper_dist(apex, points[cidx])
        scored.append((d, synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def estimate_hyper_sigma(points, indices):
    sample = random.sample(indices, min(len(indices), 100))
    dists = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dists.append(hyper_dist(points[sample[i]], points[sample[j]]))
    dists.sort()
    return dists[len(dists) // 2] if dists else 1.0


def run(
    max_depth_wn=6,
    max_nodes_wn=1000,
    max_depth_gasket=7,
    k=10,
    sample_size=150,
    seed=0,
):
    print("Loading WordNet + gasket + v2 embedding...")
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    emb = embed_wordnet_on_gasket_v2(sub, g)
    print(f"  {len(g.circles)} circles, {len(emb.wn_to_circle)} embedded")

    print("Lifting all circles to H^3...")
    points = all_hyperboloid_points(g)
    indices = list(emb.circle_to_wn.keys())
    sigma = estimate_hyper_sigma(points, indices)
    print(f"  median pairwise hyperbolic distance: {sigma:.4f}")

    print("Training Poincaré baseline...")
    edges = [(n, node.parent) for n, node in sub.nodes.items() if node.parent]
    pmodel = PoincareModel(edges, size=10, negative=10)
    pmodel.train(epochs=100, print_every=50)

    vocab = set(emb.wn_to_circle) & set(pmodel.kv.key_to_index)
    candidates = [
        n for n in vocab
        if len(set(sub.siblings(n)) & vocab) >= 2 and sub.nodes[n].parent in vocab
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating {len(queries)} queries, k={k}")

    methods = [
        ("AEL 2D euclidean",         lambda q: [n for n, _ in euclidean_knn(
            g, emb.wn_to_circle[q], k,
            restrict_to=set(emb.circle_to_wn.keys()))]),
        ("AEL 3D geodesic",          lambda q: hyper_dist_retrieve(points, emb, q, k)),
        ("AEL 3D cone (omni, 0.4)",  lambda q: cone3d_omnidirectional(points, emb, q, k, 0.4, sigma)),
        ("AEL 3D cone (omni, 1.5)",  lambda q: cone3d_omnidirectional(points, emb, q, k, 1.5, sigma)),
        ("AEL 3D cone (->parent,0.3)", lambda q: cone3d_axis_to_parent(points, emb, sub, q, k, 0.3, sigma * 2)),
        ("AEL 3D cone (->parent,0.8)", lambda q: cone3d_axis_to_parent(points, emb, sub, q, k, 0.8, sigma * 2)),
        ("AEL 3D cone (->parent,1.5)", lambda q: cone3d_axis_to_parent(points, emb, sub, q, k, 1.5, sigma * 2)),
        ("Poincaré (10d)",           lambda q: [n for n, _ in pmodel.kv.most_similar(q, topn=len(vocab))][:k]),
    ]

    totals = {name: {"sp": 0.0, "sr": 0.0, "hp": 0.0} for name, _ in methods}

    for q in queries:
        # Map 2D euclidean result-indices back to synsets.
        true_sibs = set(sub.siblings(q)) & vocab
        true_hyps = set(sub.hypernyms(q)) & vocab

        for name, fn in methods:
            retr_raw = fn(q)
            if name == "AEL 2D euclidean":
                retr = [emb.circle_to_wn[j] for j in retr_raw if j in emb.circle_to_wn]
            else:
                retr = [r for r in retr_raw if r in vocab]
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
