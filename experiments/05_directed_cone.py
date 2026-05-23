"""Directed cone: axis points from query toward its hypernym (parent).

If the gasket placement has *any* coherent axial structure (children of a node
roughly opposite from that node's parent direction), a narrow cone pointed
away-from-parent should isolate siblings, while a wide cone should pull in
the hypernym chain.

We measure this directly: for each query, set the axis to the unit vector
from the query's circle toward its parent's circle. Then retrieval with this
axis-pointed cone is compared to omnidirectional and to plain euclidean k-NN.
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.cone import Cone, axis_to
from src.ael.embed_v2 import embed_wordnet_on_gasket_v2
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.retrieval import euclidean_knn
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


def directed_cone_retrieve(g, emb, q, k, axis: complex, aperture: float, sigma: float):
    qidx = emb.wn_to_circle[q]
    apex = g.circles[qidx].z
    cone = Cone(apex=apex, axis=axis, aperture=aperture, sigma=sigma)
    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q:
            continue
        w = cone.weight(g.circles[cidx].z)
        scored.append((-w, synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def estimate_sigma(g, emb):
    idxs = list(emb.circle_to_wn.keys())[:200]
    dists = []
    for i in range(len(idxs)):
        zi = g.circles[idxs[i]].z
        for j in range(i + 1, len(idxs)):
            dists.append(abs(g.circles[idxs[j]].z - zi))
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
    print("Setting up...")
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    emb = embed_wordnet_on_gasket_v2(sub, g)
    sigma = estimate_sigma(g, emb)
    print(f"  {len(emb.wn_to_circle)} embedded, sigma={sigma:.4f}")

    vocab = set(emb.wn_to_circle)
    candidates = [
        n for n in vocab
        if len(set(sub.siblings(n)) & vocab) >= 2
        and sub.nodes[n].parent in vocab
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\n{len(queries)} queries, k={k}")

    # Two cone configurations:
    #   away-from-parent  -- axis = unit(query - parent), aperture small,
    #     should isolate siblings (the part of space *opposite* the parent).
    #   toward-parent     -- axis = unit(parent - query), aperture small,
    #     should hit hypernym chain.
    #   wide              -- aperture ~ pi, basically omnidirectional.

    apertures = [0.2, 0.5, 1.0, 1.6, math.pi]

    print("\n" + "=" * 80)
    print("Axis = away-from-parent (predicted to isolate siblings at narrow aperture)")
    print("=" * 80)
    print(f"{'aperture':>10}  {'sibling-P':>10}  {'sibling-R':>10}  {'hyp-P':>10}")
    for ap in apertures:
        sp = sr = hp = 0.0
        for q in queries:
            true_sibs = set(sub.siblings(q)) & vocab
            true_hyps = set(sub.hypernyms(q)) & vocab
            parent_z = g.circles[emb.wn_to_circle[sub.nodes[q].parent]].z
            query_z = g.circles[emb.wn_to_circle[q]].z
            axis_away = axis_to(query_z, query_z + (query_z - parent_z))  # away from parent
            retr = directed_cone_retrieve(g, emb, q, k, axis=axis_away, aperture=ap, sigma=sigma * 2)
            sp += precision_at_k(retr, true_sibs, k)
            sr += recall_at_k(retr, true_sibs, k)
            hp += precision_at_k(retr, true_hyps, k)
        n = len(queries)
        print(f"  {ap:>8.3f}  {sp/n:>10.4f}  {sr/n:>10.4f}  {hp/n:>10.4f}")

    print("\n" + "=" * 80)
    print("Axis = toward-parent (predicted to climb hypernym chain at narrow aperture)")
    print("=" * 80)
    print(f"{'aperture':>10}  {'sibling-P':>10}  {'sibling-R':>10}  {'hyp-P':>10}")
    for ap in apertures:
        sp = sr = hp = 0.0
        for q in queries:
            true_sibs = set(sub.siblings(q)) & vocab
            true_hyps = set(sub.hypernyms(q)) & vocab
            parent_z = g.circles[emb.wn_to_circle[sub.nodes[q].parent]].z
            query_z = g.circles[emb.wn_to_circle[q]].z
            axis_toward = axis_to(query_z, parent_z)
            retr = directed_cone_retrieve(g, emb, q, k, axis=axis_toward, aperture=ap, sigma=sigma * 3)
            sp += precision_at_k(retr, true_sibs, k)
            sr += recall_at_k(retr, true_sibs, k)
            hp += precision_at_k(retr, true_hyps, k)
        n = len(queries)
        print(f"  {ap:>8.3f}  {sp/n:>10.4f}  {sr/n:>10.4f}  {hp/n:>10.4f}")


if __name__ == "__main__":
    run()
