"""Iteration-3: cone attention prototype on the gasket.

Two demonstrations:

  A. Omnidirectional multi-head cone retrieval.
     Smooth alternative to k-NN; aperture is a single learnable knob.
     Compared to hard euclidean k-NN under varying aperture.

  B. Aperture-as-temperature trade.
     Narrow aperture: only nearest tangent circles (siblings).
     Wide aperture: pulls in ancestor-chain circles (hypernyms).
     Single dial controls sibling-vs-hypernym balance.
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.cone import Cone, axis_to, make_multihead, multihead_score
from src.ael.embed_v2 import embed_wordnet_on_gasket_v2
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
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


def cone_retrieve(g, emb, q, k, aperture, sigma, n_heads=8):
    """Retrieve top-k synsets by multi-head cone weight."""
    qidx = emb.wn_to_circle[q]
    apex = g.circles[qidx].z
    cones = make_multihead(apex, n_heads=n_heads, aperture=aperture, sigma=sigma)
    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q:
            continue
        z = g.circles[cidx].z
        w = multihead_score(cones, z, reduce="max")
        scored.append((-w, synset))  # negative for sort = descending
    scored.sort()
    return [s for _, s in scored[:k]]


def estimate_sigma(g, emb):
    """Median pairwise distance between embedded circle centers (heuristic sigma)."""
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
    sample_size=100,
    seed=0,
):
    print("Loading WordNet + gasket + embedding (v2)...")
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    emb = embed_wordnet_on_gasket_v2(sub, g)
    print(f"  {len(g.circles)} circles, {len(emb.wn_to_circle)} embedded synsets")

    sigma = estimate_sigma(g, emb)
    print(f"  median pairwise dist (used as sigma): {sigma:.4f}")

    vocab = set(emb.wn_to_circle)
    candidates = [
        n for n in vocab
        if len(set(sub.siblings(n)) & vocab) >= 2 and sub.nodes[n].parent
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating {len(queries)} queries, k={k}")

    # --- Demo A: aperture sweep showing the trade ---
    print("\n" + "=" * 70)
    print("Aperture sweep (smaller = sharper, larger = more diffuse)")
    print("=" * 70)
    print(f"{'aperture':>10}  {'sibling-P@k':>12}  {'sibling-R@k':>12}  {'hypernym-P@k':>13}")
    print("-" * 70)

    apertures = [0.05, 0.1, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.0, math.pi]
    best = {"sib_p": (0.0, 0.0), "hyp_p": (0.0, 0.0)}  # (value, aperture)

    for ap in apertures:
        sp = sr = hp = 0.0
        for q in queries:
            true_sibs = set(sub.siblings(q)) & vocab
            true_hyps = set(sub.hypernyms(q)) & vocab
            retr = cone_retrieve(g, emb, q, k, aperture=ap, sigma=sigma)
            sp += precision_at_k(retr, true_sibs, k)
            sr += recall_at_k(retr, true_sibs, k)
            hp += precision_at_k(retr, true_hyps, k)
        sp /= len(queries); sr /= len(queries); hp /= len(queries)
        if sp > best["sib_p"][0]:
            best["sib_p"] = (sp, ap)
        if hp > best["hyp_p"][0]:
            best["hyp_p"] = (hp, ap)
        print(f"  {ap:>8.3f}  {sp:>12.4f}  {sr:>12.4f}  {hp:>13.4f}")

    print("-" * 70)
    print(f"best sibling-P:  {best['sib_p'][0]:.4f}  @ aperture={best['sib_p'][1]:.3f}")
    print(f"best hypernym-P: {best['hyp_p'][0]:.4f}  @ aperture={best['hyp_p'][1]:.3f}")
    print("=" * 70)

    # --- Demo B: qualitative examples for narrow vs wide aperture ---
    print("\nQualitative: narrow (0.1) vs wide (1.6) aperture on 5 queries")
    print("-" * 70)
    samples = queries[:5]
    for q in samples:
        true_sibs = set(sub.siblings(q))
        true_hyps = set(sub.hypernyms(q))
        narrow = cone_retrieve(g, emb, q, 5, aperture=0.1, sigma=sigma)
        wide = cone_retrieve(g, emb, q, 5, aperture=1.6, sigma=sigma * 3)

        def mark(name):
            if name in true_sibs: return f"{name}[S]"
            if name in true_hyps: return f"{name}[H]"
            return name

        print(f"  {q}")
        print(f"    narrow -> {[mark(x) for x in narrow]}")
        print(f"    wide   -> {[mark(x) for x in wide]}")
    print("\n  [S]=true sibling, [H]=true hypernym")


if __name__ == "__main__":
    run()
