"""Phase B: twin-prime addressing experiment.

Two demonstrations:
  1. Sanity: every circle gets a well-defined address. Twin anchors are
     identified; descent paths reconstruct from root.
  2. Twin-anchor-boosted cone retrieval: in UHS-cone scoring, scale weights
     of twin-prime-anchor circles up by a small factor. Test if this helps
     pick out 'landmark' siblings.
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

from src.ael.cone_uhs import ConeUHS, six_axes_3d
from src.ael.embed_v2 import embed_wordnet_on_gasket_v2
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.prime_addr import TwinAddressBook, address_all
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


def cone_with_twin_boost(points, emb, sub, q, k, aperture, sigma, anchors: set[int], boost: float):
    qidx = emb.wn_to_circle[q]
    apex = points[qidx]
    parent = sub.nodes[q].parent
    if parent and parent in emb.wn_to_circle:
        target = points[emb.wn_to_circle[parent]]
        axis = uhs_log(apex, target)
        if float(np.linalg.norm(axis)) < 1e-9:
            cones = [ConeUHS(apex=apex, axis=a, aperture=aperture, sigma=sigma) for a in six_axes_3d()]
        else:
            cones = [ConeUHS(apex=apex, axis=axis, aperture=aperture, sigma=sigma)]
    else:
        cones = [ConeUHS(apex=apex, axis=a, aperture=aperture, sigma=sigma) for a in six_axes_3d()]

    scored = []
    for synset, cidx in emb.wn_to_circle.items():
        if synset == q: continue
        w = max(c.weight(points[cidx]) for c in cones)
        if cidx in anchors:
            w *= boost
        scored.append((-w, synset))
    scored.sort()
    return [s for _, s in scored[:k]]


def run(max_depth_wn=6, max_nodes_wn=1000, max_depth_gasket=7,
        k=10, sample_size=150, seed=0):
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    emb = embed_wordnet_on_gasket_v2(sub, g)
    points = all_uhs_points(g)

    # Build address book.
    book = TwinAddressBook.build(g)
    print(f"\nGasket address book:")
    print(f"  circles: {len(g.circles)}")
    print(f"  twin-anchor circles: {len(book.anchors)} ({len(book.anchors)/len(g.circles):.1%})")
    print(f"  distinct twin pairs:  {len(book.by_twin_pair)}")

    # Show some example addresses.
    addrs = address_all(g)
    print(f"\nSample addresses (first 10 non-root circles):")
    for i in range(4, 14):
        print(f"  c[{i:3d}] k={int(round(g.circles[i].k)):>5d}  {addrs[i]}  path_len={len(addrs[i].descent_path)}")

    # How many embedded synsets ended up on twin-anchor circles?
    anchor_set = set(book.anchors)
    anchored_synsets = [n for n, cidx in emb.wn_to_circle.items() if cidx in anchor_set]
    print(f"\nWordNet synsets placed on twin-anchor circles: {len(anchored_synsets)} / {len(emb.wn_to_circle)} ({len(anchored_synsets)/len(emb.wn_to_circle):.1%})")
    print(f"  sample anchored synsets: {anchored_synsets[:5]}")

    # Retrieval with and without twin-anchor boost.
    sigma_uhs = 11.35  # from previous experiment
    vocab = set(emb.wn_to_circle)
    candidates = [n for n in vocab
                  if len(set(sub.siblings(n)) & vocab) >= 2
                  and sub.nodes[n].parent in vocab]
    rng = random.Random(seed); rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating {len(queries)} queries, k={k}")

    methods = [
        ("UHS cone -> parent (no boost, ap 1.5)",
            lambda q: cone_with_twin_boost(points, emb, sub, q, k, 1.5, sigma_uhs * 2, anchor_set, 1.0)),
        ("UHS cone -> parent (boost 1.5x)",
            lambda q: cone_with_twin_boost(points, emb, sub, q, k, 1.5, sigma_uhs * 2, anchor_set, 1.5)),
        ("UHS cone -> parent (boost 3.0x)",
            lambda q: cone_with_twin_boost(points, emb, sub, q, k, 1.5, sigma_uhs * 2, anchor_set, 3.0)),
        ("UHS cone -> parent (boost 10x)",
            lambda q: cone_with_twin_boost(points, emb, sub, q, k, 1.5, sigma_uhs * 2, anchor_set, 10.0)),
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
    print("\n" + "=" * 80)
    print(f"Method                                  sibling-P  sibling-R  hyp-P")
    print("-" * 80)
    for name, _ in methods:
        m = totals[name]
        print(f"{name:<40}  {m['sp']/n:>8.4f}   {m['sr']/n:>8.4f}  {m['hp']/n:>8.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run()
