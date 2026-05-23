"""Phase E: Hopf-fibration placement vs UHS placement vs Poincaré.

Hypothesis: snapping a learned point to the gasket lattice has been the
single biggest source of performance loss in AEL. Hopf placement should
substantially reduce that loss because:
  - the codomain is compact (S^3) -- no t blow-up
  - base snap is over S^2 (not UHS) -- gasket bases are also on S^2
  - phase stays continuous (we don't round it)
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim.models.poincare import PoincareModel

from src.ael.cone_hopf import ConeS2, four_axes_at, log_map, make_multihead_s2
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.hopf_lift import assign_curvature_phases, gasket_to_s2
from src.ael.placement_head import (
    PlacementHead,
    TrainConfig,
    build_edge_lists,
    export_points,
    snap_to_nearest_circle,
    train_placement_head,
)
from src.ael.placement_head_hopf import (
    HopfPlacementHead,
    HopfTrainConfig,
    export_bases,
    snap_to_gasket_base,
    train_hopf_head,
)
from src.ael.uhs import all_uhs_points, uhs_dist
from src.ael.wordnet_data import load_noun_subset


def p_at_k(retr, rel, k):
    if not rel: return float("nan")
    top = retr[:k]
    if not top: return 0.0
    return sum(1 for r in top if r in rel) / k


def r_at_k(retr, rel, k):
    if not rel: return float("nan")
    return sum(1 for r in retr[:k] if r in rel) / len(rel)


def s2_dist(p, q):
    ip = float(np.clip(np.dot(p, q), -1.0, 1.0))
    return math.acos(ip)


def hopf_direct_retrieve(bases, idx_to_name, q_idx, k):
    qp = bases[q_idx]
    scored = []
    for j, name in enumerate(idx_to_name):
        if j == q_idx: continue
        scored.append((s2_dist(qp, bases[j]), name))
    scored.sort()
    return [n for _, n in scored[:k]]


def hopf_snapped_retrieve(gasket_s2, snapped, idx_to_name, q_idx, k):
    qp = gasket_s2[snapped[q_idx]]
    scored = []
    for j, name in enumerate(idx_to_name):
        if j == q_idx: continue
        scored.append((s2_dist(qp, gasket_s2[snapped[j]]), name))
    scored.sort()
    return [n for _, n in scored[:k]]


def hopf_cone_to_parent(
    gasket_s2, snapped, bases, idx_to_name, name_to_idx, sub,
    q_name, q_idx, k, aperture, sigma, use_snapped=True
):
    """Cone with axis pointing from query toward its parent's base point."""
    qp = gasket_s2[snapped[q_idx]] if use_snapped else bases[q_idx]
    parent = sub.nodes[q_name].parent
    if parent and parent in name_to_idx:
        pidx = name_to_idx[parent]
        target = gasket_s2[snapped[pidx]] if use_snapped else bases[pidx]
        axis = log_map(qp, target)
        nrm = float(np.linalg.norm(axis))
        if nrm < 1e-9:
            cones = make_multihead_s2(qp, n_heads=4, aperture=aperture, sigma=sigma)
        else:
            cones = [ConeS2(apex=qp, axis=axis, aperture=aperture, sigma=sigma)]
    else:
        cones = make_multihead_s2(qp, n_heads=4, aperture=aperture, sigma=sigma)
    scored = []
    for j, name in enumerate(idx_to_name):
        if j == q_idx: continue
        v = gasket_s2[snapped[j]] if use_snapped else bases[j]
        w = max(c.weight(v) for c in cones)
        scored.append((-w, name))
    scored.sort()
    return [n for _, n in scored[:k]]


def run(max_depth_wn=6, max_nodes_wn=1000, max_depth_gasket=7,
        k=10, sample_size=150, seed=0,
        hopf_epochs=100, uhs_epochs=80):
    torch.manual_seed(seed)
    print("Loading WordNet + gasket...")
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    gasket_s2 = gasket_to_s2(g)
    print(f"  {len(g.circles)} circles, {len(sub)} synsets")

    idx_to_name = list(sub.nodes.keys())
    name_to_idx = {n: i for i, n in enumerate(idx_to_name)}
    vocab_size = len(idx_to_name)

    pos_edges, sib_edges = build_edge_lists(sub, name_to_idx)
    print(f"  hypernym edges: {len(pos_edges)}, sibling pairs: {len(sib_edges)}")

    # --- Train Hopf head ---
    print(f"\nTraining Hopf placement head ({hopf_epochs} epochs)...")
    hopf_head = HopfPlacementHead(vocab_size=vocab_size, hidden_dim=64)
    hopf_cfg = HopfTrainConfig(epochs=hopf_epochs, lr=5e-3, log_every=hopf_epochs // 4)
    train_hopf_head(hopf_head, pos_edges, sib_edges, vocab_size, hopf_cfg)
    hopf_bases = export_bases(hopf_head, vocab_size)
    hopf_snapped, hopf_snap_dist = snap_to_gasket_base(hopf_bases, gasket_s2)
    print(f"  Hopf snap: {len(set(hopf_snapped))} distinct circles used / {vocab_size}")
    print(f"  median snap distance: {np.median(hopf_snap_dist):.4f} rad")

    # --- Train UHS head (Phase C, for direct comparison) ---
    print(f"\nTraining UHS placement head ({uhs_epochs} epochs)...")
    uhs_head = PlacementHead(vocab_size=vocab_size, hidden_dim=64)
    uhs_cfg = TrainConfig(epochs=uhs_epochs, lr=1e-3, log_every=uhs_epochs // 4)
    train_placement_head(uhs_head, pos_edges, sib_edges, vocab_size, uhs_cfg)
    uhs_pts = export_points(uhs_head, vocab_size)
    uhs_gasket_points = all_uhs_points(g)
    uhs_snapped = snap_to_nearest_circle(uhs_pts, uhs_gasket_points)
    print(f"  UHS snap: {len(set(uhs_snapped))} distinct circles used / {vocab_size}")

    # --- Poincaré baseline ---
    print("\nTraining Poincaré baseline...")
    edges = [(n, sub.nodes[n].parent) for n in sub.nodes if sub.nodes[n].parent]
    pmodel = PoincareModel(edges, size=10, negative=10)
    pmodel.train(epochs=100, print_every=50)

    vocab = set(idx_to_name) & set(pmodel.kv.key_to_index)
    candidates = [n for n in vocab
                  if len(set(sub.siblings(n)) & vocab) >= 2
                  and sub.nodes[n].parent in vocab]
    rng = random.Random(seed); rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating {len(queries)} queries, k={k}")

    def uhs_direct_retrieve(q):
        qp = uhs_pts[name_to_idx[q]]
        class _P: __slots__ = ("x", "y", "t");
        class _P:
            def __init__(self, p): self.x = float(p[0]); self.y = float(p[1]); self.t = max(float(p[2]), 1e-6)
        qpt = _P(qp)
        scored = []
        for j, name in enumerate(idx_to_name):
            if name == q: continue
            pt = _P(uhs_pts[j])
            scored.append((uhs_dist(qpt, pt), name))
        scored.sort()
        return [n for _, n in scored[:k]]

    methods = [
        ("UHS direct (free)",        lambda q: uhs_direct_retrieve(q)),
        ("UHS snapped",              lambda q: [idx_to_name[i] for i in
            sorted(range(vocab_size),
                   key=lambda j: uhs_dist(uhs_gasket_points[uhs_snapped[name_to_idx[q]]],
                                          uhs_gasket_points[uhs_snapped[j]])) if i != name_to_idx[q]][:k]),
        ("Hopf direct (free)",       lambda q: hopf_direct_retrieve(hopf_bases, idx_to_name, name_to_idx[q], k)),
        ("Hopf snapped",             lambda q: hopf_snapped_retrieve(gasket_s2, hopf_snapped, idx_to_name, name_to_idx[q], k)),
        ("Hopf snapped cone 0.5",    lambda q: hopf_cone_to_parent(gasket_s2, hopf_snapped, hopf_bases, idx_to_name,
                                                                   name_to_idx, sub, q, name_to_idx[q], k, 0.5, 0.6)),
        ("Hopf snapped cone 1.2",    lambda q: hopf_cone_to_parent(gasket_s2, hopf_snapped, hopf_bases, idx_to_name,
                                                                   name_to_idx, sub, q, name_to_idx[q], k, 1.2, 0.6)),
        ("Poincaré (10d)",           lambda q: [n for n, _ in pmodel.kv.most_similar(q, topn=len(vocab))][:k]),
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
