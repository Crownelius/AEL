"""Phase C: learnable placement + final eval vs Poincaré.

Pipeline:
  1. Load WordNet noun subtree.
  2. Build gasket + lift all circles to UHS.
  3. Train PlacementHead on WordNet edges (hypernyms + siblings).
  4. Snap each learned UHS point to the nearest gasket circle.
  5. Evaluate retrieval (UHS distance + cone) against Poincaré.
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim.models.poincare import PoincareModel

from src.ael.cone_uhs import ConeUHS, six_axes_3d
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.placement_head import (
    PlacementHead,
    TrainConfig,
    build_edge_lists,
    export_points,
    snap_to_nearest_circle,
    train_placement_head,
)
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


def uhs_dist_retrieve(learned_pts, idx_to_name, q_idx, k):
    """k-NN by UHS distance between learned points directly (no snap)."""
    qp = learned_pts[q_idx]
    class _P:
        def __init__(self, xyt):
            self.x = float(xyt[0]); self.y = float(xyt[1]); self.t = max(float(xyt[2]), 1e-6)
    qpt = _P(qp)
    scored = []
    for j, name in enumerate(idx_to_name):
        if j == q_idx: continue
        pt = _P(learned_pts[j])
        d = uhs_dist(qpt, pt)
        scored.append((d, name))
    scored.sort()
    return [n for _, n in scored[:k]]


def snapped_uhs_retrieve(gasket_points, snapped_idx, idx_to_name, q_idx, k):
    """k-NN after snapping every learned point to its nearest gasket circle."""
    qp = gasket_points[snapped_idx[q_idx]]
    scored = []
    for j, name in enumerate(idx_to_name):
        if j == q_idx: continue
        d = uhs_dist(qp, gasket_points[snapped_idx[j]])
        scored.append((d, name))
    scored.sort()
    return [n for _, n in scored[:k]]


def snapped_cone_to_parent(
    gasket_points, snapped_idx, idx_to_name, name_to_idx, sub, q_name, q_idx,
    k, aperture, sigma,
):
    qp = gasket_points[snapped_idx[q_idx]]
    parent = sub.nodes[q_name].parent
    if parent and parent in name_to_idx:
        target = gasket_points[snapped_idx[name_to_idx[parent]]]
        axis = uhs_log(qp, target)
        nrm = float(np.linalg.norm(axis))
        if nrm < 1e-9:
            cones = [ConeUHS(apex=qp, axis=a, aperture=aperture, sigma=sigma) for a in six_axes_3d()]
        else:
            cones = [ConeUHS(apex=qp, axis=axis, aperture=aperture, sigma=sigma)]
    else:
        cones = [ConeUHS(apex=qp, axis=a, aperture=aperture, sigma=sigma) for a in six_axes_3d()]
    scored = []
    for j, name in enumerate(idx_to_name):
        if j == q_idx: continue
        pt = gasket_points[snapped_idx[j]]
        w = max(c.weight(pt) for c in cones)
        scored.append((-w, name))
    scored.sort()
    return [n for _, n in scored[:k]]


def run(max_depth_wn=6, max_nodes_wn=1000, max_depth_gasket=7,
        k=10, sample_size=150, seed=0,
        train_epochs=80, train_lr=5e-3):
    print("Loading WordNet + gasket...")
    sub = load_noun_subset(max_depth=max_depth_wn, max_nodes=max_nodes_wn)
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=max_depth_gasket)
    gasket_points = all_uhs_points(g)
    print(f"  {len(g.circles)} circles, {len(sub)} synsets")

    # Index synsets.
    idx_to_name = list(sub.nodes.keys())
    name_to_idx = {n: i for i, n in enumerate(idx_to_name)}
    vocab_size = len(idx_to_name)
    print(f"  vocab: {vocab_size} synsets")

    # Build edge lists.
    pos_edges, sib_edges = build_edge_lists(sub, name_to_idx)
    print(f"  hypernym edges: {len(pos_edges)}, sibling pairs: {len(sib_edges)}")

    # Train placement head.
    head = PlacementHead(vocab_size=vocab_size, hidden_dim=64)
    cfg = TrainConfig(epochs=train_epochs, lr=train_lr, log_every=20)
    print(f"\nTraining placement head on device={cfg.device}...")
    train_placement_head(head, pos_edges, sib_edges, vocab_size, cfg)

    # Export learned points.
    learned_pts = export_points(head, vocab_size)
    print(f"\nLearned points: shape {learned_pts.shape}")
    print(f"  t range: [{learned_pts[:,2].min():.3f}, {learned_pts[:,2].max():.3f}]")

    # Snap to gasket.
    print("Snapping learned points to nearest gasket circles...")
    snapped = snap_to_nearest_circle(learned_pts, gasket_points)
    print(f"  distinct circles used: {len(set(snapped))} / {len(snapped)} synsets")

    # Train Poincaré baseline (same data, same vocab).
    print("\nTraining Poincaré baseline...")
    edges = [(n, sub.nodes[n].parent) for n in sub.nodes if sub.nodes[n].parent]
    pmodel = PoincareModel(edges, size=10, negative=10)
    pmodel.train(epochs=100, print_every=50)

    # Pick queries.
    vocab = set(idx_to_name)
    candidates = [n for n in vocab
                  if n in pmodel.kv.key_to_index
                  and len(set(sub.siblings(n)) & vocab) >= 2
                  and sub.nodes[n].parent in vocab]
    rng = random.Random(seed); rng.shuffle(candidates)
    queries = candidates[:sample_size]
    print(f"\nEvaluating {len(queries)} queries, k={k}")

    methods = [
        ("AEL learned-UHS direct",     lambda q: uhs_dist_retrieve(learned_pts, idx_to_name, name_to_idx[q], k)),
        ("AEL learned+snap UHS dist",  lambda q: snapped_uhs_retrieve(gasket_points, snapped, idx_to_name, name_to_idx[q], k)),
        ("AEL learned+snap cone 0.4",  lambda q: snapped_cone_to_parent(
            gasket_points, snapped, idx_to_name, name_to_idx, sub, q, name_to_idx[q], k, 0.4, 1.5)),
        ("AEL learned+snap cone 1.0",  lambda q: snapped_cone_to_parent(
            gasket_points, snapped, idx_to_name, name_to_idx, sub, q, name_to_idx[q], k, 1.0, 1.5)),
        ("Poincaré (10d)",             lambda q: [n for n, _ in pmodel.kv.most_similar(q, topn=len(vocab))][:k]),
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
