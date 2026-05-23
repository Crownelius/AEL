"""k-NN retrieval over an Apollonian gasket embedding.

Two distance variants for iteration 1:
  - euclidean: |z_a - z_b| in the gasket plane
  - graph:     shortest-path hop count in the tangency adjacency graph
"""

from __future__ import annotations

import heapq
from collections import deque

from .embed import Embedding
from .gasket import Gasket


def euclidean_knn(g: Gasket, query_idx: int, k: int, restrict_to: set[int] | None = None) -> list[tuple[int, float]]:
    """k nearest circles by Euclidean distance between centers."""
    qz = g.circles[query_idx].z
    pool = restrict_to if restrict_to is not None else range(len(g.circles))
    scored = []
    for j in pool:
        if j == query_idx:
            continue
        d = abs(g.circles[j].z - qz)
        scored.append((d, j))
    scored.sort()
    return [(j, d) for d, j in scored[:k]]


def graph_knn(g: Gasket, query_idx: int, k: int, restrict_to: set[int] | None = None) -> list[tuple[int, int]]:
    """k nearest by tangency-graph BFS distance."""
    dist: dict[int, int] = {query_idx: 0}
    queue: deque[int] = deque([query_idx])
    out: list[tuple[int, int]] = []

    while queue and len(out) < k + 1:  # +1 because query itself sits in dist
        cur = queue.popleft()
        d = dist[cur]
        if cur != query_idx and (restrict_to is None or cur in restrict_to):
            out.append((cur, d))
        for j in g.adj[cur]:
            if j not in dist:
                dist[j] = d + 1
                queue.append(j)

    return out[:k]


def wn_knn_euclidean(g: Gasket, emb: Embedding, query_synset: str, k: int) -> list[tuple[str, float]]:
    """k nearest WordNet synsets to a query, by Euclidean gasket distance."""
    if query_synset not in emb.wn_to_circle:
        return []
    qidx = emb.wn_to_circle[query_synset]
    pool = set(emb.circle_to_wn.keys())
    hits = euclidean_knn(g, qidx, k, restrict_to=pool)
    return [(emb.circle_to_wn[j], d) for j, d in hits]


def wn_knn_graph(g: Gasket, emb: Embedding, query_synset: str, k: int) -> list[tuple[str, int]]:
    """k nearest WordNet synsets to a query, by tangency-graph hop distance."""
    if query_synset not in emb.wn_to_circle:
        return []
    qidx = emb.wn_to_circle[query_synset]
    pool = set(emb.circle_to_wn.keys())
    hits = graph_knn(g, qidx, k, restrict_to=pool)
    return [(emb.circle_to_wn[j], d) for j, d in hits]
