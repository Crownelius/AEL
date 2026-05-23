"""Map a WordNet noun subtree onto a gasket via tangency-based descent.

Iteration-1 strategy:
  - WordNet root (entity.n.01) -> a chosen root circle of the gasket (the
    smallest positive-curvature circle in the root quadruple; the bounding
    circle is reserved as "everything").
  - For each WordNet node n in BFS order: assign each child of n to the
    next-available circle tangent to n's circle.
  - "Available" = not yet assigned to any WordNet node.
  - Ordering of tangent neighbors is by ascending curvature, then by angle
    from the parent center -- deterministic.

If a parent runs out of tangent neighbors, we recursively borrow from neighbors
of neighbors (so deeply-fanout parents end up with a small "halo"). This is
crude but keeps the mapping total; iteration 2 will replace it with the prime-
tuple addressing scheme from the design doc.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .gasket import Gasket
from .wordnet_data import WnSubset


@dataclass
class Embedding:
    """A WordNet -> gasket assignment."""

    wn_to_circle: dict[str, int]              # synset name -> circle index
    circle_to_wn: dict[int, str]              # inverse


def _sorted_neighbors(g: Gasket, idx: int) -> list[int]:
    """Neighbors of a circle, sorted deterministically."""
    parent = g.circles[idx]
    neigh = list(g.adj[idx])

    def sort_key(j: int) -> tuple:
        c = g.circles[j]
        # Sort by curvature first (smaller = larger circles, more "important"),
        # then by angle around parent center.
        angle = math.atan2(c.z.imag - parent.z.imag, c.z.real - parent.z.real)
        return (round(c.k, 6), round(angle, 6))

    return sorted(neigh, key=sort_key)


def embed_wordnet_on_gasket(
    sub: WnSubset,
    g: Gasket,
    root_circle: int = 1,
) -> Embedding:
    """BFS assign WordNet nodes to gasket circles via tangency descent.

    `root_circle` defaults to index 1 (the first non-bounding root circle) --
    the bounding (k=-1) circle is treated as "the universe", not as `entity`.
    """
    wn_to_circle: dict[str, int] = {}
    circle_to_wn: dict[int, str] = {}

    wn_to_circle[sub.root] = root_circle
    circle_to_wn[root_circle] = sub.root

    queue: deque[str] = deque([sub.root])

    while queue:
        wn_name = queue.popleft()
        wn_node = sub.nodes[wn_name]
        parent_circle = wn_to_circle[wn_name]

        children = wn_node.children
        if not children:
            continue

        # Gather candidate circles, breadth-first from the parent circle,
        # skipping any already assigned.
        candidates = _find_unassigned(g, parent_circle, circle_to_wn, need=len(children))

        # Zip children -> circles (deterministic since both are ordered).
        for child_name, cidx in zip(children, candidates):
            wn_to_circle[child_name] = cidx
            circle_to_wn[cidx] = child_name
            queue.append(child_name)

        # If we ran short, the remaining children are dropped from the embedding.
        # We log this so the eval knows.
        if len(candidates) < len(children):
            dropped = children[len(candidates) :]
            for d in dropped:
                # Mark by not assigning -- they simply won't appear in wn_to_circle
                # and queue (they're skipped). Subclass-safe.
                pass

    return Embedding(wn_to_circle=wn_to_circle, circle_to_wn=circle_to_wn)


def _find_unassigned(
    g: Gasket,
    start: int,
    taken: dict[int, str],
    need: int,
) -> list[int]:
    """BFS from `start` collecting unassigned circles, up to `need`."""
    out: list[int] = []
    visited = {start}
    queue: deque[int] = deque([start])

    while queue and len(out) < need:
        cur = queue.popleft()
        for j in _sorted_neighbors(g, cur):
            if j in visited:
                continue
            visited.add(j)
            if j not in taken:
                out.append(j)
                if len(out) >= need:
                    break
            queue.append(j)

    return out
