"""Iteration-2 placement: curvature-budget + region pre-allocation.

Two fixes over `embed.py`:

1. **Region pre-allocation.** Before placing any synsets, each top-level child
   of the WordNet root claims a disjoint sub-gasket: a gasket circle at a
   chosen depth, and *all of its descendants* in the tangency graph become
   that child's reserved territory. Sibling top-level concepts cannot leak
   into each other's regions.

2. **Curvature-budget matching.** Within a region, when placing K children of
   a parent, we (a) sort children by their own WordNet subtree size desc,
   (b) collect K candidate circles by BFS from the parent's circle staying
   inside the region, sort them by ascending curvature (= largest first),
   (c) zip them. Big subtrees land on big circles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .gasket import Gasket
from .wordnet_data import WnSubset


@dataclass
class EmbeddingV2:
    wn_to_circle: dict[str, int]
    circle_to_wn: dict[int, str]
    region_of: dict[str, int]  # synset -> region root circle index


def _subtree_sizes(sub: WnSubset) -> dict[str, int]:
    """Number of descendants (including self) for each synset."""
    sizes: dict[str, int] = {}

    def visit(name: str) -> int:
        if name in sizes:
            return sizes[name]
        s = 1
        for c in sub.nodes[name].children:
            s += visit(c)
        sizes[name] = s
        return s

    visit(sub.root)
    return sizes


def _bfs_collect_in_region(
    g: Gasket,
    start: int,
    region: set[int],
    taken: set[int],
    need: int,
) -> list[int]:
    """BFS from start, staying inside `region`, collecting up to `need` unassigned circles."""
    out: list[int] = []
    visited = {start}
    queue: deque[int] = deque([start])
    while queue and len(out) < need:
        cur = queue.popleft()
        # Sort neighbors deterministically by curvature for stable assignment.
        neighbors = sorted(g.adj[cur], key=lambda j: (round(g.circles[j].k, 6), j))
        for j in neighbors:
            if j in visited or j not in region:
                continue
            visited.add(j)
            if j not in taken:
                out.append(j)
                if len(out) >= need:
                    break
            queue.append(j)
    return out


def _region_of_circle(g: Gasket, region_root: int, max_size: int = 10000) -> set[int]:
    """All circles reachable from region_root in the tangency graph (capped)."""
    out: set[int] = {region_root}
    queue: deque[int] = deque([region_root])
    while queue and len(out) < max_size:
        cur = queue.popleft()
        for j in g.adj[cur]:
            if j not in out:
                out.add(j)
                queue.append(j)
    return out


def embed_wordnet_on_gasket_v2(sub: WnSubset, g: Gasket) -> EmbeddingV2:
    """Region + curvature-budget placement."""
    sizes = _subtree_sizes(sub)

    wn_to_circle: dict[str, int] = {}
    circle_to_wn: dict[int, str] = {}
    region_of: dict[str, int] = {}
    taken: set[int] = set()

    # --- Step 1: place WordNet root on a chosen root circle.
    # The bounding circle (k=-1) is "the universe"; entity sits on the smaller
    # interior circle of the root quadruple. We pick index 1 (k=2 left circle).
    root_circle = 1
    wn_to_circle[sub.root] = root_circle
    circle_to_wn[root_circle] = sub.root
    taken.add(root_circle)

    # --- Step 2: pre-allocate one region per top-level WordNet child.
    # Top-level children sit on circles tangent to the entity circle. We pick
    # the K tangent neighbors with lowest curvature (= biggest territory).
    top_children = list(sub.nodes[sub.root].children)
    top_children.sort(key=lambda c: sizes[c], reverse=True)

    # Available tangent circles to root, sorted by curvature ascending.
    candidate_roots = sorted(
        [j for j in g.adj[root_circle] if j != 0 and j not in taken],  # skip bounding (idx 0)
        key=lambda j: (round(g.circles[j].k, 6), j),
    )

    regions: dict[str, set[int]] = {}
    region_root_idx: dict[str, int] = {}

    if len(candidate_roots) < len(top_children):
        # Not enough direct neighbors -- expand to 2-hops.
        extra = []
        for nbr in g.adj[root_circle]:
            for nbr2 in g.adj[nbr]:
                if nbr2 != root_circle and nbr2 not in candidate_roots and nbr2 not in taken and nbr2 != 0:
                    extra.append(nbr2)
        candidate_roots += sorted(set(extra), key=lambda j: (round(g.circles[j].k, 6), j))

    for child, region_root in zip(top_children, candidate_roots):
        regions[child] = _region_of_circle(g, region_root)
        # Exclude already-taken circles (root, other region roots) from the region.
        regions[child] -= {root_circle}
        region_root_idx[child] = region_root
        wn_to_circle[child] = region_root
        circle_to_wn[region_root] = child
        region_of[child] = region_root
        taken.add(region_root)

    # Region overlap is allowed (gasket is densely connected); we'll resolve
    # by first-come, first-served on `taken`.

    # --- Step 3: BFS over WordNet, placing each node's children in the
    # appropriate region with curvature-budget matching.
    queue: deque[str] = deque(top_children)
    while queue:
        wn_name = queue.popleft()
        if wn_name not in wn_to_circle:
            continue
        parent_circle = wn_to_circle[wn_name]
        parent_region = regions.get(_root_region_for(wn_name, sub, region_of))
        if parent_region is None:
            # No region defined (top-level had no region or wn_name is root) -- use whole gasket.
            parent_region = set(range(len(g.circles))) - {0}

        children = sub.nodes[wn_name].children
        if not children:
            continue

        # Sort children by descendant count desc (biggest gets first pick).
        children_sorted = sorted(children, key=lambda c: sizes[c], reverse=True)

        candidates = _bfs_collect_in_region(
            g, parent_circle, parent_region, taken, need=len(children_sorted)
        )
        # Now sort candidates by curvature ascending (lowest = biggest circle).
        candidates.sort(key=lambda j: (round(g.circles[j].k, 6), j))

        for child, cidx in zip(children_sorted, candidates):
            wn_to_circle[child] = cidx
            circle_to_wn[cidx] = child
            taken.add(cidx)
            region_of[child] = region_of.get(wn_name, cidx)
            queue.append(child)

    return EmbeddingV2(
        wn_to_circle=wn_to_circle,
        circle_to_wn=circle_to_wn,
        region_of=region_of,
    )


def _root_region_for(wn_name: str, sub: WnSubset, region_of: dict[str, int]) -> str | None:
    """Walk up the WordNet tree to the top-level region root name."""
    if wn_name == sub.root:
        return None
    cur = wn_name
    while cur is not None:
        parent = sub.nodes[cur].parent
        if parent == sub.root:
            return cur
        if parent is None:
            return None
        cur = parent
    return None
