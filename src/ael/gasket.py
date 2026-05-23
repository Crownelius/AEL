"""Apollonian gasket generator via Vieta-jumping reflection.

Given a Descartes quadruple Q = (c0, c1, c2, c3) of mutually tangent circles,
replacing c_i with its Apollonian companion yields another valid quadruple. The
companion is given by the linear reflection:

    k_i' = 2 (k_j + k_k + k_l) - k_i
    k_i' z_i' = 2 (k_j z_j + k_k z_k + k_l z_l) - k_i z_i

where (j, k, l) are the other three indices. This avoids re-solving Descartes
and keeps integer curvatures exactly integer.

The recursion expands each quadruple into 4 child quadruples (one per reflected
slot). We BFS to bounded depth, dedup circles by canonical (k, z) hash, and
record adjacency (which circles are mutually tangent).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .descartes import Circle, descartes_fourth_circle


CANONICAL_PRECISION = 9  # decimal places for hashing centers


def _canon(c: Circle) -> tuple:
    """Hashable key for a circle, robust to floating noise."""
    return (
        round(c.k, CANONICAL_PRECISION),
        round(c.z.real, CANONICAL_PRECISION),
        round(c.z.imag, CANONICAL_PRECISION),
    )


def reflect(quad: tuple[Circle, Circle, Circle, Circle], i: int) -> Circle:
    """Reflect c_i across the other three, returning its Apollonian companion."""
    others = [quad[j] for j in range(4) if j != i]
    c_i = quad[i]
    k_sum = sum(c.k for c in others)
    kz_sum = sum(c.k * c.z for c in others)
    k_new = 2.0 * k_sum - c_i.k
    kz_new = 2.0 * kz_sum - c_i.k * c_i.z
    z_new = kz_new / k_new if k_new != 0 else complex(float("inf"))
    return Circle(k_new, z_new)


@dataclass
class Gasket:
    """A generated Apollonian gasket up to bounded depth."""

    root: tuple[Circle, Circle, Circle, Circle]
    circles: list[Circle] = field(default_factory=list)
    index_of: dict[tuple, int] = field(default_factory=dict)
    # adjacency: i -> set of indices tangent to circle i
    adj: dict[int, set[int]] = field(default_factory=dict)
    # depth at which each circle was first discovered (root = 0)
    depth: dict[int, int] = field(default_factory=dict)
    # parents: for non-root circles, the 3 indices it was generated from
    parents: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    def add_circle(self, c: Circle, depth: int, parents: tuple[int, int, int] | None = None) -> int:
        key = _canon(c)
        if key in self.index_of:
            return self.index_of[key]
        idx = len(self.circles)
        self.circles.append(c)
        self.index_of[key] = idx
        self.adj[idx] = set()
        self.depth[idx] = depth
        if parents is not None:
            self.parents[idx] = parents
        return idx

    def link(self, i: int, j: int) -> None:
        if i == j:
            return
        self.adj[i].add(j)
        self.adj[j].add(i)


def build_gasket(
    root: tuple[Circle, Circle, Circle, Circle],
    max_depth: int = 5,
    max_curvature: float | None = None,
) -> Gasket:
    """BFS expansion of the gasket from a root Descartes quadruple."""
    g = Gasket(root=root)

    # Seed: add the 4 root circles, all mutually tangent.
    root_idxs = [g.add_circle(c, depth=0) for c in root]
    for a in range(4):
        for b in range(a + 1, 4):
            g.link(root_idxs[a], root_idxs[b])

    # BFS over quadruples.
    queue: deque[tuple[tuple[int, int, int, int], int]] = deque()
    queue.append((tuple(root_idxs), 0))  # type: ignore[arg-type]
    seen_quads: set[tuple[int, int, int, int]] = set()

    while queue:
        quad_idxs, d = queue.popleft()
        canon_q = tuple(sorted(quad_idxs))
        if canon_q in seen_quads:
            continue
        seen_quads.add(canon_q)
        if d >= max_depth:
            continue
        quad = tuple(g.circles[i] for i in quad_idxs)

        for i in range(4):
            c_new = reflect(quad, i)
            if max_curvature is not None and abs(c_new.k) > max_curvature:
                continue
            others_idx = tuple(quad_idxs[j] for j in range(4) if j != i)
            new_idx = g.add_circle(c_new, depth=d + 1, parents=others_idx)
            for j in others_idx:
                g.link(new_idx, j)
            new_quad = tuple(others_idx) + (new_idx,)
            queue.append((new_quad, d + 1))  # type: ignore[arg-type]

    return g


def standard_root_neg1_2_2_3() -> tuple[Circle, Circle, Circle, Circle]:
    """The canonical (-1, 2, 2, 3) integer Apollonian root quadruple.

    Bounding circle of radius 1 centered at 0. Two interior circles of radius 1/2
    at +/- 0.5. One k=3 circle in the upper gap.
    """
    c0 = Circle(k=-1, z=0 + 0j)
    c1 = Circle(k=2, z=-0.5 + 0j)
    c2 = Circle(k=2, z=+0.5 + 0j)
    # Pick the upper of the two k=3 companions.
    upper, _lower = descartes_fourth_circle(c0, c1, c2)
    if upper.z.imag < 0:
        upper, _lower = _lower, upper
    return (c0, c1, c2, upper)
