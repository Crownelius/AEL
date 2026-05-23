"""Lift each gasket circle to a Cl(3,1) Descartes 4-vector.

For a non-root circle c with parents (p1, p2, p3), the natural Descartes
4-tuple is (k_c, k_p1, k_p2, k_p3) -- a valid quadruple by construction.

For root circles, we use the root Descartes 4-tuple (b0, b1, b2, b3) with
the self-curvature placed first.
"""

from __future__ import annotations

import numpy as np

from .descartes_3d import HyperboloidPoint, to_hyperboloid
from .gasket import Gasket


def circle_4vector(g: Gasket, idx: int) -> np.ndarray:
    """The Descartes 4-vector for circle idx."""
    c = g.circles[idx]
    if idx in g.parents:
        p1, p2, p3 = g.parents[idx]
        return np.array(
            [c.k, g.circles[p1].k, g.circles[p2].k, g.circles[p3].k],
            dtype=float,
        )
    # Root circle: 4-tuple = the root quadruple itself, with self first.
    root_ks = [rc.k for rc in g.root]
    self_k = c.k
    others = [k for k in root_ks if k != self_k]
    # Handle duplicates (the root has two k=2 circles).
    if root_ks.count(self_k) > 1:
        # Remove only one occurrence.
        removed = False
        others = []
        for k in root_ks:
            if k == self_k and not removed:
                removed = True
                continue
            others.append(k)
    return np.array([self_k] + others[:3], dtype=float)


def all_4vectors(g: Gasket) -> np.ndarray:
    """Stack of 4-vectors, one per circle."""
    return np.stack([circle_4vector(g, i) for i in range(len(g.circles))])


def all_hyperboloid_points(g: Gasket) -> list[HyperboloidPoint]:
    """Project each circle's 4-vector to a unit-norm point on H^3."""
    return [to_hyperboloid(circle_4vector(g, i)) for i in range(len(g.circles))]
