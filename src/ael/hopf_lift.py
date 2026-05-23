"""Lift gasket circles to Hopf fibers in S^3.

Each gasket circle has a planar center z = x + i y. We:
  1. stereographically project z to a point on S^2 (the Riemann sphere)
  2. pick the canonical phase-0 representative on the Hopf fiber over that point

The result: each circle gets a base point in S^2 plus its fiber S^1 in S^3.
For most retrieval purposes we only need the base point (the phase carries
information that the learnable head adds later). The fiber identity remains
available for link-number-based metrics.

The radius of the gasket circle does NOT play a role in this lift -- it's
already encoded in the gasket's adjacency structure. Concept hierarchy
shows up as spherical distance on S^2, not as radial distance.
"""

from __future__ import annotations

import numpy as np

from .gasket import Gasket
from .hopf import hopf_map, hopf_section, stereographic_to_sphere


def gasket_to_s2(g: Gasket) -> np.ndarray:
    """For each circle, return its base point on S^2 (shape: N x 3)."""
    return np.array([stereographic_to_sphere(c.z) for c in g.circles], dtype=float)


def gasket_to_s3_section(g: Gasket) -> np.ndarray:
    """Each circle's phase-0 representative on its Hopf fiber (shape: N x 4)."""
    s2 = gasket_to_s2(g)
    return hopf_section(s2)


def assign_curvature_phases(
    g: Gasket, twin_phase_offset: float = np.pi / 2
) -> np.ndarray:
    """Optional: assign a deterministic phase per circle based on curvature.

    Twin-prime-paired circles get phases offset by `twin_phase_offset`
    (pi/2 by default) so the fiber identifies them as a paired set.
    Other circles get phase = atan2(z.imag, z.real), giving a smooth phase
    field across the gasket plane.
    """
    from .primes import twin_pair_of
    phases = np.zeros(len(g.circles), dtype=float)
    for i, c in enumerate(g.circles):
        k = int(round(c.k))
        if k > 0:
            pair = twin_pair_of(k)
            if pair is not None:
                # Lower member -> 0, upper -> offset
                phases[i] = 0.0 if pair[0] == k else twin_phase_offset
                continue
        # Default: planar angle of the center
        phases[i] = np.arctan2(c.z.imag, c.z.real)
    return phases
