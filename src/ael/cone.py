"""Cone attention over the gasket.

A cone is parameterized by:
  apex     -- a circle (or its center z_apex)
  axis     -- a unit direction in the plane (complex number, |axis| = 1)
  aperture -- a half-angle, in radians

The smooth weight a cone assigns to a candidate point z is:

    w_dir = exp(- (theta / aperture)^2)        # angular falloff (Gaussian)
    w_rad = exp(- (r / sigma)^2)               # radial falloff
    w = w_dir * w_rad

where theta is the angle between (z - z_apex) and axis, and r = |z - z_apex|.
Both factors are smooth in their parameters, so this layer is gradient-friendly
in (axis, aperture, sigma) -- the "trainable temperature" the design called for.

Multi-head: H cones from the same apex with axes spread around the circle.
A point's score is the max (or softmax-sum) of per-cone weights, giving an
omnidirectional but still directional-aware similarity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Cone:
    apex: complex            # z_apex (the apex circle's center)
    axis: complex            # unit-magnitude direction
    aperture: float          # half-angle in radians (>0)
    sigma: float = 1.0       # radial extent

    def weight(self, z: complex) -> float:
        """Smooth weight assigned to a point z by this cone."""
        d = z - self.apex
        r = abs(d)
        if r == 0:
            return 1.0
        # Angle between d and axis.
        # cos(theta) = Re((d/r) * conj(axis))
        unit_d = d / r
        cos_theta = max(-1.0, min(1.0, (unit_d * self.axis.conjugate()).real))
        theta = math.acos(cos_theta)
        w_dir = math.exp(-(theta / max(self.aperture, 1e-9)) ** 2)
        w_rad = math.exp(-(r / max(self.sigma, 1e-9)) ** 2)
        return w_dir * w_rad


def make_multihead(
    apex: complex,
    n_heads: int = 8,
    aperture: float = 0.4,
    sigma: float = 1.0,
    base_axis_angle: float = 0.0,
) -> list[Cone]:
    """Build n_heads cones evenly distributed in angle around the apex."""
    cones = []
    for h in range(n_heads):
        angle = base_axis_angle + 2.0 * math.pi * h / n_heads
        axis = complex(math.cos(angle), math.sin(angle))
        cones.append(Cone(apex=apex, axis=axis, aperture=aperture, sigma=sigma))
    return cones


def multihead_score(cones: list[Cone], z: complex, reduce: str = "max") -> float:
    """Aggregate per-head weights for a point."""
    weights = [c.weight(z) for c in cones]
    if reduce == "max":
        return max(weights)
    elif reduce == "sum":
        return sum(weights)
    elif reduce == "softmax":
        # log-sum-exp normalization (avoids overflow).
        m = max(weights)
        return m + math.log(sum(math.exp(w - m) for w in weights) / len(weights))
    raise ValueError(f"unknown reduce: {reduce}")


def axis_to(apex: complex, target: complex) -> complex:
    """Unit-axis pointing from apex toward target."""
    d = target - apex
    n = abs(d)
    if n == 0:
        return complex(1.0, 0.0)
    return d / n
