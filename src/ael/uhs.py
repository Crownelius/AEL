"""Upper half-space model of hyperbolic 3-space.

Each circle in the gasket (center z = x + i*y, radius r) lifts to the point
(x, y, r) in H^3_UHS = { (x, y, t) : t > 0 }. The hyperbolic metric is

    ds^2 = (dx^2 + dy^2 + dt^2) / t^2.

Closed-form distance:

    cosh(d(p, q)) = 1 + |p_xy - q_xy|^2 / (2 p_t q_t) + (p_t - q_t)^2 / (2 p_t q_t)
                  = 1 + ((p_xy - q_xy)^2 + (p_t - q_t)^2) / (2 p_t q_t)

For tangent circles, this is small. For deeply nested circles (vastly
different radii), this grows. This is the natural Apollonian hyperbolic
geometry: the gasket sits on the boundary plane t=0.

Geodesics: either vertical lines or Euclidean semicircles whose centers lie
in the boundary plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gasket import Gasket


@dataclass(frozen=True)
class UHSPoint:
    x: float
    y: float
    t: float  # > 0


def circle_to_uhs(z: complex, r: float, eps: float = 1e-6) -> UHSPoint:
    """Lift a circle (center z, radius r) to its half-ball apex in H^3."""
    return UHSPoint(x=z.real, y=z.imag, t=max(abs(r), eps))


def uhs_dist(p: UHSPoint, q: UHSPoint) -> float:
    dx = p.x - q.x
    dy = p.y - q.y
    dt = p.t - q.t
    num = dx * dx + dy * dy + dt * dt
    cosh_d = 1.0 + num / (2.0 * p.t * q.t)
    cosh_d = max(cosh_d, 1.0)
    return math.acosh(cosh_d)


def uhs_log(p: UHSPoint, q: UHSPoint) -> np.ndarray:
    """Tangent vector at p pointing toward q along the geodesic to q.

    The exponential map in UHS is closed-form but messy; for cone attention
    we only need a tangent vector whose Euclidean direction matches the
    geodesic direction at p, and whose Euclidean length equals d(p, q).

    Approximation that suffices for our use:
      direction = (q - p) in ambient (x, y, t)
      length    = d(p, q)
    This is accurate near p and degrades smoothly with distance. For our
    attention-weighting purposes (Gaussian falloff), this is sufficient and
    fully differentiable.
    """
    d = uhs_dist(p, q)
    raw = np.array([q.x - p.x, q.y - p.y, q.t - p.t], dtype=float)
    n = float(np.linalg.norm(raw))
    if n < 1e-12:
        return np.zeros(3)
    return (d / n) * raw


def all_uhs_points(g: Gasket) -> list[UHSPoint]:
    """Lift every gasket circle to UHS."""
    out = []
    for c in g.circles:
        r = abs(1.0 / c.k) if c.k != 0 else 1e6
        # The bounding circle (k=-1) is special -- it contains everything.
        # We lift it to a high-altitude point above the origin.
        out.append(circle_to_uhs(c.z, r))
    return out
