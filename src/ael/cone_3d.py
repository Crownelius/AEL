"""Cone attention on the hyperboloid H^3 ⊂ R^{3,1}.

A cone is parameterised by:
  apex     -- a point u in H^3 (timelike unit Minkowski vector)
  axis     -- a unit tangent vector at u, in T_u H (spacelike, <axis,u>_M=0)
  aperture -- a geodesic half-angle, in radians
  sigma    -- a geodesic-distance scale

For a candidate point v in H^3, write log_u(v) in T_u H. This tangent
vector has Euclidean-style norm equal to the geodesic distance d(u, v) and
direction = the geodesic direction at u toward v. We compute

    theta = angle in tangent space between axis and log_u(v)
    w_dir = exp(- (theta / aperture)^2)
    w_rad = exp(- (d / sigma)^2)
    w     = w_dir * w_rad.

This is smooth, differentiable in (axis, aperture, sigma), and reduces to
the 2D cone when the hyperboloid is restricted to a 2-plane.

Multi-head: a list of cones at the same apex, axes spaced around T_u.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .descartes_3d import (
    HyperboloidPoint,
    hyper_dist,
    hyper_log,
    minkowski_inner,
    tangent_unit_axis,
    to_hyperboloid,
)


@dataclass
class Cone3D:
    apex: HyperboloidPoint
    axis: np.ndarray       # tangent vector at apex, Minkowski-orthogonal to apex.u
    aperture: float        # half-angle (radians)
    sigma: float = 1.0     # geodesic-distance scale

    def __post_init__(self) -> None:
        # Ensure axis is in the tangent space and unit-norm in tangent metric.
        self.axis = tangent_unit_axis(self.apex, np.asarray(self.axis, dtype=float))

    def weight(self, v: HyperboloidPoint) -> float:
        # Distance term.
        d = hyper_dist(self.apex, v)
        w_rad = math.exp(-(d / max(self.sigma, 1e-9)) ** 2)

        # Direction term: angle between log_u(v) and axis in the tangent space.
        # The Minkowski form is NEGATIVE-definite on T_u (the single positive
        # eigendirection is spanned by u itself), so the induced Riemannian
        # metric is g(a, b) = -<a, b>_M.  Normalise and take the cosine in g.
        log_v = hyper_log(self.apex, v)
        g_log = -float(minkowski_inner(log_v, log_v))  # >= 0 on the tangent space
        n_log = math.sqrt(max(g_log, 0.0))
        if n_log < 1e-12:
            return w_rad  # v == apex (or almost)
        log_unit = log_v / n_log
        # cos(theta) = g(log_unit, axis) = -<log_unit, axis>_M.  Do NOT take the
        # absolute value: a cone pointing away from v must score lower than one
        # pointing toward it, which |cos| would destroy.
        cos_theta = -float(minkowski_inner(log_unit, self.axis))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        w_dir = math.exp(-(theta / max(self.aperture, 1e-9)) ** 2)

        return w_dir * w_rad


def make_multihead_3d(
    apex: HyperboloidPoint,
    axes: list[np.ndarray],
    aperture: float = 0.4,
    sigma: float = 1.0,
) -> list[Cone3D]:
    """Build a multi-head cone from an apex and a list of ambient axis vectors."""
    return [Cone3D(apex=apex, axis=a, aperture=aperture, sigma=sigma) for a in axes]


def default_basis_axes(apex: HyperboloidPoint) -> list[np.ndarray]:
    """Three tangent axes at apex via Gram-Schmidt on the standard basis.

    Returns three Minkowski-orthogonal-to-apex vectors that span T_u.
    """
    out: list[np.ndarray] = []
    for e in np.eye(4):
        v = e - minkowski_inner(e, apex.u) * apex.u  # remove apex-component
        for w in out:
            # Subtract projection (in tangent metric).
            # Use Minkowski inner; tangent vectors at a timelike point.
            ip = float(minkowski_inner(v, w))
            v = v - ip * w
        n2 = abs(float(minkowski_inner(v, v)))
        if n2 < 1e-12:
            continue
        out.append(v / math.sqrt(n2))
        if len(out) == 3:
            break
    return out


def multihead_score(cones: list[Cone3D], v: HyperboloidPoint, reduce: str = "max") -> float:
    weights = [c.weight(v) for c in cones]
    if reduce == "max":
        return max(weights)
    if reduce == "sum":
        return sum(weights)
    raise ValueError(f"unknown reduce: {reduce}")
