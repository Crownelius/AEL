"""Cone attention in the upper half-space model of H^3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .uhs import UHSPoint, uhs_dist, uhs_log


@dataclass
class ConeUHS:
    apex: UHSPoint
    axis: np.ndarray       # 3D unit direction in ambient tangent space at apex
    aperture: float        # half-angle (radians)
    sigma: float = 1.0     # hyperbolic-distance scale

    def __post_init__(self) -> None:
        a = np.asarray(self.axis, dtype=float)
        n = float(np.linalg.norm(a))
        self.axis = a / n if n > 1e-12 else a

    def weight(self, v: UHSPoint) -> float:
        d = uhs_dist(self.apex, v)
        w_rad = math.exp(-(d / max(self.sigma, 1e-9)) ** 2)
        log_v = uhs_log(self.apex, v)
        nlog = float(np.linalg.norm(log_v))
        if nlog < 1e-12:
            return w_rad
        cos_theta = float(np.dot(log_v / nlog, self.axis))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        w_dir = math.exp(-(theta / max(self.aperture, 1e-9)) ** 2)
        return w_dir * w_rad


def basis_axes_3d() -> list[np.ndarray]:
    return [np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])]


def six_axes_3d() -> list[np.ndarray]:
    """Six axes along +/-x, +/-y, +/-t."""
    e = []
    for a in basis_axes_3d():
        e.append(a); e.append(-a)
    return e


def multihead_score(cones: list[ConeUHS], v: UHSPoint, reduce: str = "max") -> float:
    weights = [c.weight(v) for c in cones]
    if reduce == "max":
        return max(weights)
    return sum(weights) if reduce == "sum" else max(weights)
