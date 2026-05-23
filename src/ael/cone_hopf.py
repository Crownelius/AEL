"""Cone attention on S^2 with optional phase weighting along Hopf fibers.

A cone on S^2 is parameterised by:
  apex     -- a base point u in S^2 (unit vector in R^3)
  axis     -- a unit tangent direction at u (in T_u S^2)
  aperture -- a geodesic half-angle on S^2 (radians)
  sigma    -- a spherical-distance scale (radians)

Weight at v in S^2:
    theta = angle between log_u(v) and axis in T_u S^2
    d     = sphere_dist(u, v)
    w_dir = exp(- (theta / aperture)^2)
    w_rad = exp(- (d / sigma)^2)
    w     = w_dir * w_rad

Optional phase weight: if both apex and v carry phases (along their Hopf
fibers), include a small term penalising large phase differences. Phase is
periodic so we use 1 - cos(phi_a - phi_v) as a distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ConeS2:
    apex: np.ndarray            # unit vector in R^3 (S^2)
    axis: np.ndarray            # unit tangent at apex (3-vector)
    aperture: float             # half-angle, radians
    sigma: float = 0.5          # spherical distance scale
    apex_phase: float | None = None   # optional phase along apex fiber
    phase_weight: float = 0.0   # 0 disables phase term

    def __post_init__(self) -> None:
        a = np.asarray(self.apex, dtype=float); a = a / max(float(np.linalg.norm(a)), 1e-12)
        self.apex = a
        ax = np.asarray(self.axis, dtype=float)
        # Project axis to tangent space at apex (remove apex-component).
        ax = ax - float(np.dot(ax, a)) * a
        n = float(np.linalg.norm(ax))
        self.axis = ax / n if n > 1e-12 else ax

    def weight(self, v: np.ndarray, v_phase: float | None = None) -> float:
        v = np.asarray(v, dtype=float)
        ip = float(np.clip(np.dot(self.apex, v), -1.0, 1.0))
        d = math.acos(ip)
        w_rad = math.exp(-(d / max(self.sigma, 1e-9)) ** 2)

        # log_apex(v) = (theta / sin(theta)) * (v - cos(theta) * apex), tangent.
        if d < 1e-9:
            return w_rad
        sd = math.sin(d)
        if sd < 1e-12:
            return w_rad
        log_v = (d / sd) * (v - ip * self.apex)
        n_log = float(np.linalg.norm(log_v))
        if n_log < 1e-12:
            return w_rad
        cos_theta = float(np.clip(np.dot(log_v / n_log, self.axis), -1.0, 1.0))
        theta = math.acos(cos_theta)
        w_dir = math.exp(-(theta / max(self.aperture, 1e-9)) ** 2)

        w = w_dir * w_rad

        if self.phase_weight > 0 and self.apex_phase is not None and v_phase is not None:
            phase_diff_d = 1.0 - math.cos(self.apex_phase - v_phase)  # in [0, 2]
            w_phase = math.exp(-self.phase_weight * phase_diff_d)
            w *= w_phase

        return w


def tangent_basis_at(u: np.ndarray) -> list[np.ndarray]:
    """Two orthonormal tangent vectors at u in S^2."""
    u = np.asarray(u, dtype=float); u = u / max(float(np.linalg.norm(u)), 1e-12)
    # Pick a vector not parallel to u.
    seed = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, u) * u
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(u, e1)
    return [e1, e2]


def four_axes_at(u: np.ndarray) -> list[np.ndarray]:
    e1, e2 = tangent_basis_at(u)
    return [e1, -e1, e2, -e2]


def make_multihead_s2(
    apex: np.ndarray, n_heads: int = 4, aperture: float = 0.5, sigma: float = 0.5
) -> list[ConeS2]:
    if n_heads == 2:
        e1, e2 = tangent_basis_at(apex)
        axes = [e1, e2]
    elif n_heads == 4:
        axes = four_axes_at(apex)
    else:
        # Evenly-spaced axes in T_u using a rotation of e1.
        e1, e2 = tangent_basis_at(apex)
        axes = []
        for h in range(n_heads):
            angle = 2.0 * math.pi * h / n_heads
            axes.append(math.cos(angle) * e1 + math.sin(angle) * e2)
    return [ConeS2(apex=apex, axis=a, aperture=aperture, sigma=sigma) for a in axes]


def multihead_score(
    cones: list[ConeS2], v: np.ndarray, v_phase: float | None = None, reduce: str = "max"
) -> float:
    weights = [c.weight(v, v_phase) for c in cones]
    if reduce == "max":
        return max(weights)
    return sum(weights) if reduce == "sum" else max(weights)


def log_map(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Logarithm map on S^2 at u, pointing toward v. Length = sphere_dist(u, v)."""
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    ip = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if ip > 1.0 - 1e-12:
        return np.zeros(3)
    d = math.acos(ip)
    sd = math.sin(d)
    if sd < 1e-12:
        return np.zeros(3)
    return (d / sd) * (v - ip * u)
