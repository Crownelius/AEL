"""Hopf fibration: S^3 -> S^2 with S^1 fibers.

A unit quaternion q = (a, b, c, d) in R^4 with |q|=1 represents a point on
the 3-sphere. The Hopf map

    eta : S^3 -> S^2
    eta(q) = (2(ac + bd), 2(bc - ad), a^2 + b^2 - c^2 - d^2)

is the canonical principal U(1)-bundle. Two preimage points are on the same
fiber iff they differ by a unit-complex multiplication on (z1, z2) where
z1 = a + bi, z2 = c + di.

Key facts used by AEL:
  - Tangent circles on the Riemann sphere have Hopf-linked fibers.
  - The fiber over any p in S^2 is a great circle in S^3.
  - We canonically identify the gasket plane C with S^2 via stereographic
    projection from the north pole.

This module is pure numpy. Torch versions live in placement_head_hopf.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion + S^3 utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quaternion:
    a: float
    b: float
    c: float
    d: float

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c, self.d], dtype=float)

    @classmethod
    def from_array(cls, q: np.ndarray) -> "Quaternion":
        return cls(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def norm(self) -> float:
        return float(math.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2))

    def normalized(self) -> "Quaternion":
        n = self.norm()
        if n < 1e-12:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.a / n, self.b / n, self.c / n, self.d / n)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d
        return Quaternion(
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        )


# ---------------------------------------------------------------------------
# Hopf map and inverse
# ---------------------------------------------------------------------------

def hopf_map(q: np.ndarray) -> np.ndarray:
    """eta : S^3 -> S^2 via the standard formula on (a,b,c,d) coords."""
    q = np.asarray(q, dtype=float)
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x = 2.0 * (a * c + b * d)
    y = 2.0 * (b * c - a * d)
    z = a ** 2 + b ** 2 - c ** 2 - d ** 2
    return np.stack([x, y, z], axis=-1)


def hopf_section(p: np.ndarray) -> np.ndarray:
    """A canonical section S^2 -> S^3: pick the phase-0 representative on
    the fiber over p.

    For p = (x, y, z) on S^2 with z != -1:
        q = (1/sqrt(2(1+z))) * (1 + z, 0, x, -y)

    The negation on the y -> d component is required for the section to
    actually satisfy eta(q) = p with our Hopf-map convention
    eta(a,b,c,d) = (2(ac+bd), 2(bc-ad), a^2+b^2-c^2-d^2).

    For z = -1 (south pole) we return (0, 0, 1, 0) by convention.
    """
    p = np.asarray(p, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    out = np.zeros(p.shape[:-1] + (4,), dtype=float)
    south = z < -1.0 + 1e-9
    not_south = ~south
    denom = np.sqrt(np.clip(2.0 * (1.0 + z), 1e-12, None))
    a = (1.0 + z) / denom
    c = x / denom
    d = -y / denom
    out[..., 0] = np.where(not_south, a, 0.0)
    out[..., 1] = 0.0
    out[..., 2] = np.where(not_south, c, 1.0)
    out[..., 3] = np.where(not_south, d, 0.0)
    return out


def fiber_at(p: np.ndarray, phase: float) -> np.ndarray:
    """Quaternion on the Hopf fiber over p in S^2 at U(1) phase angle 'phase'."""
    q0 = hopf_section(p)                  # phase-0 representative
    # Rotate by e^{i*phase/2} = (cos(phase/2), sin(phase/2)) acting on (z1, z2)
    # where z1 = q[0] + q[1] i, z2 = q[2] + q[3] i.
    c = math.cos(phase / 2.0)
    s = math.sin(phase / 2.0)
    a0, b0, c0, d0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    a = c * a0 - s * b0
    b = c * b0 + s * a0
    cc = c * c0 - s * d0
    dd = c * d0 + s * c0
    return np.stack([a, b, cc, dd], axis=-1)


# ---------------------------------------------------------------------------
# Riemann sphere stereographic projection (plane -> S^2)
# ---------------------------------------------------------------------------

def stereographic_to_sphere(z: complex) -> np.ndarray:
    """Inverse stereographic from N=(0,0,1): C -> S^2 \\ {N}.

    For z = x + i y:
        X = 2x / (|z|^2 + 1)
        Y = 2y / (|z|^2 + 1)
        Z = (|z|^2 - 1) / (|z|^2 + 1)
    """
    x, y = z.real, z.imag
    r2 = x * x + y * y
    denom = r2 + 1.0
    return np.array([2.0 * x / denom, 2.0 * y / denom, (r2 - 1.0) / denom], dtype=float)


def sphere_to_stereographic(p: np.ndarray) -> complex:
    """S^2 \\ {N} -> C. Inverse of stereographic_to_sphere."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if z >= 1.0 - 1e-9:
        return complex(float("inf"))
    denom = 1.0 - z
    return complex(x / denom, y / denom)


# ---------------------------------------------------------------------------
# Distances on S^2 and S^3
# ---------------------------------------------------------------------------

def sphere_dist(p: np.ndarray, q: np.ndarray) -> float:
    """Great-circle distance on S^2: arccos(p . q), clamped."""
    ip = float(np.clip(np.dot(p, q), -1.0, 1.0))
    return math.acos(ip)


def s3_dist(p: np.ndarray, q: np.ndarray) -> float:
    """Great-circle distance on S^3."""
    ip = float(np.clip(np.dot(p, q), -1.0, 1.0))
    return math.acos(ip)


# ---------------------------------------------------------------------------
# Phase along a fiber
# ---------------------------------------------------------------------------

def phase_of(q: np.ndarray) -> float:
    """Recover the U(1) phase of q relative to the canonical section.

    Returns phi in (-pi, pi].
    """
    p = hopf_map(q)
    q0 = hopf_section(p)
    # q and q0 differ by left multiplication by (cos(phi/2), sin(phi/2), 0, 0)
    # acting on (z1, z2). Recover phi from q * q0^{-1} = e^{i phi/2}.
    # In quaternion terms: q = e^{i phi/2} * q0 (where i = first imag unit)
    # Equivalently: q * conj(q0) = (cos(phi/2), sin(phi/2), 0, 0).
    a0, b0, c0, d0 = q0[0], q0[1], q0[2], q0[3]
    a, b, c, d = q[0], q[1], q[2], q[3]
    # q * conj(q0)
    r0 = a * a0 + b * b0 + c * c0 + d * d0
    r1 = b * a0 - a * b0 + d * c0 - c * d0  # i-component, with sign convention
    phi = 2.0 * math.atan2(r1, r0)
    return float(phi)


# ---------------------------------------------------------------------------
# Hopf linking
# ---------------------------------------------------------------------------

def fibers_linked(p: np.ndarray, q: np.ndarray) -> bool:
    """True iff the Hopf fibers over p and q are distinct (and thus linked).

    Any two distinct Hopf fibers in S^3 have linking number 1.
    """
    return float(np.linalg.norm(p - q)) > 1e-9
