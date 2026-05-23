"""Descartes circle theorem.

Given four mutually tangent circles with curvatures k1..k4 (curvature = 1/radius,
signed: negative if the circle contains the other three),

    (k1 + k2 + k3 + k4)^2 = 2 (k1^2 + k2^2 + k3^2 + k4^2).

Solving for k4 given k1, k2, k3:

    k4 = k1 + k2 + k3 +/- 2 * sqrt(k1*k2 + k2*k3 + k3*k1).

The two roots are the two circles tangent to the first three. Apollonian
recursion: replace one of {k1,k2,k3} with the new k4 and recurse.

Complex Descartes extension (Lagarias-Mallows-Wilks) lifts curvatures and
curvature*center products into the same identity, giving us the centers:

    (k1 z1 + k2 z2 + k3 z3 + k4 z4)^2
        = 2 (k1^2 z1^2 + k2^2 z2^2 + k3^2 z3^2 + k4^2 z4^2),

where z_i is the center as a complex number. Solving:

    k4 z4 = k1 z1 + k2 z2 + k3 z3 +/- 2 * sqrt(k1 k2 z1 z2 + k2 k3 z2 z3 + k3 k1 z3 z1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Circle:
    """Curvature is signed (negative = bounding circle). Center is complex."""

    k: float       # curvature
    z: complex     # center as complex number

    @property
    def r(self) -> float:
        return abs(1.0 / self.k) if self.k != 0 else math.inf

    def __repr__(self) -> str:
        return f"Circle(k={self.k:g}, z={self.z.real:.4f}{self.z.imag:+.4f}j)"


def descartes_fourth_curvature(k1: float, k2: float, k3: float) -> tuple[float, float]:
    """Return the two solutions for k4 given three mutually tangent curvatures."""
    s = k1 + k2 + k3
    disc = k1 * k2 + k2 * k3 + k3 * k1
    if disc < 0:
        # Numerical noise on integer inputs; clamp.
        if disc > -1e-9:
            disc = 0.0
        else:
            raise ValueError(f"Negative discriminant: {disc}")
    root = 2.0 * math.sqrt(disc)
    return s + root, s - root


def descartes_fourth_circle(c1: Circle, c2: Circle, c3: Circle) -> tuple[Circle, Circle]:
    """Return both Apollonian companions of three mutually tangent circles."""
    k1, k2, k3 = c1.k, c2.k, c3.k
    z1, z2, z3 = c1.z, c2.z, c3.z

    k_sum = k1 + k2 + k3
    k_disc = k1 * k2 + k2 * k3 + k3 * k1
    if k_disc < 0 and k_disc > -1e-9:
        k_disc = 0.0
    k_root = 2.0 * math.sqrt(k_disc)
    k4_plus = k_sum + k_root
    k4_minus = k_sum - k_root

    kz_sum = k1 * z1 + k2 * z2 + k3 * z3
    kz_disc = k1 * k2 * z1 * z2 + k2 * k3 * z2 * z3 + k3 * k1 * z3 * z1
    # Complex sqrt — principal branch.
    kz_root = 2.0 * _csqrt(kz_disc)
    # Two sign combinations consistent with the curvature signs.
    kz4_plus = kz_sum + kz_root
    kz4_minus = kz_sum - kz_root

    z4_plus = kz4_plus / k4_plus if k4_plus != 0 else complex(math.inf)
    z4_minus = kz4_minus / k4_minus if k4_minus != 0 else complex(math.inf)

    return Circle(k4_plus, z4_plus), Circle(k4_minus, z4_minus)


def _csqrt(w: complex) -> complex:
    """Complex sqrt, principal branch (real part >= 0)."""
    return w ** 0.5


def verify_descartes(c1: Circle, c2: Circle, c3: Circle, c4: Circle, tol: float = 1e-6) -> bool:
    """Sanity check: do four circles satisfy the Descartes identity?"""
    k = [c1.k, c2.k, c3.k, c4.k]
    lhs = sum(k) ** 2
    rhs = 2.0 * sum(x * x for x in k)
    return abs(lhs - rhs) < tol
