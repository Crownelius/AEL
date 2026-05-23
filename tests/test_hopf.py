"""Sanity tests for Hopf fibration module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.hopf import (
    Quaternion,
    fiber_at,
    fibers_linked,
    hopf_map,
    hopf_section,
    phase_of,
    s3_dist,
    sphere_dist,
    sphere_to_stereographic,
    stereographic_to_sphere,
)


def test_hopf_map_lands_on_s2():
    """eta(q) is a unit vector for any unit quaternion."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        p = hopf_map(q)
        assert abs(float(np.linalg.norm(p)) - 1.0) < 1e-9


def test_section_recovers_point():
    """eta(hopf_section(p)) == p."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        q = hopf_section(v)
        assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-9, "section not on S^3"
        p2 = hopf_map(q)
        assert np.allclose(v, p2, atol=1e-8), f"{v} != {p2}"


def test_fiber_at_stays_on_fiber():
    """fiber_at(p, phi) projects back to the same p for any phi."""
    rng = np.random.default_rng(2)
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        q = fiber_at(v, float(phi))
        p = hopf_map(q)
        assert np.allclose(p, v, atol=1e-7), f"phi={phi}: {v} vs {p}"


def test_phase_round_trip():
    """phase_of(fiber_at(p, phi)) == phi (mod 2pi)."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    for phi in [0.1, 0.7, 1.5, -1.0, 2.5, -2.5]:
        q = fiber_at(v, phi)
        phi_back = phase_of(q)
        # Compare mod 2pi.
        diff = ((phi - phi_back + math.pi) % (2 * math.pi)) - math.pi
        assert abs(diff) < 1e-6, f"phi={phi}, recovered {phi_back}, diff={diff}"


def test_stereographic_round_trip():
    """C -> S^2 -> C is identity."""
    for z in [0+0j, 1+0j, 0+1j, 0.5-0.3j, 2.0+1.5j, -0.7-1.2j]:
        p = stereographic_to_sphere(z)
        assert abs(float(np.linalg.norm(p)) - 1.0) < 1e-9
        z2 = sphere_to_stereographic(p)
        assert abs(z - z2) < 1e-8, f"{z} != {z2}"


def test_origin_maps_to_south_pole():
    """z=0 -> (0,0,-1)."""
    p = stereographic_to_sphere(0 + 0j)
    assert np.allclose(p, [0, 0, -1])


def test_infinity_corresponds_to_north_pole():
    """Far-away z -> approaches (0,0,1)."""
    p = stereographic_to_sphere(1000 + 0j)
    assert p[2] > 0.99


def test_sphere_dist_symmetric():
    p = np.array([1, 0, 0]); q = np.array([0, 1, 0])
    assert abs(sphere_dist(p, q) - math.pi / 2) < 1e-9
    assert abs(sphere_dist(p, p) - 0.0) < 1e-9


def test_fibers_linked_when_different():
    """Any two distinct points on S^2 give distinct fibers."""
    p = np.array([1, 0, 0])
    q = np.array([0, 1, 0])
    assert fibers_linked(p, q)
    assert not fibers_linked(p, p)


def test_section_at_north_pole():
    """The Hopf section is well-defined at all points except possibly south."""
    p = np.array([0.0, 0.0, 1.0])  # north
    q = hopf_section(p)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-9
    assert np.allclose(hopf_map(q), p, atol=1e-9)


if __name__ == "__main__":
    tests = [
        ("Hopf maps to S^2", test_hopf_map_lands_on_s2),
        ("section recovers point", test_section_recovers_point),
        ("fiber stays on fiber", test_fiber_at_stays_on_fiber),
        ("phase round-trip", test_phase_round_trip),
        ("stereographic round-trip", test_stereographic_round_trip),
        ("origin -> south pole", test_origin_maps_to_south_pole),
        ("infinity -> near north", test_infinity_corresponds_to_north_pole),
        ("sphere dist symmetric", test_sphere_dist_symmetric),
        ("distinct fibers linked", test_fibers_linked_when_different),
        ("section at north pole", test_section_at_north_pole),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} Hopf tests passed.")
