"""Sanity tests for Cl(3,1) Descartes module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.descartes_3d import (
    REFLECTION_MATRICES,
    apollonian_reflection,
    descartes_form,
    hyper_dist,
    hyper_log,
    is_descartes,
    minkowski_inner,
    minkowski_norm_sq,
    tangent_unit_axis,
    to_hyperboloid,
)


def test_root_quadruple_is_descartes():
    """The (-1,2,2,3) integral quadruple satisfies the Descartes form."""
    b = np.array([-1, 2, 2, 3], dtype=float)
    assert abs(descartes_form(b)) < 1e-9


def test_non_degenerate_quadruple():
    """(-1, 2, 3, 6) is in the (-1,2,2,3) gasket."""
    b = np.array([-1, 2, 3, 6], dtype=float)
    assert is_descartes(b)


def test_reflection_preserves_descartes():
    """The four Apollonian reflections all preserve F."""
    b = np.array([-1, 2, 3, 6], dtype=float)
    for i in range(4):
        b2 = apollonian_reflection(b, i)
        assert is_descartes(b2), f"S_{i}({b}) = {b2} not Descartes"


def test_reflection_matrix_matches_function():
    """REFLECTION_MATRICES applied as linear ops match the function form."""
    b = np.array([-1, 2, 3, 6], dtype=float)
    for i in range(4):
        b_fn = apollonian_reflection(b, i)
        b_mat = REFLECTION_MATRICES[i] @ b
        assert np.allclose(b_fn, b_mat), f"mismatch at i={i}"


def test_reflection_is_involution():
    """S_i^2 = I."""
    b = np.array([-1, 2, 3, 6], dtype=float)
    for i in range(4):
        b2 = apollonian_reflection(apollonian_reflection(b, i), i)
        assert np.allclose(b, b2)


def test_known_reflection_value():
    """S_0 on (-1, 2, 3, 6) replaces -1 with 23 (the next ancestor in the gasket)."""
    b = np.array([-1, 2, 3, 6], dtype=float)
    b2 = apollonian_reflection(b, 0)
    assert b2[0] == 23.0
    assert (b2[1:] == b[1:]).all()


def test_descartes_minkowski_relation():
    """F(b) = -<b, b>_M with our convention."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = rng.normal(size=4)
        assert abs(descartes_form(b) + minkowski_norm_sq(b)) < 1e-9


def test_hyperboloid_unit_norm():
    """to_hyperboloid produces <u, u>_M = +1."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        b = np.abs(rng.normal(size=4)) + 0.1  # ensure timelike with positive sum
        u = to_hyperboloid(b)
        assert abs(minkowski_norm_sq(u.u) - 1.0) < 1e-9, f"<u,u> = {minkowski_norm_sq(u.u)}"


def test_hyper_distance_symmetric():
    rng = np.random.default_rng(2)
    a = to_hyperboloid(np.abs(rng.normal(size=4)) + 0.1)
    b = to_hyperboloid(np.abs(rng.normal(size=4)) + 0.1)
    d_ab = hyper_dist(a, b)
    d_ba = hyper_dist(b, a)
    assert abs(d_ab - d_ba) < 1e-9
    assert d_ab >= 0


def test_log_lies_in_tangent_space():
    """log_u(v) is Minkowski-orthogonal to u."""
    rng = np.random.default_rng(3)
    u = to_hyperboloid(np.abs(rng.normal(size=4)) + 0.1)
    v = to_hyperboloid(np.abs(rng.normal(size=4)) + 0.1)
    w = hyper_log(u, v)
    assert abs(minkowski_inner(w, u.u)) < 1e-6, f"<log, u>_M = {minkowski_inner(w, u.u)}"


def test_tangent_axis_orthogonal():
    rng = np.random.default_rng(4)
    u = to_hyperboloid(np.abs(rng.normal(size=4)) + 0.1)
    w = tangent_unit_axis(u, rng.normal(size=4))
    assert abs(minkowski_inner(w, u.u)) < 1e-6


def test_minkowski_negative_definite_on_tangent_space():
    """The induced metric on T_u is -<.,.>_M, i.e. the Minkowski form is
    negative-definite there. Guards the sign convention the cone relies on."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        u = to_hyperboloid(np.abs(rng.normal(size=4)) + 0.1)
        w = tangent_unit_axis(u, rng.normal(size=4))
        # tangent_unit_axis normalises in g = -<w,w>_M, so <w,w>_M must be <= 0.
        assert minkowski_inner(w, w) <= 1e-9, f"<w,w>_M = {minkowski_inner(w, w)} > 0 on tangent space"


def test_tangent_axis_is_g_unit():
    """tangent_unit_axis returns a vector of unit length in the induced metric
    g(w,w) = -<w,w>_M."""
    rng = np.random.default_rng(8)
    u = to_hyperboloid(np.abs(rng.normal(size=4)) + 0.1)
    w = tangent_unit_axis(u, rng.normal(size=4))
    g = -minkowski_inner(w, w)
    assert abs(g - 1.0) < 1e-6, f"g-norm^2 = {g}, expected 1"


def test_cone3d_directional_selectivity():
    """A Cone3D pointing toward a point must score it higher than a point in
    the opposite direction. Regression test for the abs(cos) bug that made
    forward and backward indistinguishable."""
    from src.ael.cone_3d import Cone3D
    from src.ael.descartes_3d import hyper_log

    apex = to_hyperboloid(np.array([3.0, 1.0, 1.0, 1.0]))
    p_fwd = to_hyperboloid(np.array([3.0, 1.2, 1.0, 1.0]))
    p_bwd = to_hyperboloid(np.array([3.0, 0.8, 1.0, 1.0]))
    axis = hyper_log(apex, p_fwd)
    cone = Cone3D(apex=apex, axis=axis, aperture=0.5, sigma=5.0)
    w_fwd = cone.weight(p_fwd)
    w_bwd = cone.weight(p_bwd)
    assert w_fwd > w_bwd + 0.1, f"forward {w_fwd:.4f} not clearly > backward {w_bwd:.4f}"


if __name__ == "__main__":
    tests = [
        ("root quadruple", test_root_quadruple_is_descartes),
        ("non-degenerate quadruple", test_non_degenerate_quadruple),
        ("reflection preserves descartes", test_reflection_preserves_descartes),
        ("reflection matrix = function", test_reflection_matrix_matches_function),
        ("reflection involution", test_reflection_is_involution),
        ("known reflection value", test_known_reflection_value),
        ("descartes/minkowski relation", test_descartes_minkowski_relation),
        ("hyperboloid unit norm", test_hyperboloid_unit_norm),
        ("hyper distance symmetric", test_hyper_distance_symmetric),
        ("log in tangent space", test_log_lies_in_tangent_space),
        ("tangent axis orthogonal", test_tangent_axis_orthogonal),
        ("minkowski neg-definite on T_u", test_minkowski_negative_definite_on_tangent_space),
        ("tangent axis is g-unit", test_tangent_axis_is_g_unit),
        ("cone3d directional selectivity", test_cone3d_directional_selectivity),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
    print("\nAll Cl(3,1) tests complete.")
