"""Sanity tests for the gasket generator against known math."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.descartes import verify_descartes
from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3


def test_root_quadruple_satisfies_descartes():
    root = standard_root_neg1_2_2_3()
    assert verify_descartes(*root)


def test_integer_curvatures_preserved():
    """The (-1,2,2,3) gasket has all-integer curvatures."""
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=5)
    for c in g.circles:
        assert abs(c.k - round(c.k)) < 1e-6, f"Non-integer curvature: {c.k}"


def test_known_curvature_prefix():
    """Curvature spectrum prefix matches the documented (-1,2,2,3) sequence.

    From Graham, Lagarias, Mallows, Wilks, Yan (2003) and OEIS-style listings.
    """
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=6)
    ks = sorted(set(round(c.k) for c in g.circles))
    expected_prefix = [-1, 2, 3, 6, 11, 14, 15, 18, 23, 26, 27, 30, 35, 38, 39, 42]
    assert ks[: len(expected_prefix)] == expected_prefix, (
        f"Got: {ks[:len(expected_prefix)]}"
    )


def test_residue_classes_mod_12():
    """All curvatures must be in residue classes {2, 3, 6, 11} mod 12, plus -1."""
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=6)
    allowed = {2, 3, 6, 11}
    for c in g.circles:
        k = round(c.k)
        if k == -1:
            continue
        assert k % 12 in allowed, f"Curvature {k} not in allowed residue classes"


def test_every_non_root_circle_has_three_tangencies():
    """Each generated circle is tangent to exactly the 3 parents that birthed it,
    plus any later-discovered circles that happen to be tangent to it."""
    g = build_gasket(standard_root_neg1_2_2_3(), max_depth=4)
    for idx in range(4, len(g.circles)):
        # At minimum, the 3 parents.
        assert len(g.adj[idx]) >= 3


if __name__ == "__main__":
    test_root_quadruple_satisfies_descartes()
    test_integer_curvatures_preserved()
    test_known_curvature_prefix()
    test_residue_classes_mod_12()
    test_every_non_root_circle_has_three_tangencies()
    print("All gasket tests passed.")
