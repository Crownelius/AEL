"""Twin-prime addressing for gasket circles.

Every circle gets an address tuple

    Address(curvature, twin_partner, descent_path)

where:
  - curvature: signed integer (matches gasket curvature)
  - twin_partner: the OTHER member of the twin-prime pair if this curvature
    participates in one, else None
  - descent_path: tuple of parent indices in the gasket (the recursive
    'reflection history' that produced this circle)

Twin-prime-anchored circles act as 'named landmarks' in the gasket. Circles
without twin-prime curvatures still have addresses, but reference their
nearest twin-prime anchor for retrieval / similarity boosts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gasket import Gasket
from .primes import is_twin_either, twin_pair_of


@dataclass(frozen=True)
class Address:
    curvature: int
    twin_partner: int | None
    descent_path: tuple[int, ...]   # gasket BFS path
    is_twin_anchor: bool

    def __str__(self) -> str:
        if self.twin_partner is not None:
            return f"#({self.curvature},{self.twin_partner})"
        return f"#{self.curvature}"


def address_of_circle(g: Gasket, idx: int, path_cache: dict[int, tuple[int, ...]] | None = None) -> Address:
    """Compute the canonical address of circle idx."""
    c = g.circles[idx]
    k_int = int(round(c.k))
    pair = twin_pair_of(k_int) if k_int > 0 else None
    if pair is not None:
        partner = pair[1] if pair[0] == k_int else pair[0]
    else:
        partner = None

    # Reconstruct descent path: chain of parent triples back to root.
    if path_cache is None:
        path_cache = {}
    path = _descent_path(g, idx, path_cache)
    return Address(
        curvature=k_int,
        twin_partner=partner,
        descent_path=path,
        is_twin_anchor=pair is not None,
    )


def _descent_path(g: Gasket, idx: int, cache: dict[int, tuple[int, ...]]) -> tuple[int, ...]:
    if idx in cache:
        return cache[idx]
    if idx not in g.parents:
        cache[idx] = (idx,)
        return cache[idx]
    # Choose the deterministic-canonical parent (lowest index).
    parents = g.parents[idx]
    primary_parent = min(parents)
    parent_path = _descent_path(g, primary_parent, cache)
    full = parent_path + (idx,)
    cache[idx] = full
    return full


def address_all(g: Gasket) -> list[Address]:
    cache: dict[int, tuple[int, ...]] = {}
    return [address_of_circle(g, i, cache) for i in range(len(g.circles))]


def twin_anchor_indices(g: Gasket) -> list[int]:
    """Indices of circles whose curvature participates in a twin-prime pair."""
    return [i for i, c in enumerate(g.circles)
            if c.k > 0 and is_twin_either(int(round(c.k)))]


@dataclass
class TwinAddressBook:
    """Lookup table from address curvature -> list of circle indices."""

    by_curvature: dict[int, list[int]] = field(default_factory=dict)
    by_twin_pair: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    anchors: list[int] = field(default_factory=list)

    def add(self, g: Gasket, idx: int, addr: Address) -> None:
        self.by_curvature.setdefault(addr.curvature, []).append(idx)
        if addr.twin_partner is not None:
            pair = tuple(sorted((addr.curvature, addr.twin_partner)))
            self.by_twin_pair.setdefault(pair, []).append(idx)  # type: ignore[arg-type]
        if addr.is_twin_anchor:
            self.anchors.append(idx)

    @classmethod
    def build(cls, g: Gasket) -> "TwinAddressBook":
        book = cls()
        cache: dict[int, tuple[int, ...]] = {}
        for i in range(len(g.circles)):
            addr = address_of_circle(g, i, cache)
            book.add(g, i, addr)
        return book
