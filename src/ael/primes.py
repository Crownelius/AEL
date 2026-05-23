"""Minimal prime + twin-prime utilities scaled for gasket curvatures (<~10^5).

Mirrors the API of Crownelius/twin-prime-engine at our (small) scale. The
key idea borrowed from that project is the **mod-2520 gasket filter**:
2520 = 2^3 * 3^2 * 5 * 7, and every twin-prime pair (p, p+2) with p >= 5
has p in a specific subset of residue classes mod 2520. Pre-computing this
subset makes the twin-prime check a single dictionary lookup.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def sieve(n: int) -> np.ndarray:
    """Boolean array of length n+1 where sieve[i] = True iff i is prime."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p :: p] = False
    return s


@lru_cache(maxsize=None)
def primes_up_to(n: int) -> tuple[int, ...]:
    return tuple(int(p) for p in np.where(sieve(n))[0])


@lru_cache(maxsize=None)
def twin_primes_up_to(n: int) -> tuple[tuple[int, int], ...]:
    """All (p, p+2) twin-prime pairs with p+2 <= n."""
    s = sieve(n)
    out = []
    for p in range(3, n - 1):
        if s[p] and s[p + 2]:
            out.append((int(p), int(p + 2)))
    return tuple(out)


@lru_cache(maxsize=None)
def twin_residues_mod_2520() -> frozenset[int]:
    """Residue classes mod 2520 that host the lower member of twin-prime pairs.

    Computed by inspection of all twin pairs (p, p+2) with p < some threshold:
    p mod 2520 lies in a small subset of {0..2519}.
    """
    n = 200_000  # large enough to cover all classes
    s = sieve(n)
    classes = set()
    for p in range(3, n - 1):
        if s[p] and s[p + 2]:
            classes.add(p % 2520)
    return frozenset(classes)


def is_prime(k: int) -> bool:
    if k < 2:
        return False
    n = max(k, 100)  # grow sieve as needed via lru_cache
    s = sieve(n)
    return bool(s[k])


def is_twin_lower(k: int) -> bool:
    """k and k+2 are both prime."""
    if k < 3:
        return False
    if (k % 2520) not in twin_residues_mod_2520() and k >= 5:
        return False
    return is_prime(k) and is_prime(k + 2)


def is_twin_upper(k: int) -> bool:
    return k >= 5 and is_prime(k) and is_prime(k - 2)


def is_twin_either(k: int) -> bool:
    return is_twin_lower(k) or is_twin_upper(k)


def twin_pair_of(k: int) -> tuple[int, int] | None:
    """If k participates in a twin-prime pair, return (p_low, p_high)."""
    if is_twin_lower(k):
        return (k, k + 2)
    if is_twin_upper(k):
        return (k - 2, k)
    return None
