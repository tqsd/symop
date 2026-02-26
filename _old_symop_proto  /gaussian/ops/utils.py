from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def permutation_matrix(p: Sequence[int]) -> np.ndarray:
    """Return permutation matrix P such that (P v)[i] = v[p[i]].

    Parameters
    ----------
    p:
        A permutation of 0..n-1.

    Returns
    -------
    np.ndarray
        Complex matrix of shape (n, n).

    """
    p_list = list(int(i) for i in p)
    n = len(p_list)
    if sorted(p_list) != list(range(n)):
        raise ValueError("p must be a permutation of 0..n-1")

    P = np.zeros((n, n), dtype=complex)
    for i, j in enumerate(p_list):
        P[i, j] = 1.0 + 0.0j
    return P


def invert_permutation(p: Sequence[int]) -> list[int]:
    """Invert a permutation p where (P v)[i] = v[p[i]].

    Returns p_inv such that p_inv[p[i]] = i.
    """
    p_list = list(int(i) for i in p)
    n = len(p_list)
    if sorted(p_list) != list(range(n)):
        raise ValueError("p must be a permutation of 0..n-1")

    inv = [0] * n
    for i, j in enumerate(p_list):
        inv[j] = i
    return inv


def swap_matrix(n: int, i: int, j: int) -> np.ndarray:
    """Return n x n permutation matrix that swaps indices i and j."""
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"swap indices out of range: i={i}, j={j}, n={n}")
    p = list(range(n))
    p[i], p[j] = p[j], p[i]
    return permutation_matrix(p)
