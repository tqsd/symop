"""Helpers for assembling mode-space unitary matrices.

Includes block-diagonal composition and embedding of local k×k
unitaries into a larger identity acting on multiple modes.
"""

from __future__ import annotations

import numpy as np

from .conventions import require_square, require_unitary_optional


def block_diag(*blocks: np.ndarray) -> np.ndarray:
    r"""Return the block-diagonal concatenation of multiple square matrices.

    If blocks are U1, U2, ..., then:

    .. math::

        U = \mathrm{diag}(U_1, U_2, \ldots).

    Parameters
    ----------
    blocks:
        Square matrices.

    Returns
    -------
    numpy.ndarray
        Block-diagonal matrix.

    Raises
    ------
    ValueError
        If any block is not square.

    """
    if len(blocks) == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    sizes = []
    for B in blocks:
        require_square(B)
        sizes.append(B.shape[0])

    n = int(sum(sizes))
    U = np.zeros((n, n), dtype=np.complex128)

    i = 0
    for B in blocks:
        m = B.shape[0]
        U[i : i + m, i : i + m] = B
        i += m
    return U


def embed_1(
    *,
    n: int,
    i: int,
    u1: complex,
) -> np.ndarray:
    r"""Embed a 1x1 unitary into an n-dimensional identity.

    .. math::

        U = I, \quad U_{i i} \leftarrow u_1.

    Parameters
    ----------
    n:
        Total dimension.
    i:
        Index to replace (0-based).
    u1:
        Complex scalar of modulus 1 (not enforced here).

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n).

    """
    nn = int(n)
    ii = int(i)
    if nn < 1:
        raise ValueError("n must be >= 1")
    if not (0 <= ii < nn):
        raise ValueError("i out of range")
    U = np.eye(nn, dtype=np.complex128)
    U[ii, ii] = complex(u1)
    return U


def embed_2(
    *,
    n: int,
    i: int,
    j: int,
    U2: np.ndarray,
) -> np.ndarray:
    r"""Embed a 2x2 unitary U2 acting on indices (i, j) into an n-dimensional identity.

    The embedding acts on the subspace spanned by basis vectors i and j, leaving
    all other basis vectors unchanged.

    Parameters
    ----------
    n:
        Total dimension.
    i, j:
        Indices to couple (0-based, distinct).
    U2:
        2x2 matrix.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n).

    """
    nn = int(n)
    ii = int(i)
    jj = int(j)
    if nn < 2:
        raise ValueError("n must be >= 2")
    if ii == jj:
        raise ValueError("i and j must be distinct")
    if not (0 <= ii < nn and 0 <= jj < nn):
        raise ValueError("indices out of range")

    U2 = np.asarray(U2, dtype=np.complex128)
    if U2.shape != (2, 2):
        raise ValueError(f"Expected U2 shape (2, 2), got {U2.shape}")

    U = np.eye(nn, dtype=np.complex128)
    idx = [ii, jj]
    U[np.ix_(idx, idx)] = U2
    return U


def embed_u(
    *,
    n: int,
    indices: list[int] | tuple[int, ...],
    Uk: np.ndarray,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> np.ndarray:
    r"""Embed a kxk unitary Uk onto a subset of indices inside an n-dimensional identity.

    Parameters
    ----------
    n:
        Total dimension.
    indices:
        Distinct indices (0-based) where Uk acts, in the order matching Uk.
    Uk:
        Square matrix of shape (k, k).
    check_unitary:
        If True, validate Uk is unitary.
    atol:
        Tolerance for optional unitary check.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (n, n).

    Raises
    ------
    ValueError
        If indices are invalid or Uk shape mismatches.

    """
    nn = int(n)
    idx = [int(x) for x in indices]
    if nn < 1:
        raise ValueError("n must be >= 1")
    if len(idx) == 0:
        return np.eye(nn, dtype=np.complex128)
    if len(set(idx)) != len(idx):
        raise ValueError("indices must be distinct")
    if min(idx) < 0 or max(idx) >= nn:
        raise ValueError("indices out of range")

    Uk = np.asarray(Uk, dtype=np.complex128)
    require_square(Uk)
    k = Uk.shape[0]
    if k != len(idx):
        raise ValueError(f"Uk is {k}x{k} but got {len(idx)} indices")

    require_unitary_optional(Uk, check_unitary=check_unitary, atol=atol)

    U = np.eye(nn, dtype=np.complex128)
    U[np.ix_(idx, idx)] = Uk
    return U
