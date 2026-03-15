"""Matrix validation utilities.

Helper functions used by the unitary construction modules to validate
matrix shape, dimensions, and optional unitarity constraints.
"""

from __future__ import annotations

import numpy as np


def require_square(U: np.ndarray) -> None:
    r"""Raise ValueError unless U is a square matrix.

    Parameters
    ----------
    U:
        Candidate matrix.

    Raises
    ------
    ValueError
        If U is not 2D or not square.

    """
    if U.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got ndim={U.ndim}")
    if U.shape[0] != U.shape[1]:
        raise ValueError(f"Expected square matrix, got shape={U.shape}")


def require_dim(U: np.ndarray, *, n: int) -> None:
    r"""Raise ValueError unless U has shape (n, n).

    Parameters
    ----------
    U:
        Candidate matrix.
    n:
        Required dimension.

    Raises
    ------
    ValueError
        If shape mismatch.

    """
    require_square(U)
    if U.shape != (n, n):
        raise ValueError(f"Expected shape {(n, n)}, got {U.shape}")


def require_unitary_optional(
    U: np.ndarray,
    *,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> None:
    r"""Optionally validate U is unitary, i.e. U^H U = I.

    This is deliberately optional because (a) it costs time, and (b) some uses
    may pass approximate numerics.

    Parameters
    ----------
    U:
        Candidate matrix.
    check_unitary:
        If True, validate unitarity.
    atol:
        Absolute tolerance for np.allclose.

    Raises
    ------
    ValueError
        If check_unitary is True and U is not unitary within atol.

    """
    require_square(U)
    if not check_unitary:
        return
    n = U.shape[0]
    Id = np.eye(n, dtype=np.complex128)
    UU = U.conjugate().T @ U
    if not np.allclose(UU, Id, atol=atol):
        raise ValueError("Matrix is not unitary within tolerance.")
