from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import numpy as np

from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.core import GaussianCore


def as_index_list(core: GaussianCore, modes: Sequence[ModeOpProto]) -> List[int]:
    return [core.basis.require_index_of(m) for m in modes]


def check_is_square_matrix(X: np.ndarray, *, name: Optional[str] = None) -> None:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        if name is not None:
            raise ValueError(f"{name} must be square, got shape {X.shape}")
        raise ValueError(f"matrix must be square, got shape {X.shape}")


def check_is_unitary(U: np.ndarray, *, atol: float = 1e-12) -> None:
    check_is_square_matrix(U)
    k = U.shape[0]
    I = np.eye(k, dtype=complex)
    lhs = U.conj().T @ U
    if not np.allclose(lhs, I, atol=atol, rtol=0.0):
        err = float(np.max(np.abs(lhs - I)))
        raise ValueError(f"U is not unitary; max abs(U^dag U - I) = {err}")


def check_is_same_shape(U: np.ndarray, V: np.ndarray) -> None:
    if U.shape != V.shape:
        raise ValueError(f"U and V must have same shapes, got {U.shape} vs {V.shape}")


def check_is_subset_indices(n: int, idx: Sequence[int]) -> None:
    idx_list = list(idx)
    if len(idx_list) == 0:
        raise ValueError("idx must not be empty")
    if any((i < 0 or i >= n) for i in idx_list):
        raise IndexError(f"idx out of range for n={n}: {idx_list}")
    if len(set(idx_list)) != len(idx_list):
        raise ValueError(f"idx contains duplicates: {idx_list}")


def embed_subset_matrices(
    n: int, idx: Sequence[int], U: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds full (n,n) matrices `U_full` and `V_full` that act as (U,V) on idx
    and as (I,0) on the complement.
    """
    U_full = np.eye(n, dtype=complex)
    V_full = np.zeros((n, n), dtype=complex)
    idx_list = list(idx)
    U_full[np.ix_(idx_list, idx_list)] = U
    V_full[np.ix_(idx_list, idx_list)] = V
    return U_full, V_full


def check_ccr_preservation(
    G: np.ndarray, U_full: np.ndarray, V_full: np.ndarray, *, atol: float
) -> None:
    r"""
    Checks CCR preservation in a general (possibly non-orthogonal) basis.

    For ladder operators with :math:`[a_i, a_j^\dagger] = G_{ij}`, a Bogoliubov
    map

    .. math::

        a' = U a + V a^\dagger

    preserves CCR if:

    .. math::

        U G U^\dagger - V G^* V^\dagger = G, \quad
        U G V^\dagger - V G^* U^T = 0

    where * is elementwise complex conjugation.

    This reduces to the standard symplectic comstraints in the canonical case
    :math:`G=I`
    """
    G = np.asarray(G, dtype=complex)

    lhs1 = U_full @ G @ U_full.conj().T - V_full @ G.conj() @ V_full.conj().T
    if not np.allclose(lhs1, G, atol=atol, rtol=0.0):
        err = float(np.max(np.abs(lhs1 - G)))
        raise ValueError(f"Bogoliubov map violates [a,a^dag]=G, max abs error: {err}")

    lhs2 = U_full @ G @ V_full.T - V_full @ G.conj() @ U_full.T
    if not np.allclose(lhs2, 0.0, atol=atol, rtol=0.0):
        err = float(np.max(np.abs(lhs2)))
        raise ValueError(f"Bogoliubov map violates [a,a]=0; max abs error: {err}")


def alpha_from_quadrature_mean(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).reshape(-1)
    n2 = d.shape[0]
    if n2 % 2 != 0:
        raise ValueError("Quadrature mean must have even length")
    n = n2 // 2
    x = d[0:n]
    p = d[n : 2 * n]
    return (x + 1j * p) / np.sqrt(2.0)


def ladder_indices(n: int, idx_list: List[int]) -> List[int]:
    r"""
    Ladder ordering indices for

    .. math::

        r = (a_1, \ldots, a_n, a_1^\dagger, \ldots, a_n^\dagger)

    """
    return idx_list + [i + n for i in idx_list]


def as_complex_matrix(
    a: np.ndarray, *, shape: Tuple[int, int], name: str
) -> np.ndarray:
    out = np.asarray(a, dtype=complex)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out


def as_optional_complex_vector(
    a: Optional[np.ndarray], *, k2: int, name: str
) -> Optional[np.ndarray]:
    if a is None:
        return None
    out = np.asarray(a, dtype=complex).reshape(-1)
    if out.shape != (k2,):
        raise ValueError(f"{name} must have shape ({k2},), got {out.shape}")
    return out


def check_finite_all(X: np.ndarray, Y: np.ndarray, d0: Optional[np.ndarray]) -> None:
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains NaN/Inf")
    if d0 is not None and (not np.isfinite(d0).all()):
        raise ValueError("d0 contains NaN/Inf")


def check_shape_mat(A: np.ndarray, shape: Tuple[int, int], name) -> np.ndarray:
    out = np.asarray(A)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out


def check_shape_vec(
    v: Optional[np.ndarray], k2: int, name: str
) -> Optional[np.ndarray]:
    if v is None:
        return None
    out = np.asarray(v).reshape(-1)
    if out.shape != (k2,):
        raise ValueError(f"{name} must have shape ({k2}, ), got {out.shape}")
    return out


def quadrature_indices(n: int, idx_list: List[int], which: str) -> List[int]:
    which = str(which)
    if which == "x":
        return list(idx_list)
    if which == "p":
        return [i + n for i in idx_list]
    if which == "xp":
        return list(idx_list) + [i + n for i in idx_list]
    raise ValueError("which must be one of: 'x', 'p', 'xp'")


def as_real_vec(y: np.ndarray, *, m: int, name: str) -> np.ndarray:
    out = np.asarray(y, dtype=float).reshape(-1)
    if out.shape != (m,):
        raise ValueError(f"{name} must have shape ({m},), got {out.shape}")
    return out


def as_real_mat(A: np.ndarray, *, shape: Tuple[int, int], name: str) -> np.ndarray:
    out = np.asarray(A, dtype=float)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out


def complement_indices(n: int, sel: Sequence[int]) -> np.ndarray:
    sel_set = set(int(i) for i in sel)
    return np.asarray([i for i in range(n) if i not in sel_set], dtype=int)
