from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from symop_proto.gaussian.ops.common import (
    check_finite_all,
    check_is_square_matrix,
    check_shape_mat,
    check_shape_vec,
)


def ladder_change_of_basis(k: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Return the linear maps between quadratures and ladder operators
    (per k modes).

    Conventions
    -----------

    Quadrature operator vector is ordered as

    .. math::

        r_q = (x_1, \ldots, x_k, p_1, \ldots, p_k)^{\mathsf{T}}.

    Ladder operator vector is ordered as

    .. math::

        r_\ell = (a_1, \ldots, a_k,
        a_1^\dagger, \ldots, a_k^\dagger)^{\mathsf{T}}.

    with

    .. math::

        a = \frac{x - i p}{\sqrt{2}},\quad
        a^\dagger = \frac{x + i p}{\sqrt{2}},

    this function implements a linear relation

    .. math::

        r_\ell = T r_q, \qquad r_q = T^{-1}r_\ell.

    Here

    .. math::

        T = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
        I & -i I \\
        I & i I
        \end{pmatrix},\qquad
        T^{-1} = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
        I&I\\
        iI&-iI
        \end{pmatrix}.

    Returns
    -------
    (T, Tinv):
        Complex matrices of shape ``(2k,2k)``.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    Id = np.eye(k, dtype=complex)
    s = 1.0 / np.sqrt(2.0)

    T = s * np.block([[Id, -1j * Id], [Id, 1j * Id]])
    Tinv = s * np.block([[Id, Id], [1j * Id, -1j * Id]])

    return T, Tinv


def quadrature_to_ladder_affine(
    Xq: np.ndarray,
    Yq: np.ndarray,
    dq: Optional[np.ndarray] = None,
    *,
    check_finite=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Converts an affine Gaussian map from quadratures to ladder-operator
    coortinates


    Input map in quadrature coordinates
    -----------------------------------

    The map is specified on the quadrature operator vector

    .. math::

        r_q = (x_1, \ldots, x_k, p_1,\ldots, p_k)^{\mathsf{T}}

    as an affine transformation of first and second moments:

    .. math::

        \mu_q' = X_q \mu_q + d_q, \qquad
        V_q' = X_q V_q X_q^{\mathsf T} + Y_q.

    Output map in ladder coordinates
    --------------------------------

    Define the ladder operator vector

    .. math::

        r_\ell = (a_1, \ldots, a_k,a_1^\dagger, \ldots a_k^\dagger)^{\mathsf{T}}

    related by the change of basis

    .. math::

        r_\ell = T r_q, \qquad
        r_q = T^{-1}r_\ell

    where ``T`` is given by :func:`ladder_change_of_basis`.

    Then the same affine map written in ladder coordinates is

    .. math::

        \mu_\ell' = X_\ell \mu_\ell +d_\ell \quad
        V_\ell' = X_\ell V_\ell X_\ell^{mathsf{T}} + Y_\ell

    with

    .. math::

        X_\ell = T X_q T^{-1}, \qquad
        d_\ell = T d_q, \qquad
        Y_\ell = T Y_q T^{\mathsf{T}}


    Parameters
    ----------
    Xq, Yq:
        Quadrature matrices of shape ``(2k,2k)`` (typically real-valued).
    dq:
        Optional quadrature displacement of shape ``(2k,)``.
    check_finite:
        If True, reject NaN/Inf.

    Returns
    -------
    (Xl, Yl, dl):
        Ladder-coordinate matrices/vectors, complex-valued, shapes 
        ``(2k,2k)``, ``(2k,2k)``, ``(2k,)``.

    Examples
    --------
    Single-mode attenuation in quadratures:

    .. math::

        X_q = \sqrt{\eta}\, I_2,\qquad Y_q = (1-\eta)(2\bar n + 1) I_2,\qquad d_q=0.

    Convert to ladder coordinates and inspect:

    .. jupyter-execute::

        import numpy as np
        from symop_proto.gaussian.ops.transform import quadrature_to_ladder_affine

        eta = 0.8
        nbar = 0.0

        Xq = np.sqrt(eta) * np.eye(2)
        Yq = (1.0 - eta) * (2.0 * nbar + 1.0) * np.eye(2)

        Xl, Yl, dl = quadrature_to_ladder_affine(Xq, Yq)

        print("Xl:")
        print(Xl)
        print("Yl:")
        print(Yl)
        print("dl:")
        print(dl)

    Phase-space rotation (single mode) in quadratures:

    .. math::

        X_q(\theta) =
        \begin{pmatrix}
          \cos\theta & -\sin\theta \\
          \sin\theta & \cos\theta
        \end{pmatrix}, \quad Y_q=0.

    This becomes a diagonal phase on ladder operators.

    """
    Xq = np.asarray(Xq)
    Yq = np.asarray(Yq)
    check_is_square_matrix(Xq)
    check_shape_mat(Yq, Xq.shape, "Yq")

    k2 = Xq.shape[0]
    if k2 % 2 != 0:
        raise ValueError("Xq must be of shape (2k,2k) with even dimensions")
    k = k2 // 2

    Xq = check_shape_mat(Xq, (k2, k2), "Xq")
    Yq = check_shape_mat(Yq, (k2, k2), "Yq")
    dq = check_shape_vec(dq, k2, "dq")

    if check_finite:
        check_finite_all(Xq, Yq, dq)

    T, Tinv = ladder_change_of_basis(k)

    Xl = T @ Xq @ Tinv
    Yl = T @ Yq @ T.conj().T

    dl = np.zeros((k2,), dtype=complex) if dq is None else (T @ dq.astype(complex))

    return (
        np.asarray(Xl, dtype=complex),
        np.asarray(Yl, dtype=complex),
        np.asarray(dl, dtype=complex),
    )


def ladder_to_quadrature_affine(
    Xl: np.ndarray,
    Yl: np.ndarray,
    dl: Optional[np.ndarray] = None,
    *,
    check_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Inverse of :func:`quadrature_to_ladder_affine`.

    Given an affine map in ladder coordinates,

    .. math::

        \mu_\ell' = X_\ell \mu_\ell +d_\ell,\qquad
        V_\ell' = X_\ell V_\ell X_\ell^{\mathsf{T}}
        +Y_\ell,

    return the equivalent map in quadratures using

    .. math::

        X_q = T^{-1}X_ell T,\qquad
        d_q = T^{-1}d_\ell,\qquad
        Y_q = T^{-1} Y_\ell (T^{-1})^{\mathsf{T}}

    """
    Xl = np.asarray(Xl)
    Yl = np.asarray(Yl)

    if Xl.ndim != 2 or Xl.shape[0] != Xl.shape[1]:
        raise ValueError(f"Xl must be square, got shape {Xl.shape}")
    if Yl.shape != Xl.shape:
        raise ValueError(f"Yl must have shape {Xl.shape}, got {Yl.shape}")

    k2 = Xl.shape[0]
    if k2 % 2 != 0:
        raise ValueError("Xl must be of shape (2k,2k) with even dimension")
    k = k2 // 2

    dl = check_shape_vec(dl, k2, "dl")
    if check_finite:
        check_finite_all(Xl, Yl, dl)

    T, Tinv = ladder_change_of_basis(k)

    Xq = Tinv @ Xl @ T
    Yq = Tinv @ Yl @ Tinv.conj().T
    dq = np.zeros((k2,), dtype=complex) if dl is None else (Tinv @ dl.astype(complex))

    return (
        np.asarray(Xq, dtype=complex),
        np.asarray(Yq, dtype=complex),
        np.asarray(dq, dtype=complex),
    )
