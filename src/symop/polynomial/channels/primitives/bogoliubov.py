r"""Bogoliubov rewrites for CCR polynomial representations.

This module implements active linear bosonic transformations that mix
creation and annihilation operators. In an ordered basis of modes, the
transformation is specified by matrices ``X`` and ``Y`` such that

.. math::

    a_k^\dagger
    \;\mapsto\;
    \sum_j X_{j k}\, a_j^\dagger + Y_{j k}\, a_j,

with the annihilation transformation determined by adjunction,

.. math::

    a_k
    \;\mapsto\;
    \sum_j \overline{Y_{j k}}\, a_j^\dagger
    +
    \overline{X_{j k}}\, a_j.

When ``Y = 0``, the transformation reduces to a passive linear mode map.

Notes
-----
This primitive is naturally suited for density and operator polynomial
rewrites. For ket polynomials, substitution may generate annihilation
operators even when acting on creator-only expressions, so an
additional CCR reduction step is required after rewriting.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.ccr.protocols.op import OpPoly as OpPolyProtocol
from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops import (
    ModeOp as ModeOpProtocol,
)
from symop.core.types.operator_kind import OperatorKind
from symop.polynomial.rewrites.substitution import (
    rewrite_densitypoly,
    rewrite_ketpoly,
    rewrite_oppoly,
)

SubstFn = Callable[[LadderOpProtocol], list[tuple[complex, LadderOpProtocol]]]
KetReducer = Callable[[KetPoly], KetPoly]


@dataclass(frozen=True)
class BogoliubovMap:
    r"""Represent an active linear bosonic map on an ordered mode basis.

    The ordered tuple ``modes`` defines the basis in which the matrices
    ``X`` and ``Y`` are interpreted. Column ``k`` of these matrices gives
    the image of input mode ``k`` in the ordered output basis.

    In the Heisenberg picture, the ladder operators transform as

    .. math::

        a_k^\dagger
        \;\mapsto\;
        \sum_j X_{j k}\, a_j^\dagger + Y_{j k}\, a_j,

    and

    .. math::

        a_k
        \;\mapsto\;
        \sum_j \overline{Y_{j k}}\, a_j^\dagger
        +
        \sum_j \overline{X_{j k}}\, a_j.

    Optional CCR validation checks the bosonic constraints

    .. math::

        X^\dagger X - Y^\dagger Y = I,

    and

    .. math::

        X^T Y - Y^T X = 0,

    within the numerical tolerance ``atol``.

    Parameters
    ----------
    modes : tuple of ModeOpProtocol
        Ordered mode basis used for the matrix representation.
    X : ndarray
        Matrix multiplying creation operators in the Bogoliubov
        transformation.
    Y : ndarray
        Matrix mixing creation and annihilation operators in the
        Bogoliubov transformation.
    check_ccr : bool, default=False
        If ``True``, validate the bosonic CCR consistency conditions.
    atol : float, default=1e-10
        Absolute tolerance used for optional CCR validation.

    Notes
    -----
    When ``Y = 0``, this object describes an ordinary passive linear mode
    transformation.

    """

    modes: tuple[ModeOpProtocol, ...]
    X: np.ndarray
    Y: np.ndarray
    check_ccr: bool = False
    atol: float = 1e-10

    def __post_init__(self) -> None:
        """Validate matrix dimensions, mode uniqueness, and optional CCR constraints."""
        n = len(self.modes)

        if len({m.signature for m in self.modes}) != n:
            raise ValueError("BogoliubovMap: modes must be distinct")

        if self.X.shape != (n, n):
            raise ValueError(
                f"BogoliubovMap: expected X shape {(n, n)}, got {self.X.shape}"
            )
        if self.Y.shape != (n, n):
            raise ValueError(
                f"BogoliubovMap: expected Y shape {(n, n)}, got {self.Y.shape}"
            )

        if self.check_ccr:
            eye = np.eye(n, dtype=np.complex128)

            lhs_1 = self.X.conjugate().T @ self.X - self.Y.conjugate().T @ self.Y
            lhs_2 = self.X.T @ self.Y - self.Y.T @ self.X

            if not np.allclose(lhs_1, eye, atol=self.atol):
                raise ValueError("BogoliubovMap: X^H X - Y^H Y != I within tolerance")
            if not np.allclose(lhs_2, 0.0, atol=self.atol):
                raise ValueError("BogoliubovMap: X^T Y - Y^T X != 0 within tolerance")


def make_substitution(bmap: BogoliubovMap) -> SubstFn:
    r"""Construct the ladder-operator substitution induced by a Bogoliubov map.

    The returned substitution function rewrites ladder operators
    according to the Heisenberg action of ``bmap``. If an operator acts
    on a mode not present in ``bmap.modes``, it is left unchanged.

    For creation operators, the substitution is

    .. math::

        a_k^\dagger
        \;\mapsto\;
        \sum_j X_{j k}\, a_j^\dagger + Y_{j k}\, a_j.

    For annihilation operators, the substitution is

    .. math::

        a_k
        \;\mapsto\;
        \sum_j \overline{Y_{j k}}\, a_j^\dagger
        +
        \sum_j \overline{X_{j k}}\, a_j.

    Parameters
    ----------
    bmap : BogoliubovMap
        Bogoliubov transformation specification.

    Returns
    -------
    callable
        Substitution function mapping a ladder operator to a list of
        ``(coefficient, operator)`` pairs.

    Notes
    -----
    The substitution acts only on operators whose mode signatures are
    present in ``bmap.modes``. Other operators pass through unchanged.

    """
    modes = bmap.modes
    X = bmap.X
    Y = bmap.Y
    index = {m.signature: i for i, m in enumerate(modes)}

    def subst(op: LadderOpProtocol) -> list[tuple[complex, LadderOpProtocol]]:
        k = index.get(op.mode.signature, None)
        if k is None:
            return [(1.0 + 0.0j, op)]

        xcol = X[:, k]
        ycol = Y[:, k]
        out: list[tuple[complex, LadderOpProtocol]] = []

        if op.kind == OperatorKind.CRE:
            for j in range(len(modes)):
                if xcol[j] != 0:
                    out.append((complex(xcol[j]), modes[j].create))
                if ycol[j] != 0:
                    out.append((complex(ycol[j]), modes[j].ann))
            return out

        for j in range(len(modes)):
            if ycol[j] != 0:
                out.append((complex(np.conjugate(ycol[j])), modes[j].create))
            if xcol[j] != 0:
                out.append((complex(np.conjugate(xcol[j])), modes[j].ann))
        return out

    return subst


def apply_to_densitypoly(
    rho: DensityPolyProtocol, *, bmap: BogoliubovMap
) -> DensityPolyProtocol:
    r"""Apply a Bogoliubov rewrite to a density polynomial.

    The density polynomial is rewritten by substituting ladder operators
    on both the left and right monomials according to the Bogoliubov
    transformation defined by ``bmap``.

    Parameters
    ----------
    rho : DensityPolyProtocol
        Input density polynomial.
    bmap : BogoliubovMap
        Bogoliubov transformation to apply.

    Returns
    -------
    DensityPolyProtocol
        Rewritten density polynomial after the active linear bosonic
        transformation.

    """
    subst = make_substitution(bmap)
    return rewrite_densitypoly(rho, subst)


def apply_to_oppoly(op: OpPolyProtocol, *, bmap: BogoliubovMap) -> OpPolyProtocol:
    r"""Apply a Bogoliubov rewrite to an operator polynomial.

    Each ladder operator in the operator polynomial is rewritten
    according to the Bogoliubov transformation defined by ``bmap``.

    Parameters
    ----------
    op : OpPolyProtocol
        Input operator polynomial.
    bmap : BogoliubovMap
        Bogoliubov transformation to apply.

    Returns
    -------
    OpPolyProtocol
        Rewritten operator polynomial after the active linear bosonic
        transformation.

    """
    subst = make_substitution(bmap)
    return rewrite_oppoly(op, subst)


def apply_to_ketpoly(
    poly: KetPoly,
    *,
    bmap: BogoliubovMap,
    reduce_ketpoly: KetReducer,
) -> KetPoly:
    r"""Apply a Bogoliubov rewrite to a ket polynomial and reduce the result.

    The ket polynomial is first rewritten by substituting ladder
    operators according to the Bogoliubov transformation. Because active
    transformations may generate annihilation operators even when acting
    on creator-only expressions, the rewritten polynomial must then be
    reduced using a caller-supplied CCR reduction routine.

    Parameters
    ----------
    poly : KetPoly
        Input ket polynomial.
    bmap : BogoliubovMap
        Bogoliubov transformation to apply.
    reduce_ketpoly : callable
        Callback that performs the required CCR reduction or normal
        ordering after substitution.

    Returns
    -------
    KetPoly
        Reduced ket polynomial after substitution and CCR reduction.

    Notes
    -----
    Unlike passive linear mode maps, a Bogoliubov transformation cannot
    in general be handled by simply discarding annihilator-containing
    terms. The reduction callback is responsible for applying the CCR
    relations and returning a valid ket representation.

    """
    subst = make_substitution(bmap)
    rewritten = rewrite_ketpoly(poly, subst)
    return reduce_ketpoly(rewritten)
