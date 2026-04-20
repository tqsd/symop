"""Linear mode-map rewrites for CCR polynomial representations.

This module defines passive linear transformations on a finite ordered
set of modes and provides helpers for applying such transformations to
symbolic CCR polynomial objects.

The linear map acts in the Heisenberg picture by rewriting ladder
operators according to a unitary mode mixing matrix. The same
substitution machinery is then reused to transform ket, density, and
operator polynomials.

Notes
-----
The ordered tuple ``modes`` defines the basis in which the matrix
representation is interpreted. Column ``k`` of the matrix gives the
image of input mode ``k`` expressed in the ordered output basis.

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


@dataclass(frozen=True)
class LinearModeMap:
    r"""Represent a passive linear transformation on an ordered mode basis.

    The ordered tuple ``modes`` defines both the input and output basis
    used to interpret the matrix ``U``. The convention is that column
    ``k`` of ``U`` gives the image of input mode ``k`` in the ordered
    output basis.

    In the Heisenberg picture, the ladder operators transform as

    .. math::

        a_k^\dagger
        \;\mapsto\;
        \sum_j U_{j k}\, a_j^\dagger,

    and

    .. math::

        a_k
        \;\mapsto\;
        \sum_j \overline{U_{j k}}\, a_j.

    Parameters
    ----------
    modes : tuple of ModeOpProtocol
        Ordered mode basis used for the matrix representation.
    U : ndarray
        Square matrix representing the passive linear transformation in
        the ordered basis given by ``modes``.
    check_unitary : bool, default=False
        If ``True``, validate that ``U`` is unitary within the tolerance
        specified by ``atol``.
    atol : float, default=1e-10
        Absolute tolerance used for optional unitary validation.

    Notes
    -----
    This class stores only the linear transformation data and does not
    itself apply the rewrite. Use ``make_substitution`` or one of the
    ``apply_to_*`` helpers to act on symbolic polynomial objects.

    """

    modes: tuple[ModeOpProtocol, ...]
    U: np.ndarray
    check_unitary: bool = False
    atol: float = 1e-10

    def __post_init__(self) -> None:
        """Validate the matrix shape and optionally check unitarity."""
        n = len(self.modes)
        if self.U.shape != (n, n):
            raise ValueError(
                f"LinearModeMap: expected U shape {(n, n)}, got {self.U.shape}"
            )
        if self.check_unitary:
            Id = np.eye(n, dtype=np.complex128)
            UU = self.U.conjugate().T @ self.U
            if not np.allclose(UU, Id, atol=self.atol):
                raise ValueError("LinearModeMap: U is not unitary within tolerance")


def make_substitution(lmap: LinearModeMap) -> SubstFn:
    r"""Construct the ladder-operator substitution induced by a linear mode map.

    The returned substitution function rewrites ladder operators
    according to the Heisenberg action of ``lmap``. If an operator acts
    on a mode not present in ``lmap.modes``, it is left unchanged.

    For creation operators, the substitution is

    .. math::

        a_k^\dagger
        \;\mapsto\;
        \sum_j U_{j k}\, a_j^\dagger.

    For annihilation operators, the substitution is

    .. math::

        a_k
        \;\mapsto\;
        \sum_j \overline{U_{j k}}\, a_j.

    Parameters
    ----------
    lmap : LinearModeMap
        Linear mode transformation specification.

    Returns
    -------
    callable
        Substitution function mapping a ladder operator to a list of
        ``(coefficient, operator)`` pairs.

    Notes
    -----
    The substitution acts only on operators whose mode signatures are
    present in ``lmap.modes``. Other operators pass through unchanged.

    """
    modes = lmap.modes
    U = lmap.U
    index = {m.signature: i for i, m in enumerate(modes)}

    def subst(op: LadderOpProtocol) -> list[tuple[complex, LadderOpProtocol]]:
        k = index.get(op.mode.signature)
        if k is None:
            return [(1.0 + 0.0j, op)]
        col = U[:, k]

        if op.kind == OperatorKind.CRE:
            return [(complex(col[j]), modes[j].create) for j in range(len(modes))]
        else:
            return [
                (complex(np.conjugate(col[j])), modes[j].ann) for j in range(len(modes))
            ]

    return subst


def make_right_substitution(lmap: LinearModeMap) -> SubstFn:
    r"""Construct the bra-side substitution induced by a linear mode map.

    This is the substitution used on the right monomial of a density term,
    corresponding to the adjoint-side action needed for

        rho -> U rho U^\dagger.

    For creation operators, the coefficients are conjugated relative to the
    ket-side map. For annihilation operators, the unconjugated coefficients
    are used.

    """
    modes = lmap.modes
    U = lmap.U
    index = {m.signature: i for i, m in enumerate(modes)}

    def subst(op: LadderOpProtocol) -> list[tuple[complex, LadderOpProtocol]]:
        k = index.get(op.mode.signature)
        if k is None:
            return [(1.0 + 0.0j, op)]
        col = U[:, k]

        if op.kind == OperatorKind.CRE:
            return [
                (complex(np.conjugate(col[j])), modes[j].create)
                for j in range(len(modes))
            ]

        return [(complex(col[j]), modes[j].ann) for j in range(len(modes))]

    return subst


def _apply_ketpoly_to_vacuum(poly: KetPoly) -> KetPoly:
    """Project a rewritten ket polynomial onto the vacuum reference sector.

    After operator substitution and normal ordering, terms containing
    annihilation operators acting on the vacuum are discarded. Only
    creator-only monomials and the identity term are retained.

    Parameters
    ----------
    poly : KetPoly
        Input ket polynomial after symbolic rewriting.

    Returns
    -------
    KetPoly
        Ket polynomial containing only vacuum-compatible terms.

    """
    return KetPoly(
        tuple(
            t
            for t in poly.terms
            if t.monomial.is_creator_only or t.monomial.is_identity
        )
    ).combine_like_terms()


def apply_to_ketpoly(poly: KetPoly, *, lmap: LinearModeMap) -> KetPoly:
    r"""Apply a passive linear mode map to a ket polynomial.

    The transformation is implemented by rewriting all ladder operators
    in the polynomial according to the Heisenberg action of ``lmap`` and
    then discarding terms incompatible with the vacuum reference state.

    Parameters
    ----------
    poly : KetPoly
        Input ket polynomial.
    lmap : LinearModeMap
        Passive linear mode transformation to apply.

    Returns
    -------
    KetPoly
        Rewritten ket polynomial after vacuum-compatible projection.

    Notes
    -----
    The final projection step keeps only creator-only monomials and the
    identity term, corresponding to the usual vacuum-based ket
    polynomial semantics.

    """
    subst = make_substitution(lmap)
    rewritten = rewrite_ketpoly(poly, subst)
    return _apply_ketpoly_to_vacuum(rewritten)


def apply_to_densitypoly(
    rho: DensityPolyProtocol, *, lmap: LinearModeMap
) -> DensityPolyProtocol:
    r"""Apply a passive linear mode map to a density polynomial.

    The density polynomial is rewritten by substituting ladder operators
    on both the left and right monomials according to the Heisenberg
    action of ``lmap``.

    Parameters
    ----------
    rho : DensityPolyProtocol
        Input density polynomial.
    lmap : LinearModeMap
        Passive linear mode transformation to apply.

    Returns
    -------
    DensityPolyProtocol
        Rewritten density polynomial after the linear mode
        transformation.

    Notes
    -----
    This function applies the same substitution map to both sides of the
    density polynomial, with annihilation operators transformed using
    the conjugated matrix elements as required by the Heisenberg
    convention.

    """
    left_subst = make_substitution(lmap)
    right_subst = make_right_substitution(lmap)
    return rewrite_densitypoly(rho, left_subst, right_subst)


def apply_to_oppoly(op: OpPolyProtocol, *, lmap: LinearModeMap) -> OpPolyProtocol:
    r"""Apply a passive linear mode map to an operator polynomial.

    Each ladder operator in the operator polynomial is rewritten
    according to the Heisenberg action of ``lmap``.

    Parameters
    ----------
    op : OpPolyProtocol
        Input operator polynomial.
    lmap : LinearModeMap
        Passive linear mode transformation to apply.

    Returns
    -------
    OpPolyProtocol
        Rewritten operator polynomial after the linear mode
        transformation.

    """
    subst = make_substitution(lmap)
    return rewrite_oppoly(op, subst)
