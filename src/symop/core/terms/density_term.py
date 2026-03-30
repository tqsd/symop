"""Density-operator terms built from left and right monomials.

This module defines :class:`DensityTerm`, a structural building block for
density-operator expressions in the operator algebra.

A density term consists of a complex coefficient together with two monomials,
one acting on the left and one acting on the right. This makes it suitable
for representing operator-basis expansions of density operators, outer-product
like expressions, and intermediate forms used in symbolic manipulations of
quantum channels and states.

The class provides structural queries such as identity detection, creator-only
and annihilator-only checks, degree counts, mode extraction, and exact or
approximate signatures for hashing and merging.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from symop.core.monomial import Monomial
from symop.core.protocols.ops import (
    ModeOp as ModeOpProtocol,
)
from symop.core.protocols.ops import (
    Monomial as MonomialProtocol,
)
from symop.core.types.signature import Signature


@dataclass(frozen=True)
class DensityTerm:
    r"""A single term in a density-operator expression.

    A ``DensityTerm`` represents a coefficient multiplying a left and right
    monomial, matching the common pattern for outer-product-like or
    superoperator bookkeeping:

    .. math::

        \rho \;\sim\; c \, L \, (\cdot) \, R,

    or, in a bra-ket flavored picture, a term proportional to
    :math:`c\, L \lvert \psi \rangle \langle \phi \rvert R` depending on the
    surrounding formalism.

    The class stores left/right monomials explicitly, and provides helpers for
    structural queries (identity, creator-only, degree counts, etc.).

    Notes
    -----
    ``DensityTerm`` is immutable (``frozen=True``). Use :meth:`scaled` to
    obtain a modified copy with a scaled coefficient.

    This class implements
    :class:`~symop.core.terms.density_term.DensityTerm` protocol.

    """

    coeff: complex
    left: MonomialProtocol
    right: MonomialProtocol

    @staticmethod
    def identity() -> DensityTerm:
        r"""Return the multiplicative identity density term.

        This is the term with unit coefficient and identity monomials on both
        sides:

        .. math::

            1 \cdot I \;\; \text{(left)} \quad,\quad 1 \cdot I \;\; \text{(right)}.
        """
        return DensityTerm(1.0, Monomial(), Monomial())

    def adjoint(self) -> DensityTerm:
        r"""Return the adjoint (Hermitian conjugate) of this density term.

        The adjoint conjugates the coefficient and swaps left/right:

        .. math::

            (c, L, R)^\dagger = (c^*, R, L).
        """
        return DensityTerm(self.coeff.conjugate(), self.right, self.left)

    def scaled(self, s: complex) -> DensityTerm:
        r"""Return a copy with the coefficient scaled by ``s``.

        .. math::

            (c, L, R) \mapsto (s\,c, L, R).
        """
        return replace(self, coeff=self.coeff * s)

    @property
    def signature(self) -> Signature:
        """Return a signature."""
        return ("DT", "L", self.left.signature, "R", self.right.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        """Return an approximate signature."""
        return (
            "DT_approx",
            "L",
            self.left.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
            "R",
            self.right.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )

    @property
    def is_creator_only_left(self) -> bool:
        """Return ``True`` if the left monomial is creator-only or identity."""
        return self.left.is_creator_only or self.left.is_identity

    @property
    def is_creator_only_right(self) -> bool:
        """Return ``True`` if the right monomial is creator-only or identity."""
        return self.right.is_creator_only or self.right.is_identity

    @property
    def is_creator_only(self) -> bool:
        """Return ``True`` if both left and right are creator-only or identity."""
        return self.is_creator_only_left and self.is_creator_only_right

    @property
    def is_annihilator_only_left(self) -> bool:
        """Return ``True`` if the left monomial is annihilator-only or identity."""
        return self.left.is_annihilator_only or self.left.is_identity

    @property
    def is_annihilator_only_right(self) -> bool:
        """Return ``True`` if the right monomial is annihilator-only or identity."""
        return self.right.is_annihilator_only or self.right.is_identity

    @property
    def is_annihilator_only(self) -> bool:
        """Return ``True`` if both left and right are annihilator-only or identity."""
        return self.is_annihilator_only_left and self.is_annihilator_only_right

    @property
    def is_identity_left(self) -> bool:
        """Return ``True`` if the left monomial is the identity."""
        return self.left.is_identity

    @property
    def is_identity_right(self) -> bool:
        """Return ``True`` if the right monomial is the identity."""
        return self.right.is_identity

    @property
    def is_diagonal_in_monomials(self) -> bool:
        r"""Return ``True`` if left and right monomials are structurally identical.

        This checks equality at the signature level:

        .. math::

            \mathrm{sig}(L) = \mathrm{sig}(R).
        """
        return self.left.signature == self.right.signature

    @property
    def creation_count_left(self) -> int:
        """Return the number of creation operators in the left monomial."""
        return len(self.left.creators)

    @property
    def creation_count_right(self) -> int:
        """Return the number of creation operators in the right monomial."""
        return len(self.right.creators)

    @property
    def annihilation_count_left(self) -> int:
        """Return the number of creation operators in the right monomial."""
        return len(self.left.annihilators)

    @property
    def annihilation_count_right(self) -> int:
        """Return the number of annihilation operators in the right monomial."""
        return len(self.right.annihilators)

    @property
    def mode_ops_left(self) -> tuple[ModeOpProtocol, ...]:
        """Return the ordered mode operators appearing in the left monomial."""
        return self.left.mode_ops

    @property
    def mode_ops_right(self) -> tuple[ModeOpProtocol, ...]:
        """Return the ordered mode operators appearing in the right monomial."""
        return self.right.mode_ops


if TYPE_CHECKING:
    from symop.core.protocols.terms import (
        DensityTerm as DensityTermProtocol,
    )
    from symop.core.protocols.terms import (
        KetTerm as KetTermProtocol,
    )
    from symop.core.terms import KetTerm

    _ket_term_check: KetTermProtocol = KetTerm(1.0, Monomial())
    _density_term_check: DensityTermProtocol = DensityTerm(1.0, Monomial(), Monomial())
