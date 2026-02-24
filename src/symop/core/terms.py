"""Ket and density terms for operator-algebra expressions.

This module defines atomic algebraic terms used to build state and
density-operator expansions in the symbolic ladder-operator formalism.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from symop.core.monomial import Monomial
from symop.core.protocols import (
    DensityTermProto,
    KetTermProto,
    ModeOpProto,
    MonomialProto,
    SignatureProto,
)


@dataclass(frozen=True)
class KetTerm(KetTermProto):
    r"""A single term in a ket-like operator expression.

    A ``KetTerm`` represents a coefficient multiplied by a monomial in ladder
    operators (typically in normal-ordered form, depending on how
    :class:`~symop.core.monomial.Monomial` is defined):

    .. math::

        \lvert \psi \rangle \;\sim\; c \, M,

    where :math:`c \in \mathbb{C}` and :math:`M` is a monomial of ladder
    operators (e.g., products of :math:`a_i` and :math:`a_i^\dagger`).

    Notes
    -----
    ``KetTerm`` is immutable (``frozen=True``). Use :meth:`scaled` to obtain a
    modified copy with a scaled coefficient.

    """

    coeff: complex
    monomial: MonomialProto

    @staticmethod
    def identity() -> KetTermProto:
        r"""Return the multiplicative identity term.

        This is the term with unit coefficient and the identity monomial:

        .. math::

            1 \cdot I.
        """
        return KetTerm(1.0, Monomial())

    def adjoint(self) -> KetTermProto:
        r"""Return the adjoint (Hermitian conjugate) of this term.

        The adjoint is defined by conjugating the scalar coefficient and
        taking the monomial adjoint:

        .. math::

            (c\,M)^\dagger = c^*\, M^\dagger.
        """
        return KetTerm(coeff=self.coeff.conjugate(), monomial=self.monomial.adjoint())

    def scaled(self, s: complex) -> KetTerm:
        r"""Return a copy with the coefficient scaled by ``s``.

        The monomial is unchanged:

        .. math::

            (c\,M) \mapsto (s\,c)\,M.
        """
        return replace(self, coeff=self.coeff * s)

    @property
    def signature(self) -> SignatureProto:
        """Return a signature."""
        return ("KT", self.monomial.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> SignatureProto:
        """Return an approximate signature."""
        return (
            "KT_approx",
            self.monomial.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )

    @property
    def is_creator_only(self) -> bool:
        """Return ``True`` if the monomial contains only creation operators."""
        return self.monomial.is_creator_only

    @property
    def is_annihilator_only(self) -> bool:
        """Return ``True`` if the monomial contains only annihilation operators."""
        return self.monomial.is_annihilator_only

    @property
    def is_identity(self) -> bool:
        """Return ``True`` if the monomial is the identity."""
        return self.monomial.is_identity

    @property
    def creation_count(self) -> int:
        r"""Return the number of creation operators in the monomial.

        .. math::

            n_\mathrm{create} = \#\{\text{creators in } M\}.
        """
        return len(self.monomial.creators)

    @property
    def annihilation_count(self) -> int:
        r"""Return the number of annihilation operators in the monomial.

        .. math::

            n_\mathrm{ann} = \#\{\text{annihilators in } M\}.
        """
        return len(self.monomial.annihilators)

    @property
    def total_degree(self) -> int:
        r"""Return the total ladder-operator degree of the monomial.

        .. math::

            \deg(M) = n_\mathrm{create} + n_\mathrm{ann}.
        """
        return self.creation_count + self.annihilation_count

    @property
    def mode_ops(self) -> tuple[ModeOpProto, ...]:
        """Return the ordered mode operators appearing in the monomial."""
        return self.monomial.mode_ops


@dataclass(frozen=True)
class DensityTerm(DensityTermProto):
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

    """

    coeff: complex
    left: MonomialProto
    right: MonomialProto

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
    def signature(self) -> SignatureProto:
        """Return a signature."""
        return ("DT", "L", self.left.signature, "R", self.right.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> SignatureProto:
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
    def mode_ops_left(self) -> tuple[ModeOpProto, ...]:
        """Return the ordered mode operators appearing in the left monomial."""
        return self.left.mode_ops

    @property
    def mode_ops_right(self) -> tuple[ModeOpProto, ...]:
        """Return the ordered mode operators appearing in the right monomial."""
        return self.right.mode_ops
