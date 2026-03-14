"""Density term for operator-algebra expressions."""

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
class KetTerm:
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

    This class implements
    :class:`~symop.core.protocols.terms.ket_term.KetTerm` protocol.

    """

    coeff: complex
    monomial: MonomialProtocol

    @staticmethod
    def identity() -> KetTerm:
        r"""Return the multiplicative identity term.

        This is the term with unit coefficient and the identity monomial:

        .. math::

            1 \cdot I.
        """
        return KetTerm(1.0, Monomial())

    def adjoint(self) -> KetTerm:
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
    def signature(self) -> Signature:
        """Return a signature."""
        return ("KT", self.monomial.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
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
    def mode_ops(self) -> tuple[ModeOpProtocol, ...]:
        """Return the ordered mode operators appearing in the monomial."""
        return self.monomial.mode_ops


if TYPE_CHECKING:
    from symop.core.protocols.terms import KetTerm as KetTermProtocol

    _ket_term_check: KetTermProtocol = KetTerm.identity()
