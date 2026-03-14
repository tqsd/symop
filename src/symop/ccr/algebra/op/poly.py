r"""Operator polynomials.

This module defines :class:`OpPoly`, a finite linear combination of operator
words (:class:`~symop.core.terms.op_term.OpTerm`).

.. math::

    \mathcal{O} = \sum_k c_k \, W_k,

where each :math:`W_k` is an ordered product of ladder operators.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import TYPE_CHECKING, Self

from symop.ccr.protocols.actions import (
    SupportsLeftWordAction,
    SupportsRightWordAction,
)
from symop.ccr.protocols.common import Additive, Scaled
from symop.ccr.protocols.ket import KetPoly
from symop.core.protocols.ops.operators import LadderOp, ModeOp
from symop.core.terms.op_term import OpTerm
from symop.core.types.signature import Signature

from .combine import combine_like_terms
from .from_words import from_words
from .multiply import multiply


@dataclass(frozen=True)
class OpPoly:
    r"""Finite linear combination of :class:`OpTerm` objects.

    Parameters
    ----------
    terms:
        Tuple of operator terms.

    Notes
    -----
    Multiplication does not automatically merge like terms; call
    :meth:`combine_like_terms` to normalize.

    """

    terms: tuple[OpTerm, ...] = ()

    @staticmethod
    def from_words(
        words: Iterable[Iterable[LadderOp]],
        coeffs: Iterable[complex] | None = None,
    ) -> OpPoly:
        """Construct an operator polynomial from words and coefficients."""
        return OpPoly(from_words(words, coeffs, term_factory=OpTerm))

    @staticmethod
    def identity(c: complex = 1.0) -> OpPoly:
        """Return the identity operator polynomial."""
        return OpPoly((OpTerm.identity(c),))

    @staticmethod
    def zero() -> OpPoly:
        """Return the zero operator polynomial."""
        return OpPoly(())

    @staticmethod
    def a(mode: ModeOp) -> OpPoly:
        """Return the annihilation operator for ``mode``."""
        return OpPoly.from_words([[mode.ann]])

    @staticmethod
    def adag(mode: ModeOp) -> OpPoly:
        """Return the creation operator for ``mode``."""
        return OpPoly.from_words([[mode.create]])

    @staticmethod
    def n(mode: ModeOp) -> OpPoly:
        r"""Return the number operator :math:`a^\dagger a` for ``mode``."""
        return OpPoly.from_words([[mode.create, mode.ann]])

    @staticmethod
    def q(mode: ModeOp) -> OpPoly:
        r"""Return the quadrature :math:`q = (a + a^\dagger)/\sqrt{2}`."""
        return (
            OpPoly.from_words([[mode.ann]]) + OpPoly.from_words([[mode.create]])
        ) * (1.0 / sqrt(2))

    @staticmethod
    def x(mode: ModeOp) -> OpPoly:
        """Alias for :meth:`q`."""
        return OpPoly.q(mode)

    @staticmethod
    def p(mode: ModeOp) -> OpPoly:
        r"""Return the quadrature :math:`p = (i a^\dagger - i a)/\sqrt{2}`."""
        return OpPoly.from_words([[mode.create]]) * (1j / sqrt(2)) + OpPoly.from_words(
            [[mode.ann]]
        ) * (-1j / sqrt(2))

    @staticmethod
    def X_theta(mode: ModeOp, theta: float) -> OpPoly:
        r"""Return the rotated quadrature :math:`X_\theta`."""
        e_m = cos(theta) - 1j * sin(theta)
        e_p = cos(theta) + 1j * sin(theta)
        return (
            OpPoly.from_words([[mode.ann]]) * e_m
            + OpPoly.from_words([[mode.create]]) * e_p
        ) * (1.0 / sqrt(2))

    @staticmethod
    def q2(mode: ModeOp) -> OpPoly:
        """Return :math:`q^2`."""
        q = OpPoly.q(mode)
        return (q * q).combine_like_terms()

    @staticmethod
    def p2(mode: ModeOp) -> OpPoly:
        """Return :math:`p^2`."""
        p = OpPoly.p(mode)
        return (p * p).combine_like_terms()

    @staticmethod
    def n2(mode: ModeOp) -> OpPoly:
        """Return :math:`n^2`."""
        n = OpPoly.n(mode)
        return (n * n).combine_like_terms()

    def scaled(self, c: complex) -> OpPoly:
        """Return a scaled copy."""
        return OpPoly(tuple(term.scaled(c) for term in self.terms))

    def adjoint(self) -> OpPoly:
        """Return the Hermitian adjoint polynomial."""
        return OpPoly(tuple(term.adjoint() for term in self.terms))

    def combine_like_terms(
        self,
        *,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> OpPoly:
        r"""Combine operator terms with identical signatures.

        Terms are bucketed either by their exact signatures or, when
        ``approx=True``, by approximate signatures obtained via
        :meth:`OpTermProto.approx_signature`.

        When approximate matching is enabled, the parameters
        ``decimals`` and ``ignore_global_phase`` are forwarded to the
        underlying signature computation of the ladder operators.

        Parameters
        ----------
        approx:
            If ``True``, use approximate signatures for bucketing.
            Otherwise, exact signatures are used.
        decimals:
            Number of decimal places used when rounding floating-point
            quantities in approximate signatures.
        ignore_global_phase:
            If ``True``, component signatures may ignore global phase
            factors when constructing approximate signatures.

        Returns
        -------
        OpPoly
            A new polynomial with coefficients of identical words summed
            and zero-sum buckets removed.

        Notes
        -----
        This operation performs purely signature-based merging. It does
        not reorder words or apply commutation relations.

        See Also
        --------
        combine_like_terms (:func:`~symop.ccr.algebra.combine.combine_like_terms`)

        """
        return OpPoly(
            combine_like_terms(
                self.terms,
                approx=approx,
                term_factory=OpTerm,
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            )
        )

    def normalize(
        self,
        *,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> OpPoly:
        r"""Return a normalized operator polynomial.

        This is a convenience alias for :meth:`combine_like_terms`. It
        produces a polynomial in which identical operator words have been
        merged and zero-coefficient terms removed.

        Parameters
        ----------
        approx:
            If ``True``, approximate signatures are used for merging.
        decimals:
            Decimal precision used for approximate signature rounding.
        ignore_global_phase:
            If ``True``, global phase factors may be ignored in
            approximate signature comparisons.

        Returns
        -------
        OpPoly
            Normalized operator polynomial.

        """
        return self.combine_like_terms(
            approx=approx,
            decimals=decimals,
            ignore_global_phase=ignore_global_phase,
        )

    @property
    def is_zero(self) -> bool:
        """Return ``True`` iff there are no terms."""
        return len(self.terms) == 0

    @property
    def is_identity(self) -> bool:
        """Return ``True`` iff all terms are empty words and there is at least one."""
        return len(self.terms) > 0 and all(len(t.ops) == 0 for t in self.terms)

    @property
    def unique_modes(self) -> tuple[ModeOp, ...]:
        """Return unique modes appearing in the polynomial (first-seen order)."""
        seen: dict[Signature, ModeOp] = {}
        for t in self.terms:
            for lop in t.ops:
                seen.setdefault(lop.mode.signature, lop.mode)
        return tuple(seen.values())

    @property
    def mode_count(self) -> int:
        """Return the number of unique modes appearing in the polynomial."""
        return len(self.unique_modes)

    def __add__(self, other: Self) -> OpPoly:
        r"""Return the symbolic sum ``self + other``.

        Addition concatenates term tuples and does not automatically merge
        identical operator words. Use :meth:`combine_like_terms` (or
        :meth:`normalize`) to bucket by signatures and sum coefficients.

        Parameters
        ----------
        other:
            Another operator polynomial.

        Returns
        -------
        OpPoly
            The formal sum as an :class:`OpPoly`.

        """
        return OpPoly((*self.terms, *other.terms))

    def __sub__(self, other: OpPoly) -> OpPoly:
        r"""Return the symbolic difference ``self - other``.

        Defined as

        .. math::

            self - other \equiv self + (-1)\,other.

        Parameters
        ----------
        other:
            Another operator polynomial.

        Returns
        -------
        OpPoly
            The formal difference polynomial.

        """
        return self + (-1.0) * other

    def __neg__(self) -> OpPoly:
        r"""Return the additive inverse ``-self``.

        Equivalent to scalar multiplication by ``-1``.
        """
        return self.scaled(-1.0)

    def __mul__(self, other: Self | complex) -> OpPoly:
        r"""Return the symbolic product.

        This operator implements two cases:

        1. **Scalar scaling** (right scalar):

           .. math::

               \mathcal{O}\,c \equiv c\,\mathcal{O}.

        2. **Polynomial multiplication**:

           .. math::

               \left(\sum_i c_i W_i\right)\left(\sum_j d_j V_j\right)
               = \sum_{i,j} (c_i d_j)\,(W_i V_j),

           where word multiplication is word concatenation.

        Notes
        -----
        - No CCR commutation or normal ordering is applied here.
        - Like terms are not merged automatically; call
          :meth:`combine_like_terms` to normalize.

        Parameters
        ----------
        other:
            Either a scalar (``int``, ``float``, ``complex``) or another
            operator polynomial.

        Returns
        -------
        OpPoly
            Product polynomial.

        """
        if isinstance(other, int | float | complex):
            return self.scaled(other)
        return OpPoly(multiply(self.terms, other.terms, term_factory=OpTerm))

    def __rmul__(self, other: complex) -> OpPoly:
        r"""Return scalar multiplication with scalar on the left.

        Implements

        .. math::

            c\,\mathcal{O} \equiv \mathcal{O}\,c.

        Parameters
        ----------
        other:
            Scalar multiplier.

        Returns
        -------
        OpPoly
            Scaled polynomial.

        Returns
        -------
        NotImplemented
            If ``other`` is not a scalar.

        """
        if isinstance(other, int | float | complex):
            return self.scaled(other)
        return NotImplemented

    def __truediv__(self, other: complex) -> OpPoly:
        r"""Return scalar division ``self / c``.

        Parameters
        ----------
        other:
            Scalar divisor.

        Returns
        -------
        OpPoly
            Scaled polynomial.

        Raises
        ------
        TypeError
            If ``other`` is not a scalar.
        ZeroDivisionError
            If ``other`` is numerically zero.

        """
        if not isinstance(other, int | float | complex):
            raise TypeError("OpPoly can only be divided by a scalar.")
        if abs(other) == 0:
            raise ZeroDivisionError("Division by zero scalar.")
        return self.scaled(1.0 / other)

    def __bool__(self) -> bool:
        r"""Return ``False`` for the zero polynomial, ``True`` otherwise.

        The polynomial is considered zero iff it contains no terms.
        """
        return bool(self.terms)

    def __eq__(self, other: object) -> bool:
        r"""Return structural equality.

        Two polynomials are equal iff their term tuples are exactly equal.
        No automatic normalization or approximate matching is performed.

        Notes
        -----
        For semantic equality up to merging, compare normalized copies, e.g.

        .. math::

            \mathrm{normalize}(A) == \mathrm{normalize}(B).

        Returns
        -------
        bool
            Structural equality.

        """
        if not isinstance(other, OpPoly):
            return NotImplemented
        return self.terms == other.terms

    def __repr__(self) -> str:
        r"""Return an unambiguous developer representation.

        This representation is structural and intentionally minimal, since
        pretty-printing is out of scope for this layer.
        """
        return f"{self.__class__.__name__}(terms={self.terms!r})"

    def __len__(self) -> int:
        r"""Return the number of operator terms.

        This equals the number of words in the formal expansion

        .. math::

            \mathcal{O} = \sum_k c_k W_k.

        Notes
        -----
        This is purely structural. Terms are not automatically merged,
        so the length may decrease after calling
        :meth:`combine_like_terms`.

        """
        return len(self.terms)

    def __iter__(self) -> Iterator[OpTerm]:
        r"""Iterate over operator terms.

        Yields
        ------
        OpTermProto
            The operator terms in their stored order.

        Notes
        -----
        The iteration order reflects the internal tuple order and
        carries no mathematical meaning.

        """
        return iter(self.terms)

    def __matmul__(self, other: object) -> object:
        r"""Compose/apply using matrix-multiplication semantics.

        Supported cases
        ---------------
        - ``OpPoly @ OpPoly``:
          Composition by distributing over terms and concatenating words.
        - ``OpPoly @ KetPolyProto``:
          Left action on a ket via :meth:`KetPolyProto.apply_words`.
        - ``OpPoly @ SupportsLeftActionDensity``:
          Left action on a density via :meth:`apply_left`.

        Notes
        -----
        This is symbolic and linear. It does not automatically merge like
        terms; call normalization explicitly where desired.

        """
        if isinstance(other, OpPoly):
            return OpPoly(multiply(self.terms, other.terms, term_factory=OpTerm))

        if isinstance(other, KetPoly):
            return other.apply_words((t.coeff, t.ops) for t in self.terms)

        if (
            isinstance(other, SupportsLeftWordAction)
            and isinstance(other, Scaled)
            and isinstance(other, Additive)
        ):
            out = other.zero()
            for t in self.terms:
                out = out + other.apply_left(t.ops).scaled(t.coeff)
            return out

        return NotImplemented

    def __rmatmul__(self, other: object) -> object:
        r"""Right-action dispatcher for ``other @ self``.

        Supported cases
        ---------------
        - ``SupportsRightActionDensity @ OpPoly``:
          Right action on a density via :meth:`apply_right`.
        """
        if (
            isinstance(other, SupportsRightWordAction)
            and isinstance(other, Scaled)
            and isinstance(other, Additive)
        ):
            out = other.zero()
            for t in self.terms:
                out = out + other.apply_right(t.ops).scaled(t.coeff)
            return out

        return NotImplemented


if TYPE_CHECKING:
    from symop.ccr.protocols.op import OpPoly as OpPolyProtocol

    _op_check: OpPolyProtocol = OpPoly.identity()
