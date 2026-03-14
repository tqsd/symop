r"""Symbolic ket polynomials for CCR algebra.

This module defines :class:`~symop.ccr.ket.poly.KetPoly`, a small immutable
wrapper around a tuple of ket terms.

A ket polynomial represents a finite expansion

.. math::

    \lvert \psi \rangle \;\sim\; \sum_k c_k\, M_k,

where each :math:`M_k` is a normally ordered monomial of ladder operators.
All operations are purely symbolic and based on CCR normal ordering and
term canonicalization; no matrix representations are used.

Presentation / pretty-printing is intentionally out of scope for this package.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from symop.ccr.algebra.ket.apply import ket_apply_word, ket_apply_words_linear
from symop.ccr.protocols.op import OpPoly as OpPolyProtocol
from symop.core.protocols.ops.operators import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops.operators import (
    ModeOp,
)
from symop.core.terms.ket_term import KetTerm
from symop.core.types.signature import Signature

from .combine import combine_like_terms_ket
from .from_ops import ket_from_ops
from .from_word import ket_from_word
from .inner import ket_inner
from .multiply import ket_multiply
from .scale import ket_scale


@dataclass(frozen=True)
class KetPoly:
    r"""Symbolic ket polynomial.

    Parameters
    ----------
    terms:
        Tuple of ket terms. Terms are not automatically canonicalized on
        construction; call :meth:`combine_like_terms` when needed.

    Notes
    -----
    The class is intentionally logic-only. It provides algebraic operations
    and structural queries, but no string/latex rendering.

    """

    terms: tuple[KetTerm, ...] = ()

    @staticmethod
    def identity() -> KetPoly:
        """Construct identity object ```KetPoly```."""
        return KetPoly.from_ops(creators=(), annihilators=(), coeff=1.0)

    @staticmethod
    def from_ops(
        *,
        creators: Iterable[LadderOpProtocol] = (),
        annihilators: Iterable[LadderOpProtocol] = (),
        coeff: complex = 1.0,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> KetPoly:
        r"""Construct from explicit creators and annihilators."""
        return KetPoly(
            ket_from_ops(
                creators=creators,
                annihilators=annihilators,
                coeff=coeff,
                approx=approx,
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            )
        )

    @staticmethod
    def from_word(
        *,
        ops: Iterable[LadderOpProtocol],
        eps: float = 1e-12,
    ) -> KetPoly:
        r"""Construct by normal-ordering an operator word."""
        return KetPoly(ket_from_word(ops=ops, eps=eps))

    def combine_like_terms(
        self,
        *,
        eps: float = 1e-12,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> KetPoly:
        r"""Return a canonicalized polynomial by merging identical monomials."""
        return KetPoly(
            combine_like_terms_ket(
                self.terms,
                eps=eps,
                approx=approx,
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            )
        )

    def scaled(self, c: complex) -> KetPoly:
        r"""Return ``c * self``."""
        return KetPoly(ket_scale(self.terms, c))

    def multiply(
        self,
        other: KetPoly,
        *,
        eps: float = 1e-12,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> KetPoly:
        r"""Return the symbolic product ``self * other``."""
        return KetPoly(
            ket_multiply(
                self.terms,
                other.terms,
                eps=eps,
                approx=approx,
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            )
        )

    def apply_word(
        self, word: Iterable[LadderOpProtocol], *, eps: float = 1e-12
    ) -> KetPoly:
        r"""Apply an operator word on the left: ``word * self``.

        Parameters
        ----------
        word:
            Ordered ladder-operator word to apply.
        eps:
            Numerical tolerance forwarded to normal ordering / canonicalization.

        Returns
        -------
        KetPoly
            Resulting ket polynomial.

        """
        return KetPoly(ket_apply_word(self.terms, word, eps=eps))

    def apply_words(
        self,
        terms: Iterable[tuple[complex, Iterable[LadderOpProtocol]]],
        *,
        eps: float = 1e-12,
    ) -> KetPoly:
        r"""Apply a linear combination of operator words on the left.

        Interprets ``terms`` as an operator polynomial

        .. math::

            \left(\sum_i c_i W_i\right)\lvert \psi \rangle.

        Parameters
        ----------
        terms:
            Iterable of pairs ``(coeff, word)``.
        eps:
            Numerical tolerance forwarded to normal ordering / canonicalization.

        Returns
        -------
        KetPoly
            Resulting ket polynomial in canonical form.

        """
        return KetPoly(ket_apply_words_linear(self.terms, terms, eps=eps))

    def inner(self, other: KetPoly, *, eps: float = 1e-12) -> complex:
        r"""Return the symbolic inner product ``<self|other>``."""
        return ket_inner(self.terms, other.terms, eps=eps)

    def norm2(self, *, eps: float = 1e-12) -> float:
        r"""Return ``<self|self>`` as a real float."""
        return float(self.inner(self, eps=eps).real)

    def normalize(self, *, eps: float = 1e-14) -> KetPoly:
        r"""Return a normalized copy of the polynomial."""
        n2 = self.norm2(eps=eps)
        if n2 < eps:
            raise ValueError("Cannot normalize: near-zero norm.")
        return self.scaled(1.0 / (n2**0.5))

    def is_normalized(self, *, eps: float = 1e-14) -> bool:
        r"""Return True if the polynomial has unit norm within tolerance."""
        return abs(self.norm2(eps=eps) - 1.0) <= eps

    @property
    def is_creator_only(self) -> bool:
        r"""Return True if every term is creators-only or identity."""
        return all(
            t.monomial.is_creator_only or t.monomial.is_identity for t in self.terms
        )

    @property
    def is_annihilator_only(self) -> bool:
        r"""Return True if every term is annihilators-only or identity."""
        return all(
            t.monomial.is_annihilator_only or t.monomial.is_identity for t in self.terms
        )

    @property
    def is_identity(self) -> bool:
        r"""Return True if every term is the identity monomial."""
        return all(t.monomial.is_identity for t in self.terms)

    @property
    def creation_count(self) -> int:
        r"""Total number of creation operators across all terms."""
        return sum(len(t.monomial.creators) for t in self.terms)

    @property
    def annihilation_count(self) -> int:
        r"""Total number of annihilation operators across all terms."""
        return sum(len(t.monomial.annihilators) for t in self.terms)

    @property
    def total_degree(self) -> int:
        r"""Total ladder-operator degree across all terms."""
        return self.creation_count + self.annihilation_count

    @property
    def unique_modes(self) -> tuple[ModeOp, ...]:
        r"""Return unique modes appearing in the polynomial (first-seen order)."""
        seen: dict[Signature, ModeOp] = {}
        for t in self.terms:
            for m in t.monomial.mode_ops:
                seen.setdefault(m.signature, m)
        return tuple(seen.values())

    @property
    def mode_count(self) -> int:
        r"""Return the number of unique modes appearing in the polynomial."""
        return len(self.unique_modes)

    def require_creator_only(self) -> None:
        r"""Raise ValueError unless the polynomial is creators-only (or identity)."""
        if not self.is_creator_only:
            raise ValueError("Expected creators-only ket (plus identities).")

    def __add__(self, other: KetPoly) -> KetPoly:
        r"""Return the symbolic difference ``self - other``.

        This is defined as addition with a negated polynomial:

        .. math::

            self - other \equiv self + (-1) \cdot other.

        Notes
        -----
        Terms are concatenated and canonicalized via
        :func:`combine_like_terms_ket`.

        """
        return KetPoly(combine_like_terms_ket((*self.terms, *other.terms)))

    def __sub__(self, other: KetPoly) -> KetPoly:
        r"""Return the symbolic difference ``self - other``.

        This is defined as addition with a negated polynomial:

        .. math::

            self - other \equiv self + (-1) \cdot other.

        Notes
        -----
        Terms are concatenated and canonicalized via
        :func:`combine_like_terms_ket`.

        """
        return self + (-1.0) * other

    def __neg__(self) -> KetPoly:
        """Return the additive inverse ``-self``.

        Equivalent to scalar multiplication by ``-1``.
        """
        return self.scaled(-1.0)

    def __truediv__(self, other: complex) -> KetPoly:
        """Return scalar division ``self / c``.

        Parameters
        ----------
        other:
            Scalar divisor.

        Raises
        ------
        TypeError
            If ``other`` is not a scalar.
        ZeroDivisionError
            If ``other`` is numerically zero.

        """
        if not isinstance(other, int | float | complex):
            raise TypeError("KetPoly can only be divided by a scalar.")
        if abs(other) == 0:
            raise ZeroDivisionError("Division by zero scalar.")
        return self.scaled(1.0 / other)

    def __eq__(self, other: object) -> bool:
        """Return structural equality.

        Two polynomials are considered equal if their term tuples are
        exactly equal. No automatic canonicalization or tolerance-based
        comparison is performed.

        Notes
        -----
        For semantic equality, call :meth:`combine_like_terms` first.

        """
        if not isinstance(other, KetPoly):
            return NotImplemented
        return self.terms == other.terms

    def __mul__(self, other: KetPoly | complex) -> KetPoly:
        r"""Return the symbolic product.

        This operator implements two distinct cases:

        1. **Scalar multiplication**

           .. math::

               \lvert \psi \rangle \cdot c
               \;\equiv\;
               c \, \lvert \psi \rangle,

           where ``c`` is a real or complex scalar.

        2. **Polynomial multiplication**

           .. math::

               \lvert \psi \rangle \cdot \lvert \phi \rangle,

           defined symbolically via CCR normal ordering and term
           canonicalization.

        Parameters
        ----------
        other:
            Either a scalar (``int``, ``float``, ``complex``) or another
            :class:`KetPoly`.

        Returns
        -------
        KetPoly
            Resulting symbolic ket polynomial.

        Notes
        -----
        No automatic normalization is performed.

        """
        if isinstance(other, int | float | complex):
            return self.scaled(other)
        return self.multiply(other)

    def __rmul__(self, other: complex) -> KetPoly:
        r"""Return scalar multiplication with scalar on the left.

        Implements

        .. math::

            c \cdot \lvert \psi \rangle
            \;\equiv\;
            \lvert \psi \rangle \cdot c.

        Parameters
        ----------
        other:
            Scalar multiplier.

        Returns
        -------
        KetPoly
            Scaled polynomial.

        Notes
        -----
        Only scalar-left multiplication is supported. Polynomial-left
        multiplication is handled by :meth:`__mul__`.

        """
        return self.scaled(other)

    def __bool__(self) -> bool:
        """Return False for the zero polynomial, True otherwise.

        The polynomial is considered zero if it contains no terms.
        """
        return bool(self.terms)

    def __repr__(self) -> str:
        """Return an unambiguous developer representation.

        This representation is structural and intentionally minimal,
        since pretty-printing is out of scope for this layer.
        """
        return f"{self.__class__.__name__}(terms={self.terms!r})"

    def __rmatmul__(self, other: object) -> object:
        r"""Left action by an operator polynomial: ``other @ self``.

        Supported cases
        ---------------
        - ``OpPolyProto @ KetPoly``:
          Applies the linear combination of operator words to this ket via
          :meth:`KetPoly.apply_words`.

        Notes
        -----
        This is a symbolic linear action. Like terms are not necessarily merged
        unless your :meth:`apply_words` implementation canonicalizes.

        """
        if isinstance(other, OpPolyProtocol):
            return self.apply_words((t.coeff, t.ops) for t in other.terms)
        return NotImplemented


if TYPE_CHECKING:
    from symop.ccr.protocols.ket import KetPoly as KetPolyProtocol

    _ket_poly_check: KetPolyProtocol = KetPoly.identity()
