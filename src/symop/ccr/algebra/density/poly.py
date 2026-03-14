r"""Symbolic density polynomials for CCR algebra.

A density polynomial is represented as a finite linear combination of outer
products of normally ordered monomials,

.. math::

    \rho \sim \sum_i c_i \, |L_i\rangle\langle R_i|.

This module defines :class:`~symop.ccr.density.poly.DensityPoly`, an immutable
wrapper around a tuple of density terms. It delegates algebraic operations to
the pure functional implementations in :mod:`symop.ccr.algebra.density`.

Only symbolic CCR logic is performed here. No matrices are constructed.
Presentation / pretty-printing is intentionally out of scope.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from symop.ccr.protocols.op import OpPoly as OpPolyProtocol
from symop.core.monomial import Monomial
from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops import (
    ModeOp as ModeOpProtocol,
)
from symop.core.protocols.terms import KetTerm as KetTermProtocol
from symop.core.terms import DensityTerm
from symop.core.types.signature import Signature

from .apply_left import apply_left
from .apply_right import apply_right
from .combine import combine_like_terms_density
from .inner import density_inner
from .normalize_trace import density_normalize_trace
from .partial_trace import density_partial_trace
from .pure import density_pure
from .purity import density_purity
from .scale import density_scale
from .trace import density_trace


@dataclass(frozen=True)
class DensityPoly:
    r"""Symbolic density polynomial.

    A :class:`DensityPoly` stores a tuple of density terms, each term being a
    coefficient times an outer product of monomials:

    .. math::

        c \, |L\rangle\langle R|.

    Parameters
    ----------
    terms:
        Tuple of density terms. Terms are not automatically canonicalized on
        construction; call :meth:`combine_like_terms` (or :meth:`normalize`)
        when needed.

    Notes
    -----
    This class is intentionally logic-only. It wraps functional algebra
    routines and provides convenient structural queries and standard
    linear-algebra-like operations (addition and scalar scaling).

    """

    terms: tuple[DensityTerm, ...] = ()

    @staticmethod
    def pure(ket_terms: tuple[KetTermProtocol, ...]) -> DensityPoly:
        r"""Construct a pure-state density polynomial ``|psi><psi|``.

        Parameters
        ----------
        ket_terms:
            Ket polynomial terms describing :math:`|\psi\rangle`.

        Returns
        -------
        DensityPoly
            Density polynomial representing :math:`|\psi\rangle\langle\psi|`.

        """
        return DensityPoly(density_pure(ket_terms))

    @staticmethod
    def zero() -> DensityPoly:
        r"""Return the zero density operator.

        Returns
        -------
        DensityPoly
            The empty polynomial (no terms).

        """
        return DensityPoly(())

    @staticmethod
    def identity() -> DensityPoly:
        r"""Return the identity operator as a density polynomial.

        This returns the single-term polynomial

        .. math::

            |I\rangle\langle I|,

        where :math:`I` is the identity monomial (empty word).

        Returns
        -------
        DensityPoly
            Identity density polynomial.

        """
        Id = Monomial.identity()
        return DensityPoly((DensityTerm(coeff=1.0 + 0.0j, left=Id, right=Id),))

    def combine_like_terms(
        self,
        *,
        eps: float = 1e-12,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> DensityPoly:
        r"""Return a canonicalized polynomial by merging identical terms.

        Density terms are bucketed by signatures of their left/right monomials,
        and coefficients within each bucket are summed.

        Parameters
        ----------
        eps:
            Buckets whose summed coefficient magnitude is smaller than ``eps``
            are removed.
        approx:
            If ``True``, use approximate signatures for matching.
        decimals:
            Number of decimals used when rounding floating components in
            approximate signatures.
        ignore_global_phase:
            If ``True``, component signatures may ignore global phase when
            constructing approximate signatures.

        Returns
        -------
        DensityPoly
            Polynomial with like terms merged and near-zero terms removed.

        Notes
        -----
        This is purely signature-based merging. It does not apply CCR
        commutation relations or reorder monomials.

        """
        return DensityPoly(
            combine_like_terms_density(
                self.terms,
                eps=eps,
                approx=approx,
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            )
        )

    def normalize(
        self,
        *,
        eps: float = 1e-12,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> DensityPoly:
        r"""Return a normalized structural form of the polynomial.

        This is a convenience alias for :meth:`combine_like_terms`.

        Returns
        -------
        DensityPoly
            Polynomial with merged like terms.

        """
        return self.combine_like_terms(
            eps=eps,
            approx=approx,
            decimals=decimals,
            ignore_global_phase=ignore_global_phase,
        )

    def scaled(self, c: complex) -> DensityPoly:
        r"""Return ``c * self``.

        Parameters
        ----------
        c:
            Scalar multiplier.

        Returns
        -------
        DensityPoly
            Scaled density polynomial.

        """
        return DensityPoly(density_scale(self.terms, c))

    def apply_left(self, word: Iterable[LadderOpProtocol]) -> DensityPoly:
        r"""Apply an operator word to the left: ``word * rho``.

        This symbolically applies the ladder-operator word to each term's left
        monomial.

        Parameters
        ----------
        word:
            Ordered ladder-operator word to apply on the left.

        Returns
        -------
        DensityPoly
            Resulting density polynomial.

        """
        return DensityPoly(apply_left(self.terms, word))

    def apply_right(self, word: Iterable[LadderOpProtocol]) -> DensityPoly:
        r"""Apply an operator word to the right: ``rho * word``.

        This symbolically applies the ladder-operator word to each term's right
        monomial.

        Parameters
        ----------
        word:
            Ordered ladder-operator word to apply on the right.

        Returns
        -------
        DensityPoly
            Resulting density polynomial.

        """
        return DensityPoly(apply_right(self.terms, word))

    def trace(self) -> complex:
        r"""Return the trace :math:`\mathrm{Tr}(\rho)`.

        Returns
        -------
        complex
            Trace of the density polynomial.

        """
        return density_trace(self.terms)

    def normalize_trace(self, *, eps: float = 1e-14) -> DensityPoly:
        r"""Return a trace-normalized density polynomial.

        This scales the density so that its trace is unity.

        Parameters
        ----------
        eps:
            If ``|Tr(rho)| < eps`` a :class:`ValueError` is raised.

        Returns
        -------
        DensityPoly
            Trace-normalized density polynomial.

        Raises
        ------
        ValueError
            If the trace is too small to safely normalize.

        """
        return DensityPoly(density_normalize_trace(self.terms, eps=eps))

    def inner(self, other: DensityPoly) -> complex:
        r"""Return the Hilbert-Schmidt inner product ``<self, other>``.

        This delegates to :func:`density_inner` and is typically equivalent to

        .. math::

            \langle A, B \rangle_{\mathrm{HS}} = \mathrm{Tr}(A^\dagger B).

        Parameters
        ----------
        other:
            Another density polynomial.

        Returns
        -------
        complex
            Hilbert-Schmidt inner product.

        """
        return density_inner(self.terms, other.terms)

    def purity(self) -> float:
        r"""Return the purity :math:`\mathrm{Tr}(\rho^2)`.

        Returns
        -------
        float
            Purity of the density polynomial.

        """
        return density_purity(self.terms)

    def partial_trace(self, trace_over_modes: Iterable[object]) -> DensityPoly:
        r"""Return the partial trace over a set of modes.

        Parameters
        ----------
        trace_over_modes:
            Iterable of mode identifiers to trace out. The expected identifier
            type is determined by the underlying term/mode implementation
            (typically objects with a stable ``signature``).

        Returns
        -------
        DensityPoly
            Reduced density polynomial.

        """
        return DensityPoly(density_partial_trace(self.terms, trace_over_modes))

    def hs_norm2(self) -> float:
        r"""Return the squared Hilbert-Schmidt norm.

        .. math::

            \| \rho \|_{\mathrm{HS}}^2 = \langle \rho, \rho \rangle_{\mathrm{HS}}.

        Returns
        -------
        float
            Squared Hilbert-Schmidt norm (real-valued).

        """
        return float(self.inner(self).real)

    def hs_norm(self) -> float:
        r"""Return the Hilbert-Schmidt norm.

        Returns
        -------
        float
            :math:`\|\rho\|_{\mathrm{HS}}`.

        """
        return float(self.hs_norm2() ** 0.5)

    @property
    def unique_modes(self) -> tuple[ModeOpProtocol, ...]:
        r"""Return unique modes appearing in the density (first-seen order).

        Modes are extracted from both left and right monomials and uniqued
        by exact mode signature.

        Returns
        -------
        tuple[ModeOpProto, ...]
            Unique modes in first-seen order.

        """
        seen: dict[Signature, ModeOpProtocol] = {}
        for dt in self.terms:
            for m in (*dt.left.mode_ops, *dt.right.mode_ops):
                seen.setdefault(m.signature, m)
        return tuple(seen.values())

    @property
    def mode_count(self) -> int:
        r"""Return the number of unique modes appearing in the density."""
        return len(self.unique_modes)

    @property
    def is_diagonal_in_monomials(self) -> bool:
        r"""Return True iff each term has identical left/right monomials."""
        return all(dt.left.signature == dt.right.signature for dt in self.terms)

    @property
    def is_identity_left(self) -> bool:
        r"""Return True iff every term has identity on the left."""
        return all(dt.left.is_identity for dt in self.terms)

    @property
    def is_identity_right(self) -> bool:
        r"""Return True iff every term has identity on the right."""
        return all(dt.right.is_identity for dt in self.terms)

    @property
    def is_creator_only_left(self) -> bool:
        """Return True iff every term is creator-only or identity on the left."""
        return all(dt.is_creator_only_left for dt in self.terms)

    @property
    def is_creator_only_right(self) -> bool:
        """Return True iff every term is creator-only or identity on the right."""
        return all(dt.is_creator_only_right for dt in self.terms)

    @property
    def is_creator_only(self) -> bool:
        """Return True iff every term is creator-only or identity on both sides."""
        return all(dt.is_creator_only for dt in self.terms)

    def is_trace_normalized(self, eps: float = 1e-12) -> bool:
        r"""Return True if :math:`\mathrm{Tr}(\rho)=1` within tolerance."""
        return abs(self.trace() - (1.0 + 0.0j)) <= eps

    def is_pure(self, eps: float = 1e-12) -> bool:
        r"""Return True if purity is 1 within tolerance."""
        return abs(self.purity() - 1.0) <= eps

    def require_trace_normalized(self, eps: float = 1e-12) -> None:
        r"""Raise :class:`ValueError` unless the density is trace-normalized."""
        if not self.is_trace_normalized(eps=eps):
            raise ValueError("Density is not trace-normalized.")

    def is_block_diagonal_by_modes(self) -> bool:
        r"""Return True iff each term matches left/right mode lists.

        This checks that the ordered list of mode signatures appearing in the
        left monomial matches that of the right monomial for every term.
        """
        for dt in self.terms:
            if tuple(m.signature for m in dt.left.mode_ops) != tuple(
                m.signature for m in dt.right.mode_ops
            ):
                return False
        return True

    def __len__(self) -> int:
        r"""Return the number of density terms (structural length)."""
        return len(self.terms)

    def __iter__(self) -> Iterator[DensityTerm]:
        r"""Iterate over density terms in stored order."""
        return iter(self.terms)

    def __bool__(self) -> bool:
        r"""Return ``False`` for the zero polynomial, ``True`` otherwise."""
        return bool(self.terms)

    def __add__(self, other: DensityPoly) -> DensityPoly:
        r"""Return the symbolic sum ``self + other``.

        Notes
        -----
        The result is returned in a canonicalized form by calling
        :meth:`combine_like_terms`.

        """
        return DensityPoly((*self.terms, *other.terms)).combine_like_terms()

    def __sub__(self, other: DensityPoly) -> DensityPoly:
        r"""Return the symbolic difference ``self - other``."""
        return self + (-1.0) * other

    def __neg__(self) -> DensityPoly:
        r"""Return the additive inverse ``-self``."""
        return self.scaled(-1.0)

    def __mul__(self, other: complex) -> DensityPoly:
        r"""Return scalar scaling ``self * c``.

        Parameters
        ----------
        other:
            Scalar multiplier.

        Returns
        -------
        DensityPoly
            Scaled polynomial.

        Returns
        -------
        NotImplemented
            If ``other`` is not a scalar.

        """
        if isinstance(other, int | float | complex):
            return self.scaled(other)
        return NotImplemented

    def __rmul__(self, other: complex) -> DensityPoly:
        r"""Return scalar-left scaling ``c * self``."""
        return self.__mul__(other)

    def __truediv__(self, other: complex) -> DensityPoly:
        r"""Return scalar division ``self / c``."""
        if not isinstance(other, int | float | complex):
            raise TypeError("DensityPoly can only be divided by a scalar.")
        if abs(other) == 0:
            raise ZeroDivisionError("Division by zero scalar.")
        return self.scaled(1.0 / other)

    def __eq__(self, other: object) -> bool:
        r"""Return structural equality.

        Equality is exact and structural: term tuples must match exactly.
        No normalization or approximate merging is performed.
        """
        if not isinstance(other, DensityPoly):
            return NotImplemented
        return self.terms == other.terms

    def __repr__(self) -> str:
        r"""Return an unambiguous developer representation."""
        return f"{self.__class__.__name__}(terms={self.terms!r})"

    def __matmul__(self, other: object) -> object:
        r"""Right action / composition using matrix-multiplication semantics.

        Supported cases
        ---------------
        - ``DensityPoly @ OpPolyProto``:
          Apply the operator polynomial to the right of the density:

          .. math::

              \rho @ \mathcal{O} \equiv \rho \mathcal{O}.

          This distributes linearly over the operator polynomial terms and
          applies each word via :meth:`apply_right`.

        Notes
        -----
        This is purely symbolic and linear. Like terms are merged using
        :meth:`combine_like_terms` at the end.

        Returns
        -------
        DensityPoly
            Result of the right action.

        Returns
        -------
        NotImplemented
            If the operand type is unsupported.

        """
        if isinstance(other, OpPolyProtocol):
            out_terms: tuple[DensityTerm, ...] = ()
            for t in other.terms:
                if t.coeff == 0:
                    continue
                part = apply_right(self.terms, t.ops)
                if t.coeff != 1:
                    part = density_scale(part, t.coeff)
                out_terms = (*out_terms, *part)
            return DensityPoly(out_terms).combine_like_terms()
        return NotImplemented

    def __rmatmul__(self, other: object) -> object:
        r"""Dispatch for ``other @ self``.

        Notes
        -----
        Left action of operator polynomials is implemented on :class:`OpPoly`
        (via ``OpPoly.__matmul__``). This method is provided for symmetry and
        returns ``NotImplemented`` by default so Python can fall back to the
        left operand implementation.

        """
        return NotImplemented


if TYPE_CHECKING:
    from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol

    _density_poly_check: DensityPolyProtocol = DensityPoly.identity()
