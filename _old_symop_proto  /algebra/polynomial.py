from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Union

from symop_proto.algebra.density.inner import density_inner
from symop_proto.algebra.density.normalize_trace import density_normalize_trace
from symop_proto.algebra.density.partial_trace import density_partial_trace
from symop_proto.algebra.density.purity import density_purity
from symop_proto.algebra.density.scale import density_scale
from symop_proto.algebra.density.trace import density_trace
from symop_proto.algebra.ket.apply import (
    ket_apply_word,
    ket_apply_words_linear,
)
from symop_proto.algebra.pretty.ket import ket_latex, ket_repr
from symop_proto.algebra.protocols import DensityPolyProto, KetPolyProto
from symop_proto.core.protocols import (
    DensityTermProto,
    KetTermProto,
    LadderOpProto,
    ModeOpProto,
)

from .ket.from_ops import ket_from_ops
from .ket.from_word import ket_from_word
from .ket.combine import combine_like_terms_ket
from .ket.scale import ket_scale
from .ket.multiply import ket_multiply
from .ket.inner import ket_inner

from .density.pure import density_pure
from .density.combine import combine_like_terms_density
from .density.apply_left import density_apply_left
from .density.apply_right import density_apply_right


@dataclass(frozen=True)
class KetPoly(KetPolyProto):
    r"""Symbolic ket polynomial :math:`|\psi\rangle = \sum_k c_k\,|M_k\rangle`.

    The object stores a tuple of :class:`KetTerm` and provides common
    algebraic operations:

    - Construction from operators via :meth:`from_ops` or :meth:`from_word`.
    - Linear algebra: :meth:`scaled`, :meth:`__add__`, :meth:`__mul__`,
      :meth:`multiply`, :meth:`inner`, :meth:`norm2`, :meth:`normalize`.
    - Operator application: :meth:`apply_word`, :meth:`apply_words`.
    - Canonicalization and queries: :meth:`combine_like_terms`,
      :py:attr:`is_creator_only`, :py:attr:`is_annihilator_only`,
      :py:attr:`is_identity`, :py:attr:`unique_modes`, :py:attr:`mode_count`.

    All manipulations are purely symbolic (normal ordering plus
    commutator contractions) and do not rely on matrix representations.
    """

    terms: Tuple[KetTermProto, ...] = ()

    @staticmethod
    def from_ops(
        *,
        creators: Iterable[LadderOpProto] = (),
        annihilators: Iterable[LadderOpProto] = (),
        coeff: complex = 1.0,
    ) -> "KetPoly":
        return KetPoly(
            ket_from_ops(creators=creators, annihilators=annihilators, coeff=coeff)
        )

    @staticmethod
    def from_word(*, ops: Iterable[LadderOpProto]) -> KetPoly:
        return KetPoly(ket_from_word(ops=ops))

    def combine_like_terms(self, **kw) -> KetPoly:
        return KetPoly(combine_like_terms_ket(self.terms, **kw))

    def scaled(self, c: complex) -> KetPoly:
        return KetPoly(ket_scale(self.terms, c))

    def multiply(self, other: KetPolyProto) -> KetPoly:
        return KetPoly(ket_multiply(self.terms, other.terms))

    def inner(self, other: KetPolyProto) -> complex:
        return ket_inner(self.terms, other.terms)

    def norm2(self) -> float:
        return float(self.inner(self).real)

    def normalize(self, *, eps: float = 1e-14) -> KetPoly:
        n2 = self.norm2()
        if n2 < eps:
            raise ValueError("Cannot Normalize: ~0 norm")
        return self.scaled(1.0 / (n2**0.5))

    def apply_word(self, word: Iterable[LadderOpProto]) -> KetPoly:
        return KetPoly(ket_apply_word(self.terms, word))

    def apply_words(
        self, terms: Iterable[Tuple[complex, Iterable[LadderOpProto]]]
    ) -> KetPoly:
        return KetPoly(ket_apply_words_linear(self.terms, terms))

    def __add__(self, other: KetPolyProto) -> KetPoly:
        return KetPoly(combine_like_terms_ket((*self.terms, *other.terms)))

    def __mul__(self, other: Union[KetPolyProto, complex]) -> KetPoly:
        return (
            self.scaled(other)
            if isinstance(other, (int, float, complex))
            else self.multiply(other)
        )

    def __rmul__(self, other: complex) -> KetPoly:
        return self.scaled(other)

    def __repr__(self) -> str:
        return ket_repr(self.terms)

    @property
    def latex(self) -> str:
        return ket_latex(self.terms)

    def _repr_latex_(self) -> str:
        return f"${self.latex}$"

    def is_normalized(self, eps: float = 1e-14) -> bool:
        return abs(self.norm2() - 1) < eps

    @property
    def is_creator_only(self) -> bool:
        # allow identity terms, too
        return all(
            t.monomial.is_creator_only or t.monomial.is_identity for t in self.terms
        )

    @property
    def is_annihilator_only(self) -> bool:
        return all(
            t.monomial.is_annihilator_only or t.monomial.is_identity for t in self.terms
        )

    @property
    def is_identity(self) -> bool:
        return all(t.monomial.is_identity for t in self.terms)

    @property
    def creation_count(self) -> int:
        return sum(len(t.monomial.creators) for t in self.terms)

    @property
    def annihilation_count(self) -> int:
        return sum(len(t.monomial.annihilators) for t in self.terms)

    @property
    def total_degree(self) -> int:
        return self.creation_count + self.annihilation_count

    @property
    def unique_modes(self) -> Tuple[ModeOpProto, ...]:
        seen: dict[tuple, ModeOpProto] = {}
        for t in self.terms:
            for m in t.monomial.mode_ops:
                seen.setdefault(m.signature, m)
        return tuple(seen.values())

    @property
    def mode_count(self) -> int:
        return len(self.unique_modes)

    def require_creator_only(self) -> None:
        if not self.is_creator_only:
            raise ValueError("Expected creators-only ket (plus identities).")


@dataclass(frozen=True)
class DensityPoly(DensityPolyProto):
    r"""Symbolic density polynomial :math:`\rho = \sum_i c_i\,|L_i\rangle\langle R_i|`.

    Provides:

    - Construction of pure states via :meth:`pure(psi)`.
    - Linear algebra and canonicalization: :meth:`scaled`,
      :meth:`combine_like_terms`, :meth:`inner`, :meth:`hs_norm2`,
      :meth:`hs_norm`.
    - Physical operations: :meth:`trace`, :meth:`normalize_trace`,
      :meth:`purity`, :meth:`partial_trace`, left/right application of
      operator words (:meth:`apply_left`, :meth:`apply_right`).
    - Structural queries: :py:attr:`unique_modes`, :py:attr:`mode_count`,
      :py:attr:`is_diagonal_in_monomials`, :py:attr:`is_creator_only`,
      :meth:`is_trace_normalized`, :meth:`require_trace_normalized`,
      :meth:`is_block_diagonal_by_modes`.

    All computations are symbolic using normal ordering and commutator
    relations; no matrix representations are required.
    """

    terms: Tuple[DensityTermProto, ...] = ()

    @staticmethod
    def pure(psi: KetPolyProto) -> DensityPoly:
        return DensityPoly(density_pure(psi.terms))

    @staticmethod
    def zero() -> "DensityPoly":
        r"""
        Return the zero density polynomial.

        This is the additive identity: it has no terms and represents the
        zero operator.
        """
        return DensityPoly(())

    @staticmethod
    def identity() -> "DensityPoly":
        r"""
        Return the identity operator as a density polynomial.

        The identity is represented as a single term

        .. math::

            I = 1 \cdot |I\rangle\langle I|,

        where both monomials are the empty monomial (the identity monomial).

        Notes
        -----
        Do not confuse this with ``DensityPoly()`` (empty terms), which
        represents the zero operator.
        """
        from symop_proto.core.terms import DensityTerm

        return DensityPoly((DensityTerm.identity(),))

    def scaled(self, c: complex) -> DensityPoly:
        return DensityPoly(density_scale(self.terms, c))

    def combine_like_terms(self, **kw) -> DensityPoly:
        return DensityPoly(combine_like_terms_density(self.terms, **kw))

    def apply_left(self, word: Iterable[LadderOpProto]) -> DensityPoly:
        return DensityPoly(density_apply_left(self.terms, word))

    def apply_right(self, word: Iterable[LadderOpProto]) -> DensityPoly:
        return DensityPoly(density_apply_right(self.terms, word))

    def trace(self) -> complex:
        return density_trace(self.terms)

    def partial_trace(self, trace_over_modes: Iterable[object]) -> DensityPoly:
        return DensityPoly(density_partial_trace(self.terms, trace_over_modes))

    def inner(self, other: DensityPolyProto) -> complex:
        return density_inner(self.terms, other.terms)

    def purity(self) -> float:
        return density_purity(self.terms)

    def normalize_trace(self, *, eps: float = 1e-14) -> "DensityPoly":
        return DensityPoly(density_normalize_trace(self.terms, eps=eps))

    def hs_norm2(self) -> float:
        return float(self.inner(self).real)

    def hs_norm(self) -> float:
        return self.hs_norm2() ** 0.5

    @property
    def is_creator_only_left(self) -> bool:
        return all(dt.left.is_creator_only or dt.left.is_identity for dt in self.terms)

    @property
    def is_creator_only_right(self) -> bool:
        return all(
            dt.right.is_creator_only or dt.right.is_identity for dt in self.terms
        )

    @property
    def is_creator_only(self) -> bool:
        return self.is_creator_only_left and self.is_creator_only_right

    @property
    def is_identity_left(self) -> bool:
        return all(dt.left.is_identity for dt in self.terms)

    @property
    def is_identity_right(self) -> bool:
        return all(dt.right.is_identity for dt in self.terms)

    @property
    def is_diagonal_in_monomials(self) -> bool:
        return all(dt.left.signature == dt.right.signature for dt in self.terms)

    @property
    def unique_modes(self) -> Tuple[ModeOpProto, ...]:
        seen: dict[tuple, ModeOpProto] = {}
        for dt in self.terms:
            for m in (*dt.left.mode_ops, *dt.right.mode_ops):
                seen.setdefault(m.signature, m)
        return tuple(seen.values())

    @property
    def mode_count(self) -> int:
        return len(self.unique_modes)

    # ---- Physical sanity checks ---------------------------------------------
    def is_trace_normalized(self, eps: float = 1e-12) -> bool:
        return abs(self.trace() - 1.0) <= eps

    def is_pure(self, eps: float = 1e-12) -> bool:
        return abs(self.purity() - 1.0) <= eps

    def require_trace_normalized(self, eps: float = 1e-12) -> None:
        if not self.is_trace_normalized(eps):
            raise ValueError("Density is not trace-normalized.")

    def is_block_diagonal_by_modes(self) -> bool:
        for dt in self.terms:
            if tuple(m.signature for m in dt.left.mode_ops) != tuple(
                m.signature for m in dt.right.mode_ops
            ):
                return False
        return True

    def __repr__(self) -> str:
        from symop_proto.algebra.pretty.density import density_repr

        return density_repr(self.terms)

    def _repr_latex_(self) -> str:
        from symop_proto.algebra.pretty.density import density_latex

        return rf"${density_latex(self.terms, style='brackets')}$"
