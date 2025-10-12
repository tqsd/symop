from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple, Union, Optional
from numpy import sqrt, cos, sin

from symop_proto.core.operators import LadderOp, ModeOp
from symop_proto.core.protocols import LadderOpProto


def op_from_words(
    words: Iterable[Iterable[LadderOpProto]],
    coeffs: Optional[Iterable[complex]] = None,
) -> Tuple[OpTerm, ...]:
    ws = [tuple(w) for w in words]
    if coeffs is None:
        coeffs = [1.0] * len(ws)
    return tuple(OpTerm(w, c) for w, c in zip(ws, coeffs))


def op_multiply(
    a: Tuple[OpTerm, ...], b: Tuple[OpTerm, ...]
) -> Tuple[OpTerm, ...]:
    return tuple(
        OpTerm(ti.ops + tj.ops, ti.coeff * tj.coeff) for ti in a for tj in b
    )


def op_combine_like_terms(
    terms: Tuple[OpTerm, ...], *, use_approx: bool = False, **env_kw
) -> Tuple[OpTerm, ...]:
    key = (
        (lambda t: t.approx_signature(**env_kw))
        if use_approx
        else (lambda t: t.signature)
    )
    buckets: Dict[tuple, complex] = {}
    reps: Dict[tuple, Tuple[LadderOp, ...]] = {}
    for t in terms:
        k = key(t)
        buckets[k] = buckets.get(k, 0.0j) + t.coeff
        reps.setdefault(k, t.ops)
    return tuple(OpTerm(reps[k], c) for k, c in buckets.items() if c != 0.0)


@dataclass(frozen=True)
class OpTerm:
    r"""
    Single operator "word" with a complex coefficient.

    An :class:`OpTerm` represents a monomial in ladder operators:

    .. math::

        T \equiv c \, \hat o_1 \hat o_2 \cdots \hat o_L,

    where ``ops = (o_1, o_2, ..., o_L)`` and ``coeff = c``.

    Adjoint is defined by reversing the operator order, taking daggers,
    and conjugating the scalar:

    .. math::

        T^\dagger
        = c^* \, \hat o_L^\dagger \cdots \hat o_2^\dagger \hat o_1^\dagger.

    The (exact) signature is a tuple of the exact operator signatures in order.
    The approximate signature delegates to each operator's approximate signature.

    Notes
    -----
    - ``identity()`` creates the empty word (no operators) with a given coeff.
    - This class is immutable; scaling and adjoint return new instances.
    """

    ops: Tuple[LadderOpProto, ...]
    coeff: complex = 1.0

    @staticmethod
    def identity(c: complex = 1.0) -> OpTerm:
        return OpTerm(ops=(), coeff=c)

    def scaled(self, c: complex) -> OpTerm:
        return OpTerm(self.ops, c * self.coeff)

    def adjoint(self) -> OpTerm:
        return OpTerm(
            ops=tuple(op.dagger() for op in reversed(self.ops)),
            coeff=self.coeff.conjugate(),
        )

    @property
    def signature(self) -> Tuple[Any, ...]:
        return ("OP", tuple(op.signature for op in self.ops))

    def approx_signature(self, **env_kw) -> Tuple[Any, ...]:
        return (
            "OP_approx",
            tuple(op.approx_signature(**env_kw) for op in self.ops),
        )


@dataclass(frozen=True)
class OpPoly:
    r"""
    Finite linear combination of :class:`OpTerm` objects.

    .. math::

        \mathcal{O} \equiv \sum_k c_k \, \hat o_{k,1}\hat o_{k,2}\cdots \hat o_{k,L_k}.

    Construction helpers
    --------------------
    - ``from_words(words, coeffs=None)`` builds an operator polynomial from
      a list of words (iterables of ladder ops) with optional coefficients.
    - ``identity(c)`` and ``zero()`` create the identity and the zero operator.
    - ``a(mode)``, ``adag(mode)``, ``n(mode)`` give standard monomials.
    - ``q(mode) = (a + a^\dag)/sqrt(2)``, ``p(mode) = i a^\dag/sqrt(2) - i a/sqrt(2)``.
    - ``X_theta(mode, theta) = ( e^{-i\theta} a + e^{+i\theta} a^\dag ) / sqrt(2)``.

    Algebra
    -------
    - Addition concatenates term lists:  ``O1 + O2``.
    - Scalar multiplication scales coefficients: ``c * O`` or ``O * c``.
    - Multiplication forms all concatenations of words with multiplied coeffs:
      ``(sum_i c_i W_i) * (sum_j d_j V_j) = sum_{i,j} (c_i d_j) (W_i || V_j)``.

    Normalization
    -------------
    ``combine_like_terms(use_approx=False, **env_kw)`` merges terms with identical
    (exact or approximate) signatures by summing their coefficients and discarding
    zeros.

    Properties
    ----------
    - ``is_zero``: True iff there are no terms.
    - ``is_identity``: True iff there is at least one term and all terms are empty words.

    Notes
    -----
    - Multiplication does not automatically merge terms; call
      ``combine_like_terms()`` to collapse identical words.
    - All helpers accept any iterables; generators are safely materialized.
    """

    terms: Tuple[OpTerm, ...] = ()

    @staticmethod
    def from_words(
        words: Iterable[Iterable[LadderOp]],
        coeffs: Optional[Iterable[complex]] = None,
    ) -> OpPoly:
        return OpPoly(op_from_words(words, coeffs))

    @staticmethod
    def identity(c: complex = 1.0) -> OpPoly:
        return OpPoly((OpTerm.identity(c),))

    @staticmethod
    def zero() -> OpPoly:
        return OpPoly(())

    @staticmethod
    def a(mode: ModeOp) -> OpPoly:
        return OpPoly.from_words([[mode.ann]])

    @staticmethod
    def adag(mode: ModeOp) -> OpPoly:
        return OpPoly.from_words([[mode.create]])

    @staticmethod
    def n(mode: ModeOp) -> OpPoly:
        return OpPoly.from_words([[mode.create, mode.ann]])

    @staticmethod
    def q(mode: ModeOp) -> OpPoly:
        return (
            OpPoly.from_words([[mode.ann]])
            + OpPoly.from_words([[mode.create]])
        ) * (1.0 / sqrt(2))

    @staticmethod
    def x(mode: ModeOp) -> OpPoly:
        return OpPoly.q(mode)

    @staticmethod
    def p(mode: ModeOp) -> OpPoly:
        return OpPoly.from_words([[mode.create]]) * (
            1j / sqrt(2)
        ) + OpPoly.from_words([[mode.ann]]) * (-1j / sqrt(2))

    @staticmethod
    def X_theta(mode: "ModeOp", theta: float) -> OpPoly:
        e_m = cos(theta) - 1j * sin(theta)
        e_p = cos(theta) + 1j * sin(theta)
        return (
            OpPoly.from_words([[mode.ann]]) * e_m
            + OpPoly.from_words([[mode.create]]) * e_p
        ) * (1.0 / sqrt(2))

    @staticmethod
    def q2(mode: ModeOp) -> OpPoly:
        q = OpPoly.q(mode)
        return (q * q).combine_like_terms()

    @staticmethod
    def p2(mode: ModeOp) -> OpPoly:
        p = OpPoly.p(mode)
        return (p * p).combine_like_terms()

    @staticmethod
    def n2(mode: ModeOp) -> OpPoly:
        n = OpPoly.n(mode)
        return (n * n).combine_like_terms()

    def scaled(self, c: complex) -> OpPoly:
        return OpPoly(tuple(term.scaled(c) for term in self.terms))

    def adjoint(self) -> OpPoly:
        return OpPoly(tuple(term.adjoint() for term in self.terms))

    def combine_like_terms(self, **kw):
        return OpPoly(op_combine_like_terms(self.terms, **kw))

    @property
    def is_zero(self) -> bool:
        return len(self.terms) == 0

    @property
    def is_identity(self) -> bool:
        return len(self.terms) > 0 and all(len(t.ops) == 0 for t in self.terms)

    def __add__(self, other: OpPoly) -> OpPoly:
        return OpPoly((*self.terms, *other.terms))

    def __mul__(self, other: Union[OpPoly, complex]) -> OpPoly:
        if isinstance(other, (int, float, complex)):
            return self.scaled(other)
        return OpPoly(op_multiply(self.terms, other.terms))

    def __rmul__(self, other: complex) -> "OpPoly":
        if isinstance(other, (int, float, complex)):
            return self.scaled(other)
        return NotImplemented
