r"""Operator-term primitives.

This module defines :class:`OpTerm`, a single operator "word" (ordered product
of ladder operators) together with a complex coefficient.

A term corresponds to

.. math::

    T = c \, o_1 o_2 \cdots o_L,

where each :math:`o_k` is a ladder operator instance.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from symop.core.protocols.ops.operators import LadderOp
from symop.core.types.signature import Signature


@dataclass(frozen=True)
class OpTerm:
    r"""Single operator word with a complex coefficient.

    An :class:`OpTerm` represents an ordered product of ladder operators:

    .. math::

        T \equiv c \, \hat o_1 \hat o_2 \cdots \hat o_L.

    The adjoint is defined by reversing the operator order, taking daggers,
    and conjugating the scalar:

    .. math::

        T^\dagger
        = c^* \, \hat o_L^\dagger \cdots \hat o_2^\dagger \hat o_1^\dagger.

    Notes
    -----
    ``identity()`` creates the empty word (no operators) with a given
    coefficient. The class is immutable.

    """

    ops: tuple[LadderOp, ...]
    coeff: complex = 1.0

    @staticmethod
    def identity(c: complex = 1.0) -> OpTerm:
        """Return the identity term (empty word) with coefficient ``c``."""
        return OpTerm(ops=(), coeff=c)

    def scaled(self, c: complex) -> OpTerm:
        """Return a copy with coefficient multiplied by ``c``."""
        return OpTerm(self.ops, c * self.coeff)

    def adjoint(self) -> OpTerm:
        """Return the Hermitian adjoint term."""
        return OpTerm(
            ops=tuple(op.dagger() for op in reversed(self.ops)),
            coeff=self.coeff.conjugate(),
        )

    @property
    def signature(self) -> Signature:
        """Exact signature, stable under structural equality."""
        return ("op_term", tuple(op.signature for op in self.ops))

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature:
        """Approximate signature, delegating to ladder-operator components."""
        return (
            "op_term_approx",
            tuple(
                op.approx_signature(
                    decimals=decimals, ignore_global_phase=ignore_global_phase
                )
                for op in self.ops
            ),
        )

    def __len__(self) -> int:
        r"""Return the word length.

        This equals the number of ladder operators in the ordered product

        .. math::

            T = c \, o_1 o_2 \cdots o_L.

        Returns
        -------
        int
            Number of operators in the word.

        """
        return len(self.ops)

    def __iter__(self) -> Iterator[LadderOp]:
        r"""Iterate over ladder operators in word order.

        Yields
        ------
        LadderOpProto
            Operators :math:`o_1, \dots, o_L` in stored order.

        """
        return iter(self.ops)

    def __neg__(self) -> OpTerm:
        r"""Return the additive inverse ``-T``.

        Equivalent to multiplying the coefficient by ``-1``.
        """
        return self.scaled(-1.0)

    def __mul__(self, other: complex) -> OpTerm:
        r"""Return scalar multiplication ``T * c``.

        Parameters
        ----------
        other:
            Scalar multiplier.

        Returns
        -------
        OpTerm
            Term with scaled coefficient.

        Raises
        ------
        TypeError
            If ``other`` is not a scalar.

        """
        if not isinstance(other, int | float | complex):
            return NotImplemented
        return self.scaled(other)

    def __rmul__(self, other: complex) -> OpTerm:
        r"""Return scalar-left multiplication ``c * T``.

        Equivalent to ``T * c``.
        """
        if not isinstance(other, int | float | complex):
            return NotImplemented
        return self.scaled(other)

    def __truediv__(self, other: complex) -> OpTerm:
        r"""Return scalar division ``T / c``.

        Parameters
        ----------
        other:
            Scalar divisor.

        Returns
        -------
        OpTerm
            Term with coefficient divided by ``other``.

        Raises
        ------
        TypeError
            If ``other`` is not a scalar.
        ZeroDivisionError
            If ``other`` is numerically zero.

        """
        if not isinstance(other, int | float | complex):
            raise TypeError("OpTerm can only be divided by a scalar.")
        if abs(other) == 0:
            raise ZeroDivisionError("Division by zero scalar.")
        return self.scaled(1.0 / other)

    def __eq__(self, other: object) -> bool:
        r"""Return structural equality.

        Two terms are equal iff:

        - Their operator words are identical (exact order).
        - Their coefficients are exactly equal.

        No approximate comparison is performed.
        """
        if not isinstance(other, OpTerm):
            return NotImplemented
        return self.ops == other.ops and self.coeff == other.coeff

    def __bool__(self) -> bool:
        r"""Return ``False`` if the coefficient is exactly zero.

        A term with zero coefficient is considered algebraically zero,
        even if operators are present.
        """
        return self.coeff != 0

    def __repr__(self) -> str:
        r"""Return an unambiguous developer representation.

        This representation is structural and intentionally minimal,
        as pretty-printing belongs to a higher layer.
        """
        return f"{self.__class__.__name__}(ops={self.ops!r}, coeff={self.coeff!r})"
