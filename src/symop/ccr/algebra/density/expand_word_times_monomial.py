r"""Expansion of an operator word acting on a monomial from the left.

This module provides a symbolic normal-ordering helper for products of the form

.. math::

    W \cdot \hat{M},

where :math:`W` is an operator word (a finite ordered sequence of ladder
operators) and :math:`\hat{M}` is a normally-ordered monomial in ladder
operators.

The result is returned as a list of ket terms representing the normally-ordered
expansion of :math:`W \hat{M}`.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.ccr.algebra.ket.apply import ket_apply_word
from symop.core.protocols import KetTermProto, LadderOpProto, MonomialProto


def expand_word_times_monomial(
    word: Iterable[LadderOpProto],
    M: MonomialProto,
) -> list[KetTermProto]:
    r"""Expand an operator word acting on a monomial from the left.

    Computes the normally-ordered expansion of

    .. math::

        W \cdot \hat{M},

    where ``W`` is a sequence of ladder operators and :math:`\hat{M}` is the
    target monomial.

    Internally, this starts from the one-term ket representing :math:`\hat{M}`
    and applies the operators in ``word`` one-by-one (right-to-left) using
    :func:`~symop.ccr.algebra.ket.apply.ket_apply_word`.

    Parameters
    ----------
    word:
        Operator word to apply on the left.
    M:
        Target monomial.

    Returns
    -------
    list[KetTermProto]
        List of normally-ordered ket terms for :math:`W \hat{M}`.

    """
    from symop.core.terms import KetTerm

    terms: tuple[KetTermProto, ...] = (KetTerm(coeff=1.0, monomial=M),)

    # Left multiplication by W = o1 o2 ... oL means we apply operators starting
    # from the rightmost operator on the current expression.
    for op in reversed(tuple(word)):
        terms = tuple(ket_apply_word(terms, (op,)))

    return list(terms)
