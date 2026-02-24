from __future__ import annotations
from typing import Iterable, List
from symop_proto.algebra.ket.apply import ket_apply_word
from symop_proto.algebra.ket.multiply import ket_multiply
from symop_proto.core.protocols import (
    KetTermProto,
    LadderOpProto,
    MonomialProto,
)
from symop_proto.algebra.ket.from_word import ket_from_word


def expand_word_times_monomial(
    word: Iterable[LadderOpProto], M: MonomialProto
) -> List[KetTermProto]:
    r"""Expand an operator word acting on a monomial from the left.

    Computes the normally ordered expansion of

    .. math::

        W \cdot \hat{M},

    where ``W`` is a sequence of ladder operators and :math:`\hat{M}` is the
    monomial. Internally this builds the ket for ``W`` via
    :func:`symop_proto.algebra.ket.from_word.ket_from_word`, multiplies by the
    one-term ket representing :math:`\hat{M}`, and expands with
    :func:`symop_proto.algebra.ket.multiply.ket_multiply`.

    Parameters
    ----------
    word : Iterable[LadderOpProto]
        Operator word to apply on the left.
    M : MonomialProto
        Target monomial.

    Returns
    -------
    List[KetTermProto]
        List of normally ordered ket terms for :math:`W \hat{M}`.
    """
    from symop_proto.core.terms import KetTerm

    terms: List[KetTermProto] = [KetTerm(1.0, M)]
    for op in reversed(tuple(word)):
        terms = list(ket_apply_word(tuple(terms), (op,)))
    return terms
