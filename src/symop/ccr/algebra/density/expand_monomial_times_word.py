r"""Expansion of a monomial multiplied by an operator word.

This module provides a helper for symbolic normal ordering of products

.. math::

    \hat{M} \cdot W,

where :math:`\hat{M}` is a normally-ordered monomial in ladder operators and
:math:`W` is an operator word (an ordered sequence of ladder operators).

The result is returned as a list of ket terms representing the normally-ordered
expansion of :math:`\hat{M} W`.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.ccr.algebra.ket.from_word import ket_from_word
from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops import (
    Monomial as MonomialProtocol,
)
from symop.core.terms.ket_term import KetTerm


def expand_monomial_times_word(
    M: MonomialProtocol,
    word: Iterable[LadderOpProtocol],
) -> list[KetTerm]:
    r"""Expand a monomial multiplied by an operator word.

    Produces all normally ordered ket terms resulting from

    .. math::

        \hat{M} \cdot W =
        \left(\prod_i \hat{a}_i^\dagger \prod_j \hat{a}_j\right) W,

    where :math:`\hat{M}` is the given monomial and ``W`` is a sequence of
    ladder operators.

    The result is obtained by converting the concatenated operator sequence
    into a normally-ordered ket expansion using
    :func:`~symop.ccr.ket.from_word.ket_from_word`.

    Parameters
    ----------
    M:
        Base monomial to expand from.
    word:
        Operator word applied on the right.

    Returns
    -------
    list[KetTerm]
        Expanded ket terms in normal order.

    """
    ops = (*M.creators, *M.annihilators, *tuple(word))
    return list(ket_from_word(ops=ops))
