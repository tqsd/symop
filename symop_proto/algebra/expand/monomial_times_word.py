from __future__ import annotations
from typing import Iterable, List
from symop_proto.core.monomial import Monomial
from symop_proto.core.protocols import KetTermProto, LadderOpProto
from symop_proto.algebra.ket.from_word import ket_from_word


def expand_monomial_times_word(
    M: Monomial, word: Iterable[LadderOpProto]
) -> List[KetTermProto]:
    r"""Expand a monomial multiplied by an operator word.

    Produces all normally ordered :class:`KetTerm` instances resulting from

    .. math::

        \hat{M} \cdot W =
        (\prod_i \hat{a}_i^\dagger \prod_j \hat{a}_j) W,

    where :math:`\hat{M}` is the given monomial and ``W`` is a sequence of
    ladder operators. The result is obtained using
    :func:`symop_proto.algebra.ket.from_word.ket_from_word`.

    Parameters
    ----------
    M : Monomial
        The base monomial to expand from.
    word : Iterable[LadderOpProto]
        The operator word applied to the right.

    Returns
    -------
    List[KetTermProto]
        Expanded ket terms in normal order.
    """
    return list(
        ket_from_word(ops=(*M.creators, *M.annihilators, *tuple(word)))
    )
