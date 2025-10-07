from __future__ import annotations
from typing import List, Tuple
from symop_proto.core.terms import KetTerm
from symop_proto.algebra.ket.from_word import ket_from_word
from .combine import combine_like_terms_ket


def ket_multiply(
    a: Tuple[KetTerm, ...], b: Tuple[KetTerm, ...]
) -> Tuple[KetTerm, ...]:
    """Multiply two symbolic ket expansions

    Forms the product of two tuples of :class:`KetTerm` objects by
    concatenating their creation and annihilation operators, expanding
    the resulting operator words via :func:`ket_from_word`, and summing
    all resulting terms. The output is normalized and simplified with
    :func:`combine_like_terms_ket`.

    Args:
        a: Left hand factor terms in the product.
        b: Right hand factor terms in the product.

    Returns:
        The combined ket terms obtained from all pairwise products between
        ``a`` and ``b``. Equivalent monomials are merged and coefficients
        summed.

    Complexity:
        O(N_a * N_b * C(L)), where N_a and N_b are the numbers of terms
        in ``a`` and ``b``, and C(L) is the cost of expanding a single
        operator word of length L via :func:`ket_from_word`.
    """
    out: List[KetTerm] = []
    for ti in a:
        for tj in b:
            word_terms = ket_from_word(
                ops=(
                    *ti.monomial.creators,
                    *ti.monomial.annihilators,
                    *tj.monomial.creators,
                    *tj.monomial.annihilators,
                )
            )
            for tk in word_terms:
                out.append(
                    KetTerm(ti.coeff * tj.coeff * tk.coeff, tk.monomial)
                )
    return combine_like_terms_ket(out)
