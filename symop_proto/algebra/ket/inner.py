from __future__ import annotations
from typing import Tuple
from symop_proto.algebra.ket.from_word import ket_from_word
from symop_proto.algebra.ket.identity_coeff import identity_coeff
from symop_proto.core.protocols import KetTermProto


def ket_inner(
    a: Tuple[KetTermProto, ...],
    b: Tuple[KetTermProto, ...],
    *,
    eps: float = 1e-12,
) -> complex:
    """Compute the inner product between two symbolic ket expansions.

    Evalutes the overlap <a|b> between the two tuples of :class:`KetTerm`
    instances by contracting their creation and annihilation operators.
    Each pair of terms is combined into a full operator word, expanded
    via :func:`ket_from_word`, and the coefficient of the identity term
    is extracted to yield the scalar contribution to the inner product.

    Args:
        a: First Tuple of terms representing the bra (conjugated) side
            of the inner product.
        b: Second tuple of terms representing the ket side of the inner
            product.

    Returns:
        complex: The total inner product value <a|b> obtained by summing
        over all term-wise contractions.

    Notes:
        - The adjoint of each term in ``a`` is constructed by taking the
          Hermitian conjugate of its operators. This is done by the function.
        - Only the scalar (identity) component of each resulting operator
          contributes to the overlap. The identity coefficient corresponds to
          the term where all creation-annihilation pairs have been contracted
          away.
        - This function performs a purely symbolic evaluation of the inner
          product; no explicit matrix representation is required.

    Complexity:
        O(N_a  N_b  C(L)), where N_a and N_b are the number of terms in
        ``a`` and ``b``, and C(L) is the cost of a single call to
        :func:`ket_from_word`, typically scaling between O(L) and O(L^2)
        for short operator sequences.
    """
    total: complex = 0.0 + 0.0j
    for ti in a:
        if abs(ti.coeff) <= eps:
            continue
        mi = ti.monomial

        L_cre = tuple(op.dagger() for op in reversed(mi.annihilators))
        L_ann = tuple(op.dagger() for op in reversed(mi.creators))
        for tj in b:
            if abs(tj.coeff) <= eps:
                continue
            mj = tj.monomial
            word_terms = ket_from_word(
                ops=(
                    *L_cre,
                    *L_ann,
                    *mj.creators,
                    *mj.annihilators,
                ),
                eps=eps,
            )
            total += (
                ti.coeff.conjugate() * tj.coeff * identity_coeff(word_terms)
            )
    return total
