from __future__ import annotations
from typing import Iterable, List, Tuple
from symop_proto.algebra.expand.word_times_monomial import (
    expand_word_times_monomial,
)
from symop_proto.core.protocols import DensityTermProto, LadderOpProto
from .combine import combine_like_terms_density


def density_apply_left(
    terms: Tuple[DensityTermProto, ...], word: Iterable[LadderOpProto]
) -> Tuple[DensityTermProto, ...]:
    """Apply Operator word on the left of a density polynomial

    For each density term, this function expands the product into
    a linear combination of normally-ordered monomials via
    :func:`expand_word_times_monomial()`. Each expanded ket-term
    ``kt`` contributes a density term with
    ``coeff = t.coeff*kt.coeff``, ``left = kt.monomial``, and the
    original right factor unchandeg. All contributions from all
    input terms are concatenated and ordered with
    :func:`combine_like_terms_density`.

    Args:
        terms: Input density polynomial as tuple of :class:`DensityTrem`
        word: Ordered list of :class:`LadderOp` forming the operator
        product applied on the left of each term's left monomial.

    Returns:
        A tuple of :class:`DensityTerm`

    Notes:
        - This is purely symbolic: it uses commutators to normal order
    """
    from symop_proto.core.terms import DensityTerm

    out: List[DensityTermProto] = []

    for t in terms:
        for kt in expand_word_times_monomial(word, t.left):
            out.append(
                DensityTerm(
                    coeff=t.coeff * kt.coeff, left=kt.monomial, right=t.right
                )
            )
    return combine_like_terms_density(out)
