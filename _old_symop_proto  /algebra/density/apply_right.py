from __future__ import annotations

from collections.abc import Iterable

from symop_proto.algebra.expand.word_times_monomial import (
    expand_word_times_monomial,
)
from symop_proto.core.protocols import DensityTermProto, LadderOpProto

from .combine import combine_like_terms_density


def density_apply_right(
    terms: tuple[DensityTermProto, ...], word: Iterable[LadderOpProto]
) -> tuple[DensityTermProto, ...]:
    """Apply an operator word on the right of density polynomial

    For each density term, this computes the right action. Operationaly
    we expand that product by applying the reversed, daggered word to the
    right monomial from the left using :func:`expand_word_times_monomial()`.
    Each expanded ket-term contributes a density term, while keeping the left
    monomial unchanged. Contributions from all inputs are then combined and
    ordered with :func:`combine_like_terms_density()`.

    Args:
        terms: Input density polynomial as tuple
        word: Ordered list of :class:`LadderOp` forming the operator
            product

    Returns:
        a tuple of :class:`DensityTerm` forming the DensityPolynomial

    Notes:
        - Purely symbolic normal ordering: no matrix representations.

    """
    from symop_proto.core.terms import DensityTerm

    w = tuple(word)
    dag_word = tuple(op.dagger() for op in reversed(w))

    out: list[DensityTermProto] = []
    for t in terms:
        for kt in expand_word_times_monomial(dag_word, t.right):
            out.append(
                DensityTerm(
                    coeff=t.coeff * kt.coeff,
                    left=t.left,
                    right=kt.monomial,
                )
            )
    return combine_like_terms_density(out)
