r"""Left-application of operator words to density polynomials.

A density polynomial is a finite linear combination of terms

.. math::

    \rho \sim \sum_k c_k\, L_k\, R_k,

where each term stores a complex coefficient ``c_k`` and two normally-ordered
monomials ``(left=L_k, right=R_k)``.

This module provides the symbolic action of an operator word
:math:`W = \hat o_1 \hat o_2 \cdots \hat o_L` on the *left*:

.. math::

    \rho \mapsto W\,\rho.

For each term, the product :math:`W L_k` is expanded and normal-ordered using
commutation relations via :func:`~symop.ccr.algebra.expand.word_times_monomial.expand_word_times_monomial`.
Each expansion contribution updates the term's left monomial and coefficient,
while keeping the right monomial unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.ccr.algebra.density.expand_word_times_monomial import (
    expand_word_times_monomial,
)
from symop.core.protocols import DensityTermProto, LadderOpProto
from symop.core.terms import DensityTerm

from .combine import combine_like_terms_density


def apply_left(
    terms: tuple[DensityTermProto, ...],
    word: Iterable[LadderOpProto],
) -> tuple[DensityTermProto, ...]:
    """Apply an operator word on the left of a density polynomial.

    For each density term ``t``, this function expands the product
    ``word * t.left`` into a linear combination of normally-ordered monomials
    via :func:`~symop.ccr.algebra.expand.word_times_monomial.expand_word_times_monomial`.

    Each expanded ket-term ``kt`` contributes a density term with:

    - ``coeff = t.coeff * kt.coeff``
    - ``left  = kt.monomial``
    - ``right = t.right`` (unchanged)

    All contributions from all input terms are concatenated and then merged
    using :func:`~symop.ccr.algebra.density.combine.combine_like_terms_density`.

    Parameters
    ----------
    terms:
        Input density polynomial terms.
    word:
        Iterable of ladder operators forming the operator product applied on
        the left of each term's left monomial.

    Returns
    -------
    tuple[DensityTermProto, ...]
        Output density polynomial terms after left application and combination.

    Notes
    -----
    This is purely symbolic. Normal ordering is performed using commutation
    relations inside ``expand_word_times_monomial``.

    """
    out: list[DensityTermProto] = []

    for t in terms:
        for kt in expand_word_times_monomial(word, t.left):
            out.append(
                DensityTerm(
                    coeff=t.coeff * kt.coeff,
                    left=kt.monomial,
                    right=t.right,
                )
            )

    return combine_like_terms_density(out)
