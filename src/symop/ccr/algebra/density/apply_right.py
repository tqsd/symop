r"""Right-application of operator words to density polynomials.

A density polynomial is a finite linear combination of terms

.. math::

    \rho \sim \sum_k c_k\, L_k\, R_k,

where each term stores a complex coefficient ``c_k`` and two normally-ordered
monomials ``(left=L_k, right=R_k)``.

This module provides the symbolic action of an operator word
:math:`W = \hat o_1 \hat o_2 \cdots \hat o_L` on the *right*:

.. math::

    \rho \mapsto \rho\, W.

Operationally, right-action is implemented using the identity

.. math::

    R_k\, W \;\equiv\; \left(W^\dagger R_k^\dagger\right)^\dagger,

which, at the level of symbolic normal ordering, can be computed by expanding
the left action of the reversed daggered word on the right monomial.

Concretely we form

.. math::

    W_{\mathrm{dag}} = \hat o_L^\dagger \cdots \hat o_2^\dagger \hat o_1^\dagger

and expand :math:`W_{\mathrm{dag}} \cdot R_k` via the normal-ordering engine.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.ccr.algebra.density.expand_word_times_monomial import (
    expand_word_times_monomial,
)
from symop.core.protocols import DensityTermProto, LadderOpProto
from symop.core.terms import DensityTerm

from .combine import combine_like_terms_density


def apply_right(
    terms: tuple[DensityTermProto, ...],
    word: Iterable[LadderOpProto],
) -> tuple[DensityTermProto, ...]:
    """Apply an operator word on the right of a density polynomial.

    For each density term ``t``, this function computes the symbolic right
    action by expanding the left action of the reversed, daggered word on the
    right monomial:

    - form ``dag_word = reversed(word)`` with each operator daggered
    - expand ``dag_word * t.right`` via
      :func:`~symop.ccr.algebra.expand.word_times_monomial.expand_word_times_monomial`

    Each expanded ket-term ``kt`` contributes an output density term with:

    - ``coeff = t.coeff * kt.coeff``
    - ``left  = t.left`` (unchanged)
    - ``right = kt.monomial``

    All contributions are concatenated and then canonicalized using
    :func:`~symop.ccr.algebra.density.combine.combine_like_terms_density`.

    Parameters
    ----------
    terms:
        Input density polynomial terms.
    word:
        Iterable of ladder operators forming the operator product applied on
        the right.

    Returns
    -------
    tuple[DensityTermProto, ...]
        Output density polynomial terms after right application and combination.

    Notes
    -----
    This is purely symbolic. Normal ordering is performed using commutation
    relations inside ``expand_word_times_monomial``.

    """
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
