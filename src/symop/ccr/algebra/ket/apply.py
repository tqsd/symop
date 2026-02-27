r"""Apply operator words to symbolic kets.

A symbolic ket is represented as a finite linear combination of normally-ordered
monomials,

.. math::

    \lvert \psi \rangle \sim \sum_k c_k\, M_k,

where each :math:`M_k` is a normally-ordered monomial in ladder operators.

This module implements left-multiplication by an operator word

.. math::

    W = \hat o_1 \hat o_2 \cdots \hat o_L,

acting as :math:`\lvert \psi \rangle \mapsto W \lvert \psi \rangle`, and also
the action of a linear combination of words.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.core.protocols import KetTermProto, LadderOpProto

from .combine import combine_like_terms_ket
from .from_word import ket_from_word
from .multiply import ket_multiply
from .scale import ket_scale


def ket_apply_word(
    ket: tuple[KetTermProto, ...],
    word: Iterable[LadderOpProto],
    *,
    eps: float = 1e-12,
) -> tuple[KetTermProto, ...]:
    """Apply a single operator word to a symbolic ket.

    The operator word (an ordered sequence of ladder operators) is expanded into
    a normally-ordered ket via :func:`~symop.ccr.ket.from_word.ket_from_word`.
    The resulting expansion is then multiplied with the input ``ket`` using
    :func:`~symop.ccr.algebra.ket.multiply.ket_multiply`, yielding the normally-ordered
    result of left multiplication by the given operator product.

    Parameters
    ----------
    ket:
        Input ket as a tuple of ket terms.
    word:
        Iterable of ladder operators forming the operator product to apply on
        the left.
    eps:
        Numerical tolerance forwarded to normal ordering / canonicalization

    Returns
    -------
    tuple[KetTermProto, ...]
        The resulting ket terms, normally ordered.

    Notes
    -----
    This performs symbolic normal ordering only; no matrix representation is
    involved.

    """
    word_terms = ket_from_word(ops=word, eps=eps)
    return ket_multiply(word_terms, ket)


def ket_apply_words_linear(
    ket: tuple[KetTermProto, ...],
    terms: Iterable[tuple[complex, Iterable[LadderOpProto]]],
    *,
    eps: float = 1e-12,
) -> tuple[KetTermProto, ...]:
    """Apply a linear combination of operator words to a ket.

    For each pair ``(c, word)`` in ``terms``, this computes
    :func:`~symop.ccr.algebra.ket.apply.ket_apply_word` and scales the result by ``c``.
    All partial results are concatenated and canonicalized using
    :func:`~symop.ccr.algebra.ket.combine.combine_like_terms_ket`.

    Parameters
    ----------
    ket:
        Input ket as a tuple of ket terms.
    terms:
        Iterable of pairs ``(coeff, word)`` describing the linear combination.
    eps:
        Numerical tolerance forwarded to normal ordering / canonicalization

    Returns
    -------
    tuple[KetTermProto, ...]
        The resulting ket terms, combined into a canonical form.

    """
    out: tuple[KetTermProto, ...] = ()

    for coeff, word in terms:
        if coeff == 0:
            continue
        part = ket_apply_word(ket, word, eps=eps)
        if coeff != 1:
            part = ket_scale(part, coeff)
        out = (*out, *part)

    return combine_like_terms_ket(out, eps=eps)
