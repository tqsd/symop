from __future__ import annotations

from collections.abc import Iterable

from symop_proto.core.protocols import KetTermProto, LadderOpProto

from .combine import combine_like_terms_ket
from .from_word import ket_from_word
from .multiply import ket_multiply
from .scale import ket_scale


def ket_apply_word(
    ket: tuple[KetTermProto, ...], word: Iterable[LadderOpProto]
) -> tuple[KetTermProto, ...]:
    """Apply a single operator word to a symbolic ket.

    The word (list of ladder operators) is first expanded
    into a normally ordereb ket via :func:`ket_from_word`. Then expansion
    is multiplied with the input ket using :func:`ket_multiply`,
    yielding the normally ordered resultt of left multiplication by
    given operator product.

    Args:
        ket: Input ket given as a tuple of :class:`KetTerm`
        word: Iterable of :class:`LadderOp` instances forming the
            product to apply on the left.

    Returns:
        A putle of :class:`KetTerm` representing the result, already
        combined and normally ordered

    Notes:
        - This performs symbolic normal-ordering only; no matrix reps
          needed

    """
    word_terms = ket_from_word(ops=word)
    return ket_multiply(word_terms, ket)


def ket_apply_words_linear(
    ket: tuple[KetTermProto, ...],
    terms: Iterable[tuple[complex, Iterable[LadderOpProto]]],
) -> tuple[KetTermProto, ...]:
    """Aplly a linear combination of operator words to a ket

    For each pair (c,word) in ``terms``, this computes
    :func:`ket_apply_word(ket, word)` and scales the result by ``c``.
    All partial results are concatenated and ordered with
    :func:`combine_like_terms_ket`.

    Args:
        ket: Input ket
        terms: Iterable of (coeff[complex], word)

    Returns:
        A tuple of :class:`KetTerm`

    """
    out: tuple[KetTermProto, ...] = ()
    for coeff, word in terms:
        part = ket_apply_word(ket, word)
        if coeff != 1:
            part = ket_scale(part, coeff)
        out = (*out, *part)
    return combine_like_terms_ket(out)
