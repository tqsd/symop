from collections.abc import Callable, Iterable

from symop_proto.algebra.protocols import OpTermProto
from symop_proto.core.protocols import LadderOpProto

TermFactory = Callable[[tuple[LadderOpProto, ...], complex], OpTermProto]


def op_from_words(
    words: Iterable[Iterable[LadderOpProto]],
    coeffs: Iterable[complex] | None = None,
    *,
    term_factory: TermFactory | None = None,
) -> tuple[OpTermProto, ...]:
    """Build operator terms from raw operator words and coefficients

    Each input word is materialized to a tuple of ``LadderOp`` and
    paired with a coefficent. If coeffs is None, a unit coefficient
    is provided for each word. The output term objects are constructed
    via term_factory, if not providded lazy import is used.

    Args:
        words: list of words
        coeffs: Optional list of coefficients, must match ``len(words)``,
             if not given 1 is used for each term
        term_factory: Factory for building the outputs

    Returns:
        Tuple[OpTermProto, ...]
            Tuple of constructed operator terms.

    """
    if term_factory is None:
        from symop_proto.algebra.operator_polynomial import OpTerm as _OpTerm

        term_factory = _OpTerm
    ws = [tuple(w) for w in words]
    if coeffs is None:
        coeffs = [1.0] * len(ws)
    return tuple(term_factory(w, c) for w, c in zip(ws, coeffs, strict=False))
