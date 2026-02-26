from collections.abc import Callable

from symop_proto.algebra.protocols import OpTermProto
from symop_proto.core.protocols import LadderOpProto

TermFactory = Callable[[tuple[LadderOpProto, ...], complex], OpTermProto]


def op_combine_like_terms(
    terms: tuple[OpTermProto, ...],
    *,
    approx: bool = False,
    term_factory: TermFactory | None = None,
    **env_kw,
) -> tuple[OpTermProto, ...]:
    """Combine like operator-word terms

    This function buckets input terms by either their exact signatures
    or their approximate signatures when ``approx=True``. Coefficients
    are accumulated per bucket, and the term is operator term is emited.

    Args:
        terms: List of operator terms to combine
        approx: If true approximate signatures are used
        term_factory: Optional callable used to build output
            terms. If not provided, a lazy import of ``OpTerm``
            is performed to avoid a hard dependency.
        **env_kw: Extra keyword used for the envelope comparison
            when ``approx=True``.

    Returns:
        Combined list of terms with summed coefficient, zero-sum buckets
            are dropped

    """
    if term_factory is None:
        from symop_proto.algebra.operator_polynomial import OpTerm as _OpTerm

        term_factory = _OpTerm
    key = (
        (lambda t: t.approx_signature(**env_kw)) if approx else (lambda t: t.signature)
    )
    buckets: dict[tuple, complex] = {}
    reps: dict[tuple, tuple[LadderOpProto, ...]] = {}
    for t in terms:
        k = key(t)
        buckets[k] = buckets.get(k, 0.0j) + t.coeff
        reps.setdefault(k, t.ops)
    return tuple(term_factory(reps[k], c) for k, c in buckets.items() if c != 0.0)
