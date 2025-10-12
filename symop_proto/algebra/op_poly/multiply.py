from typing import Callable, Optional, Tuple

from symop_proto.algebra.protocols import OpTermProto
from symop_proto.core.protocols import LadderOpProto


TermFactory = Callable[[Tuple[LadderOpProto, ...], complex], OpTermProto]


def op_multiply(
    a: Tuple[OpTermProto, ...],
    b: Tuple[OpTermProto, ...],
    *,
    term_factory: Optional[TermFactory] = None,
) -> Tuple[OpTermProto, ...]:
    """Cartesizn product of two ``OperatorTerm`` tuples

    For each pair ``(ti, tj)`` in a times b this function forms
    a new term whose operator word is ``ti.ops+tj.ops`` and whose
    coefficient is ``ti.coeff*tj.coeff``. The constructor used for
    output terms is provided by ``term_factory``, if omitted, a lazy
    import isused.

    Args:
        a: Tuple of left operator terms
        b: Tuple of right operator terms
        term_factory: OpTermProto used to build output

    Returns:
        Tuple containing ``len(a)*len(b)`` multiplied terms in row-major
        order

    """
    if term_factory is None:
        from symop_proto.algebra.operator_polynomial import OpTerm as _OpTerm

        term_factory = _OpTerm
    return tuple(
        term_factory(ti.ops + tj.ops, ti.coeff * tj.coeff)
        for ti in a
        for tj in b
    )
