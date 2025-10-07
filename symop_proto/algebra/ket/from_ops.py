from __future__ import annotations
from typing import Iterable, Tuple
from symop_proto.core.operators import LadderOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm
from .combine import combine_like_terms_ket


def ket_from_ops(
    *,
    creators: Iterable[LadderOp] = (),
    annihilators: Iterable[LadderOp] = (),
    coeff: complex = 1.0,
) -> Tuple[KetTerm, ...]:
    """Construct a ket term from creation and annihilation operators

    Creates a single :class:`KetTerm` with the given coefficient and
    operator sequence defined by the provided operators. The resulting
    term is passed through :func:`combine_like_terms_ket` for normalization
    and consistency of output format.

    Args:
        creators: Creation operators applied in order.
        annihilators: Annihilation operators applied in order.
        coeff: Complex coefficient of the resulting term.

    Returns:
        A tuple containing the constructed ket term. Tuple is chosen, so
        that the output can easily be forwarded to the KetPoly.

    """
    m = Monomial(tuple(creators), tuple(annihilators))
    return combine_like_terms_ket((KetTerm(coeff, m),))
