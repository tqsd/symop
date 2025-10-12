from __future__ import annotations
from typing import Iterable, Tuple
from symop_proto.core.protocols import LadderOpProto, KetTermProto
from .combine import combine_like_terms_ket


def ket_from_ops(
    *,
    creators: Iterable[LadderOpProto] = (),
    annihilators: Iterable[LadderOpProto] = (),
    coeff: complex = 1.0,
    approx: bool = False,
    **env_kw,
) -> Tuple[KetTermProto, ...]:
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
    from symop_proto.core.monomial import Monomial
    from symop_proto.core.terms import KetTerm

    creators_t = tuple(creators)
    annihilators_t = tuple(annihilators)

    for op in creators:
        if not op.is_creation:
            raise ValueError("creators must be creation operators")
    for op in annihilators:
        if not op.is_annihilation:
            raise ValueError("annihilators must be annihilation operators")

    m = Monomial(tuple(creators_t), tuple(annihilators_t))
    return combine_like_terms_ket(
        (KetTerm(coeff, m),), approx=approx, **env_kw
    )
