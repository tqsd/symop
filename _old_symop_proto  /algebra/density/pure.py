from __future__ import annotations
from typing import List, Tuple
from symop_proto.core.protocols import DensityTermProto, KetTermProto
from .combine import combine_like_terms_density


def density_pure(
    ket_terms: Tuple[KetTermProto, ...],
) -> Tuple[DensityTermProto, ...]:
    r"""Construct a pure-state density polynomial.

    Given a ket expansion

    .. math::

        |\psi\rangle = \sum_i c_i \, |M_i\rangle,

    this returns

    .. math::

        \rho = |\psi\rangle \langle \psi|
        = \sum_{i,j} c_i c_j^* \, |M_i\rangle \langle M_j|.

    Parameters
    ----------
    ket_terms : Tuple[KetTermProto, ...]
        Terms forming the state vector.

    Returns
    -------
    Tuple[DensityTermProto, ...]
        Density terms representing :math:`|\psi\rangle\langle\psi|`,
        combined with :func:`combine_like_terms_density`.
    """
    from symop_proto.core.terms import DensityTerm

    out: List[DensityTermProto] = []
    for ti in ket_terms:
        for tj in ket_terms:
            out.append(
                DensityTerm(ti.coeff * tj.coeff.conjugate(), ti.monomial, tj.monomial)
            )
    return combine_like_terms_density(out)
