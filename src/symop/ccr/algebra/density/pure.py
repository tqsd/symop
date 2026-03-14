r"""Pure-state density construction for symbolic kets.

Given a ket expansion

.. math::

    |\psi\rangle = \sum_i c_i\,|M_i\rangle,

this module constructs the corresponding rank-1 density polynomial

.. math::

    \rho = |\psi\rangle\langle\psi|
         = \sum_{i,j} c_i c_j^*\, |M_i\rangle\langle M_j|.

All terms are merged using symbolic density combination.
"""

from __future__ import annotations

from symop.core.protocols.terms import KetTerm as KetTermProtocol
from symop.core.terms.density_term import DensityTerm

from .combine import combine_like_terms_density


def density_pure(
    ket_terms: tuple[KetTermProtocol, ...],
) -> tuple[DensityTerm, ...]:
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
    ket_terms:
        Terms forming the state vector.

    Returns
    -------
    tuple[DensityTermProto, ...]
        Density terms representing :math:`|\psi\rangle\langle\psi|`,
        combined via
        :func:`~symop.ccr.algebra.density.combine.combine_like_terms_density`.

    Notes
    -----
    - No assumption of normalization is made.
    - The output is Hermitian by construction.

    """
    out: list[DensityTerm] = []

    for ti in ket_terms:
        for tj in ket_terms:
            out.append(
                DensityTerm(
                    coeff=ti.coeff * tj.coeff.conjugate(),
                    left=ti.monomial,
                    right=tj.monomial,
                )
            )

    return combine_like_terms_density(tuple(out))
