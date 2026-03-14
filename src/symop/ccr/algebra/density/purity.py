r"""Purity of symbolic density polynomials.

The purity of a density operator is defined as

.. math::

    \mathrm{Purity}(\rho) = \mathrm{Tr}(\rho^2)
                           = \langle \rho, \rho \rangle,

i.e. the Hilbert--Schmidt inner product of the density with itself.

For normalized states:

- Pure states satisfy :math:`\mathrm{Tr}(\rho^2) = 1`
- Mixed states satisfy :math:`\mathrm{Tr}(\rho^2) < 1`

This module computes purity symbolically using the density inner product.
"""

from __future__ import annotations

from symop.core.protocols.terms import DensityTerm as DensityTermProtocol

from .inner import density_inner


def density_purity(
    terms: tuple[DensityTermProtocol, ...],
) -> float:
    r"""Compute the purity :math:`\mathrm{Tr}(\rho^2)`.

    Parameters
    ----------
    terms:
        Density polynomial terms representing :math:`\rho`.

    Returns
    -------
    float
        The real part of :math:`\mathrm{Tr}(\rho^2)`.

    Notes
    -----
    - Computed as ``density_inner(terms, terms)``.
    - The result is returned as a real float.
    - No Hermiticity assumption is enforced here.

    """
    return float(density_inner(terms, terms).real)
