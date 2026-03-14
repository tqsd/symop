r"""Trace normalization for symbolic density polynomials.

This module provides a helper that rescales a density polynomial so that its
trace equals one.

Given a density polynomial :math:`\rho`, we compute :math:`\mathrm{Tr}(\rho)`
symbolically and return

.. math::

    \rho' = \frac{1}{\mathrm{Tr}(\rho)}\,\rho,

whenever :math:`|\mathrm{Tr}(\rho)|` is not too small.
"""

from __future__ import annotations

from symop.core.protocols.terms import DensityTerm as DensityTermProtocol
from symop.core.terms.density_term import DensityTerm

from .scale import density_scale
from .trace import density_trace


def density_normalize_trace(
    terms: tuple[DensityTermProtocol, ...],
    *,
    eps: float = 1e-14,
) -> tuple[DensityTerm, ...]:
    r"""Scale a density polynomial so that its trace equals 1.

    This function computes :math:`\mathrm{Tr}(\rho)` via
    :func:`~symop.ccr.algebra.density.trace.density_trace` and returns
    :math:`\rho' = \rho / \mathrm{Tr}(\rho)`. If
    :math:`|\mathrm{Tr}(\rho)| < \text{eps}`, a :class:`ValueError` is raised.

    Mathematically,

    .. math::

        \rho' \;=\; \frac{1}{\mathrm{Tr}(\rho)} \, \rho,
        \qquad
        \mathrm{Tr}(\rho') \;=\; 1.

    Parameters
    ----------
    terms:
        Input density polynomial :math:`\rho`.
    eps:
        Minimum absolute threshold for :math:`|\mathrm{Tr}(\rho)|`. If the
        trace magnitude is smaller than ``eps``, normalization is refused.

    Returns
    -------
    tuple[DensityTerm, ...]
        The scaled density polynomial with unit trace.

    Raises
    ------
    ValueError
        If :math:`|\mathrm{Tr}(\rho)| < \text{eps}`.

    Notes
    -----
    - Scaling is performed by :func:`~symop.ccr.algebra.density.scale.density_scale`
      with factor :math:`1/\mathrm{Tr}(\rho)`.
    - Works with complex traces; no assumption of Hermiticity is required.

    """
    tr = density_trace(terms)
    if abs(tr) < eps:
        raise ValueError(f"Cannot normalize trace: |Tr(rho)| < {eps}")
    return density_scale(terms, 1.0 / tr)
