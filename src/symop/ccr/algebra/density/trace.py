r"""Trace of symbolic density polynomials.

Given a density polynomial

.. math::

    \rho = \sum_i c_i \, |L_i\rangle\langle R_i|,

the trace is computed symbolically as

.. math::

    \mathrm{Tr}(\rho)
    = \sum_i c_i \, \langle L_i \mid R_i \rangle.

The bra–ket overlap is evaluated purely symbolically via
:func:`overlap_right_left`.
"""

from __future__ import annotations

from symop.core.protocols import DensityTermProto

from .overlap_right_left import overlap_right_left


def density_trace(
    terms: tuple[DensityTermProto, ...],
) -> complex:
    r"""Compute the trace of a density polynomial.

    Evaluates

    .. math::

        \mathrm{Tr}(\rho)
        =
        \sum_i c_i \, \langle L_i \mid R_i \rangle,

    where each term corresponds to :math:`c_i |L_i\rangle\langle R_i|`.

    Parameters
    ----------
    terms:
        Density terms forming the operator.

    Returns
    -------
    complex
        Complex trace value.

    Notes
    -----
    - Computed symbolically via :func:`overlap_right_left`.
    - No Hermiticity assumption is required.

    """
    total: complex = 0.0 + 0.0j

    for t in terms:
        total += t.coeff * overlap_right_left(t.right, t.left)

    return total
