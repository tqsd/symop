r"""Scaling of symbolic density polynomials.

Given a density polynomial

.. math::

    \rho = \sum_i c_i \, |L_i\rangle\langle R_i|,

this module provides scalar multiplication

.. math::

    c \rho = \sum_i (c\,c_i)\,|L_i\rangle\langle R_i|.

The structure of monomials is unchanged; only coefficients are scaled.
"""

from __future__ import annotations

from symop.core.protocols import DensityTermProto


def density_scale(
    terms: tuple[DensityTermProto, ...],
    c: complex,
) -> tuple[DensityTermProto, ...]:
    r"""Scale all density terms by a complex coefficient.

    Multiplies each term's coefficient by :math:`c` and returns a new tuple.

    Parameters
    ----------
    terms:
        Input density polynomial.
    c:
        Complex scalar multiplier.

    Returns
    -------
    tuple[DensityTermProto, ...]
        New density terms with coefficients scaled by :math:`c`.

    Notes
    -----
    - The original terms are not modified.
    - Left and right monomials are preserved.

    """
    from symop.core.terms import DensityTerm

    return tuple(
        DensityTerm(
            coeff=c * t.coeff,
            left=t.left,
            right=t.right,
        )
        for t in terms
    )
