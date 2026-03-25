r"""Density polynomial multiplication.

This module provides functionality for multiplying density polynomials
in the CCR-based symbolic framework.

A density polynomial is represented as a linear combination of terms of
the form :math:`c \, |L\rangle\langle R|`, where :math:`|L\rangle` and
:math:`|R\rangle` are ket polynomials.

The product of two density terms is defined via contraction of the inner
indices:

.. math::

    \left(c_1 \, |L\rangle\langle R|\right)
    \left(c_2 \, |A\rangle\langle B|\right)
    =
    c_1 c_2 \, \langle R | A \rangle \, |L\rangle\langle B|.

This corresponds to standard operator composition in Dirac notation,
where the inner product :math:`\langle R | A \rangle` contracts the
adjacent bra–ket pair.

The multiplication is extended linearly over all pairs of terms from the
left and right density polynomials. Contributions with negligible overlap
are discarded based on a numerical threshold.

Notes
-----
- The inner product :math:`\langle R | A \rangle` is evaluated using
  :func:`symop.ccr.algebra.ket.inner.ket_inner`.
- Terms are canonicalized after multiplication using
  :func:`combine_like_terms_density`.
- This operation preserves the density-operator structure
  :math:`|\,\cdot\,\rangle\langle\,\cdot\,|`.

"""

from __future__ import annotations

from symop.ccr.algebra.ket.inner import ket_inner
from symop.core.terms import DensityTerm, KetTerm

from .combine import combine_like_terms_density


def density_multiply(
    left_terms: tuple[DensityTerm, ...],
    right_terms: tuple[DensityTerm, ...],
    *,
    eps: float = 1e-12,
) -> tuple[DensityTerm, ...]:
    """Return the symbolic product of two density polynomials.

    For density terms

    ``c1 |L><R|`` and ``c2 |A><B|``,

    the product is defined as

    ``c1 c2 <R|A> |L><B|``,

    extended linearly over all pairs of terms.

    Parameters
    ----------
    left_terms:
        Terms of the left density polynomial.
    right_terms:
        Terms of the right density polynomial.
    eps:
        Contributions with overlap magnitude below ``eps`` are discarded.

    Returns
    -------
    tuple[DensityTerm, ...]
        Canonicalized output density terms.

    """
    out: list[DensityTerm] = []

    for t_left in left_terms:
        ket_r = (KetTerm(coeff=1.0 + 0.0j, monomial=t_left.right),)

        for t_right in right_terms:
            ket_a = (KetTerm(coeff=1.0 + 0.0j, monomial=t_right.left),)
            overlap = ket_inner(ket_r, ket_a, eps=eps)

            if abs(overlap) <= eps:
                continue

            out.append(
                DensityTerm(
                    coeff=t_left.coeff * t_right.coeff * overlap,
                    left=t_left.left,
                    right=t_right.right,
                )
            )

    return combine_like_terms_density(out, eps=eps)
