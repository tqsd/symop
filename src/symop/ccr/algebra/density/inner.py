r"""Hilbert--Schmidt inner product for symbolic density polynomials.

A density polynomial is represented as a finite sum of rank-1 outer products

.. math::

    A \sim \sum_i c_i \, |L_i\rangle \langle R_i|,

with complex coefficients ``c_i`` and normally-ordered monomials ``L_i, R_i``.
This module evaluates the symbolic Hilbert--Schmidt inner product

.. math::

    \langle A, B \rangle = \mathrm{Tr}(A^\dagger B),

without constructing any explicit matrix representations.
"""

from __future__ import annotations

from symop.core.protocols import DensityTermProto

from .overlap_right_left import overlap_right_left


def density_inner(
    a: tuple[DensityTermProto, ...],
    b: tuple[DensityTermProto, ...],
) -> complex:
    r"""Compute the Hilbert--Schmidt inner product between two density polynomials.

    This function evaluates the symbolic Hilbert--Schmidt inner product

    .. math::

        \langle A, B \rangle = \mathrm{Tr}(A^\dagger B)

    by expanding density terms of both operands. Each density term is of the
    form

    .. math::

        t_i = c_i \, |L_i\rangle \langle R_i|,

    where :math:`c_i` is a complex coefficient. The total inner product is

    .. math::

        \langle A, B \rangle
        = \sum_{i \in A} \sum_{j \in B}
          c_i^* c_j \,
          \langle L_i | L_j \rangle \,
          \langle R_j | R_i \rangle.

    The bra--ket overlaps are evaluated symbolically using
    :func:`~symop.ccr.algebra.density.overlap_right_left.overlap_right_left`.

    Parameters
    ----------
    a:
        Left operand (bra side) of the inner product, corresponding to
        :math:`A^\dagger`.
    b:
        Right operand (ket side) of the inner product, corresponding to
        :math:`B`.

    Returns
    -------
    complex
        The Hilbert--Schmidt inner product :math:`\mathrm{Tr}(A^\dagger B)`.

    Notes
    -----
    * The coefficients of the left operand ``a`` are conjugated to form
      :math:`A^\dagger`.
    * Overlaps between left and right monomials are computed symbolically via
      commutation rules, without explicit matrix representations.
    * This inner product is Hermitian:
      :math:`\langle A, B \rangle = \langle B, A \rangle^*`.

    """
    total: complex = 0.0 + 0.0j

    for ti in a:
        Li, Ri, ci = ti.left, ti.right, ti.coeff
        for tj in b:
            Lj, Rj, cj = tj.left, tj.right, tj.coeff
            ov_L = overlap_right_left(Li, Lj)
            ov_R = overlap_right_left(Rj, Ri)
            total += ci.conjugate() * cj * ov_L * ov_R

    return total
