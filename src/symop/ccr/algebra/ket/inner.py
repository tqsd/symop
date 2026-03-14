r"""Symbolic inner products for CCR ket expansions.

This module provides a purely symbolic routine to compute inner products
between ket expansions represented as sums of normally ordered monomials.

Given two expansions

.. math::

    \lvert a \rangle \;\sim\; \sum_i c_i\, M_i,
    \qquad
    \lvert b \rangle \;\sim\; \sum_j d_j\, N_j,

the inner product is

.. math::

    \langle a \mid b \rangle
    = \sum_{i,j} c_i^* d_j \,\langle 0 \mid M_i^\dagger N_j \mid 0 \rangle.

Each contribution is evaluated by forming the operator word corresponding
to :math:`M_i^\dagger N_j`, expanding it into normal order using
:func:`symop.ccr.algebra.ket.from_word.ket_from_word`, and extracting the scalar
(identity) coefficient, which is the only part that survives the vacuum
sandwich.

"""

from __future__ import annotations

from symop.core.protocols.terms import KetTerm as KetTermProtocol

from .from_word import ket_from_word
from .identity_coeff import identity_coeff


def ket_inner(
    a: tuple[KetTermProtocol, ...],
    b: tuple[KetTermProtocol, ...],
    *,
    eps: float = 1e-12,
) -> complex:
    r"""Compute the symbolic inner product :math:`\langle a \mid b \rangle`.

    The inputs are tuples of ket terms representing formal expansions

    .. math::

        \lvert a \rangle \;\sim\; \sum_i c_i\, M_i,
        \qquad
        \lvert b \rangle \;\sim\; \sum_j d_j\, N_j,

    where :math:`M_i` and :math:`N_j` are normally ordered monomials of
    ladder operators.

    For each pair of terms, the contribution to the inner product is computed as

    .. math::

        c_i^* d_j \,\langle 0 \mid M_i^\dagger N_j \mid 0 \rangle,

    which equals the coefficient of the identity operator in the normal-ordered
    expansion of the word :math:`M_i^\dagger N_j`. This identity coefficient is
    obtained by expanding the word with :func:`symop.ccr.ket.from_word.ket_from_word`
    and extracting the identity term via :func:`symop.ccr.ket.identity_coeff.identity_coeff`.

    Parameters
    ----------
    a :
        Tuple of ket terms representing :math:`\lvert a \rangle`.
    b :
        Tuple of ket terms representing :math:`\lvert b \rangle`.
    eps :
        Coefficient threshold used to skip negligible input terms and to
        discard negligible terms during intermediate normal ordering.

    Returns
    -------
    complex
        The inner product :math:`\langle a \mid b \rangle`.

    Notes
    -----
    - The adjoint :math:`M^\dagger` is formed by reversing operator order,
      swapping creators and annihilators, and applying ``dagger()`` to each
      ladder operator.
    - Only the scalar (identity) component of :math:`M_i^\dagger N_j` contributes
      to the vacuum sandwich.
    - This routine is purely symbolic and does not form matrix representations.

    """
    total: complex = 0.0 + 0.0j

    for ti in a:
        if abs(ti.coeff) <= eps:
            continue

        mi = ti.monomial
        left_cre = tuple(op.dagger() for op in reversed(mi.annihilators))
        left_ann = tuple(op.dagger() for op in reversed(mi.creators))

        for tj in b:
            if abs(tj.coeff) <= eps:
                continue

            mj = tj.monomial
            word_terms = ket_from_word(
                ops=(
                    *left_cre,
                    *left_ann,
                    *mj.creators,
                    *mj.annihilators,
                ),
                eps=eps,
            )
            total += ti.coeff.conjugate() * tj.coeff * identity_coeff(word_terms)

    return total
