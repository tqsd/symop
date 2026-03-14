r"""Scalar multiplication for CCR ket expansions.

This module provides scalar multiplication of ket expansions represented
as tuples of ket terms.

Given

.. math::

    \lvert \psi \rangle \;\sim\; \sum_i c_i\, M_i,

scalar multiplication by :math:`\lambda \in \mathbb{C}` yields

.. math::

    \lambda \lvert \psi \rangle
    \;\sim\; \sum_i (\lambda c_i)\, M_i.

The monomials are left unchanged; only the coefficients are scaled.
"""

from __future__ import annotations

from symop.core.protocols.terms import KetTerm as KetTermProtocol
from symop.core.terms import KetTerm


def ket_scale(
    terms: tuple[KetTermProtocol, ...],
    c: complex,
) -> tuple[KetTerm, ...]:
    r"""Scale a ket expansion by a complex scalar.

    Parameters
    ----------
    terms :
        Tuple of ket terms representing a symbolic ket expansion.
    c :
        Complex scalar multiplier.

    Returns
    -------
    tuple[KetTerm, ...]
        New tuple where each term coefficient has been multiplied by ``c``.

    Notes
    -----
    - Monomials are preserved exactly.
    - This corresponds to scalar multiplication
      :math:`c \lvert \psi \rangle`.
    - If ``terms`` is empty, an empty tuple is returned.

    """
    return tuple(KetTerm(coeff=c * t.coeff, monomial=t.monomial) for t in terms)
