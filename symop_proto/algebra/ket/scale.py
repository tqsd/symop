from __future__ import annotations
from dataclasses import replace
from typing import Tuple
from symop_proto.core.terms import KetTerm


def ket_scale(terms: Tuple[KetTerm, ...], c: complex) -> Tuple[KetTerm, ...]:
    """Scale a tuple of ket terms by a complex coefficient

    Multiplies the coefficient of each :class:`KetTerm` given by the
    complex scalar ``c`` and returns a new tuple with the updated terms.
    The operation perserves the monomials and all other attributes of
    each terms.

    Args:
        terms: Terms to be scaled.
        c: Complex scalar for scaling.

    Returns:
        New tuple of scaled terms.

    Notes:
        - the function uses :func:`dataclasses.replace` to preserve the
          immutability of :class:`KetTerm` instances.
        - This operation corresponds to the scalar multiplication
          :math:`c \\cdot |\\psi\\rangle`.

    """
    return tuple(replace(t, coeff=c * t.coeff) for t in terms)
