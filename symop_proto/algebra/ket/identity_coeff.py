from __future__ import annotations
from typing import Tuple

from symop_proto.core.protocols import KetTermProto


def identity_coeff(terms: Tuple[KetTermProto, ...]) -> complex:
    """Extract the coefficient of the identity term from a list of ket terms

    Searches the given tuple of :class:`KetTerm` instances for the
    monomial corresponding to the identity operator - a term without a
    creation and annihilation operators, and returns its complex
    coefficient.

    Args:
        terms: Tuple of terms to search.

    Returns:
        The coefficient associated with the identity monomial. Returns
        ``0+0j`` if no such term is present.

    Notes:
        - The identity term represents the scalar (commutator-contracted)
          component of the operator expansion.
        - Useful for detecting constant offsets, such as those arisin from
          bosonig commutation relations.
    """
    for t in terms:
        if t.monomial.is_identity:
            return t.coeff
    return 0.0 + 0.0j
