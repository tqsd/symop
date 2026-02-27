r"""Identity-coefficient extraction for CCR ket expansions.

This module provides helpers to extract the scalar (identity) component
from a normally ordered ket-term expansion.

Given

.. math::

    \lvert \psi \rangle \;\sim\; \sum_k c_k\, M_k,

the identity coefficient corresponds to the term where

.. math::

    M_k = \mathbb{I},

i.e. the monomial has no creation or annihilation operators.

This scalar term typically arises from commutator contractions in
normal ordering and represents the c-number component of the expansion.
"""

from __future__ import annotations

from symop.core.protocols import KetTermProto


def identity_coeff(terms: tuple[KetTermProto, ...]) -> complex:
    r"""Return the coefficient of the identity monomial.

    Parameters
    ----------
    terms :
        Tuple of ket terms forming a normally ordered expansion.

    Returns
    -------
    complex
        The coefficient associated with the identity monomial
        (no creators and no annihilators). Returns ``0.0 + 0.0j`` if
        no identity term is present.

    Notes
    -----
    - The identity term represents the scalar contribution of an
      operator expansion after commutator contractions.
    - If multiple identity terms exist (which should not occur after
      canonicalization), the first match is returned.

    """
    for t in terms:
        if t.monomial.is_identity:
            return t.coeff
    return 0.0 + 0.0j
