r"""Symbolic overlaps between monomials via normal ordering.

This module provides the scalar overlap

.. math::

    \langle R \,|\, L \rangle,

computed as the identity coefficient of the normal-ordered expansion of
:math:`R^\dagger L`.

No matrix representations are formed; the result is obtained purely from the
symbolic normal-ordering engine.
"""

from __future__ import annotations

from symop.ccr.algebra.ket.from_word import ket_from_word
from symop.ccr.algebra.ket.identity_coeff import identity_coeff
from symop.core.protocols import MonomialProto


def overlap_right_left(R: MonomialProto, L: MonomialProto) -> complex:
    r"""Symbolic overlap :math:`\langle R \,|\, L \rangle` via normal ordering.

    This routine computes the scalar overlap between a "right" monomial
    :math:`R` and a "left" monomial :math:`L` by building an operator word that,
    when normally ordered, yields a linear combination of ket terms. The
    returned value is the coefficient in front of the identity monomial.

    Operationally, we form the operator word

    .. math::

        R^\dagger \, L,

    expand it symbolically with :func:`~symop.ccr.ket.from_word.ket_from_word`, and
    extract the identity coefficient with
    :func:`~symop.ccr.ket.identity_coeff.identity_coeff`. This matches the
    intuition that :math:`\langle R \,|\, L \rangle` is the scalar part of
    :math:`R^\dagger L`.

    Parameters
    ----------
    R:
        Right monomial (appears conjugated as :math:`R^\dagger`).
    L:
        Left monomial.

    Returns
    -------
    complex
        The scalar overlap :math:`\langle R \,|\, L \rangle`, i.e. the identity
        coefficient of the normal-ordered expansion of :math:`R^\dagger L`.

    Notes
    -----
        - The computation is purely symbolic, relying on commutators and normal
          ordering; no matrix representations are used.
        - Orthogonal modes (zero label overlap) lead to vanishing contraction terms
          and thus zero overlap unless the word is already the identity.

    """
    Radj = R.adjoint()
    ops = (
        *Radj.creators,
        *Radj.annihilators,
        *L.creators,
        *L.annihilators,
    )
    terms = ket_from_word(ops=ops)
    return identity_coeff(terms)
