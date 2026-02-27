r"""Construct ket terms directly from creators and annihilators.

This module provides a small constructor that builds a normally ordered
monomial from explicitly provided creation and annihilation operators and
wraps it into a single ket term.

Mathematically, the constructor represents

.. math::

    \lvert \psi \rangle \;\sim\; c \,
    \hat a_{i_1}^\dagger \cdots \hat a_{i_m}^\dagger
    \hat a_{j_1} \cdots \hat a_{j_n},

where the monomial is assumed to already be in normal order (all creators to
the left of all annihilators). No commutation or contraction is performed
here; for symbolic normal ordering of arbitrary words, use
:func:`symop.ccr.ket.from_word.ket_from_word`.

The output is passed through :func:`symop.ccr.ket.combine.combine_like_terms_ket`
so callers get a canonical, de-duplicated tuple of terms (even though this
constructor usually produces a single term).
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.core.monomial import Monomial
from symop.core.protocols import LadderOpProto
from symop.core.terms import KetTerm

from .combine import combine_like_terms_ket


def ket_from_ops(
    *,
    creators: Iterable[LadderOpProto] = (),
    annihilators: Iterable[LadderOpProto] = (),
    coeff: complex = 1.0,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> tuple[KetTerm, ...]:
    r"""Construct a (normally ordered) ket term from creators and annihilators.

    This helper builds a single monomial from the provided operator lists:

    .. math::

        M =
        \hat a_{i_1}^\dagger \cdots \hat a_{i_m}^\dagger
        \hat a_{j_1} \cdots \hat a_{j_n},

    and returns the corresponding ket term :math:`c\,M` as a tuple.

    No commutation is performed. The function assumes that the operator
    sequence provided is already in normal order (creators first, then
    annihilators). For rewriting arbitrary words into normal order, use
    :func:`symop.ccr.ket.from_word.ket_from_word`.

    Parameters
    ----------
    creators:
        Iterable of creation operators (each must satisfy ``op.is_creation``).
        The iterable is materialized and the order is preserved.
    annihilators:
        Iterable of annihilation operators (each must satisfy
        ``op.is_annihilation``). The iterable is materialized and the order is
        preserved.
    coeff:
        Scalar coefficient multiplying the resulting monomial.
    approx:
        Forwarded to :func:`combine_like_terms_ket`. If ``True``, like terms are
        merged using approximate signatures.
    decimals:
        Forwarded to approximate signature selection when ``approx=True``.
    ignore_global_phase:
        Forwarded to approximate signature selection when ``approx=True``.

    Returns
    -------
    tuple[KetTerm, ...]
        Canonicalized tuple of ket terms (typically of length 1).

    Raises
    ------
    ValueError
        If any operator in ``creators`` is not a creation operator, or any
        operator in ``annihilators`` is not an annihilation operator.

    """
    creators_t = tuple(creators)
    annihilators_t = tuple(annihilators)

    for op in creators_t:
        if not op.is_creation:
            raise ValueError("creators must be creation operators")

    for op in annihilators_t:
        if not op.is_annihilation:
            raise ValueError("annihilators must be annihilation operators")

    m = Monomial(creators=creators_t, annihilators=annihilators_t)
    return combine_like_terms_ket(
        (KetTerm(coeff, m),),
        approx=approx,
        decimals=decimals,
        ignore_global_phase=ignore_global_phase,
    )
