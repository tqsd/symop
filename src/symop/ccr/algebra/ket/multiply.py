r"""Symbolic multiplication of CCR ket expansions.

This module provides a purely symbolic product for ket expansions represented
as finite sums of ket terms.

Given

.. math::

    \lvert a \rangle \;\sim\; \sum_i c_i\, M_i,
    \qquad
    \lvert b \rangle \;\sim\; \sum_j d_j\, N_j,

their product is defined as the algebraic product of the underlying operator
expressions,

.. math::

    \lvert a \rangle \lvert b \rangle \;\sim\;
    \sum_{i,j} c_i d_j \,(M_i N_j),

where each concatenated word :math:`M_i N_j` is expanded into normal order via
the CCR using :func:`symop.ccr.ket.from_word.ket_from_word`.

The resulting terms are then canonicalized by merging identical monomials.
"""

from __future__ import annotations

from symop.core.protocols import KetTermProto
from symop.core.terms import KetTerm

from .combine import combine_like_terms_ket
from .from_word import ket_from_word


def ket_multiply(
    a: tuple[KetTermProto, ...],
    b: tuple[KetTermProto, ...],
    *,
    eps: float = 1e-12,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> tuple[KetTerm, ...]:
    r"""Multiply two symbolic ket expansions.

    The inputs represent formal sums of normally ordered monomials:

    .. math::

        \lvert a \rangle \;\sim\; \sum_i c_i\, M_i,
        \qquad
        \lvert b \rangle \;\sim\; \sum_j d_j\, N_j.

    The product is computed by expanding each concatenated word
    :math:`M_i N_j` into normal order using :func:`ket_from_word` and
    accumulating the resulting terms:

    .. math::

        M_i N_j \;=\; \sum_k e_{ijk}\, P_{ijk}
        \quad\Rightarrow\quad
        \lvert a \rangle \lvert b \rangle \;\sim\;
        \sum_{i,j,k} (c_i d_j e_{ijk})\, P_{ijk}.

    Parameters
    ----------
    a :
        Left factor as a tuple of ket terms.
    b :
        Right factor as a tuple of ket terms.
    eps :
        Threshold used to skip negligible input terms and to discard
        negligible combined terms during canonicalization.
    approx :
        If ``True``, canonicalization merges terms using approximate monomial
        signatures.
    decimals :
        Forwarded to approximate signature selection when ``approx=True``.
    ignore_global_phase :
        Forwarded to approximate signature selection when ``approx=True``.

    Returns
    -------
    tuple[KetTerm, ...]
        Canonicalized product expansion.

    Notes
    -----
    This routine is symbolic. It does not apply the result to a vacuum state.

    """
    out: list[KetTermProto] = []

    for ti in a:
        if abs(ti.coeff) <= eps:
            continue
        mi = ti.monomial

        for tj in b:
            if abs(tj.coeff) <= eps:
                continue
            mj = tj.monomial

            word_terms = ket_from_word(
                ops=(
                    *mi.creators,
                    *mi.annihilators,
                    *mj.creators,
                    *mj.annihilators,
                ),
                eps=eps,
            )

            pref = ti.coeff * tj.coeff
            for tk in word_terms:
                out.append(KetTerm(pref * tk.coeff, tk.monomial))

    return combine_like_terms_ket(
        out,
        eps=eps,
        approx=approx,
        decimals=decimals,
        ignore_global_phase=ignore_global_phase,
    )
