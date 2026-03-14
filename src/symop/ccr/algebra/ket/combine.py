r"""Canonicalization helpers for CCR ket terms.

This module provides utilities to canonicalize finite sums of ket terms by
merging terms that share the same monomial (exactly or approximately) and
discarding numerically negligible coefficients.

Given a formal sum

.. math::

    \lvert \psi \rangle \;\sim\; \sum_i c_i\, M_i,

where each :math:`M_i` is a normally ordered monomial of ladder operators,
canonicalization groups all terms with identical monomials and replaces them
by a single term with summed coefficient.

Approximate canonicalization
----------------------------

When ``approx=True``, monomials are grouped using their
``approx_signature(decimals=..., ignore_global_phase=...)``. This is intended
for numerically robust merging when mode parameters differ only by small
floating errors and when global phase parameters should be ignored.

This file is logic-only and contains no formatting or pretty-printing.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.ccr.common.signatures import sig_mono
from symop.core.protocols.ops import Monomial
from symop.core.protocols.terms import KetTerm as KetTermProtocol
from symop.core.terms.ket_term import KetTerm
from symop.core.types.signature import Signature


def combine_like_terms_ket(
    terms: Iterable[KetTermProtocol],
    eps: float = 1e-12,
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> tuple[KetTerm, ...]:
    r"""Combine like terms in a ket-term expansion.

    Terms are grouped by monomial identity and their coefficients are summed:

    .. math::

        \sum_{i:\,\mathrm{sig}(M_i)=k} c_i\,M_i
        \;\mapsto\;
        \left(\sum_{i:\,\mathrm{sig}(M_i)=k} c_i\right) M_k.

    Grouping can be either exact (``approx=False``) or approximate
    (``approx=True``) via the monomial's approximate signature.

    Parameters
    ----------
    terms:
        Iterable of ket terms to combine.
    eps:
        Discard threshold. Any combined coefficient with
        :math:`|c| \le \texttt{eps}` is dropped.
    approx:
        If ``True``, group terms using ``approx_signature`` instead of
        ``signature``.
    decimals:
        Rounding precision forwarded to ``approx_signature`` when
        ``approx=True``.
    ignore_global_phase:
        Forwarded to ``approx_signature`` when ``approx=True``.

    Returns
    -------
    tuple[KetTerm, ...]
        Combined terms sorted by monomial exact signature.

    """
    acc_coeff: dict[Signature, complex] = {}
    acc_mono: dict[Signature, Monomial] = {}

    for t in terms:
        k = sig_mono(
            t.monomial,
            approx=approx,
            decimals=decimals,
            ignore_global_phase=ignore_global_phase,
        )
        acc_coeff[k] = acc_coeff.get(k, 0.0j) + t.coeff
        acc_mono.setdefault(k, t.monomial)

    out: list[KetTerm] = []
    for k, c in acc_coeff.items():
        if abs(c) <= eps:
            continue
        out.append(KetTerm(c, acc_mono[k]))

    out.sort(key=lambda t: t.monomial.signature)
    return tuple(out)
