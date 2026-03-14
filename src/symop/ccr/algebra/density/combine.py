r"""Combine like terms in symbolic density-operator polynomials.

A density polynomial in this package is represented as a finite linear
combination of outer products of normally-ordered monomials,

.. math::

    \rho \sim \sum_k c_k\, L_k\, R_k,

where each term stores a complex coefficient ``c_k`` and two monomials
``(left=L_k, right=R_k)``. Two terms are considered "like" if their left and
right monomials match under either an exact signature or an approximate
signature (for example after rounding envelope-dependent quantities).

This module provides a canonicalization step that groups like terms, sums their
coefficients, drops near-zero groups, and returns a deterministically sorted
tuple of concrete density terms.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.ccr.common.signatures import sig_density
from symop.core.protocols.ops import Monomial as MonomialProtocol
from symop.core.terms import DensityTerm
from symop.core.types.signature import Signature


def combine_like_terms_density(
    terms: Iterable[DensityTerm],
    eps: float = 1e-12,
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> tuple[DensityTerm, ...]:
    r"""Combine like density terms by summing coefficients and dropping zeros.

    This function groups density terms by a (left, right) signature key and
    sums complex coefficients within each group. Groups whose total coefficient
    magnitude is below ``eps`` are discarded. The output is returned as a tuple
    of concrete :class:`~symop.core.terms.DensityTerm` instances
    sorted deterministically by the exact monomial signatures.

    Grouping is driven by :func:`~symop.ccr.algebra.common.signatures.sig_density`.
    When ``approx=True``, the key may be constructed from approximate monomial
    signatures (for example by rounding envelope-dependent quantities).

    Parameters
    ----------
    terms:
        Iterable of density terms to combine.
    eps:
        Threshold for dropping groups with near-zero total coefficient.
        A group is dropped if ``abs(c_total) < eps``.
    approx:
        If True, use approximate signatures for grouping.
    decimals:
        Number of decimal digits used by approximate signatures (if applicable).
    ignore_global_phase:
        Whether approximate signatures may ignore global phase (if applicable).

    Returns
    -------
    tuple[DensityTerm, ...]
        A tuple of merged density terms with like terms combined, near-zero
        groups removed, and deterministic ordering.

    Notes
    -----
    This function is purely algebraic: it does not evaluate any operator
    matrices. It only merges terms that are identical under the chosen
    signature scheme.

    Complexity
    ----------
    Let ``N`` be the number of input terms and ``K`` the number of unique
    signature keys. The runtime is :math:`O(N + K \log K)` due to one-pass
    accumulation and a final sort over the ``K`` merged terms.

    """
    acc_coeff: dict[Signature, complex] = {}
    acc_rep: dict[Signature, tuple[MonomialProtocol, MonomialProtocol]] = {}

    for t in terms:
        key = sig_density(
            t,
            approx=approx,
            decimals=decimals,
            ignore_global_phase=ignore_global_phase,
        )
        acc_coeff[key] = acc_coeff.get(key, 0j) + t.coeff
        acc_rep.setdefault(key, (t.left, t.right))

    out: list[DensityTerm] = []
    for key, c in acc_coeff.items():
        if abs(c) < eps:
            continue
        left, right = acc_rep[key]
        out.append(DensityTerm(coeff=c, left=left, right=right))

    out.sort(key=lambda t: (t.left.signature, t.right.signature))
    return tuple(out)
