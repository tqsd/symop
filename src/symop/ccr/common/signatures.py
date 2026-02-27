r"""Signature helpers for CCR algebra canonicalization.

This module provides small helpers that extract *exact* or *approximate*
signatures from algebraic atoms (monomials, ket terms, density terms).

These signatures are intended to be used as stable, hashable keys in
canonicalization routines such as "combine like terms".

Exact vs approximate
--------------------

Many objects in the CCR layer expose two related notions of identity:

- ``signature``: exact structural identity (stable, hashable).
- ``approx_signature(decimals=..., ignore_global_phase=...)``: identity up to an
  approximation scheme based on rounding floating parameters and (optionally)
  ignoring global phase.

This module standardizes the selection logic:

.. math::

    \mathrm{sig}(x) =
    \begin{cases}
        x.\mathrm{signature}, & \text{if approx is False}, \\
        x.\mathrm{approx\_signature}(\ldots), & \text{if approx is True}.
    \end{cases}

Notes
-----
This file is intentionally dependency-light and should remain stable, since it
sits at the bottom of the CCR dependency tree.

"""

from __future__ import annotations

from symop.core.protocols import (
    DensityTermProto,
    KetTermProto,
    MonomialProto,
    SignatureProto,
)


def sig_mono(
    m: MonomialProto,
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> SignatureProto:
    """Return an exact or approximate signature for a monomial.

    Parameters
    ----------
    m:
        A monomial-like object that provides the attribute ``signature`` and the
        method ``approx_signature(decimals=..., ignore_global_phase=...)``.
    approx:
        If ``True``, return ``m.approx_signature(...)``.
        If ``False``, return ``m.signature``.
    decimals:
        Number of decimals to round to (forwarded to ``approx_signature``).
    ignore_global_phase:
        If ``True``, treat global phase parameters as zero for grouping
        (forwarded to ``approx_signature``).

    Returns
    -------
    SignatureProto
        A hashable signature tuple suitable for use as a dictionary key.

    """
    if not approx:
        return m.signature
    return m.approx_signature(
        decimals=decimals, ignore_global_phase=ignore_global_phase
    )


def sig_ket(
    t: KetTermProto,
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> SignatureProto:
    """Return an exact or approximate signature for a ket term.

    Parameters
    ----------
    t:
        A ket-term-like object that provides the attribute ``signature`` and the
        method ``approx_signature(decimals=..., ignore_global_phase=...)``.
    approx:
        If ``True``, return ``t.approx_signature(...)``.
        If ``False``, return ``t.signature``.
    decimals:
        Number of decimals to round to (forwarded to ``approx_signature``).
    ignore_global_phase:
        If ``True``, treat global phase parameters as zero for grouping
        (forwarded to ``approx_signature``).

    Returns
    -------
    SignatureProto
        A hashable signature tuple suitable for use as a dictionary key.

    """
    if not approx:
        return t.signature
    return t.approx_signature(
        decimals=decimals, ignore_global_phase=ignore_global_phase
    )


def sig_density(
    t: DensityTermProto,
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> SignatureProto:
    """Return an exact or approximate signature for a density term.

    Parameters
    ----------
    t:
        A density-term-like object that provides the attribute ``signature`` and
        the method ``approx_signature(decimals=..., ignore_global_phase=...)``.
    approx:
        If ``True``, return ``t.approx_signature(...)``.
        If ``False``, return ``t.signature``.
    decimals:
        Number of decimals to round to (forwarded to ``approx_signature``).
    ignore_global_phase:
        If ``True``, treat global phase parameters as zero for grouping
        (forwarded to ``approx_signature``).

    Returns
    -------
    SignatureProto
        A hashable signature tuple suitable for use as a dictionary key.

    """
    if not approx:
        return t.signature
    return t.approx_signature(
        decimals=decimals, ignore_global_phase=ignore_global_phase
    )
