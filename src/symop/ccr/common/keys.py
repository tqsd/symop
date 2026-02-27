"""Key helpers for CCR algebra canonicalization.

This module defines small, strict helpers that produce hashable *keys* for
grouping and sorting CCR objects in algorithms such as "combine like terms".

Keys vs signatures
------------------

Objects in the CCR layer expose:

- ``signature``: exact structural identity.
- ``approx_signature(decimals=..., ignore_global_phase=...)``: identity under an
  approximation scheme based on rounding and optional phase normalization.

A *key* is the value used by algorithms to bucket or sort objects. In most
cases, the key is simply the object's signature (exact or approximate), but
this module centralizes the decision logic to keep the CCR layer consihstent.

Strict keyword convention
-------------------------

This project standardizes approximate signatures to accept only:

- ``decimals: int``
- ``ignore_global_phase: bool``

Accordingly, the helpers in this module do not accept ``**kwargs`` and forward
only these parameters.

Notes
-----
This module should remain dependency-light and must not import concrete CCR
polynomial implementations. It is intended to sit near the bottom of the CCR
dependency tree.

"""

from __future__ import annotations

from collections.abc import Iterable

from symop.core.protocols import HasSignature, SignatureProto


def sig_obj(
    obj: HasSignature,
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> SignatureProto:
    """Return an exact or approximate key for a signature-bearing object.

    Parameters
    ----------
    obj:
        Any object that provides ``signature`` and
        ``approx_signature(decimals=..., ignore_global_phase=...)``.
    approx:
        If ``True``, return ``obj.approx_signature(...)``.
        If ``False``, return ``obj.signature``.
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
        return obj.signature
    return obj.approx_signature(
        decimals=decimals, ignore_global_phase=ignore_global_phase
    )


def sig_lop(
    op: HasSignature,
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> SignatureProto:
    """Return an exact or approximate key for a ladder operator.

    Parameters
    ----------
    op:
        A ladder-operator-like object with ``signature`` and
        ``approx_signature(decimals=..., ignore_global_phase=...)``.
    approx:
        If ``True``, return ``op.approx_signature(...)``.
        If ``False``, return ``op.signature``.
    decimals:
        Number of decimals to round to (forwarded to ``approx_signature``).
    ignore_global_phase:
        If ``True``, treat global phase parameters as zero for grouping
        (forwarded to ``approx_signature``).

    Returns
    -------
    SignatureProto
        Hashable ladder-operator key.

    Notes
    -----
    This function exists mostly for readability. It is equivalent to
    :func:`sig_obj` but communicates intent at call sites.

    """
    return sig_obj(
        op,
        approx=approx,
        decimals=decimals,
        ignore_global_phase=ignore_global_phase,
    )


def sig_word(
    ops: Iterable[HasSignature],
    *,
    approx: bool = False,
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> tuple[SignatureProto, ...]:
    """Return a key for an operator word (ordered product of ladder operators).

    Parameters
    ----------
    ops:
        Iterable of ladder operators defining a word. The iterable is
        materialized into a tuple to make the key stable and hashable.
    approx:
        If ``True``, each operator key is taken from ``approx_signature(...)``.
        If ``False``, each operator key is taken from ``signature``.
    decimals:
        Number of decimals to round to (forwarded to each ``approx_signature``).
    ignore_global_phase:
        If ``True``, treat global phase parameters as zero for grouping
        (forwarded to each ``approx_signature``).

    Returns
    -------
    tuple[SignatureProto, ...]
        Hashable key representing the ordered operator word.

    """
    return tuple(
        sig_lop(
            op,
            approx=approx,
            decimals=decimals,
            ignore_global_phase=ignore_global_phase,
        )
        for op in ops
    )
