r"""Combination of like operator terms.

This module provides normalization utilities for operator polynomials.
In particular, it merges terms that share the same operator word
(up to exact or approximate signatures) and accumulates their coefficients.

Given a collection of operator terms

.. math::

    T_k = c_k \, o_{k,1} o_{k,2} \cdots o_{k,L_k},

terms with identical signatures are combined into a single term with
coefficient equal to the sum of their coefficients.
"""

from __future__ import annotations

from symop.ccr._typing import OpTermFactory, OpTermT
from symop.core.protocols.ops.operators import LadderOp
from symop.core.protocols.terms.op_term import OpTerm
from symop.core.types.signature import Signature


def combine_like_terms(
    terms: tuple[OpTerm, ...],
    *,
    approx: bool = False,
    term_factory: OpTermFactory[OpTermT],
    decimals: int = 12,
    ignore_global_phase: bool = False,
) -> tuple[OpTermT, ...]:
    """Combine operator terms with identical signatures.

    Terms are bucketed either by their exact signature or by their
    approximate signature (when ``approx=True``). Coefficients are
    accumulated per bucket and zero-sum buckets are discarded.

    Parameters
    ----------
    terms:
        Tuple of operator terms to combine.
    approx:
        If ``True``, approximate signatures are used via
        :meth:`OpTermProto.approx_signature`. Otherwise, exact
        :attr:`OpTermProto.signature` is used.
    term_factory:
        Callable used to construct output terms from
        ``(ops, coeff)``.
    decimals:
        How many decimals are used in approx case.
    ignore_global_phase:
        If ``True`` global phase is ignored in approx case.

    Returns
    -------
    tuple[OpTermT, ...]
        Combined operator terms with accumulated coefficients.
        Buckets whose summed coefficient is exactly zero are removed.

    Notes
    -----
    This function does not attempt symbolic simplification beyond
    signature-based bucketing. It assumes that input terms are already
    in a canonical word ordering if such ordering is required.

    """
    if approx:

        def key(t: OpTerm) -> Signature:
            return t.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            )

    else:

        def key(t: OpTerm) -> Signature:
            return t.signature

    buckets: dict[Signature, complex] = {}
    reps: dict[Signature, tuple[LadderOp, ...]] = {}

    for t in terms:
        k = key(t)
        buckets[k] = buckets.get(k, 0.0j) + t.coeff
        reps.setdefault(k, t.ops)

    return tuple(term_factory(reps[k], c) for k, c in buckets.items() if c != 0.0)
