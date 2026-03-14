r"""Partial trace for symbolic density polynomials.

Given a density polynomial on a factorized Hilbert space
:math:`\mathcal{H} = \mathcal{H}_K \otimes \mathcal{H}_T`, this module traces
out a subset of modes corresponding to :math:`\mathcal{H}_T`.

Each density term is :math:`c\,|L\rangle\langle R|`. We split the left/right
monomials into kept and traced parts and contract the traced subsystem:

.. math::

    \mathrm{Tr}_T\!\big(c\,|L\rangle\langle R|\big)
    =
    \big(c\,\langle R^T \mid L^T \rangle\big)\,|L^K\rangle\langle R^K|.

The overlap :math:`\langle R^T \mid L^T \rangle` is evaluated symbolically
using the ket inner product.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from symop.ccr.algebra.density.combine import combine_like_terms_density
from symop.ccr.algebra.ket.inner import ket_inner
from symop.core.monomial import Monomial
from symop.core.protocols.base import HasSignature
from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops import (
    Monomial as MonomialProtocol,
)
from symop.core.protocols.terms import DensityTerm as DensityTermProtocol
from symop.core.terms import DensityTerm, KetTerm


def _to_mode_signatures(trace_over_modes: Iterable[object]) -> set[Any]:
    r"""Normalize a list of modes to a set of ``ModeOp.signature`` keys.

    Accepts a heterogeneous iterable containing any of:

    - a ``ModeOp`` instance (its ``signature`` is used),
    - a ``LadderOp`` instance (its ``mode.signature`` is used),
    - a ``Monomial`` (all of its ``mode_ops`` signatures are added),
    - a pre-constructed signature tuple.

    Returns a set of signature-tuples suitable for membership checks.
    """
    sigs: set[Any] = set()
    for obj in trace_over_modes:
        if isinstance(obj, LadderOpProtocol):
            sigs.add(obj.mode.signature)
        elif isinstance(obj, MonomialProtocol):
            for m in obj.mode_ops:
                sigs.add(m.signature)
        elif isinstance(obj, HasSignature):
            sigs.add(obj.signature)
        else:
            sigs.add(obj)
    return sigs


def _split_monomial_by_modes(
    m: MonomialProtocol,
    trace_sigs: set[Any],
) -> tuple[Monomial, Monomial]:
    r"""Partition a monomial into kept vs traced parts.

    Operators whose ``op.mode.signature`` is **not** in ``trace_sigs`` go
    to the kept monomial; those whose signatures are in ``trace_sigs`` go
    to the traced monomial. The return value is ``(kept, traced)``.
    """
    from symop.core.monomial import Monomial

    cre_kept = tuple(op for op in m.creators if op.mode.signature not in trace_sigs)
    ann_kept = tuple(op for op in m.annihilators if op.mode.signature not in trace_sigs)
    cre_traced = tuple(op for op in m.creators if op.mode.signature in trace_sigs)
    ann_traced = tuple(op for op in m.annihilators if op.mode.signature in trace_sigs)

    return (
        Monomial(creators=cre_kept, annihilators=ann_kept),
        Monomial(creators=cre_traced, annihilators=ann_traced),
    )


def density_partial_trace(
    terms: tuple[DensityTermProtocol, ...],
    trace_over_modes: Iterable[object],
) -> tuple[DensityTerm, ...]:
    r"""Partial trace over a subset of modes for a symbolic density polynomial.

    Let the Hilbert space factor as :math:`\mathcal{H} = \mathcal{H}_K \otimes
    \mathcal{H}_T`, where :math:`\mathcal{H}_T` is the subsystem to be traced
    out. Each density term is :math:`c \, |L\rangle \langle R|`. We split the
    left/right monomials into kept/traced parts and contract the traced pieces:

    .. math::

        \mathrm{Tr}_T\!\big(c \, |L\rangle\langle R|\big)
        \;=\;
        \big(c \, \langle R^T \mid L^T \rangle\big)
        \; |L^K\rangle\langle R^K|.

    The overlap :math:`\langle R^T \mid L^T \rangle` is evaluated symbolically
    via :func:`~symop.ccr.ket.inner.ket_inner` on one-term kets built from the
    traced monomials. Terms with zero contraction are dropped, and identical
    :math:`|L^K\rangle\langle R^K|` terms are merged using
    :func:`~symop.ccr.algebra.density.combine.combine_like_terms_density`.

    Parameters
    ----------
    terms:
        Input density polynomial.
    trace_over_modes:
        Specification of the modes to trace out. May contain ``ModeOp``,
        ``LadderOp``, ``Monomial``, or raw signature tuples (or a mix).

    Returns
    -------
    tuple[DensityTerm, ...]
        The resulting density polynomial on :math:`\mathcal{H}_K` with traced
        modes removed and coefficients contracted.

    """
    trace_sigs = _to_mode_signatures(trace_over_modes)
    out: list[DensityTerm] = []

    for dt in terms:
        Lk, Lt = _split_monomial_by_modes(dt.left, trace_sigs)
        Rk, Rt = _split_monomial_by_modes(dt.right, trace_sigs)

        Lt_ket = (KetTerm(coeff=1.0, monomial=Lt),)
        Rt_ket = (KetTerm(coeff=1.0, monomial=Rt),)
        contraction = ket_inner(Rt_ket, Lt_ket)

        if contraction == 0:
            continue

        new_coeff = dt.coeff * contraction
        if new_coeff == 0:
            continue

        out.append(DensityTerm(coeff=new_coeff, left=Lk, right=Rk))

    return combine_like_terms_density(tuple(out))
