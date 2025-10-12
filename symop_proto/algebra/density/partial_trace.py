from __future__ import annotations
from typing import Iterable, Tuple, Set, Any, Dict

from symop_proto.core.protocols import (
    DensityTermProto,
    LadderOpProto,
    MonomialProto,
)

from symop_proto.algebra.ket.inner import ket_inner
from symop_proto.algebra.density.combine import combine_like_terms_density


def _to_mode_signatures(trace_over_modes: Iterable[object]) -> Set[Any]:
    r"""Normalize a list of modes to a set of ``ModeOp.signature`` keys.

    Accepts a heterogeneous iterable containing any of:

    - a ``ModeOp`` instance,
    - a ``LadderOp`` instance (its ``mode.signature`` is used),
    - a ``Monomial`` (all of its ``mode_ops`` signatures are added),
    - a pre-constructed signature tuple.

    Returns a set of signature-tuples suitable for membership checks.
    """

    sigs: Set[Any] = set()
    for obj in trace_over_modes:
        if hasattr(obj, "signature"):
            sigs.add(getattr(obj, "signature"))
        elif isinstance(obj, LadderOpProto):
            sigs.add(obj.mode.signature)
        elif isinstance(obj, MonomialProto):
            for m in obj.mode_ops:
                sigs.add(m.signature)
        else:
            # assume user passed a signature-like object already
            sigs.add(obj)
    return sigs


def _split_monomial_by_modes(
    m: MonomialProto, trace_sigs: Set[Any]
) -> tuple[MonomialProto, MonomialProto]:
    r"""Partition a monomial into kept vs traced parts.

    Operators whose ``op.mode.signature`` is **not** in ``trace_sigs`` go
    to the kept monomial; those whose signatures are in ``trace_sigs`` go
    to the traced monomial. The return value is ``(kept, traced)``.
    """
    from symop_proto.core.monomial import Monomial

    cre_kept = tuple(
        op for op in m.creators if op.mode.signature not in trace_sigs
    )
    ann_kept = tuple(
        op for op in m.annihilators if op.mode.signature not in trace_sigs
    )
    cre_traced = tuple(
        op for op in m.creators if op.mode.signature in trace_sigs
    )
    ann_traced = tuple(
        op for op in m.annihilators if op.mode.signature in trace_sigs
    )
    return Monomial(cre_kept, ann_kept), Monomial(cre_traced, ann_traced)


def density_partial_trace(
    terms: Tuple[DensityTermProto, ...],
    trace_over_modes: Iterable[object],
) -> Tuple[DensityTermProto, ...]:
    r"""Partial trace over a subset of modes for a symbolic density polynomial.

    Let the Hilbert space factor as :math:`\mathcal{H} = \mathcal{H}_K \otimes
    \mathcal{H}_T`, where :math:`\mathcal{H}_T` is the subsystem to be traced
    out. Each density term is :math:`c \, |L\rangle \langle R|`. We split the
    left/right monomials into *kept* (:math:`K`) and *traced* (:math:`T`)
    parts:

    .. math::

        L \;\mapsto\; (L^K, L^T),
        \qquad
        R \;\mapsto\; (R^K, R^T).

    The partial trace over :math:`\mathcal{H}_T` contracts the traced pieces:

    .. math::

        \mathrm{Tr}_T\!\big(c \, |L\rangle\langle R|\big)
        \;=\;
        \big(c \, \langle R^T \mid L^T \rangle\big)
        \; |L^K\rangle\langle R^K|,

    where the overlap :math:`\langle R^T \mid L^T \rangle` is evaluated
    symbolically via :func:`ket_inner` on one-term kets built from the traced
    monomials. Terms with zero contraction are dropped, and identical
    :math:`|L^K\rangle\langle R^K|` terms are merged using
    :func:`combine_like_terms_density`.

    Parameters
    ----------
    terms : Tuple[DensityTermProto, ...]
        Input density polynomial.
    trace_over_modes : Iterable[object]
        Specification of the modes to trace out. May contain ``ModeOp``,
        ``LadderOp``, ``Monomial``, or raw signature tuples (or a mix).

    Returns
    -------
    Tuple[DensityTermProto, ...]
        The resulting density polynomial on :math:`\mathcal{H}_K` with traced
        modes removed and coefficients contracted.

    Notes
    -----
    - The computation is purely symbolic (normal ordering and commutator
      contractions); no matrix representations are used.
    - If a term has no support on the traced subsystem, its contraction factor
      is :math:`\langle R^T \mid L^T \rangle = 1` and it passes through intact.
    - If the traced parts are incompatible (orthogonal), the contraction is
      zero and the term is removed.
    """

    from symop_proto.core.terms import DensityTerm, KetTerm

    trace_sigs = _to_mode_signatures(trace_over_modes)
    out: list[DensityTermProto] = []
    for dt in terms:
        # split left/right monomials into kept/traced
        Lk, Lt = _split_monomial_by_modes(dt.left, trace_sigs)
        Rk, Rt = _split_monomial_by_modes(dt.right, trace_sigs)

        # contraction over traced subsystem: <R^T | L^T>
        # build 1-term kets with coeff=1 to avoid double-counting dt.coeff
        Lt_ket = (KetTerm(1.0, Lt),)
        Rt_ket = (KetTerm(1.0, Rt),)
        contraction = ket_inner(Rt_ket, Lt_ket)

        if contraction == 0:
            continue

        new_coeff = dt.coeff * contraction
        if new_coeff == 0:
            continue

        out.append(DensityTerm(new_coeff, Lk, Rk))

    # Merge identical |L^K><R^K| after tracing
    return combine_like_terms_density(tuple(out))
