from __future__ import annotations

from collections.abc import Iterable

from symop_proto.algebra.common.signatures import sig_density
from symop_proto.core.protocols import DensityTermProto


def combine_like_terms_density(
    terms: Iterable[DensityTermProto],
    eps: float = 1e-12,
    *,
    approx: bool = False,
    **env_kw,
) -> tuple[DensityTermProto, ...]:
    """Combine like density terms by summing coefficients and dropping zeros.

    This function groups :class:`DensityTerms` instances by their
    (left, right) monomial signaturs-exactly or approximately and sums the
    complex coefficients within each group. Groups whose total coefficent
    magnitude falls below ``eps`` are discarded. The result is returned as
    a tuple of concrete :class:`DensityTerm` objects sorterd by action
    and signature.

    Grouping is driven by :func:`sig_density`, which is called with ``approx``
    and ``**env_key`` if provided. This allows apporximate merging.

    Args:
        terms: Iterable of density terms to combine.
        eps: Minimum absolute coefficient treshold
        approx: if ``True`` use approximate sinatures
        **env_kw: Extra keyword arguments forwarded to :func:`sig_density`

    Returns:
        A tuple of merged :class:`DensityTerm` instances with like terms
        combined,very small groups dropped, and deterministic sort by signature

    Complexity:
        Let N be the number of input terms and K the number of unique
        signature keys. The runtime is O(N+K log K); O(N) for the single
        pass accumulation and O(K log K) for the final sort.

    """
    from symop_proto.core.terms import DensityTerm

    acc_c: dict[tuple, complex] = {}
    acc_rep: dict[tuple, tuple] = {}
    for t in terms:
        k = sig_density(t, approx=approx, **env_kw)
        acc_c[k] = acc_c.get(k, 0j) + t.coeff
        acc_rep.setdefault(k, (t.left, t.right))
    out: list[DensityTerm] = []
    for k, c in acc_c.items():
        if abs(c.real) < eps and abs(c.imag) < eps:
            continue
        L, R = acc_rep[k]
        out.append(DensityTerm(c, L, R))
    out.sort(key=lambda t: (t.left.signature, t.right.signature))
    return tuple(out)
