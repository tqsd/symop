from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
from symop_proto.core.terms import KetTerm
from symop_proto.algebra.common.signatures import sig_mono


def combine_like_terms_ket(
    terms: Iterable[KetTerm],
    eps: float = 1e-12,
    *,
    approx: bool = False,
    **env_kw,
) -> Tuple[KetTerm, ...]:
    """Combine like terms in a list of KetTerms

    This function groups together terms that share the same monomial (up to an
    approximate equivalence if ``approx=True``) and sums their complex
    cofficients. Terms whose total coefficient magnitude falls below ``eps``
    are discarded. The result is sorted by each term's nominal signature.

    Args:
        terms: List of terms to combine.
        eps: Minimum value of ket term coefficient to keep a term.
        approx: If ``True`` use appproximate combining otherwise strict
            combining.
        **env_kw: Additional keyword arguments forwarded to :func:`sig_mono`
            for an modifying keyword arguments in the envelope signature
            constructiona.

    Returns:
        Combined and sorted ket terms.

    Complexity:
        O(N+K log K), where N is the number of given ket terms and K is
        the number of unique monomial signatures (depending on the
        approximation).

    """
    acc_c: Dict[tuple, complex] = {}
    acc_m = {}
    for t in terms:
        k = sig_mono(t.monomial, approx=approx, **env_kw)
        acc_c[k] = acc_c.get(k, 0j) + t.coeff
        acc_m.setdefault(k, t.monomial)
    out: List[KetTerm] = []
    for k, c in acc_c.items():
        if abs(c.real) < eps and abs(c.imag) < eps:
            continue
        out.append(KetTerm(c, acc_m[k]))
    out.sort(key=lambda t: t.monomial.signature)
    return tuple(out)
