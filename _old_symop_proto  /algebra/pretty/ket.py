from __future__ import annotations

from symop_proto.core.protocols import KetTermProto

from .monomial import (
    collect_mode_order,
    monomial_to_str,
)


def ket_repr(terms: tuple[KetTermProto, ...], is_state: bool = False) -> str:
    if not terms:
        return "KetPoly(0)"
    mode_index = collect_mode_order(terms)
    parts: list[str] = []
    for t in terms:
        mon_str = monomial_to_str(t.monomial, mode_index)
        parts.append(f"({t.coeff:+.3g})·{mon_str}")
    if is_state:
        return " + ".join(parts) + "·|0⟩"
    return " + ".join(parts)


def ket_latex(
    terms: tuple[KetTermProto, ...],
    is_state: bool = False,
    show_identity: bool = True,
) -> str:
    """Render a sum of ket-terms to LaTeX.

    Args:
        terms: Polynomial terms.
        is_state: If False and `terms` is empty, return "$0$".
        show_identity: Forwarded to `ketterm_latex`.

    Returns:
        LaTeX string.

    """
    from symop_proto.core.pretty.terms import ketterm_latex

    if not terms:
        if not is_state:
            return r""

    pieces: list[str] = [ketterm_latex(t, show_identity=show_identity) for t in terms]
    # Keep the same join/cleanup semantics as before.
    return r" \;+\; ".join(pieces).replace("+ -", "- ")
