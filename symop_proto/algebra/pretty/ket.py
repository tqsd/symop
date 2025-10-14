from __future__ import annotations
from typing import Tuple, List
from symop_proto.core.protocols import KetTermProto
from .monomial import (
    collect_mode_order,
    monomial_to_str,
    monomial_to_latex,
    complex_to_latex,
)


def ket_repr(terms: Tuple[KetTermProto, ...], is_state: bool = False) -> str:
    if not terms:
        return "KetPoly(0)"
    mode_index = collect_mode_order(terms)
    parts: List[str] = []
    for t in terms:
        mon_str = monomial_to_str(t.monomial, mode_index)
        parts.append(f"({t.coeff:+.3g})·{mon_str}")
    if is_state:
        return " + ".join(parts) + "·|0⟩"
    return " + ".join(parts)


def ket_latex(terms: Tuple[KetTermProto, ...], is_state: bool = False) -> str:
    if not terms:
        return r"$0$"
    mode_index = collect_mode_order(terms)
    pieces: List[str] = []
    for t in terms:
        mon_lx = monomial_to_latex(t.monomial, mode_index)
        c = complex_to_latex(t.coeff)
        needs_paren = ("+" in c[1:] or "-" in c[1:]) and not c.startswith("-")
        c_tex = rf"\left({c}\right)" if needs_paren else c
        pieces.append(rf"{c_tex}\!\cdot\!{mon_lx}")
    return r" \;+\; ".join(pieces).replace("+ -", "- ")
