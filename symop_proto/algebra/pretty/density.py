from __future__ import annotations
from typing import Tuple, List
from symop_proto.core.protocols import DensityTermProto
from .monomial import (
    collect_mode_order,
    monomial_to_str,
    monomial_to_latex,
    complex_to_latex,
)


def density_repr(terms: Tuple[DensityTermProto, ...]) -> str:
    if not terms:
        return "DensityPoly(0)"
    # Build index across left & right appearances
    fake_kets = []
    for t in terms:
        fake_kets.append(type("K", (), {"monomial": t.left}))
        fake_kets.append(type("K", (), {"monomial": t.right}))
    mode_index = collect_mode_order(fake_kets)
    parts: List[str] = []
    for t in terms:
        Ls = monomial_to_str(t.left, mode_index)
        Rs = monomial_to_str(t.right, mode_index)
        parts.append(f"({t.coeff:+.3g})·|{Ls}⟩⟨{Rs}|")
    return " + ".join(parts)


def density_latex(terms: Tuple[DensityTermProto, ...]) -> str:
    if not terms:
        return r"$0$"
    fake_kets = []
    for t in terms:
        fake_kets.append(type("K", (), {"monomial": t.left}))
        fake_kets.append(type("K", (), {"monomial": t.right}))
    mode_index = collect_mode_order(fake_kets)
    parts: List[str] = []
    for t in terms:
        c = complex_to_latex(t.coeff)
        Ls = monomial_to_latex(t.left, mode_index)
        Rs = monomial_to_latex(t.right, mode_index)
        parts.append(rf"{c}\,\lvert {Ls}\rangle\!\langle {Rs}\rvert")
    body = r" \;+\; ".join(parts).replace("+ -", "- ")
    return rf"$\displaystyle {body}$"
