from __future__ import annotations
from typing import Dict, Tuple

from symop_proto.core.protocols import DensityTermProto, KetTermProto
from symop_proto.algebra.pretty.monomial import (
    monomial_to_str,
    monomial_to_latex,
    collect_mode_order,
)
from symop_proto.core.pretty.scalars import (
    complex_to_text,
    complex_to_latex,
)


# ----- KetTerm c·M -----


def ketterm_to_str(term: KetTermProto, mode_index: Dict[Tuple, int]) -> str:
    mon = monomial_to_str(term.monomial, mode_index)
    return f"({complex_to_text(term.coeff)})·{mon}"


def ketterm_to_latex(term: KetTermProto, mode_index: Dict[Tuple, int]) -> str:
    mon = monomial_to_latex(term.monomial, mode_index)  # bare
    c = complex_to_latex(term.coeff)
    # parenthesize multi-term complex coeffs like a±bi (handled inside the latex helper if you want)
    needs_paren = ("+" in c[1:] or "-" in c[1:]) and not c.startswith("-")
    c_tex = rf"\left({c}\right)" if needs_paren else c
    return rf"{c_tex}\,\cdot\,{mon}"  # bare LaTeX


# ----- DensityTerm c·L·R^\dagger -----


class _AdjointView:
    def __init__(self, creators, annihilators):
        self.creators = tuple(creators)
        self.annihilators = tuple(annihilators)


def _right_dagger_view(dt: DensityTermProto) -> _AdjointView:
    rc = tuple(op.dagger() for op in dt.right.annihilators)
    ra = tuple(op.dagger() for op in dt.right.creators)
    return _AdjointView(creators=rc, annihilators=ra)


def densityterm_to_str(
    term: DensityTermProto, mode_index: Dict[Tuple, int]
) -> str:
    left = monomial_to_str(term.left, mode_index)
    right_dag = monomial_to_str(_right_dagger_view(term), mode_index)
    return f"({complex_to_text(term.coeff)})·{left} {right_dag}"


def densityterm_to_latex(
    term: DensityTermProto, mode_index: Dict[Tuple, int]
) -> str:
    left = monomial_to_latex(term.left, mode_index)
    right_dag = monomial_to_latex(_right_dagger_view(term), mode_index)
    c = complex_to_latex(term.coeff)
    needs_paren = ("+" in c[1:] or "-" in c[1:]) and not c.startswith("-")
    c_tex = rf"\left({c}\right)" if needs_paren else c
    return rf"{c_tex}\,\cdot\,{left}\,\cdot\,{right_dag}"  # bare LaTeX


# ----- Standalone helpers (small per-term index) -----


def ketterm_text(term: KetTermProto) -> str:
    idx = collect_mode_order([term.monomial])
    return ketterm_to_str(term, idx)


def ketterm_latex(term: KetTermProto) -> str:
    idx = collect_mode_order([term.monomial])
    return ketterm_to_latex(term, idx)


def densityterm_text(term: DensityTermProto) -> str:
    idx = collect_mode_order([term.left, term.right])
    return densityterm_to_str(term, idx)


def densityterm_latex(term: DensityTermProto) -> str:
    idx = collect_mode_order([term.left, term.right])
    return densityterm_to_latex(term, idx)
