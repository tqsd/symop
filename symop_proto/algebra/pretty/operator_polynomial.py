from __future__ import annotations
from typing import Dict, Iterable, Tuple
from symop_proto.core.protocols import LadderOpProto
from symop_proto.core.monomial import Monomial
from symop_proto.algebra.pretty.monomial import (
    monomial_to_str,
    monomial_to_latex,
    collect_mode_order,
)
from symop_proto.core.pretty.scalars import (
    complex_to_text,
    complex_to_latex,
)


def _ops_as_monomials(ops: Tuple[LadderOpProto, ...]) -> Tuple[Monomial, ...]:
    out = []
    for op in ops:
        if getattr(op, "is_creation", False):
            out.append(Monomial(creators=(op,)))
        else:
            out.append(Monomial(annihilators=(op,)))
    return tuple(out)


def _word_text(
    ops: Tuple[LadderOpProto, ...], mode_index: Dict[Tuple, int]
) -> str:
    if not ops:
        return "I"
    parts = [monomial_to_str(m, mode_index) for m in _ops_as_monomials(ops)]
    return " ".join(parts)


def _word_latex(
    ops: Tuple[LadderOpProto, ...], mode_index: Dict[Tuple, int]
) -> str:
    if not ops:
        return r"\mathbb{I}"
    parts = [monomial_to_latex(m, mode_index) for m in _ops_as_monomials(ops)]
    return r"\,".join(parts)


def opterm_to_str(
    ops: Tuple[LadderOpProto, ...],
    coeff: complex,
    mode_index: Dict[Tuple, int],
    *,
    mul: str = " ",
) -> str:
    c = complex_to_text(coeff)
    word = _word_text(ops, mode_index)
    return f"({c}){mul}{word}"


def opterm_to_latex(
    ops: Tuple[LadderOpProto, ...],
    coeff: complex,
    mode_index: Dict[Tuple, int],
) -> str:
    c = complex_to_latex(coeff)
    needs_paren = ("+" in c[1:] or "-" in c[1:]) and not c.startswith("-")
    c_tex = rf"\left({c}\right)" if needs_paren else c
    word = _word_latex(ops, mode_index)
    return rf"{c_tex}\!\cdot\!{word}"


def oppoly_to_str(
    terms: Iterable[Tuple[Tuple[LadderOpProto, ...], complex]],
    *,
    mul: str = " ",
) -> str:
    monos = []
    for ops, _ in terms:
        monos += list(_ops_as_monomials(ops))
    idx = collect_mode_order(monos)
    return " + ".join(opterm_to_str(ops, c, idx, mul=mul) for ops, c in terms)


def oppoly_to_latex(
    terms: Iterable[Tuple[Tuple[LadderOpProto, ...], complex]],
) -> str:
    monos = []
    for ops, _ in terms:
        monos += list(_ops_as_monomials(ops))
    idx = collect_mode_order(monos)
    return " + ".join(opterm_to_latex(ops, c, idx) for ops, c in terms)
