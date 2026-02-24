from __future__ import annotations
from typing import Dict, Tuple, List, Iterable
from collections import Counter

from symop_proto.core.pretty.ladder import ladder_latex, ladder_text
from symop_proto.core.protocols import MonomialProto

IDENTITY_LATEX = r"\mathbb{I}"
IDENTITY_TEXT = "I"


def collect_mode_order(monomials: Iterable[MonomialProto]) -> Dict[Tuple, int]:
    order: Dict[Tuple, int] = {}

    next_idx = 1
    for m in monomials:
        for op in (*m.creators, *m.annihilators):
            sig = op.mode.signature
            if sig not in order:
                order[sig] = next_idx
                next_idx += 1
    return order


def _pow_group_text(s: str, n: int) -> str:
    return s if n == 1 else f"({s})^{n}"


def _pow_group_latex(s: str, n: int) -> str:
    return s if n == 1 else rf"\left({s}\right)^{{{n}}}"


def monomial_to_str(
    m: MonomialProto,
    mode_index: Dict[Tuple, int],
    *,
    show_identity: bool = True,
    base: str = "m",
) -> str:
    # count by mode signature and pick a representative op for each signature
    c_count = Counter(op.mode.signature for op in m.creators)
    a_count = Counter(op.mode.signature for op in m.annihilators)

    def by_idx(sig):
        return mode_index[sig]

    c_parts: List[str] = []
    for sig in sorted(c_count, key=by_idx):
        n = c_count[sig]
        rep = next(op for op in m.creators if op.mode.signature == sig)  # creation op
        sym = ladder_text(rep, base=base)
        c_parts.append(_pow_group_text(sym, n))

    a_parts: List[str] = []
    for sig in sorted(a_count, key=by_idx):
        n = a_count[sig]
        rep = next(
            op for op in m.annihilators if op.mode.signature == sig
        )  # annihilation op
        sym = ladder_text(rep, base=base)
        a_parts.append(_pow_group_text(sym, n))

    if not c_parts and not a_parts:
        return IDENTITY_TEXT if show_identity else ""

    if c_parts and a_parts:
        return f"{' '.join(c_parts)} {' '.join(a_parts)}"
    return " ".join(c_parts) or " ".join(a_parts)


def monomial_to_latex(
    m: MonomialProto,
    mode_index: Dict[Tuple, int],
    *,
    show_identity: bool = True,
    base: str = "a",
) -> str:
    c_count = Counter(op.mode.signature for op in m.creators)
    a_count = Counter(op.mode.signature for op in m.annihilators)

    def by_idx(sig):
        return mode_index[sig]

    c_parts: List[str] = []
    for sig in sorted(c_count, key=by_idx):
        n = c_count[sig]
        rep = next(op for op in m.creators if op.mode.signature == sig)
        sym = ladder_latex(rep, base=base)
        c_parts.append(_pow_group_latex(sym, n))

    a_parts: List[str] = []
    for sig in sorted(a_count, key=by_idx):
        n = a_count[sig]
        rep = next(op for op in m.annihilators if op.mode.signature == sig)
        sym = ladder_latex(rep, base=base)
        a_parts.append(_pow_group_latex(sym, n))

    if not c_parts and not a_parts:
        return IDENTITY_LATEX if show_identity else ""

    if c_parts and a_parts:
        # small spacing between creation and annihilation blocks
        return rf"{' '.join(c_parts)} \;{' '.join(a_parts)}"
    return " ".join(c_parts) or " ".join(a_parts)
