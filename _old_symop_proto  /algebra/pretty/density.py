from __future__ import annotations
from itertools import chain
from typing import Iterable

from symop_proto.core.pretty.monomial import monomial_to_latex
from symop_proto.core.protocols import DensityTermProto
from symop_proto.algebra.pretty.monomial import collect_mode_order
from symop_proto.core.pretty.terms import (
    densityterm_to_str,
    densityterm_to_latex,
)

from symop_proto.core.pretty.terms import (
    complex_to_latex,
)
from symop_proto.core.pretty.terms import (
    _right_dagger_view,
)


def _shared_index(terms: Iterable[DensityTermProto]):
    monos = list(chain.from_iterable((t.left, t.right) for t in terms))
    return collect_mode_order(monos)


def density_repr(terms: Iterable[DensityTermProto]) -> str:
    terms = tuple(terms)
    if not terms:
        return "0"
    idx = _shared_index(terms)
    return " + ".join(
        densityterm_to_str(
            t,
            idx,
        )
        for t in terms
    )


def density_latex(
    terms: Iterable[DensityTermProto],
    *,
    style: str = "prod",  # "prod" | "braket" | "brackets"
) -> str:
    terms = tuple(terms)
    if not terms:
        return r"0"
    idx = _shared_index(terms)

    if style == "prod":

        def fmt(dt):
            return densityterm_to_latex(
                dt,
                idx,
            )

    elif style == "braket":

        def fmt(dt):
            return _densityterm_to_latex_braket(dt, idx)

    elif style == "brackets":

        def fmt(dt):
            return _densityterm_to_latex_square_brackets(dt, idx)

    else:
        raise ValueError(f"unknown style {style!r}")

    # join with a visible plus; adjust spacing to taste
    return r" + ".join(fmt(t) for t in terms)


# ---- optional alternative term styles ----


def _paren_complex(c_tex: str) -> str:
    needs = ("+" in c_tex[1:] or "-" in c_tex[1:]) and not c_tex.startswith("-")
    return rf"\left({c_tex}\right)" if needs else c_tex


def _densityterm_to_latex_braket(dt: DensityTermProto, idx, *, mul: str = r"\,") -> str:
    left = monomial_to_latex(dt.left, idx)
    right_dag = monomial_to_latex(_right_dagger_view(dt), idx)
    c_tex = _paren_complex(complex_to_latex(dt.coeff))
    return rf"{c_tex}{mul}\ket{{{left}}}\!\bra{{{right_dag}}}"


def _densityterm_to_latex_square_brackets(
    dt: DensityTermProto, idx, *, mul: str = r"\,"
) -> str:
    left = monomial_to_latex(dt.left, idx)
    right_dag = monomial_to_latex(_right_dagger_view(dt), idx)
    c_tex = _paren_complex(complex_to_latex(dt.coeff))
    return rf"{c_tex}{mul}{left}\lvert 0 \rangle \langle 0\rvert {right_dag}"
