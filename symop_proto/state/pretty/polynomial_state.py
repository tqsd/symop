from typing import Optional, Tuple
from symop_proto.algebra.pretty.density import density_latex as _density_latex
from symop_proto.core.protocols import DensityTermProto, KetTermProto


def _escape_text_for_latex(s: str) -> str:
    return s.replace("_", r"\_")


def state_latex_from_terms(
    terms: Tuple[KetTermProto, ...],
    *,
    label: Optional[str] = None,
    index: Optional[int] = None,
) -> str:
    if label is not None:
        tlabel = _escape_text_for_latex(label)
        lhs = rf"\lvert \psi_{{\text{{{tlabel}}}}}\rangle"
    elif index is not None:
        lhs = rf"\lvert \psi_{{{index}}}\rangle"
    else:
        lhs = r"\lvert \psi \rangle"

    if not terms:
        body = "0"
    else:
        body = " + ".join(t.latex for t in terms)

    return rf"$\displaystyle {lhs} \;=\; {body} \lvert 0 \rangle$"


def density_state_latex_from_terms(
    terms: Tuple[DensityTermProto, ...],
    *,
    label: Optional[str] = None,
    index: Optional[int] = None,
    style: str = "braket",
) -> str:
    if label is not None:
        tlabel = _escape_text_for_latex(label)
        lhs = rf"\rho_{{\text{{{tlabel}}}}}"
    elif index is not None:
        lhs = rf"\rho_{{{index}}}"
    else:
        lhs = r"\rho"

    body = r"0" if not terms else _density_latex(terms, style=style)
    return rf"$\displaystyle {lhs} \;=\; {body}$"
