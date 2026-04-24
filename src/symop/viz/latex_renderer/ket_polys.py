"""LaTeX rendering for polynomial ket operators.

This module provides a LaTeX dispatcher implementation for
:class:`KetPoly`. A ket polynomial is rendered as a linear combination
of monomials, each multiplied by its associated coefficient.

The renderer is purely compositional: individual monomials are rendered
via the global ``latex`` dispatcher, and coefficients are applied using
shared formatting utilities.
"""

from __future__ import annotations

from typing import Any

from symop.ccr.algebra.ket.poly import KetPoly
from symop.viz._dispatch import latex
from symop.viz.latex_renderer._latex_utils import (
    apply_coeff,
    join_signed,
    latex_config_from_kwargs,
)


@latex.register(KetPoly)
def _latex_ket_poly(obj: KetPoly, /, **kwargs: Any) -> str:
    r"""Render a :class:`KetPoly` as a LaTeX string.

    The ket polynomial is expressed as a sum of monomials with complex
    coefficients. Each monomial is rendered via the LaTeX dispatcher and
    combined into a signed expression.

    Parameters
    ----------
    obj:
        Ket polynomial to render.
    **kwargs:
        Additional keyword arguments forwarded to the LaTeX rendering
        pipeline. These may include formatting options such as
        ``decimals`` and ``eps``.

    Returns
    -------
    str
        LaTeX representation of the ket polynomial.

    Notes
    -----
    - Terms with coefficients below the configured threshold ``eps``
      are omitted.
    - If a monomial renders to an empty string, it is interpreted as the
      identity operator and rendered as ``\\mathbb{I}``.
    - If no terms remain after filtering, the result is ``"0"``.

    """
    cfg = latex_config_from_kwargs(kwargs)

    if len(obj.terms) == 0:
        return "0"

    parts: list[str] = []
    for t in obj.terms:
        c = complex(t.coeff)
        if abs(c) <= cfg.eps:
            continue
        m = latex(t.monomial, **kwargs)
        parts.append(apply_coeff(c, m, decimals=cfg.decimals, empty_body=r"\mathbb{I}"))

    return join_signed(parts) if parts else "0"
