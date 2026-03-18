"""LaTeX rendering for CCR monomials.

This module provides a LaTeX dispatcher implementation for
:class:`Monomial`. A monomial is rendered as an ordered product of
ladder operators, typically in normal order (creators followed by
annihilators).

Fallback rendering is provided for ladder operators that do not have
a registered LaTeX implementation.
"""

from __future__ import annotations

from typing import Any

from symop.core.monomial import Monomial
from symop.viz._dispatch import latex
from symop.viz.latex_renderer._latex_utils import mode_subscript


def _latex_ladder_symbol(op: Any) -> str:
    r"""Render a ladder-like operator as a LaTeX string.

    Parameters
    ----------
    op:
        Operator-like object expected to expose ``kind`` and optionally
        ``mode`` attributes.

    Returns
    -------
    str
        LaTeX representation of the operator.

    Notes
    -----
    - Recognized operator kinds include ``"a"`` (annihilation) and
      ``"adag"`` (creation).
    - If a mode is present, a subscript is generated using
      :func:`mode_subscript`.
    - Unknown operator kinds fall back to a generic
      ``\\mathrm{...}`` representation.

    """
    kind = getattr(op, "kind", None)
    kind_value = getattr(kind, "value", kind)
    kind_s = str(kind_value)

    mode = getattr(op, "mode", None)
    sub = mode_subscript(mode, latex_fn=latex) if mode is not None else ""

    if kind_s == "adag":
        if sub:
            return rf"a_{{{sub}}}^\dagger"
        return r"a^\dagger"

    if kind_s == "a":
        if sub:
            return rf"a_{{{sub}}}"
        return r"a"

    # Fallback for any other operator kinds
    if sub:
        return rf"\mathrm{{{kind_s}}}_{{{sub}}}"
    return rf"\mathrm{{{kind_s}}}"


@latex.register(Monomial)
def _latex_monomial(obj: Monomial, /, **kwargs: Any) -> str:
    r"""Render a :class:`Monomial` as a LaTeX string.

    The monomial is rendered as a product of ladder operators in
    normal order (creation operators followed by annihilation operators).

    Parameters
    ----------
    obj:
        Monomial to render.
    **kwargs:
        Additional keyword arguments forwarded to the LaTeX dispatcher
        for individual operator rendering.

    Returns
    -------
    str
        LaTeX representation of the monomial.

    Notes
    -----
    - The identity monomial is rendered as ``\\mathbb{I}``.
    - Each operator is rendered via the ``latex`` dispatcher when
      possible. If rendering fails, a fallback representation is used.
    - Operators are joined using a thin space separator (``\\,``).

    """
    # Identity
    if obj.is_identity:
        return r"\mathbb{I}"

    parts: list[str] = []

    # creators then annihilators (normal order)
    for op in obj.creators:
        try:
            parts.append(latex(op, **kwargs))
        except Exception:
            parts.append(_latex_ladder_symbol(op))

    for op in obj.annihilators:
        try:
            parts.append(latex(op, **kwargs))
        except Exception:
            parts.append(_latex_ladder_symbol(op))

    parts = [p for p in parts if p]
    if not parts:
        return r"\mathbb{I}"

    return r"\,".join(parts)
