"""LaTeX rendering for core operator types.

This module provides LaTeX dispatcher implementations for low-level
operator objects such as :class:`ModeOp` and :class:`LadderOp`. These
renderers produce compact symbolic representations suitable for use
inside larger expressions (e.g., monomials or polynomials).

Mode subscripts are derived using shared utilities to ensure consistent
label formatting across the rendering pipeline.
"""

from __future__ import annotations

from typing import Any, cast, Callable

from symop.core.operators import (
    LadderOp,
    ModeOp,
)
from symop.viz._dispatch import latex
from symop.viz.latex_renderer._latex_utils import (
    SupportsLatexLabel,
    mode_subscript,
)


def escape_latex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def mode_path_name(mode: SupportsLatexLabel) -> str | None:
    label = getattr(mode, "label", None)
    path = getattr(label, "path", None)
    name = getattr(path, "name", None)
    return str(name) if name else None


def mode_subscript_with_path(
    mode: SupportsLatexLabel,
    *,
    latex_fn: Callable[[Any], str],
    show_path: bool = True,
) -> str:
    sub = mode_subscript(mode, latex_fn=latex_fn)
    path = mode_path_name(mode)

    if not show_path or not path:
        return sub

    path_latex = r"\mathrm{" + escape_latex_text(path) + r"}"

    if sub:
        return sub + r"[" + path_latex + r"]"

    return path_latex

@latex.register(ModeOp)
def _latex_modeop(obj: ModeOp, /, **kwargs: Any) -> str:
    """Render a :class:`ModeOp` as a LaTeX string.

    Parameters
    ----------
    obj:
        Mode operator to render.
    **kwargs:
        Additional keyword arguments accepted for dispatcher compatibility.
        They are currently unused.

    Returns
    -------
    str
        LaTeX representation of the mode operator.

    Notes
    -----
    - If a mode subscript can be resolved, the operator is rendered as
      ``m_{sub}``.
    - Otherwise, a generic ``m`` symbol is returned.

    """
    #sub = mode_subscript(cast(SupportsLatexLabel, obj), latex_fn=latex)
    sub = mode_subscript_with_path(
        cast(SupportsLatexLabel, obj),
        latex_fn=latex,
        show_path=bool(kwargs.get("show_path", True)),
    )
    return (r"m_{" + sub + r"}") if sub else r"m"


@latex.register(LadderOp)
def _latex_ladderop(obj: LadderOp, /, **kwargs: Any) -> str:
    r"""Render a :class:`LadderOp` as a LaTeX string.

    Parameters
    ----------
    obj:
        Ladder operator to render.
    **kwargs:
        Additional keyword arguments accepted for dispatcher compatibility.
        They are currently unused.

    Returns
    -------
    str
        LaTeX representation of the ladder operator.

    Notes
    -----
    - Creation operators (``"adag"``) are rendered as
      ``\\hat{a}^\\dagger`` with an optional subscript.
    - Annihilation operators (``"a"``) are rendered as
      ``\\hat{a}`` with an optional subscript.
    - Other operator kinds fall back to a generic
      ``\\mathrm{...}`` representation.
    - Mode subscripts are derived from ``obj.mode`` using
      :func:`mode_subscript`.

    """
    adjoint_display = bool(kwargs.get("adjoint_display", False))

    #sub = mode_subscript(cast(SupportsLatexLabel, obj.mode), latex_fn=latex)
    sub = mode_subscript_with_path(
        cast(SupportsLatexLabel, obj.mode),
        latex_fn=latex,
        show_path=bool(kwargs.get("show_path", True)),
    )
    kind = getattr(obj.kind, "value", str(obj.kind))

    if adjoint_display:
        if kind == "adag":
            kind = "a"
        elif kind == "a":
            kind = "adag"
    if kind == "adag":
        if sub:
            return rf"\hat{{a}}_{{{sub}}}^\dagger"
        return r"\hat{a}^\dagger"

    if kind == "a":
        if sub:
            return rf"\hat{{a}}_{{{sub}}}"
        return r"\hat{a}"

    if sub:
        return r"\mathrm{" + str(kind) + r"}_{" + sub + r"}"
    return r"\mathrm{" + str(kind) + r"}"
