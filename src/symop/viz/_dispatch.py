r"""Dispatch-based visualization utilities.

Defines generic dispatchers for rendering and displaying symbolic objects
in different formats, including text, LaTeX, and plots.

The dispatch system allows type-specific implementations to be registered
externally, enabling extensible visualization across the symop framework.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import singledispatch
from html import escape
from io import BytesIO
from typing import Any


@singledispatch
def plot(obj: Any, /, **kwargs: Any) -> Any:
    raise TypeError(f"No plot() handler registered for type {type(obj)!r}")


@singledispatch
def latex(obj: Any, /, **kwargs: Any) -> str:
    return ""


@singledispatch
def text(obj: Any, /, **kwargs: Any) -> str:
    return repr(obj)


def display(obj: Any, /, **kwargs: Any) -> None | RichDisplay:
    s_latex = latex(obj, **kwargs)
    s_text = text(obj, **kwargs)

    if os.environ.get("SYMOP_DOCS_BUILD") == "1":
        return RichDisplay(latex_src=s_latex, text_src=s_text)

    from symop.viz._optional import try_ipython_display

    ipy = try_ipython_display()
    if ipy is None:
        print(s_text)
        return None

    ipy_display, IPyLatex = ipy
    if s_latex:
        ipy_display(IPyLatex("$" + s_latex + "$"))
    else:
        print(s_text)
    return None


@dataclass(frozen=True)
class RichDisplay:
    latex_src: str
    text_src: str

    def _repr_html_(self) -> str:
        if not self.latex_src:
            return f"<pre>{escape(self.text_src)}</pre>"

        try:
            svg = _latex_to_inline_svg(self.latex_src)
        except Exception as exc:
            return (
                "<pre>"
                "RichDisplay _repr_html_ failed.\n"
                f"latex_src = {escape(self.latex_src)}\n"
                f"error = {escape(str(exc))}"
                "</pre>"
            )

        return f'<div class="symop-display">{svg}</div>'

    def __repr__(self) -> str:
        return self.text_src

    def __str__(self) -> str:
        return self.text_src


@dataclass(frozen=True)
class RichDisplayGroup:
    items: tuple[RichDisplay, ...]

    def _repr_html_(self) -> str:
        return "".join(item._repr_html_() for item in self.items)

    def __repr__(self) -> str:
        return "\n\n".join(item.text_src for item in self.items)

    def __str__(self) -> str:
        return "\n\n".join(item.text_src for item in self.items)


def _to_mathtext(expr: str) -> str:
    replacements = {
        r"\lvert": "|",
        r"\rvert": "|",
        r"\left": "",
        r"\right": "",
        r"\bigl": "",
        r"\bigr": "",
        r"\Bigl": "",
        r"\Bigr": "",
        r"\biggl": "",
        r"\biggr": "",
        r"\Biggl": "",
        r"\Biggr": "",
        r"\middle": "",
    }
    out = expr
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _latex_to_inline_svg(expr: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    mathtext_expr = _to_mathtext(expr)

    fig = Figure(figsize=(0.01, 0.01))
    fig.text(0.0, 0.0, f"${mathtext_expr}$", fontsize=16, color="black")

    buf = BytesIO()
    fig.savefig(
        buf,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
    )
    svg = buf.getvalue().decode("utf-8")
    return _make_svg_theme_aware(svg)


def _make_svg_theme_aware(svg: str) -> str:
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg)

    # Remove figure/axes background patches.
    svg = re.sub(r'<path[^>]*id="patch_[^"]*"[^>]*/>', "", svg)
    svg = re.sub(r'<g id="patch_[^"]*">.*?</g>', "", svg, flags=re.DOTALL)

    if "<svg " in svg:
        svg = svg.replace(
            "<svg ",
            '<svg class="symop-math-svg" style="color: inherit;" ',
            1,
        )

    return svg


def display_many(*objs: Any, **kwargs: Any) -> None | RichDisplayGroup:
    items = tuple(
        RichDisplay(
            latex_src=latex(obj, **kwargs),
            text_src=text(obj, **kwargs),
        )
        for obj in objs
    )

    if os.environ.get("SYMOP_DOCS_BUILD") == "1":
        return RichDisplayGroup(items=items)

    from symop.viz._optional import try_ipython_display

    ipy = try_ipython_display()
    if ipy is None:
        for item in items:
            print(item.text_src)
        return None

    ipy_display, IPyLatex = ipy
    for item in items:
        if item.latex_src:
            ipy_display(IPyLatex("$" + item.latex_src + "$"))
        else:
            print(item.text_src)
    return None
