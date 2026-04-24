"""Helpers for formatting coefficients and mode labels as LaTeX strings.

This module provides small utilities used by LaTeX renderers for symbolic
objects. It includes:

- configuration for numeric formatting
- formatting of real and complex numbers
- application of scalar coefficients to symbolic bodies
- joining signed LaTeX terms
- extraction of mode subscripts from mode-like objects

The functions are intentionally lightweight and operate on generic inputs so
they can be reused across multiple symbolic representations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class SupportsLatexLabel(Protocol):
    """Protocol for objects that may expose label metadata for LaTeX output."""

    user_label: str | None
    display_index: int | None
    label: Any


@dataclass(frozen=True)
class LatexConfig:
    """Configuration for LaTeX numeric formatting.

    Parameters
    ----------
    eps:
        Absolute tolerance used for zero-like comparisons.
    decimals:
        Number of significant digits used in numeric formatting.

    """

    eps: float = 1e-15
    decimals: int = 6


def latex_config_from_kwargs(kwargs: Mapping[str, Any]) -> LatexConfig:
    """Build a :class:`LatexConfig` from a keyword mapping.

    Parameters
    ----------
    kwargs:
        Mapping that may contain the keys ``"eps"`` and ``"decimals"``.

    Returns
    -------
    LatexConfig
        Parsed configuration object.

    Notes
    -----
    Missing keys fall back to default values.

    """
    eps = float(kwargs.get("eps", 1e-15))
    decimals = int(kwargs.get("decimals", 6))
    return LatexConfig(eps=eps, decimals=decimals)


def fmt_real(x: float, *, decimals: int) -> str:
    """Format a real number for LaTeX output.

    Parameters
    ----------
    x:
        Real value to format.
    decimals:
        Number of significant digits used by general formatting.

    Returns
    -------
    str
        Formatted string representation.

    Notes
    -----
    The string ``"-0"`` is normalized to ``"0"``.

    """
    s = f"{x:.{decimals}g}"
    return "0" if s == "-0" else s


def format_complex(z: complex, *, decimals: int) -> str:
    """Format a complex number as a LaTeX string.

    Parameters
    ----------
    z:
        Complex value to format.
    decimals:
        Number of significant digits used in the output.

    Returns
    -------
    str
        LaTeX-formatted representation of the complex number.

    Notes
    -----
    Small real and imaginary parts are treated as zero using an internal
    tolerance derived from ``decimals``.

    """
    re = float(z.real)
    im = float(z.imag)

    eps = 10 ** (-(decimals + 1))
    re0 = abs(re) < eps
    im0 = abs(im) < eps

    if re0 and im0:
        return "0"
    if im0:
        return fmt_real(re, decimals=decimals)
    if re0:
        if abs(im - 1.0) < eps:
            return "i"
        if abs(im + 1.0) < eps:
            return "-i"
        return f"{fmt_real(im, decimals=decimals)}\\,i"

    sign = "+" if im >= 0 else "-"
    im_abs = abs(im)
    if abs(im_abs - 1.0) < eps:
        im_part = "i"
    else:
        im_part = f"{fmt_real(im_abs, decimals=decimals)}\\,i"
    return rf"\left({fmt_real(re, decimals=decimals)} {sign} {im_part}\right)"


def apply_coeff(coeff: complex, body: str, *, decimals: int, empty_body: str) -> str:
    """Apply a scalar coefficient to a LaTeX body string.

    Parameters
    ----------
    coeff:
        Scalar coefficient multiplying the body.
    body:
        LaTeX body to which the coefficient should be applied.
    decimals:
        Number of significant digits used when formatting nontrivial
        coefficients.
    empty_body:
        Replacement body used when ``body`` is empty.

    Returns
    -------
    str
        LaTeX string representing the scaled body.

    Notes
    -----
    Coefficients very close to ``1`` and ``-1`` are rendered without an
    explicit numeric factor.

    """
    body = body if body else empty_body

    if abs(coeff - 1.0) < 1e-15:
        return body
    if abs(coeff + 1.0) < 1e-15:
        return f"-{body}"
    return f"{format_complex(coeff, decimals=decimals)}\\,{body}"


def join_signed(parts: list[str]) -> str:
    """Join LaTeX terms while preserving explicit leading minus signs.

    Parameters
    ----------
    parts:
        Sequence of already formatted LaTeX terms.

    Returns
    -------
    str
        Joined expression. If ``parts`` is empty, returns ``"0"``.

    """
    if not parts:
        return "0"
    out = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            out += " - " + p[1:]
        else:
            out += " + " + p
    return out


def fmt_complex_latex(z: complex, *, decimals: int = 6) -> str:
    r"""Format a complex number as a compact LaTeX-compatible string.

    Parameters
    ----------
    z:
        Complex value to format.
    decimals:
        Number of decimal places used for rounding the real and imaginary
        parts before formatting.

    Returns
    -------
    str
        Compact string representation of the complex number.

    Notes
    -----
    The output is intentionally lightweight and is primarily intended for
    short inline expressions such as Jones-vector entries.

    Examples
    --------
    A purely real value is rendered as a real number, a purely imaginary
    value as an imaginary multiple, and a general complex value as a sum:

    .. math::

        1.0, \quad 0.5 i, \quad 1.0 + 0.5 i

    """
    r = round(float(z.real), decimals)
    i = round(float(z.imag), decimals)
    if i == 0.0:
        return str(r)
    if r == 0.0:
        return f"{i}i"
    sign = "+" if i >= 0 else "-"
    return f"{r}{sign}{abs(i)}i"


def mode_subscript(mode: SupportsLatexLabel, *, latex_fn: Callable[[Any], str]) -> str:
    """Return a LaTeX subscript string for a mode-like object.

    Parameters
    ----------
    mode:
        Object that may provide ``user_label``, ``display_index``, and
        ``label`` attributes.
    latex_fn:
        Callable used to format the fallback ``label`` attribute.

    Returns
    -------
    str
        Subscript string derived from the mode metadata.

    Notes
    -----
    Resolution order is:

    1. ``user_label`` if present and truthy
    2. ``display_index`` if it is an integer
    3. ``latex_fn(mode.label)`` as fallback

    If fallback formatting fails, an empty string is returned.

    """
    lab = mode.user_label
    if lab:
        return str(lab)

    idx = mode.display_index
    if isinstance(idx, int):
        return str(idx)

    try:
        return latex_fn(mode.label)
    except Exception:
        return ""
