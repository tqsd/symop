r"""Text rendering for polynomial density states.

Provides a text-dispatch implementation for :class:`DensityPoly`
objects, formatting density polynomials as readable algebraic
expressions. Coefficients are rendered with configurable precision
and small terms can be suppressed via a threshold.

The output is intended for debugging, logging, and lightweight
inspection of symbolic CCR-based density representations.
"""

from __future__ import annotations

from typing import Any

from symop.ccr.algebra.density.poly import DensityPoly
from symop.viz._dispatch import text


def _fmt_complex(c: complex, *, decimals: int = 6) -> str:
    r"""Format a complex number for compact textual display.

    Parameters
    ----------
    c:
        Complex number to format.
    decimals:
        Number of significant digits used in formatting.

    Returns
    -------
    str
        String representation of ``c`` with small real/imaginary parts
        suppressed and special handling for ``±i``.

    Notes
    -----
    - Values close to zero (within a tolerance derived from ``decimals``)
      are treated as zero.
    - The imaginary unit is rendered as ``i`` instead of ``1i``.
    - Real-only and imaginary-only numbers are simplified accordingly.

    """
    re = float(c.real)
    im = float(c.imag)

    def fmt(x: float) -> str:
        s = f"{x:.{decimals}g}"
        return "0" if s == "-0" else s

    eps = 10 ** (-(decimals + 1))
    re0 = abs(re) < eps
    im0 = abs(im) < eps

    if re0 and im0:
        return "0"
    if im0:
        return fmt(re)
    if re0:
        if abs(im - 1.0) < eps:
            return "i"
        if abs(im + 1.0) < eps:
            return "-i"
        return f"{fmt(im)}i"

    sign = "+" if im >= 0 else "-"
    im_abs = abs(im)
    im_part = "i" if abs(im_abs - 1.0) < eps else f"{fmt(im_abs)}i"
    return f"({fmt(re)} {sign} {im_part})"


@text.register(DensityPoly)
def _text_density_poly(obj: DensityPoly, /, **kwargs: Any) -> str:
    r"""Render a density polynomial as a text expression.

    Parameters
    ----------
    obj:
        Density polynomial to render.
    **kwargs:
        Optional formatting parameters:

        - eps : float, optional
            Threshold below which coefficients are treated as zero.
            Default is ``1e-12``.
        - decimals : int, optional
            Number of significant digits for coefficient formatting.
            Default is ``6``.

    Returns
    -------
    str
        Human-readable algebraic expression representing the density
        polynomial.

    Notes
    -----
    - Terms with coefficients below ``eps`` are omitted.
    - Coefficients of ``±1`` are simplified (no explicit factor).
    - Terms are combined using ``+`` and ``-`` with proper spacing.
    - Delegates term-level formatting to the ``text`` dispatcher.

    """
    eps = float(kwargs.pop("eps", 1e-12))
    decimals = int(kwargs.pop("decimals", 6))

    if len(obj.terms) == 0:
        return "0"

    parts: list[str] = []
    for t in obj.terms:
        c = complex(getattr(t, "coeff", 1.0))
        if abs(c) <= eps:
            continue

        body = text(t, **kwargs)
        if abs(c - 1.0) < 1e-15:
            parts.append(body)
        elif abs(c + 1.0) < 1e-15:
            parts.append("-" + body)
        else:
            parts.append(_fmt_complex(c, decimals=decimals) + " * " + body)

    if not parts:
        return "0"

    out = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            out += " - " + p[1:]
        else:
            out += " + " + p
    return out
