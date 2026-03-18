r"""Text rendering for polynomial term objects.

Provides text-dispatch implementations for :class:`KetTerm` and
:class:`DensityTerm`, formatting individual terms with complex
coefficients and structured operator expressions.

The output is intended for debugging and inspection of symbolic
CCR-based polynomial representations.
"""

from __future__ import annotations

from typing import Any

from symop.core.terms import DensityTerm, KetTerm
from symop.viz._dispatch import text

_MIDDOT = "\u00b7"  # ·


def _text_complex(c: complex, *, decimals: int = 6) -> str:
    r"""Format a complex number for textual display.

    Parameters
    ----------
    c:
        Complex number to format.
    decimals:
        Number of significant digits used in formatting.

    Returns
    -------
    str
        Compact string representation of ``c`` with small real/imaginary
        parts suppressed and simplified handling of ``±i``.

    Notes
    -----
    - Values close to zero are treated as zero using a tolerance derived
      from ``decimals``.
    - The imaginary unit is rendered as ``i`` instead of ``1i``.
    - Mixed real and imaginary parts are enclosed in parentheses.

    """
    re = float(c.real)
    im = float(c.imag)

    def fmt(x: float) -> str:
        s = f"{x:.{decimals}g}"
        if s == "-0":
            s = "0"
        return s

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


@text.register(KetTerm)
def _text_ket_term(obj: KetTerm, /, **kwargs: Any) -> str:
    r"""Render a ket term as a coefficient-monomial product.

    Parameters
    ----------
    obj:
        Ket term to render.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher.

    Returns
    -------
    str
        String representation of the form ``c · m`` or simplified forms
        when the coefficient is ``±1``.

    Notes
    -----
    - Monomial rendering is delegated to the ``text`` dispatcher.
    - Coefficients are formatted using :func:`_text_complex`.
    - Coefficients of ``±1`` are simplified by omitting the explicit factor.

    """
    c = complex(obj.coeff)
    m = text(obj.monomial, **kwargs)

    if abs(c - 1.0) < 1e-15:
        return m
    if abs(c + 1.0) < 1e-15:
        return "-" + m

    return f"{_text_complex(c)} {_MIDDOT} {m}"


@text.register(DensityTerm)
def _text_density_term(obj: DensityTerm, /, **kwargs: Any) -> str:
    r"""Render a density term as a bilinear operator expression.

    Parameters
    ----------
    obj:
        Density term to render.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher.

    Returns
    -------
    str
        String representation of the form ``c · L · (.) · R`` or simplified
        forms when the coefficient is ``±1``.

    Notes
    -----
    - Left and right parts are rendered via the ``text`` dispatcher.
    - The placeholder ``(.)`` denotes the density operator slot.
    - Coefficients are formatted using :func:`_text_complex`.
    - Coefficients of ``±1`` are simplified by omitting the explicit factor.

    """
    c = complex(obj.coeff)
    L = text(obj.left, **kwargs)
    R = text(obj.right, **kwargs)

    core = f"{L} {_MIDDOT} (.) {_MIDDOT} {R}"
    if abs(c - 1.0) < 1e-15:
        return core
    if abs(c + 1.0) < 1e-15:
        return "-" + core

    return f"{_text_complex(c)} {_MIDDOT} {core}"
