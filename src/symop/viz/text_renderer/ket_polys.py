r"""Text rendering for ket polynomials.

Provides a text-dispatch implementation for :class:`KetPoly`, formatting
polynomial states as sums of monomials with complex coefficients.

The output is intended for terminal-friendly inspection and debugging of
symbolic CCR-based ket representations.
"""

from __future__ import annotations

from typing import Any

from symop.ccr.algebra.ket.poly import KetPoly
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


def _term_to_text(coeff: complex, body: str, *, decimals: int = 6) -> str:
    r"""Format a single polynomial term for display.

    Parameters
    ----------
    coeff:
        Complex coefficient of the term.
    body:
        Text representation of the monomial.
    decimals:
        Number of significant digits for coefficient formatting.

    Returns
    -------
    str
        Formatted term string combining coefficient and monomial.

    Notes
    -----
    - If ``body`` is empty, it is treated as the identity ``"I"``.
    - Coefficients of ``±1`` are simplified by omitting the explicit factor.
    - Uses a middle dot (``·``) to separate coefficient and monomial.

    """
    if not body:
        body = "I"
    if abs(coeff - 1.0) < 1e-15:
        return body
    if abs(coeff + 1.0) < 1e-15:
        return "-" + body
    return f"{_text_complex(coeff, decimals=decimals)} {_MIDDOT} {body}"


@text.register(KetPoly)
def _text_ket_poly(obj: KetPoly, /, **kwargs: Any) -> str:
    r"""Render a ket polynomial as a sum of monomial terms.

    Parameters
    ----------
    obj:
        Ket polynomial to render.
    **kwargs:
        Optional formatting parameters:

        - eps : float, optional
            Threshold below which coefficients are treated as zero.
            Default is ``1e-15``.
        - decimals : int, optional
            Number of significant digits for coefficient formatting.
            Default is ``6``.

    Returns
    -------
    str
        Terminal-friendly string representation of the polynomial as a
        sum of terms.

    Notes
    -----
    - Terms with coefficients below ``eps`` are omitted.
    - Coefficients are formatted using :func:`_text_complex`.
    - Monomial rendering is delegated to the ``text`` dispatcher.
    - Terms are combined using ``+`` and ``-`` with proper spacing.

    """
    eps = float(kwargs.pop("eps", 1e-15))
    decimals = int(kwargs.pop("decimals", 6))

    if len(obj.terms) == 0:
        return "0"

    parts: list[str] = []
    for t in obj.terms:
        c = complex(t.coeff)
        if abs(c) <= eps:
            continue
        m = text(t.monomial, **kwargs)
        parts.append(_term_to_text(c, m, decimals=decimals))

    if not parts:
        return "0"

    out = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            out += " - " + p[1:]
        else:
            out += " + " + p
    return out
