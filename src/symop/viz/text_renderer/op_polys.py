r"""Text rendering for operator polynomials.

Provides a text-dispatch implementation for :class:`OpPoly`, formatting
operator polynomials as sums of ordered operator words with complex
coefficients.

The output is intended for terminal-friendly inspection and debugging of
symbolic CCR-based operator representations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from symop.ccr.algebra.op.poly import OpPoly
from symop.viz._dispatch import text

# Keep source ASCII-only; unicode output via escapes.
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


def _word_text(ops: Iterable[Any], **kwargs: Any) -> str:
    r"""Render an ordered sequence of operators as a word.

    Parameters
    ----------
    ops:
        Iterable of operator objects.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher.

    Returns
    -------
    str
        Space-separated string of operator representations, or ``"I"``
        if the sequence is empty.

    Notes
    -----
    Each operator is rendered via the ``text`` dispatcher and combined
    in order.

    """
    parts: list[str] = []
    for op in ops:
        parts.append(text(op, **kwargs))
    if not parts:
        return "I"
    return " ".join(parts)


def _term_to_text(coeff: complex, body: str, *, decimals: int = 6) -> str:
    r"""Format a single operator polynomial term for display.

    Parameters
    ----------
    coeff:
        Complex coefficient of the term.
    body:
        Text representation of the operator word.
    decimals:
        Number of significant digits for coefficient formatting.

    Returns
    -------
    str
        Formatted term string combining coefficient and operator word.

    Notes
    -----
    - If ``body`` is empty, it is treated as the identity ``"I"``.
    - Coefficients of ``±1`` are simplified by omitting the explicit factor.
    - Uses a middle dot (``·``) to separate coefficient and operator word.

    """
    if not body:
        body = "I"
    if abs(coeff - 1.0) < 1e-15:
        return body
    if abs(coeff + 1.0) < 1e-15:
        return "-" + body
    return f"{_text_complex(coeff, decimals=decimals)} {_MIDDOT} {body}"


@text.register(OpPoly)
def _text_op_poly(obj: OpPoly, /, **kwargs: Any) -> str:
    r"""Render an operator polynomial as a sum of operator words.

    Parameters
    ----------
    obj:
        Operator polynomial to render.
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
        Terminal-friendly string representation of the operator polynomial.

    Notes
    -----
    - Terms with coefficients below ``eps`` are omitted.
    - Operator sequences are rendered via :func:`_word_text`.
    - Coefficients are formatted using :func:`_text_complex`.
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
        body = _word_text(t.ops, **kwargs)
        parts.append(_term_to_text(c, body, decimals=decimals))

    if not parts:
        return "0"

    out = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            out += " - " + p[1:]
        else:
            out += " + " + p
    return out
