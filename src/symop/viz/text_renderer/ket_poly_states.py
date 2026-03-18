r"""Text rendering for ket polynomial states.

Provides a text-dispatch implementation for :class:`KetPolyState`,
formatting states as vacuum-referenced expressions of the form
``(poly)|0>``.

The output is intended for debugging and inspection of symbolic
CCR-based ket representations.
"""

from __future__ import annotations

from typing import Any

from symop.polynomial.state.ket import KetPolyState
from symop.viz._dispatch import text


@text.register(KetPolyState)
def _text_ket_poly_state(obj: KetPolyState, /, **kwargs: Any) -> str:
    r"""Render a ket polynomial state in vacuum-referenced form.

    Parameters
    ----------
    obj:
        Ket polynomial state to render.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher
        for rendering the underlying polynomial.

    Returns
    -------
    str
        String representation of the form ``(poly)|0>``. Returns ``"0"``
        if the polynomial is empty or renders to zero.

    Notes
    -----
    - The polynomial part is obtained via ``text(obj.ket)``.
    - Parentheses are added when the polynomial contains sums or
      differences to preserve readability.
    - If rendering of the polynomial fails, an empty body is assumed.
    - The output may include unicode depending on lower-level renderers.

    """
    try:
        body = text(obj.ket, **kwargs)
    except Exception:
        body = ""

    if not body or body.strip() == "0":
        return "0"

    # Parenthesize when it looks like a sum/difference.
    need_parens = (" + " in body) or (" - " in body)
    if need_parens:
        return f"({body})|0>"
    return f"{body}|0>"
