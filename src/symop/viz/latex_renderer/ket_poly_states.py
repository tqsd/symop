"""LaTeX rendering for polynomial ket states.

This module provides a LaTeX dispatcher implementation for
:class:`KetPolyState`. A ket polynomial state is rendered as an operator
polynomial acting on the vacuum state.

The rendering delegates to the underlying ket polynomial representation
(``obj.ket``) and appends the vacuum ket ``|0⟩`` to indicate state
construction.
"""

from __future__ import annotations

from typing import Any

from symop.polynomial.state.ket import KetPolyState
from symop.viz._dispatch import latex


@latex.register(KetPolyState)
def _latex_ket_poly_state(obj: KetPolyState, /, **kwargs: Any) -> str:
    r"""Render a :class:`KetPolyState` as a LaTeX string.

    The state is interpreted as a polynomial of creation operators acting
    on the vacuum:

    .. math::

        |\psi\rangle = \left(\sum_k c_k M_k\right) |0\rangle

    Parameters
    ----------
    obj:
        Polynomial ket state to render.
    **kwargs:
        Additional keyword arguments forwarded to the LaTeX dispatcher
        (e.g., formatting configuration).

    Returns
    -------
    str
        LaTeX representation of the ket state.

    Notes
    -----
    - Rendering of the polynomial part is delegated to the ``latex``
      dispatcher via ``obj.ket``.
    - If the polynomial body is empty or evaluates to zero, the function
      returns ``"0"``.
    - Parentheses are added around the polynomial if it contains multiple
      terms to preserve correct visual grouping.

    """
    try:
        body = latex(obj.ket, **kwargs)
    except Exception:
        body = ""

    if not body or body.strip() == "0":
        return r"0"

    need_parens = ("+" in body) or ("-" in body[1:])  # ignore leading '-'
    if need_parens:
        return rf"\left({body}\right)\lvert 0\rangle"
    return rf"{body}\lvert 0\rangle"
