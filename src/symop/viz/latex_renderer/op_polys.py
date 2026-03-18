"""LaTeX rendering for polynomial operator expressions.

This module provides a LaTeX dispatcher implementation for
:class:`OpPoly`. An operator polynomial is rendered as a linear
combination of operator words, where each word is an ordered product
of elementary operators.

Rendering is compositional: individual operators are rendered via the
global ``latex`` dispatcher, and coefficients are applied using shared
formatting utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from symop.ccr.algebra.op.poly import OpPoly
from symop.viz._dispatch import latex
from symop.viz.latex_renderer._latex_utils import (
    apply_coeff,
    join_signed,
    latex_config_from_kwargs,
)


def _word_latex(ops: Iterable[Any], **kwargs: Any) -> str:
    r"""Render a sequence of operators as a LaTeX word.

    Parameters
    ----------
    ops:
        Iterable of operator-like objects.
    **kwargs:
        Additional keyword arguments forwarded to the ``latex``
        dispatcher for each operator.

    Returns
    -------
    str
        LaTeX representation of the operator word.

    Notes
    -----
    - Each operator is rendered individually and concatenated with a
      space separator.
    - If no operators produce output, the identity operator
      ``\\mathbb{I}`` is returned.

    """
    parts: list[str] = []
    for op in ops:
        s = latex(op, **kwargs)
        if s:
            parts.append(s)
    if not parts:
        return "\\mathbb{I}"
    return " ".join(parts)


@latex.register(OpPoly)
def _latex_op_poly(obj: OpPoly, /, **kwargs: Any) -> str:
    r"""Render an :class:`OpPoly` as a LaTeX string.

    The operator polynomial is expressed as a sum of operator words,
    each multiplied by a complex coefficient.

    Parameters
    ----------
    obj:
        Operator polynomial to render.
    **kwargs:
        Additional keyword arguments forwarded to the LaTeX rendering
        pipeline. Supported options include:

        - ``eps`` : float
            Threshold below which coefficients are treated as zero.
        - ``decimals`` : int
            Number of significant digits for coefficient formatting.

    Returns
    -------
    str
        LaTeX representation of the operator polynomial.

    Notes
    -----
    - Terms with coefficients below the configured threshold ``eps``
      are omitted.
    - Operator words are constructed by concatenating individual
      operator renderings.
    - Empty operator words are interpreted as the identity operator
      ``\\mathbb{I}``.
    - If no terms remain after filtering, the result is ``"0"``.

    """
    cfg = latex_config_from_kwargs(kwargs)

    if len(obj.terms) == 0:
        return "0"

    parts: list[str] = []
    for t in obj.terms:
        c = complex(t.coeff)
        if abs(c) <= cfg.eps:
            continue
        body = _word_latex(t.ops, **kwargs)
        parts.append(
            apply_coeff(c, body, decimals=cfg.decimals, empty_body="\\mathbb{I}")
        )

    return join_signed(parts)
