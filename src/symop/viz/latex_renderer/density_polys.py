"""LaTeX rendering for polynomial density operators.

This module provides a LaTeX dispatcher implementation for
:class:`DensityPoly`. A density polynomial is rendered as a linear
combination of its terms, where each term contributes a coefficient
and a LaTeX-formatted body.

The rendering respects numerical thresholds and formatting precision
configured via keyword arguments.
"""

from __future__ import annotations

from typing import Any

from symop.ccr.algebra.density.poly import DensityPoly
from symop.viz._dispatch import latex
from symop.viz.latex_renderer._latex_utils import (
    join_signed,
    latex_config_from_kwargs,
)


@latex.register(DensityPoly)
def _latex_density_poly(obj: DensityPoly, /, **kwargs: Any) -> str:
    r"""Render a :class:`DensityPoly` as a LaTeX string.

    The density polynomial is represented as a sum of terms. Each term
    is rendered via the global ``latex`` dispatcher and multiplied by
    its associated coefficient.

    Parameters
    ----------
    obj:
        Density polynomial to render.
    **kwargs:
        Additional keyword arguments forwarded to the LaTeX rendering
        pipeline. These may include formatting options such as
        ``decimals`` and ``eps``.

    Returns
    -------
    str
        LaTeX representation of the density polynomial.

    Notes
    -----
    - Terms with coefficients below the configured threshold ``eps``
      are omitted.
    - If a term has an empty body, it is rendered as the identity
      operator ``\\mathbb{I}``.
    - If no terms remain after filtering, the result is ``"0"``.

    """
    cfg = latex_config_from_kwargs(kwargs)

    if len(obj.terms) == 0:
        return "0"

    parts: list[str] = []
    for t in obj.terms:
        c = complex(getattr(t, "coeff", 1.0))
        if abs(c) <= cfg.eps:
            continue
        parts.append(latex(t, **kwargs))

    return join_signed(parts)
