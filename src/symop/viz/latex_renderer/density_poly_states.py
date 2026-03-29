"""LaTeX rendering for polynomial density states.

This module provides a LaTeX dispatcher implementation for
:class:`DensityPolyState`. The rendering delegates to the underlying
density polynomial representation (``obj.rho``), which is treated as
the canonical representation of the quantum state.

The function is registered via the global ``latex`` dispatcher and is
automatically used whenever a ``DensityPolyState`` is passed to it.
"""

from __future__ import annotations

from typing import Any

from symop.polynomial.state.density import DensityPolyState
from symop.viz._dispatch import latex


@latex.register(DensityPolyState)
def _latex_density_poly_state(obj: DensityPolyState, /, **kwargs: Any) -> str:
    """Render a :class:`DensityPolyState` as a LaTeX string.

    The state is represented via its underlying density polynomial
    ``obj.rho``. Rendering is delegated to the generic ``latex``
    dispatcher, allowing reuse of lower-level formatting logic.

    Parameters
    ----------
    obj:
        Density polynomial state to render.
    **kwargs:
        Additional keyword arguments forwarded to the ``latex``
        dispatcher (e.g., formatting configuration such as precision).

    Returns
    -------
    str
        LaTeX representation of the density state.

    Notes
    -----
    If rendering of ``obj.rho`` fails or results in an empty or
    zero-like expression, the function returns ``"0"``.

    """
    subs = rf"\text{{{obj.label}}}" if obj.label else str(obj.index)
    prepend = rf"\rho_{{{subs}}}="

    try:
        body = prepend + latex(obj.rho, **kwargs)
    except Exception:
        body = ""

    if not body or body.strip() == "0":
        return r"0"

    return body
