r"""Text rendering for density polynomial states.

Provides a text-dispatch implementation for :class:`DensityPolyState`,
formatting states through their underlying density polynomial
representation.

The output is intended for debugging and inspection of symbolic
CCR-based density-state representations.
"""

from __future__ import annotations

from typing import Any

from symop.polynomial.state.density import DensityPolyState
from symop.viz._dispatch import text


@text.register(DensityPolyState)
def _text_density_poly_state(obj: DensityPolyState, /, **kwargs: Any) -> str:
    r"""Render a density polynomial state in density-operator form.

    Parameters
    ----------
    obj:
        Density polynomial state to render.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher
        for rendering the underlying density polynomial.

    Returns
    -------
    str
        String representation of the underlying density polynomial.
        Returns ``"0"`` if rendering fails or the polynomial renders
        to an empty or zero string.

    Notes
    -----
    The rendering is delegated to ``text(obj.rho)``.

    """
    try:
        body = text(obj.rho, **kwargs)
    except Exception:
        body = ""

    if not body or body.strip() == "0":
        return "0"

    return body
