"""Plotting for polynomial ket states.

This module provides a plotting dispatcher implementation for
:class:`KetPolyState`. Visualization is delegated to the underlying
ket polynomial representation (``obj.ket``), which contains the
relevant mode and envelope information.
"""

from __future__ import annotations

from typing import Any

from symop.polynomial.state.ket import KetPolyState
from symop.viz._dispatch import plot


@plot.register(KetPolyState)
def _plot_ket_poly_state(obj: KetPolyState, /, **kwargs: Any) -> Any:
    """Plot a :class:`KetPolyState`.

    The plotting is delegated to the underlying ket polynomial
    ``obj.ket``.

    Parameters
    ----------
    obj:
        Polynomial ket state to visualize.
    **kwargs:
        Additional keyword arguments forwarded to the plotting
        dispatcher.

    Returns
    -------
    Any
        Result of the underlying plotting call. The exact return type
        depends on the plotting backend and registered handler.

    Notes
    -----
    This function does not implement plotting logic directly. It relies
    on the ``plot`` dispatcher for the underlying ket polynomial object.

    """
    return plot(obj.ket, **kwargs)
