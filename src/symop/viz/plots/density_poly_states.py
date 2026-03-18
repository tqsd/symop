"""Plotting for polynomial density states.

This module provides a plotting dispatcher implementation for
:class:`DensityPolyState`. Visualization is delegated to the underlying
density polynomial representation (``obj.rho``), which contains the
relevant structural information (e.g., envelopes).
"""

from __future__ import annotations

from typing import Any

from symop.polynomial.state.density import DensityPolyState
from symop.viz._dispatch import plot


@plot.register(DensityPolyState)
def _plot_density_poly_state(obj: DensityPolyState, /, **kwargs: Any) -> Any:
    """Plot a :class:`DensityPolyState`.

    The plotting is delegated to the underlying density polynomial
    ``obj.rho``.

    Parameters
    ----------
    obj:
        Density polynomial state to visualize.
    **kwargs:
        Additional keyword arguments forwarded to the plotting
        dispatcher (e.g., styling or backend-specific options).

    Returns
    -------
    Any
        Result of the underlying plotting call. The exact return type
        depends on the plotting backend in use.

    Notes
    -----
    This function does not implement plotting logic itself. Instead, it
    relies on the ``plot`` dispatcher for :class:`DensityPoly` to handle
    visualization details such as envelope plotting.

    """
    return plot(obj.rho, **kwargs)
