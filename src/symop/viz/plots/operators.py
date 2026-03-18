"""Plotting for core operator types.

This module provides plotting dispatcher implementations for
:class:`ModeOp` and :class:`LadderOp`. The visualization is delegated
to the associated mode or label, ultimately resolving to envelope
plots.

This keeps operator-level plotting lightweight and compositional.
"""

from __future__ import annotations

from typing import Any

from symop.core.operators import (
    LadderOp,
    ModeOp,
)
from symop.viz._dispatch import plot


@plot.register(ModeOp)
def _plot_modeop(obj: ModeOp, /, **kwargs: Any) -> Any:
    """Plot a :class:`ModeOp`.

    Parameters
    ----------
    obj:
        Mode operator to visualize.
    **kwargs:
        Additional keyword arguments forwarded to the plotting
        dispatcher.

    Returns
    -------
    Any
        Result of the delegated plotting call.

    Notes
    -----
    Plotting is delegated to the associated label ``obj.label``, which
    is expected to further resolve to an envelope visualization.

    """
    return plot(obj.label, **kwargs)


@plot.register(LadderOp)
def _plot_ladderop(obj: LadderOp, /, **kwargs: Any) -> Any:
    """Plot a :class:`LadderOp`.

    Parameters
    ----------
    obj:
        Ladder operator to visualize.
    **kwargs:
        Additional keyword arguments forwarded to the plotting
        dispatcher.

    Returns
    -------
    Any
        Result of the delegated plotting call.

    Notes
    -----
    Plotting is delegated to the underlying mode ``obj.mode``.
    The visualization ultimately resolves to the envelope associated
    with that mode.

    """
    # Delegate to the underlying mode.
    return plot(obj.mode, **kwargs)
