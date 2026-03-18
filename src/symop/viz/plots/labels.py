"""Plotting for polarization and mode labels.

This module provides plotting dispatcher implementations for
:class:`Polarization` and :class:`ModeLabel`.

- Polarization is visualized as a polarization ellipse derived from the
  Jones vector.
- Mode labels delegate plotting to their associated temporal envelope,
  with a descriptive title.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.polarization import Polarization
from symop.viz._dispatch import plot
from symop.viz._optional import require_matplotlib_pyplot


@plot.register
def _(pol: Polarization, /, **kwargs: Any) -> Any:
    r"""Plot a :class:`Polarization` as a polarization ellipse.

    Parameters
    ----------
    pol:
        Polarization object defined by a Jones vector.
    **kwargs:
        Additional plotting options. Supported keys include:

        title : str or None, optional
            Title for the plot. Defaults to ``"Polarization ellipse"``.
        n : int, optional
            Number of samples used to trace the ellipse. Default is ``400``.

    Returns
    -------
    Any
        Tuple ``(figure, axes)`` where axes is a NumPy array of matplotlib axes.

    Notes
    -----
    The polarization ellipse is constructed from the Jones vector
    components :math:`(a, b)` by evaluating

    .. math::

        E_x(t) = \Re(a e^{i t}), \quad
        E_y(t) = \Re(b e^{i t})

    over one full period. The resulting curve represents the trajectory
    of the electric field vector in the transverse plane.

    """
    plt = require_matplotlib_pyplot()

    title = kwargs.pop("title", None)
    n = int(kwargs.pop("n", 400))

    a, b = pol.jones

    t = np.linspace(0.0, 2.0 * np.pi, n, dtype=float)
    e = np.exp(1j * t)
    ex = np.real(a * e)
    ey = np.real(b * e)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(ex, ey)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(Ex)")
    ax.set_ylabel("Re(Ey)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Polarization ellipse")

    ax.text(
        0.02,
        0.98,
        f"Jones = ({a}, {b})",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    return fig, np.asarray([ax], dtype=object)


@plot.register
def _(m: ModeLabel, /, **kwargs: Any) -> Any:
    """Plot a :class:`ModeLabel` via its associated envelope.

    Parameters
    ----------
    m:
        Mode label containing path, polarization, and envelope information.
    **kwargs:
        Additional keyword arguments forwarded to the envelope plotting
        dispatcher.

    Returns
    -------
    Any
        Result of the delegated plotting call (typically a figure or
        ``(figure, axes)`` tuple).

    Notes
    -----
    This function does not implement plotting directly. Instead, it
    delegates to the ``plot`` dispatcher for the associated envelope
    ``m.envelope``.

    A default title is generated if none is provided, incorporating
    basic path information.

    """
    title = kwargs.pop("title", None)
    if title is None:
        title = f"ModeLabel: path={m.path.signature[-1]}"

    return plot(m.envelope, title=title, **kwargs)
