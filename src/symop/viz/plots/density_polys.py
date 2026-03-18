r"""Plotting for polynomial density operators.

This module provides a plotting dispatcher implementation for
:class:`DensityPoly`. The visualization focuses on the temporal and
spectral envelopes referenced by the unique modes appearing in the
density polynomial.

For each mode with an associated envelope, two plots are produced:

- time-domain magnitude :math:`|\zeta(t)|`
- frequency-domain magnitude :math:`|Z(\omega)|`

The plotting logic is intentionally envelope-centric and does not
attempt to visualize algebraic coefficients or operator structure
directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symop.ccr.algebra.density.poly import DensityPoly
from symop.viz._dispatch import latex, plot


@plot.register(DensityPoly)
def _plot_density_poly(obj: DensityPoly, /, **kwargs: Any) -> Any:
    r"""Plot envelopes referenced by a :class:`DensityPoly`.

    The plotting routine extracts the unique modes appearing in the
    density polynomial and visualizes the corresponding envelopes in the
    time and frequency domains.

    Parameters
    ----------
    obj:
        Density polynomial to visualize.
    **kwargs:
        Additional plotting options. Supported keys include:

        n_time : int, optional
            Number of sample points for time-domain evaluation.
            Default is ``2048``.
        n_freq : int, optional
            Number of sample points for frequency-domain evaluation.
            Default is ``2048``.
        t_span_sigma : float, optional
            Half-width of the plotted time window in units of the
            envelope scale parameter. Default is ``6.0``.
        w_span_sigma : float, optional
            Half-width of the plotted frequency window in units of the
            frequency scale parameter. Default is ``12.0``.
        title : str or None, optional
            Figure title. If omitted, a LaTeX rendering of ``obj`` is
            attempted.
        return_axes : bool, optional
            If ``True``, return both the figure and axes array.
            Otherwise, return only the figure. Default is ``False``.

    Returns
    -------
    Any
        Either a Matplotlib figure, or a ``(figure, axes)`` tuple if
        ``return_axes`` is ``True``.

    Notes
    -----
    The plotting strategy is as follows:

    - unique modes are extracted from ``obj.unique_modes``
    - each mode contributes up to two vertically stacked axes
    - the first axis shows :math:`|\zeta(t)|`
    - the second axis shows :math:`|Z(\omega)|`

    If the density polynomial has no modes, a minimal placeholder figure
    is returned indicating either ``"zero"`` or ``"no modes"``.

    """
    n_time = int(kwargs.pop("n_time", 2048))
    n_freq = int(kwargs.pop("n_freq", 2048))
    t_span_sigma = float(kwargs.pop("t_span_sigma", 6.0))
    w_span_sigma = float(kwargs.pop("w_span_sigma", 12.0))
    title = kwargs.pop("title", None)
    return_axes = bool(kwargs.pop("return_axes", False))

    import matplotlib.pyplot as plt

    modes = getattr(obj, "unique_modes", ())
    if not modes:
        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            "zero" if len(obj.terms) == 0 else "no modes",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return (fig, np.array([ax], dtype=object)) if return_axes else fig

    n = len(modes)
    fig, axes = plt.subplots(
        nrows=2 * n,
        ncols=1,
        figsize=(9, max(3, 2 * n * 2.2)),
        sharex=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    if title is None:
        try:
            latex_title = latex(obj, **{})
        except Exception:
            latex_title = ""
        title = latex_title if latex_title else None

    if title:
        fig.suptitle(title if "$" in title else (r"$" + title + r"$"))

    def estimate_freq_window(env: Any) -> tuple[float, float]:
        w0 = float(getattr(env, "omega0", 0.0))
        sigma_w = getattr(env, "omega_sigma", None)
        if sigma_w is None:
            sigma_t = getattr(env, "sigma", None)
            sigma_w = 1.0 / max(float(sigma_t), 1e-12) if sigma_t is not None else 1.0
        sigma_w = max(float(sigma_w), 1e-12)
        W = float(w_span_sigma) * sigma_w
        return w0, W

    def mode_tag(mode: Any) -> str:
        lab = getattr(mode, "user_label", None)
        if lab:
            return str(lab)
        idx = getattr(mode, "display_index", None)
        if isinstance(idx, int):
            return str(idx)
        return ""

    for i, mode in enumerate(modes):
        label = getattr(mode, "label", None)
        env = getattr(label, "envelope", None) if label is not None else None
        if env is None:
            continue

        ax_t = axes[2 * i + 0]
        ax_w = axes[2 * i + 1]

        try:
            center, scale = env.center_and_scale()
        except Exception:
            center, scale = 0.0, 1.0
        scale = max(float(scale), 1e-12)

        t = np.linspace(
            float(center) - t_span_sigma * scale,
            float(center) + t_span_sigma * scale,
            n_time,
            dtype=float,
        )
        zt = np.asarray(env.time_eval(t), dtype=complex)
        ax_t.plot(t, np.abs(zt))
        ax_t.set_ylabel(r"$|\zeta(t)|$")

        w0, W = estimate_freq_window(env)
        w = np.linspace(w0 - W, w0 + W, n_freq, dtype=float)
        Zw = np.asarray(env.freq_eval(w), dtype=complex)
        ax_w.plot(w, np.abs(Zw))
        ax_w.set_ylabel(r"$|Z(\omega)|$")

        tag = mode_tag(mode)
        ax_t.set_title(f"mode {tag}" if tag else "mode")

        if i == n - 1:
            ax_t.set_xlabel(r"$t$")
            ax_w.set_xlabel(r"$\omega$")
        else:
            ax_t.set_xlabel("")
            ax_w.set_xlabel("")

    fig.tight_layout()
    return (fig, axes) if return_axes else fig
