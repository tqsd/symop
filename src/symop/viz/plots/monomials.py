r"""Plotting for CCR monomials.

This module provides a plotting dispatcher implementation for
:class:`Monomial`. The visualization focuses on the temporal and
spectral envelopes associated with the modes referenced by the
operators in the monomial.

For each mode appearing in the monomial, two plots are produced:

- time-domain magnitude :math:`|\zeta(t)|`
- frequency-domain magnitude :math:`|Z(\omega)|`

The plotting is mode-centric and does not attempt to visualize the
operator ordering directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symop.core.monomial import Monomial
from symop.viz._dispatch import latex, plot


def _estimate_freq_window(env: Any, *, w_span_sigma: float) -> tuple[float, float]:
    """Estimate a frequency plotting window for an envelope.

    Parameters
    ----------
    env:
        Envelope-like object expected to expose ``omega0`` and optionally
        ``omega_sigma`` or ``sigma``.
    w_span_sigma:
        Half-width multiplier for the plotted frequency window.

    Returns
    -------
    tuple of float
        Pair ``(w0, W)`` where ``w0`` is the center frequency and
        ``W`` is the half-width of the frequency range.

    Notes
    -----
    If ``omega_sigma`` is unavailable, the spectral width is
    approximated from the time-domain scale ``sigma`` using an inverse
    relation.

    """
    w0 = float(getattr(env, "omega0", 0.0))
    sigma_w = getattr(env, "omega_sigma", None)
    if sigma_w is None:
        sigma_t = getattr(env, "sigma", None)
        sigma_w = 1.0 / max(float(sigma_t), 1e-12) if sigma_t is not None else 1.0
    sigma_w = max(float(sigma_w), 1e-12)
    W = float(w_span_sigma) * sigma_w
    return w0, W


def _default_label_for_mode(mode: Any) -> str:
    """Return a short display label for a mode.

    Parameters
    ----------
    mode:
        Mode-like object that may expose ``user_label`` or
        ``display_index``.

    Returns
    -------
    str
        Human-readable label for subplot titles. Returns an empty string
        if no suitable label is available.

    """
    lab = getattr(mode, "user_label", None)
    if lab:
        return str(lab)
    idx = getattr(mode, "display_index", None)
    if isinstance(idx, int):
        return f"{idx}"
    return ""


@plot.register(Monomial)
def _plot_monomial(obj: Monomial, /, **kwargs: Any) -> Any:
    """Plot envelopes referenced by a :class:`Monomial`.

    The plotting routine extracts the modes referenced by operators in
    the monomial and visualizes their associated envelopes in the time
    and frequency domains.

    Parameters
    ----------
    obj:
        Monomial to visualize.
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
            spectral width. Default is ``12.0``.
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
    The plotting is mode-centric: each mode contributes two vertically
    stacked axes, one for the time-domain magnitude and one for the
    frequency-domain magnitude.

    If the monomial contains no modes, a minimal placeholder figure
    labeled ``"identity"`` is returned.

    """
    n_time = int(kwargs.pop("n_time", 2048))
    n_freq = int(kwargs.pop("n_freq", 2048))
    t_span_sigma = float(kwargs.pop("t_span_sigma", 6.0))
    w_span_sigma = float(kwargs.pop("w_span_sigma", 12.0))
    title = kwargs.pop("title", None)
    return_axes = bool(kwargs.pop("return_axes", False))

    # Matplotlib import stays local to keep symop core clean.
    import matplotlib.pyplot as plt

    modes = obj.mode_ops
    if len(modes) == 0:
        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "identity", ha="center", va="center")
        ax.set_axis_off()
        return (fig, np.array([ax], dtype=object)) if return_axes else fig

    n = len(modes)
    # Two rows per mode: time magnitude, freq magnitude.
    fig, axes = plt.subplots(
        nrows=2 * n,
        ncols=1,
        figsize=(9, max(3, 2 * n * 2.2)),
        sharex=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    # Default title: monomial latex, if non-empty.
    if title is None:
        try:
            latex_title = latex(obj, **{})
        except Exception:
            latex_title = ""
        title = latex_title if latex_title else None

    if title:
        if "$" in title:
            fig.suptitle(title)
        else:
            fig.suptitle(r"$" + title + r"$")

    for i, mode in enumerate(modes):
        label = getattr(mode, "label", None)
        env = getattr(label, "envelope", None) if label is not None else None
        if env is None:
            # Skip gracefully; still keep axes.
            continue

        ax_t = axes[2 * i + 0]
        ax_w = axes[2 * i + 1]

        # Time grid from center_and_scale.
        try:
            center, scale = env.center_and_scale()
        except Exception:
            center, scale = 0.0, 1.0
        scale = max(float(scale), 1e-12)
        tmin = float(center) - t_span_sigma * scale
        tmax = float(center) + t_span_sigma * scale
        t = np.linspace(tmin, tmax, n_time, dtype=float)

        zt = np.asarray(env.time_eval(t), dtype=complex)
        ax_t.plot(t, np.abs(zt))
        ax_t.set_ylabel(r"$|\zeta(t)|$")

        # Frequency grid from omega0 +/- span.
        w0, W = _estimate_freq_window(env, w_span_sigma=w_span_sigma)
        w = np.linspace(w0 - W, w0 + W, n_freq, dtype=float)
        Zw = np.asarray(env.freq_eval(w), dtype=complex)
        ax_w.plot(w, np.abs(Zw))
        ax_w.set_ylabel(r"$|Z(\omega)|$")

        # Row label on the left via y-axis title extension.
        tag = _default_label_for_mode(mode)
        if tag:
            ax_t.set_title(f"mode {tag}")
        else:
            ax_t.set_title("mode")

        if i == n - 1:
            ax_t.set_xlabel(r"$t$")
            ax_w.set_xlabel(r"$\omega$")
        else:
            ax_t.set_xlabel("")
            ax_w.set_xlabel("")

    fig.tight_layout()
    return (fig, axes) if return_axes else fig
