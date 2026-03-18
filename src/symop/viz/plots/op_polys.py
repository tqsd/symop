r"""Plotting for polynomial operator expressions.

This module provides a plotting dispatcher implementation for
:class:`OpPoly`. The visualization focuses on the temporal and spectral
envelopes associated with the unique modes referenced by operators in
the polynomial.

For each unique mode, two plots are produced:

- time-domain magnitude :math:`|\zeta(t)|`
- frequency-domain magnitude :math:`|Z(\omega)|`

Modes are deduplicated based on their signatures to avoid redundant
plots when multiple terms share the same mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symop.ccr.algebra.op.poly import OpPoly
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


def _default_mode_tag(mode: Any) -> str:
    """Return a short display tag for a mode.

    Parameters
    ----------
    mode:
        Mode-like object that may expose ``user_label`` or
        ``display_index``.

    Returns
    -------
    str
        Human-readable tag for subplot titles. Returns an empty string
        if no suitable label is available.

    """
    lab = getattr(mode, "user_label", None)
    if lab:
        return str(lab)
    idx = getattr(mode, "display_index", None)
    if isinstance(idx, int):
        return str(idx)
    return ""


def _unique_modes_from_oppoly(obj: OpPoly) -> tuple[Any, ...]:
    """Extract unique modes from an :class:`OpPoly`.

    Parameters
    ----------
    obj:
        Operator polynomial.

    Returns
    -------
    tuple
        Tuple of unique mode objects in first-seen order.

    Notes
    -----
    Modes are deduplicated based on their ``signature`` attribute to
    ensure that each physical mode is plotted only once.

    """
    seen: set[Any] = set()
    out: list[Any] = []
    for t in obj.terms:
        for op in getattr(t, "ops", ()):
            mode = getattr(op, "mode", None)
            if mode is None:
                continue
            sig = getattr(mode, "signature", None)
            if sig is None:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            out.append(mode)
    return tuple(out)


@plot.register(OpPoly)
def _plot_op_poly(obj: OpPoly, /, **kwargs: Any) -> Any:
    """Plot envelopes referenced by an :class:`OpPoly`.

    The plotting routine extracts unique modes referenced by operators
    in the polynomial and visualizes their associated envelopes in the
    time and frequency domains.

    Parameters
    ----------
    obj:
        Operator polynomial to visualize.
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
    This function plots only the unique modes appearing in the operator
    words, avoiding duplication across terms.

    If no modes are found, a placeholder figure labeled ``"no modes"``
    is returned.

    """
    n_time = int(kwargs.pop("n_time", 2048))
    n_freq = int(kwargs.pop("n_freq", 2048))
    t_span_sigma = float(kwargs.pop("t_span_sigma", 6.0))
    w_span_sigma = float(kwargs.pop("w_span_sigma", 12.0))
    title = kwargs.pop("title", None)
    return_axes = bool(kwargs.pop("return_axes", False))

    import matplotlib.pyplot as plt

    modes = _unique_modes_from_oppoly(obj)
    if len(modes) == 0:
        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "no modes", ha="center", va="center")
        ax.set_axis_off()
        return (fig, np.array([ax], dtype=object)) if return_axes else fig

    if title is None:
        latex_title = ""
        try:
            latex_title = latex(obj, **{})
        except Exception:
            latex_title = ""
        title = latex_title if latex_title else None

    n = len(modes)
    fig, axes = plt.subplots(
        nrows=2 * n,
        ncols=1,
        figsize=(10, max(3, 2 * n * 2.2)),
        sharex=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    if title:
        fig.suptitle(r"$" + title + r"$")

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
        tmin = float(center) - t_span_sigma * scale
        tmax = float(center) + t_span_sigma * scale
        t = np.linspace(tmin, tmax, n_time, dtype=float)

        zt = np.asarray(env.time_eval(t), dtype=complex)
        ax_t.plot(t, np.abs(zt))
        ax_t.set_ylabel(r"$|\zeta(t)|$")

        w0, W = _estimate_freq_window(env, w_span_sigma=w_span_sigma)
        w = np.linspace(w0 - W, w0 + W, n_freq, dtype=float)
        Zw = np.asarray(env.freq_eval(w), dtype=complex)
        ax_w.plot(w, np.abs(Zw))
        ax_w.set_ylabel(r"$|Z(\omega)|$")

        tag = _default_mode_tag(mode)
        ax_t.set_title(f"mode {tag}" if tag else "mode")

        if i == n - 1:
            ax_t.set_xlabel(r"$t$")
            ax_w.set_xlabel(r"$\omega$")
        else:
            ax_t.set_xlabel("")
            ax_w.set_xlabel("")

    fig.tight_layout()
    return (fig, axes) if return_axes else fig
