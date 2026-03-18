r"""Plotting for polynomial term objects.

This module provides plotting dispatcher implementations for
:class:`KetTerm` and :class:`DensityTerm`. The visualization focuses on
the temporal and spectral envelopes associated with the modes appearing
in the monomials of each term.

For each mode, two plots are produced:

- time-domain magnitude :math:`|\\zeta(t)|`
- frequency-domain magnitude :math:`|Z(\\omega)|`

Ket terms are shown in a single column. Density terms are shown in a
two-column layout corresponding to left and right monomials.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symop.core.terms import DensityTerm, KetTerm
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


def _draw_modes_on_axes(
    *,
    modes: tuple[Any, ...],
    axes_col: np.ndarray,
    n_time: int,
    n_freq: int,
    t_span_sigma: float,
    w_span_sigma: float,
    col_title: str | None,
) -> None:
    r"""Draw envelope plots for a sequence of modes on a column of axes.

    Parameters
    ----------
    modes:
        Tuple of mode-like objects to visualize.
    axes_col:
        One-dimensional array of axes with shape ``(2 * len(modes),)``,
        arranged as alternating time and frequency axes.
    n_time:
        Number of sample points for time-domain evaluation.
    n_freq:
        Number of sample points for frequency-domain evaluation.
    t_span_sigma:
        Half-width of the plotted time window in units of the envelope
        scale parameter.
    w_span_sigma:
        Half-width of the plotted frequency window in units of the
        spectral width.
    col_title:
        Optional prefix for subplot titles, for example ``"L"`` or
        ``"R"``.

    Returns
    -------
    None

    Notes
    -----
    Each mode contributes two plots:

    - time-domain magnitude :math:`|\zeta(t)|`
    - frequency-domain magnitude :math:`|Z(\omega)|`

    The subplot title is derived from the mode tag and optionally
    prefixed by ``col_title``.

    """
    for i, mode in enumerate(modes):
        label = getattr(mode, "label", None)
        env = getattr(label, "envelope", None) if label is not None else None
        if env is None:
            continue

        ax_t = axes_col[2 * i + 0]
        ax_w = axes_col[2 * i + 1]

        try:
            c, s = env.center_and_scale()
        except Exception:
            c, s = 0.0, 1.0

        s = max(float(s), 1e-12)
        tmin = float(c) - float(t_span_sigma) * s
        tmax = float(c) + float(t_span_sigma) * s
        t = np.linspace(tmin, tmax, int(n_time), dtype=float)

        zt = np.asarray(env.time_eval(t), dtype=complex)
        ax_t.plot(t, np.abs(zt))
        ax_t.set_ylabel(r"$|\zeta(t)|$")

        w0, W = _estimate_freq_window(env, w_span_sigma=float(w_span_sigma))
        w = np.linspace(w0 - W, w0 + W, int(n_freq), dtype=float)
        Zw = np.asarray(env.freq_eval(w), dtype=complex)
        ax_w.plot(w, np.abs(Zw))
        ax_w.set_ylabel(r"$|Z(\omega)|$")

        tag = _default_mode_tag(mode)
        title = f"mode {tag}" if tag else "mode"
        if col_title:
            title = f"{col_title} — {title}"
        ax_t.set_title(title)

        if i == len(modes) - 1:
            ax_t.set_xlabel(r"$t$")
            ax_w.set_xlabel(r"$\omega$")


@plot.register(KetTerm)
def _plot_ket_term(obj: KetTerm, /, **kwargs: Any) -> Any:
    """Plot envelopes referenced by a :class:`KetTerm`.

    The plotting routine visualizes the modes appearing in the term's
    monomial, showing time- and frequency-domain envelope magnitudes.

    Parameters
    ----------
    obj:
        Ket term to visualize.
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
    If the term monomial contains no modes, a minimal placeholder figure
    labeled ``"identity term"`` is returned.

    """
    n_time = int(kwargs.pop("n_time", 2048))
    n_freq = int(kwargs.pop("n_freq", 2048))
    t_span_sigma = float(kwargs.pop("t_span_sigma", 6.0))
    w_span_sigma = float(kwargs.pop("w_span_sigma", 12.0))
    title = kwargs.pop("title", None)
    return_axes = bool(kwargs.pop("return_axes", False))

    import matplotlib.pyplot as plt

    modes = obj.monomial.mode_ops
    if len(modes) == 0:
        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "identity term", ha="center", va="center")
        ax.set_axis_off()
        return (fig, np.array([ax], dtype=object)) if return_axes else fig

    if title is None:
        s = ""
        try:
            s = latex(obj, **{})
        except Exception:
            s = ""
        title = s if s else None

    n = len(modes)
    fig, axes = plt.subplots(
        nrows=2 * n,
        ncols=1,
        figsize=(9, max(3, 2 * n * 2.2)),
        sharex=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    if title:
        fig.suptitle(r"$" + title + r"$")

    _draw_modes_on_axes(
        modes=modes,
        axes_col=axes,
        n_time=n_time,
        n_freq=n_freq,
        t_span_sigma=t_span_sigma,
        w_span_sigma=w_span_sigma,
        col_title=None,
    )

    fig.tight_layout()
    return (fig, axes) if return_axes else fig


@plot.register(DensityTerm)
def _plot_density_term(obj: DensityTerm, /, **kwargs: Any) -> Any:
    r"""Plot envelopes referenced by a :class:`DensityTerm`.

    The plotting routine visualizes the modes appearing in the left and
    right monomials of the density term. The figure uses a two-column
    layout, with the left monomial in the first column and the right
    monomial in the second.

    Parameters
    ----------
    obj:
        Density term to visualize.
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
    The layout consists of two columns:

    - left monomial modes in the first column
    - right monomial modes in the second column

    Each mode contributes two vertically stacked plots:
    :math:`|\\zeta(t)|` and :math:`|Z(\\omega)|`.

    If both monomials contain no modes, a minimal placeholder figure
    labeled ``"identity density term"`` is returned.

    """
    n_time = int(kwargs.pop("n_time", 2048))
    n_freq = int(kwargs.pop("n_freq", 2048))
    t_span_sigma = float(kwargs.pop("t_span_sigma", 6.0))
    w_span_sigma = float(kwargs.pop("w_span_sigma", 12.0))
    title = kwargs.pop("title", None)
    return_axes = bool(kwargs.pop("return_axes", False))

    import matplotlib.pyplot as plt

    left_modes = obj.left.mode_ops
    right_modes = obj.right.mode_ops

    nL = len(left_modes)
    nR = len(right_modes)
    n = max(nL, nR)

    if n == 0:
        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "identity density term", ha="center", va="center")
        ax.set_axis_off()
        return (fig, np.array([ax], dtype=object)) if return_axes else fig

    if title is None:
        s = ""
        try:
            s = latex(obj, **{})
        except Exception:
            s = ""
        title = s if s else None

    fig, axes = plt.subplots(
        nrows=2 * n,
        ncols=2,
        figsize=(12, max(3, 2 * n * 2.2)),
        sharex=False,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]], dtype=object)

    if title:
        fig.suptitle(r"$" + title + r"$")

    # Flatten each column as (2*n,) alternating time/freq per mode slot.
    axes_left = axes[:, 0].reshape(-1)
    axes_right = axes[:, 1].reshape(-1)

    # If one side has fewer modes, hide the unused axes slots.
    for i in range(nL, n):
        axes_left[2 * i + 0].set_axis_off()
        axes_left[2 * i + 1].set_axis_off()
    for i in range(nR, n):
        axes_right[2 * i + 0].set_axis_off()
        axes_right[2 * i + 1].set_axis_off()

    _draw_modes_on_axes(
        modes=left_modes,
        axes_col=axes_left,
        n_time=n_time,
        n_freq=n_freq,
        t_span_sigma=t_span_sigma,
        w_span_sigma=w_span_sigma,
        col_title="L",
    )
    _draw_modes_on_axes(
        modes=right_modes,
        axes_col=axes_right,
        n_time=n_time,
        n_freq=n_freq,
        t_span_sigma=t_span_sigma,
        w_span_sigma=w_span_sigma,
        col_title="R",
    )

    fig.tight_layout()
    return (fig, axes) if return_axes else fig
