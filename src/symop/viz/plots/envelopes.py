r"""Plotting for Gaussian-based temporal envelopes.

This module provides plotting implementations for
:class:`GaussianEnvelope` and :class:`GaussianMixtureEnvelope`.

The visualization includes:

- time-domain magnitude :math:`|\zeta(t)|`
- optional real/imaginary components
- optional phase (masked where amplitude is small)
- frequency-domain magnitude :math:`|Z(\omega)|`
- optional spectral components and phase

The plotting is configurable and adapts automatically to envelope
parameters such as center, scale, and spectral width.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from symop.core.types.arrays import RCArray
from symop.modes.envelopes.base import BaseEnvelope
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope
from symop.modes.types import FloatArray
from symop.viz._dispatch import latex as latex_fn
from symop.viz._dispatch import plot
from symop.viz._dispatch import text as text_fn
from symop.viz._optional import require_matplotlib_pyplot


def _finite_float(x: Any, default: float) -> float:
    """Safely convert a value to a finite float.

    Parameters
    ----------
    x:
        Value to convert.
    default:
        Value returned if conversion fails or the result is not finite.

    Returns
    -------
    float
        Finite float value.

    """
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _env_center_scale(env: Any) -> tuple[float, float]:
    """Extract a robust center and scale from an envelope.

    Parameters
    ----------
    env:
        Envelope-like object expected to provide ``center_and_scale``.

    Returns
    -------
    tuple of float
        Pair ``(center, scale)`` with guaranteed finite values.

    Notes
    -----
    Falls back to ``(0.0, 1.0)`` if extraction fails.

    """
    try:
        c, s = env.center_and_scale()
        c = _finite_float(c, 0.0)
        s = max(_finite_float(s, 1.0), 1e-12)
        return c, s
    except Exception:
        return 0.0, 1.0


def _iter_components(env: Any) -> Iterable[Any]:
    """Return components of a mixture envelope.

    Parameters
    ----------
    env:
        Envelope-like object that may expose a ``components`` attribute.

    Returns
    -------
    Iterable
        Tuple of components if present, otherwise an empty iterable.

    """
    comps = getattr(env, "components", None)
    if isinstance(comps, tuple):
        return comps
    return ()


def _auto_time_window(env: Any, *, n: int, margin_sigma: float) -> np.ndarray:
    """Generate a time grid adapted to an envelope.

    Parameters
    ----------
    env:
        Envelope-like object.
    n:
        Number of time samples.
    margin_sigma:
        Window half-width in units of the envelope scale.

    Returns
    -------
    ndarray
        Time grid.

    Notes
    -----
    For mixtures, the window spans all component centers with margins.
    Otherwise, it is centered around the envelope's main support.

    """
    c, s = _env_center_scale(env)

    comps = list(_iter_components(env))
    if comps:
        taus = []
        sigmas = []
        for g in comps:
            taus.append(_finite_float(getattr(g, "tau", c), c))
            sigmas.append(max(_finite_float(getattr(g, "sigma", s), s), 1e-12))
        t_min = float(min(taus)) - float(margin_sigma) * float(max(sigmas))
        t_max = float(max(taus)) + float(margin_sigma) * float(max(sigmas))
        if t_max <= t_min:
            t_min, t_max = c - margin_sigma * s, c + margin_sigma * s
        return np.linspace(t_min, t_max, int(n), dtype=float)

    t_min = c - margin_sigma * s
    t_max = c + margin_sigma * s
    return np.linspace(float(t_min), float(t_max), int(n), dtype=float)


def _auto_freq_center_and_width(env: Any) -> tuple[float, float]:
    """Estimate frequency center and width for an envelope.

    Parameters
    ----------
    env:
        Envelope-like object.

    Returns
    -------
    tuple of float
        Pair ``(omega0, omega_sigma)``.

    Notes
    -----
    - Uses ``omega0`` and ``omega_sigma`` if available.
    - Falls back to inverse time scale.
    - Expands window for mixtures to cover all component carriers.

    """
    c, s = _env_center_scale(env)
    w0 = _finite_float(getattr(env, "omega0", 0.0), 0.0)

    ws = getattr(env, "omega_sigma", None)
    if ws is None:
        ws = 1.0 / max(s, 1e-12)
    w_sigma = max(_finite_float(ws, 1.0 / max(s, 1e-12)), 1e-12)

    # For mixtures, expand to cover all component carrier centers.
    comps = list(_iter_components(env))
    if comps:
        w0s = []
        wss = []
        for g in comps:
            w0s.append(_finite_float(getattr(g, "omega0", w0), w0))
            ws_g = getattr(g, "omega_sigma", None)
            if ws_g is None:
                sig_g = max(_finite_float(getattr(g, "sigma", s), s), 1e-12)
                ws_g = 1.0 / sig_g
            wss.append(max(_finite_float(ws_g, w_sigma), 1e-12))

        w0_center = float(np.mean(w0s))
        w_sigma = float(max(wss))
        w0_span = 0.5 * float(max(w0s) - min(w0s))
        # Add the carrier span as extra width.
        return w0_center, float(w_sigma + w0_span)

    return w0, w_sigma


def _auto_freq_grid(env: Any, *, n: int, margin_sigma: float) -> np.ndarray:
    """Generate a frequency grid adapted to an envelope.

    Parameters
    ----------
    env:
        Envelope-like object.
    n:
        Number of frequency samples.
    margin_sigma:
        Window half-width in units of frequency scale.

    Returns
    -------
    ndarray
        Frequency grid.

    """
    w0, w_sigma = _auto_freq_center_and_width(env)
    W = float(margin_sigma) * float(w_sigma)
    W = max(W, 1e-12)
    return np.linspace(w0 - W, w0 + W, int(n), dtype=float)


def _mask_phase(y: np.ndarray, *, frac: float) -> np.ndarray:
    """Mask phase where amplitude is negligible.

    Parameters
    ----------
    y:
        Complex signal values.
    frac:
        Relative amplitude threshold for masking.

    Returns
    -------
    ndarray
        Phase array with low-amplitude regions set to NaN.

    Notes
    -----
    This avoids meaningless phase values and unwrap artifacts.

    """
    amp = np.abs(y)
    if amp.size == 0:
        return np.array([], dtype=float)
    thr = float(np.max(amp)) * float(frac)
    phase = np.unwrap(np.angle(y))
    phase = phase.astype(float, copy=False)
    phase[amp < thr] = np.nan
    return cast(NDArray[np.float64], phase)


def _header_text(obj: Any, *, title: str | None) -> str | None:
    """Construct a figure header string.

    Parameters
    ----------
    obj:
        Object to render via LaTeX or text.
    title:
        Optional user-provided title.

    Returns
    -------
    str or None
        Combined title and LaTeX/text representation.

    """
    s = latex_fn(obj)
    if s:
        if title:
            return str(title) + "\n" + "$" + s + "$"
        return "$" + s + "$"
    if title:
        return str(title) + "\n" + text_fn(obj)
    return None


def _plot_env_impl(
    env: Any,
    *,
    t: FloatArray | None,
    w: FloatArray | None,
    n: int,
    nw: int,
    time_margin_sigma: float,
    freq_margin_sigma: float,
    show_freq: bool,
    show_real_imag: bool,
    show_phase: bool,
    show_freq_real_imag: bool,
    show_freq_phase: bool,
    phase_mask_frac: float,
    normalize_envelope: bool,
    normalize_spectrum: bool,
    freq_relative: bool,
    title: str | None,
    label: str | None,
    show: bool,
) -> Any:
    """Plot a temporal envelope in time and frequency domains.

    Parameters
    ----------
    env:
        Envelope object providing ``time_eval`` and ``freq_eval``.
    t, w:
        Optional user-supplied grids. If ``None``, grids are generated
        automatically.
    n, nw:
        Number of time and frequency samples.
    time_margin_sigma, freq_margin_sigma:
        Window half-widths in units of envelope scale.
    show_freq:
        If True, include frequency-domain plots.
    show_real_imag:
        If True, show real and imaginary parts in time domain.
    show_phase:
        If True, show phase in time domain.
    show_freq_real_imag:
        If True, show real and imaginary parts in frequency domain.
    show_freq_phase:
        If True, show phase in frequency domain.
    phase_mask_frac:
        Threshold for masking phase in low-amplitude regions.
    normalize_envelope:
        Normalize time-domain magnitude to 1.
    normalize_spectrum:
        Normalize frequency-domain magnitude to 1.
    freq_relative:
        If True, shift frequency axis to be relative to carrier.
    title:
        Optional figure title.
    label:
        Label used in legends.
    show:
        If True, display the plot immediately.

    Returns
    -------
    tuple
        ``(figure, axes)`` where axes is a 1D array of matplotlib axes.

    Notes
    -----
    The layout is stacked vertically:

    - time-domain magnitude
    - optional time-domain components
    - optional time-domain phase
    - frequency-domain magnitude (if enabled)
    - optional frequency components and phase

    """
    plt = require_matplotlib_pyplot()

    # Time grid
    if t is None:
        t_grid = _auto_time_window(env, n=int(n), margin_sigma=float(time_margin_sigma))
    else:
        t_grid = np.asarray(t, dtype=float)

    y = np.asarray(env.time_eval(t_grid), dtype=complex)
    y_amp = np.abs(y)

    if normalize_envelope and y_amp.size > 0:
        m = float(np.max(y_amp))
        if m > 0.0 and np.isfinite(m):
            y = cast(RCArray, y / m)
            y_amp = y_amp / m

    # Freq grid
    z = None
    w_grid = None
    z_amp = None
    w0_center = None

    if show_freq:
        if w is None:
            w_grid = _auto_freq_grid(
                env, n=int(nw), margin_sigma=float(freq_margin_sigma)
            )
        else:
            w_grid = np.asarray(w, dtype=float)

        z = np.asarray(env.freq_eval(w_grid), dtype=complex)
        z_amp = np.abs(z)

        if normalize_spectrum and z_amp.size > 0:
            m = float(np.max(z_amp))
            if m > 0.0 and np.isfinite(m):
                z = cast(RCArray, z / m)
                z_amp = z_amp / m

        if freq_relative:
            w0_center, _ = _auto_freq_center_and_width(env)
            w_grid = w_grid - float(w0_center)

    lab = label or getattr(env, "user_label", None) or type(env).__name__
    head = _header_text(env, title=title)

    n_time_rows = 1 + int(show_real_imag) + int(show_phase)
    n_freq_rows = 0
    if show_freq:
        n_freq_rows = 1 + int(show_freq_real_imag) + int(show_freq_phase)
    nrows = n_time_rows + n_freq_rows

    fig = plt.figure(figsize=(8, 2.4 * nrows))
    axs = fig.subplots(nrows=nrows, ncols=1, squeeze=False)
    ax_list = axs[:, 0]

    if head:
        fig.suptitle(head, y=0.995)

    r = 0

    # |zeta(t)|
    ax_list[r].plot(t_grid, y_amp, label=r"$|\zeta(t)|$" + " -- " + str(lab))
    ax_list[r].set_ylabel(r"$|\zeta(t)|$")
    ax_list[r].legend()
    r += 1

    # Re/Im zeta(t)
    if show_real_imag:
        ax_list[r].plot(
            t_grid, np.real(y), label=r"$\Re\,\zeta(t)$" + " -- " + str(lab)
        )
        ax_list[r].plot(
            t_grid,
            np.imag(y),
            label=r"$\Im\,\zeta(t)$" + " -- " + str(lab),
            linestyle="--",
        )
        ax_list[r].set_ylabel(r"$\zeta(t)$")
        ax_list[r].legend()
        r += 1

    # arg zeta(t)
    if show_phase:
        ph = _mask_phase(np.asarray(y, dtype=complex), frac=float(phase_mask_frac))
        ax_list[r].plot(t_grid, ph, label=r"$\arg\zeta(t)$" + " -- " + str(lab))
        ax_list[r].set_ylabel("[rad]")
        ax_list[r].legend()
        r += 1

    # Time x-label
    if show_freq:
        ax_list[r - 1].set_xlabel(r"$t$")
    else:
        ax_list[-1].set_xlabel(r"$t$")

    # Frequency panels
    if show_freq:
        assert w_grid is not None and z is not None and z_amp is not None

        w_label = r"$\Omega$" if freq_relative else r"$\omega$"

        ax_list[r].plot(w_grid, z_amp, label=r"$|Z(\omega)|$" + " -- " + str(lab))
        ax_list[r].set_ylabel(r"$|Z(\omega)|$")
        ax_list[r].legend()
        r += 1

        if show_freq_real_imag:
            ax_list[r].plot(
                w_grid,
                np.real(z),
                label=r"$\Re\,Z(\omega)$" + " -- " + str(lab),
            )
            ax_list[r].plot(
                w_grid,
                np.imag(z),
                label=r"$\Im\,Z(\omega)$" + " -- " + str(lab),
                linestyle="--",
            )
            ax_list[r].set_ylabel(r"$Z(\omega)$")
            ax_list[r].legend()
            r += 1

        if show_freq_phase:
            phw = _mask_phase(np.asarray(z, dtype=complex), frac=float(phase_mask_frac))
            ax_list[r].plot(w_grid, phw, label=r"$\arg Z(\omega)$" + " -- " + str(lab))
            ax_list[r].set_ylabel("[rad]")
            ax_list[r].legend()
            r += 1

        ax_list[-1].set_xlabel(w_label)

    if head:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()

    if show:
        plt.show()

    return fig, ax_list
@plot.register
def _(env: BaseEnvelope, /, **kwargs: Any) -> Any:
    """Plot a :class:`BaseEnvelope`.

    Parameters
    ----------
    env:
        Envelope to visualize.
    **kwargs:
        Plotting options forwarded to the internal implementation.

    Returns
    -------
    Any
        Plot result, typically ``(figure, axes)``.

    Notes
    -----
    This is the generic fallback for all envelope types that implement
    the base time/frequency interface. Specialized handlers may override
    this behavior for particular subclasses when a more tailored
    visualization is useful.
    """
    return _plot_env_impl(
        env,
        t=kwargs.pop("t", None),
        w=kwargs.pop("w", None),
        n=int(kwargs.pop("n", 2000)),
        nw=int(kwargs.pop("nw", 2000)),
        time_margin_sigma=float(kwargs.pop("time_margin_sigma", 6.0)),
        freq_margin_sigma=float(kwargs.pop("freq_margin_sigma", 6.0)),
        show_freq=bool(kwargs.pop("show_freq", True)),
        show_real_imag=bool(kwargs.pop("show_real_imag", True)),
        show_phase=bool(kwargs.pop("show_phase", False)),
        show_freq_real_imag=bool(kwargs.pop("show_freq_real_imag", False)),
        show_freq_phase=bool(kwargs.pop("show_freq_phase", False)),
        phase_mask_frac=float(kwargs.pop("phase_mask_frac", 1e-3)),
        normalize_envelope=bool(kwargs.pop("normalize_envelope", False)),
        normalize_spectrum=bool(kwargs.pop("normalize_spectrum", True)),
        freq_relative=bool(kwargs.pop("freq_relative", False)),
        title=kwargs.pop("title", None),
        label=kwargs.pop("label", None),
        show=bool(kwargs.pop("show", True)),
    )

@plot.register
def _(env: GaussianEnvelope, /, **kwargs: Any) -> Any:
    """Plot a :class:`GaussianEnvelope`.

    Parameters
    ----------
    env:
        Gaussian envelope to visualize.
    **kwargs:
        Plotting options forwarded to the internal implementation.

    Returns
    -------
    Any
        Plot result (typically ``(figure, axes)``).

    """
    return _plot_env_impl(
        env,
        t=kwargs.pop("t", None),
        w=kwargs.pop("w", None),
        n=int(kwargs.pop("n", 2000)),
        nw=int(kwargs.pop("nw", 2000)),
        time_margin_sigma=float(kwargs.pop("time_margin_sigma", 6.0)),
        freq_margin_sigma=float(kwargs.pop("freq_margin_sigma", 6.0)),
        show_freq=bool(kwargs.pop("show_freq", True)),
        show_real_imag=bool(kwargs.pop("show_real_imag", True)),
        show_phase=bool(kwargs.pop("show_phase", False)),
        show_freq_real_imag=bool(kwargs.pop("show_freq_real_imag", False)),
        show_freq_phase=bool(kwargs.pop("show_freq_phase", False)),
        phase_mask_frac=float(kwargs.pop("phase_mask_frac", 1e-3)),
        normalize_envelope=bool(kwargs.pop("normalize_envelope", False)),
        normalize_spectrum=bool(kwargs.pop("normalize_spectrum", True)),
        freq_relative=bool(kwargs.pop("freq_relative", False)),
        title=kwargs.pop("title", None),
        label=kwargs.pop("label", None),
        show=bool(kwargs.pop("show", True)),
    )


@plot.register
def _(env: GaussianMixtureEnvelope, /, **kwargs: Any) -> Any:
    """Plot a :class:`GaussianMixtureEnvelope`.

    Parameters
    ----------
    env:
        Gaussian mixture envelope to visualize.
    **kwargs:
        Plotting options forwarded to the internal implementation.

    Returns
    -------
    Any
        Plot result (typically ``(figure, axes)``).

    Notes
    -----
    The plotting window is automatically expanded to cover all mixture
    components.

    """
    return _plot_env_impl(
        env,
        t=kwargs.pop("t", None),
        w=kwargs.pop("w", None),
        n=int(kwargs.pop("n", 2000)),
        nw=int(kwargs.pop("nw", 2000)),
        time_margin_sigma=float(kwargs.pop("time_margin_sigma", 6.0)),
        freq_margin_sigma=float(kwargs.pop("freq_margin_sigma", 6.0)),
        show_freq=bool(kwargs.pop("show_freq", True)),
        show_real_imag=bool(kwargs.pop("show_real_imag", True)),
        show_phase=bool(kwargs.pop("show_phase", False)),
        show_freq_real_imag=bool(kwargs.pop("show_freq_real_imag", False)),
        show_freq_phase=bool(kwargs.pop("show_freq_phase", False)),
        phase_mask_frac=float(kwargs.pop("phase_mask_frac", 1e-3)),
        normalize_envelope=bool(kwargs.pop("normalize_envelope", False)),
        normalize_spectrum=bool(kwargs.pop("normalize_spectrum", True)),
        freq_relative=bool(kwargs.pop("freq_relative", False)),
        title=kwargs.pop("title", None),
        label=kwargs.pop("label", None),
        show=bool(kwargs.pop("show", True)),
    )
