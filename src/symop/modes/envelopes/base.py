"""Base envelope abstraction.

This module defines the abstract base class for mode envelopes.
An envelope represents a complex time-domain field and provides:

- Time-domain evaluation.
- Frequency-domain evaluation.
- Overlap computation.
- Heuristic center/scale estimation.
- Plotting utilities.

Concrete envelope types must implement time and/or frequency
evaluation consistently with the package Fourier convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from symop.core.protocols.signature import SignatureProto
from symop.modes.protocols import (
    EnvelopeProto,
    HasLatex,
    SupportsOverlapWithGeneric,
)
from symop.modes.types import (
    AxesLike,
    FloatArray,
    PlotReturn,
    RCArray,
    TimeFunc,
    coerce_axes_array,
)

plt: ModuleType | None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if TYPE_CHECKING:
    from matplotlib.figure import Figure, FigureBase


def _overlap_numeric(
    f1: TimeFunc,
    f2: TimeFunc,
    *,
    tmin: float,
    tmax: float,
    n: int = 2**16,
) -> complex:
    r"""Numerically approximate the overlap (inner product) between two time functions.

    This approximates the integral

    .. math::

        \langle f_1, f_2 \rangle
        \;=\;
        \int_{t_\mathrm{min}}^{t_\mathrm{max}} f_1(t)^*\,f_2(t)\,dt

    on a uniform grid using the trapezoidal rule.

    Parameters
    ----------
    f1, f2:
        Callables mapping a real-valued time grid to real-or-complex samples.
    tmin, tmax:
        Integration limits.
    n:
        Number of grid points used in the uniform grid (default: 2**16).

    Returns
    -------
    complex
        Approximation to the overlap integral.

    Raises
    ------
    ValueError
        If either function returns non-finite values on the grid.

    """
    t: FloatArray = np.linspace(float(tmin), float(tmax), int(n), dtype=float)
    y1: RCArray = f1(t)
    y2: RCArray = f2(t)

    if not np.isfinite(y1).all():
        bad = np.argwhere(~np.isfinite(y1))
        i = int(bad[0, 0])
        raise ValueError(f"Non-finite values from f1 at t={t[i]}: {y1[i]!r}")
    if not np.isfinite(y2).all():
        bad = np.argwhere(~np.isfinite(y2))
        i = int(bad[0, 0])
        raise ValueError(f"Non-finite values from f2 at t={t[i]}: {y2[i]!r}")

    y = cast(RCArray, np.conjugate(y1) * y2)
    return complex(np.trapezoid(y, t))


@dataclass(frozen=True)
class BaseEnvelope(EnvelopeProto, ABC):
    """Abstract base class for time/frequency envelopes.

    Subclasses must implement:
    - time_eval: evaluate the complex field on a time grid
    - freq_eval: evaluate the complex spectrum on a frequency grid
    - delayed: return a time-shifted copy
    - phased: return a phase-shifted copy
    - signature: stable identifier used for caching / comparisons
    - approx_signature: rounded or approximate identifier for grouping

    This base class provides:
    - a generic overlap implementation based on numeric quadrature
    - plotting utilities (requires matplotlib)

    Presentation:
    Objects may optionally implement :class:`symop.modes.protocols.HasLatex` to
    provide a LaTeX (mathtext) string for display in plots. If unavailable,
    plotting uses a readable textual fallback.
    """

    @abstractmethod
    def time_eval(self, t: FloatArray) -> RCArray:
        """Evaluate the envelope in the time domain on a grid of times t."""
        raise NotImplementedError

    @abstractmethod
    def freq_eval(self, w: FloatArray) -> RCArray:
        """Evaluate the envelope in the frequency domain on a grid of frequencies w."""
        raise NotImplementedError

    @abstractmethod
    def delayed(self, dt: float) -> BaseEnvelope:
        """Return a copy delayed by dt in time."""
        raise NotImplementedError

    @abstractmethod
    def phased(self, dphi: float) -> BaseEnvelope:
        """Return a copy with an added global phase dphi."""
        raise NotImplementedError

    @property
    @abstractmethod
    def signature(self) -> SignatureProto:
        """Return a stable signature for this envelope."""
        raise NotImplementedError

    @abstractmethod
    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> tuple[Any, ...]:
        """Return an approximate signature.

        Implementations typically round floating parameters to a specified number
        of decimals and/or ignore non-essential parameters (e.g., global phase),
        depending on keyword arguments.
        """
        raise NotImplementedError

    def center_and_scale(self) -> tuple[float, float]:
        """Heuristic center and scale for choosing a plotting / overlap window.

        Returns
        -------
        center:
            Center time.
        scale:
            Characteristic scale

        """
        return 0.0, 1.0

    def overlap(self, other: EnvelopeProto) -> complex:
        """Compute the overlap (inner product) with another envelope.

        If either envelope implements :class:`SupportsOverlapWithGeneric`, that
        hook is used. Otherwise, a numeric quadrature is performed over a window
        selected from center_and_scale().

        Parameters
        ----------
        other:
            The envelope to overlap with.

        Returns
        -------
        complex
            The overlap value.

        """
        if isinstance(other, SupportsOverlapWithGeneric):
            return other.overlap_with_generic(self)

        if isinstance(self, SupportsOverlapWithGeneric):
            return self.overlap_with_generic(other)

        c1, s1 = self.center_and_scale()
        c2, s2 = other.center_and_scale()
        c = 0.5 * (c1 + c2)
        S = max(s1, s2)
        T = 8.0 * S
        return _overlap_numeric(self.time_eval, other.time_eval, tmin=c - T, tmax=c + T)

    def plot(
        self,
        *,
        t: FloatArray | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        n: int = 2000,
        show_real_imag: bool = True,
        show_phase: bool = False,
        show_formula: bool = True,
        title: str | None = None,
        axes: AxesLike | None = None,
        label: str | None = None,
        normalize_envelope: bool = False,
    ) -> PlotReturn:
        """Plot the envelope in the time domain.

        If no grid is provided, a symmetric window is chosen from
        center_and_scale() as [c - 6*S, c + 6*S] and sampled with n points.

        If show_formula is True and this envelope implements HasLatex, the
        provided LaTeX is rendered in a header panel. If no LaTeX is available,
        a readable text fallback is shown instead.

        Parameters
        ----------
        t:
            Optional explicit time grid. If provided, tmin/tmax/n are ignored.
        tmin, tmax:
            Optional bounds used to build the time grid if t is not provided.
        n:
            Number of samples used when generating the grid.
        show_real_imag:
            If True, plot real and imaginary parts in a separate panel.
        show_phase:
            If True, plot unwrapped phase in a separate panel.
        show_formula:
            If True, show a header panel (LaTeX if available, otherwise fallback text).
        title:
            Optional figure title.
        axes:
            Optional existing axes to draw into.
        label:
            Optional label used for legends and fallback header.
        normalize_envelope:
            If True, normalize so max_t |field(t)| = 1 before plotting.

        Returns
        -------
        (FigureBase, ndarray)
            A tuple (fig, axs) where axs is a 1-D object array of axes.

        """
        if plt is None:
            raise RuntimeError("matplotlib is required for BaseEnvelope.plot().")

        if t is None:
            if tmin is None or tmax is None:
                c, S = self.center_and_scale()
                T = 6.0 * S
                tmin = c - T if tmin is None else tmin
                tmax = c + T if tmax is None else tmax
            t = np.linspace(float(tmin), float(tmax), int(n))

        y: RCArray = self.time_eval(t)
        amp = np.abs(y)
        if normalize_envelope and amp.max() > 0:
            y = y / amp.max()
            amp = amp / amp.max()

        n_rows = 1 + int(show_real_imag) + int(show_phase)
        header_rows = 1 if show_formula and axes is None else 0

        created_fig = False
        if axes is None:
            created_fig = True
            fig: FigureBase = plt.figure(
                figsize=(8, 2.2 * (n_rows + 0.6 * header_rows))
            )
            gs = fig.add_gridspec(
                nrows=n_rows + header_rows,
                ncols=1,
                height_ratios=([0.22] if header_rows else []) + [1] * n_rows,
            )

            if header_rows:
                ax_head = fig.add_subplot(gs[0, 0])
                ax_head.axis("off")

                latex = self.latex if isinstance(self, HasLatex) else None
                if latex:
                    head_text = "$" + latex + "$"
                else:
                    name = (
                        label
                        or getattr(self, "user_label", None)
                        or type(self).__name__
                    )
                    try:
                        sig = self.signature
                        head_text = f"{name}\n" + f"sig: {sig!r}"
                    except Exception:
                        head_text = name

                ax_head.text(
                    0.01,
                    0.60,
                    head_text,
                    transform=ax_head.transAxes,
                    ha="left",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.25", alpha=0.08, lw=0.0),
                )

            data_axes = [fig.add_subplot(gs[header_rows + i, 0]) for i in range(n_rows)]
            axs = np.asarray(data_axes, dtype=object)
        else:
            fig, axs = coerce_axes_array(axes)
            if len(axs) != n_rows:
                raise ValueError(f"Expected {n_rows} axes, got {len(axs)}.")

        lab = label or getattr(self, "user_label", None) or type(self).__name__

        r = 0
        axs[r].plot(t, amp, label="$|\\zeta(t)|$" + " -- " + str(lab))
        axs[r].set_ylabel("envelope")
        axs[r].legend()
        r += 1

        if show_real_imag:
            y_re: FloatArray = np.real(y)
            y_im: FloatArray = np.imag(y)
            axs[r].plot(t, y_re, label="$\\Re\\,\\zeta(t)$" + " -- " + str(lab))
            axs[r].plot(
                t,
                y_im,
                label="$\\Im\\,\\zeta(t)$" + " -- " + str(lab),
                linestyle="--",
            )
            axs[r].set_ylabel("field")
            axs[r].legend()
            r += 1

        if show_phase:
            axs[r].plot(t, np.unwrap(np.angle(y)), label="phase -- " + str(lab))
            axs[r].set_ylabel("[rad]")
            axs[r].legend()
            axs[r].set_xlabel("time t")

        if created_fig:
            f = cast("Figure", fig)
            if title:
                f.suptitle(title, y=0.995)
                f.tight_layout(rect=(0, 0, 1, 0.98))
            else:
                f.tight_layout()

        return fig, axs

    @staticmethod
    def plot_many(
        envelopes: Sequence[BaseEnvelope],
        *,
        t: FloatArray | None = None,
        n: int = 2000,
        show_real_imag: bool = True,
        show_phase: bool = False,
        show_formula: bool = True,
        title: str | None = None,
        labels: Sequence[str | None] | None = None,
        normalize_envelope: bool = False,
        share_window: bool = True,
        span_sigma: float = 6.0,
        axes: AxesLike | None = None,
    ) -> PlotReturn:
        r"""Plot multiple envelopes on the same axes.

        This calls plot() for the first envelope (creating the figure/axes unless
        axes is provided) and overlays the remaining envelopes onto the same axes.

        If no explicit grid t is provided and share_window is True, a common window
        is chosen using each envelope's center_and_scale() heuristic:

        .. math::

            c = \frac{1}{N}\sum_i c_i, \qquad
            S = \max_i s_i, \qquad
            T = \text{span\_sigma}\cdot S.

        Parameters
        ----------
        envelopes:
            Sequence of envelopes to plot (must be non-empty).
        t:
            Optional explicit common time grid. If provided, n/share_window/span_sigma are ignored.
        n:
            Number of samples used when generating the grid.
        show_real_imag:
            If True, plot real and imaginary parts (second panel).
        show_phase:
            If True, plot unwrapped phase (third panel).
        show_formula:
            If True, render a header panel for the first envelope (LaTeX if available).
        title:
            Optional figure title (applied when a new figure is created).
        labels:
            Optional labels, one per envelope.
        normalize_envelope:
            If True, normalize each envelope so max_t |field(t)| = 1 before plotting.
        share_window:
            If True and t is not provided, build a common window for all envelopes.
        span_sigma:
            Controls the half-width of the shared window.
        axes:
            Optional existing axes to draw into.

        Returns
        -------
        (FigureBase, ndarray)
            A tuple (fig, axs) where axs is a 1-D object array of axes.

        """
        if not envelopes:
            raise ValueError("No envelopes provided to plot_many().")

        if t is None and share_window:
            centers, scales = zip(
                *(e.center_and_scale() for e in envelopes), strict=True
            )
            c = float(np.mean(centers))
            S = float(np.max(scales))
            T = span_sigma * S
            t = np.linspace(c - T, c + T, int(n))

        fig, axs = envelopes[0].plot(
            t=t,
            n=n,
            show_real_imag=show_real_imag,
            show_phase=show_phase,
            show_formula=show_formula,
            title=title,
            axes=axes,
            label=(labels[0] if labels else None),
            normalize_envelope=normalize_envelope,
        )

        for i, env in enumerate(envelopes[1:], start=1):
            env.plot(
                t=t,
                n=n,
                show_real_imag=show_real_imag,
                show_phase=show_phase,
                show_formula=False,
                title=None,
                axes=axs,
                label=(labels[i] if labels and i < len(labels) else None),
                normalize_envelope=normalize_envelope,
            )

        return fig, axs
