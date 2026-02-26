from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.figure import Figure, FigureBase
except Exception:
    plt = None  # type: ignore
    from typing import Any as Figure
    from typing import Any as FigureBase

from symop_proto.core.protocols import (
    SignatureProto,
    TimeEvaluable,
)
from symop_proto.envelopes.protocols import (
    AxesLike,
    EnvelopeProto,
    FloatArray,
    PlotReturn,
    RCArray,
    SupportsOverlapWithGeneric,
    TimeFunc,
    _coerce_axes_array,
)


def _overlap_numeric(
    f1: TimeFunc,
    f2: TimeFunc,
    *,
    tmin: float,
    tmax: float,
    n: int = 2**16,
) -> complex:
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

    y: RCArray = np.conjugate(y1) * y2
    return complex(np.trapezoid(y, t))


@dataclass(frozen=True)
class BaseEnvelope(EnvelopeProto, ABC):
    """AbestractBaseClass to model arbitrary envelope"""

    @abstractmethod
    def time_eval(self, t: FloatArray) -> RCArray: ...

    @abstractmethod
    def freq_eval(self, w: FloatArray) -> RCArray: ...

    @abstractmethod
    def delayed(self, dt: float) -> BaseEnvelope: ...

    @abstractmethod
    def phased(self, dphi: float) -> BaseEnvelope: ...

    @property
    @abstractmethod
    def signature(self) -> SignatureProto: ...

    @abstractmethod
    def approx_signature(self, **kw: Any) -> tuple[Any, ...]: ...

    def center_and_scale(self) -> tuple[float, float]:
        """Default Center and scale

        Returns
        -------
        tuple
            Two values (``center`` and ``scale``).

        """
        return 0.0, 1.0

    def overlap(self, other: EnvelopeProto) -> complex:
        if isinstance(other, SupportsOverlapWithGeneric):
            return other.overlap_with_generic(self)

        if isinstance(self, SupportsOverlapWithGeneric):
            return self.overlap_with_generic(other)

        c1, s1 = self.center_and_scale()
        if isinstance(other, TimeEvaluable):
            c2, s2 = other.center_and_scale()
            c = 0.5 * (c1 + c2)
            S = max(s1, s2)
            T = 8.0 * S
            return _overlap_numeric(
                self.time_eval, other.time_eval, tmin=c - T, tmax=c + T
            )
        raise TypeError(
            "No overlap implementation between"
            f"{type(self).__name__} and {type(other).__name__}"
            "(other is not TimeEvaluable and provides no cross-family hook)"
        )

    @property
    def latex(self) -> str | None:
        """Return a LaTeX (mathtext) string for this envelope, or None if
        no closed-form is available. The string must be valid inside
        ``$ ... $`` (no surrounding dollar signs).
        """
        return None

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
        show_parts: bool = False,
    ) -> PlotReturn:
        """Quick visualization of this envelope.

        If no grid is provided, a symmetric window is chosen from
        :meth:`center_and_scale` as ``[c-6S, c+6S]`` and sampled with ``n``
        points.

        Args:
            t : Optional precomputed time grid. If given, ``tmin``, ``tmax``, ``n``
                are ignored.
            tmin, tmax : Optional explicit window bounds used to build ``t``.
            n : Number of samples when generating the grid (default: 2000).
            show_real_imag : Plot real/imaginary parts of the complex field
            (default: ``True``).
            show_phase : Plot unwrapped phase (default: ``True``).
            show_formula : If True and :meth:`latex_expression` returns a string,
                render it in the figure using Matplotlib mathtext.
            title : Optional figure title.

        Returns:
            Matplotlib figure and axes array (data).

        Examples:
        --------

        .. jupyter-execute::

            import matplotlib.pyplot as plt
            from symop_proto.envelopes.gaussian_envelope import (
                GaussianEnvelope,
            )

            env = GaussianEnvelope(omega0=25.0, sigma=1.0, tau=0.10, phi0=0.4)
            fig, axs = env.plot(show_formula=True, title="Gaussian envelope")

        """
        if plt is None:
            raise RuntimeError("matplotlib is required for BaseEnvelope.plot().")

        # grid
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

        # Prepare figure and axes
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
                tex = self.latex
                if tex:
                    ax_head.text(
                        0.01,
                        0.60,
                        f"${tex}$",
                        transform=ax_head.transAxes,
                        ha="left",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.25", alpha=0.08, lw=0.0),
                    )
            data_axes = [fig.add_subplot(gs[header_rows + i, 0]) for i in range(n_rows)]
            axs = np.asarray(data_axes, dtype=object)
        else:
            fig, axs = _coerce_axes_array(axes)
            if len(axs) != n_rows:
                raise ValueError(f"Expected {n_rows} axes, got {len(axs)}.")

        # Labels
        l = label or getattr(self, "user_label", None) or type(self).__name__

        # |zeta|
        r = 0
        axs[r].plot(t, amp, label=rf"$|\zeta(t)|$ -- {l}")
        axs[r].set_ylabel("envelope")
        axs[r].legend()
        r += 1

        if show_real_imag:
            y_re: FloatArray = np.real(y)
            y_im: FloatArray = np.imag(y)
            axs[r].plot(t, y_re, label=rf"$\Re\,\zeta(t)$ -- {l}")
            axs[r].plot(t, y_im, label=rf"$\Im\,\zeta(t)$ -- {l}", linestyle="--")
            axs[r].set_ylabel("field")
            axs[r].legend()
            r += 1

        # Phase
        if show_phase:
            axs[r].plot(t, np.unwrap(np.angle(y)), label=f"phase -- {l}")
            axs[r].set_ylabel("[rad]")
            axs[r].legend()
            axs[r].set_xlabel("time t")

        if created_fig:
            f = cast(Figure, fig)
            if title:
                f.suptitle(title, y=0.995)
                f.tight_layout(rect=(0, 0, 1, 0.98))
            else:
                f.tight_layout()

        return fig, axs

    @staticmethod
    def plot_many(
        envelopes: Sequence[EnvelopeProto],
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
        r"""Plot multiple envelopes on the **same** axes.

        This is a convenience wrapper that calls :meth:`BaseEnvelope.plot` for the
        first envelope (creating the figure/axes unless ``axes=`` is provided) and
        then overlays the remaining envelopes into the same axes.

        If no explicit grid ``t`` is provided and ``share_window=True``, a common
        symmetric window is chosen from the envelopes' heuristics:

        .. math::

            c = \frac{1}{N}\sum_i c_i, \qquad
            S = \max_i s_i, \qquad
            T = \text{span\_sigma}\cdot S,

        and the time grid is sampled uniformly on :math:`[c-T,\,c+T]` with ``n``
        points. If you pass an explicit ``t``, it is reused for all envelopes.

        Parameters
        ----------
        envelopes :
            Sequence of envelopes to plot. Must be non-empty.
        t :
            Optional precomputed common time grid. If given, ``n``, ``share_window``
            and ``span_sigma`` are ignored.
        n :
            Number of samples when generating the grid (default: 2000).
        show_real_imag :
            Plot real/imaginary parts (second panel) for each overlay (default: True).
        show_phase :
            Plot unwrapped phase (third panel) for each overlay (default: False).
        show_formula :
            If True, and the first envelope's :pyattr:`latex` property returns a string,
            render it in a small header panel above the data (only for the first call
            that creates the figure).
        title :
            Optional figure title (only applied when a new figure is created).
        labels :
            Optional sequence of labels, one per envelope. If omitted or shorter than
            ``len(envelopes)``, missing labels default to ``env.user_label`` if present,
            otherwise the class name.
        normalize_envelope :
            If True, normalize each envelope so that :math:`\max_t |\zeta(t)| = 1`
            before plotting (applies to all overlays).
        share_window :
            If True (default) and no ``t`` is provided, use a common window built from
            the envelopes' center/scale heuristics. If False, the first envelope's
            window is used to build ``t``.
        span_sigma :
            Controls the half-width ``T`` of the shared window when ``share_window=True``
            (default: 6.0).
        axes :
            Optional existing axes (1D array-like of :class:`matplotlib.axes.Axes`)
            to draw into. If omitted, a new figure and axes are created.

        Returns
        -------
        (Figure, ndarray)
            A tuple ``(fig, axs)`` where ``fig`` is the Matplotlib figure and
            ``axs`` is a 1-D object array of axes. The array length is
            ``1 + int(show_real_imag) + int(show_phase)``.

        Examples
        --------

        .. jupyter-execute::

            import numpy as np
            import matplotlib.pyplot as plt
            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.envelopes.base import BaseEnvelope

            # Three Gaussian envelopes with different delays/phases
            g1 = GaussianEnvelope(omega0=25.0, sigma=0.35, tau=0.00, phi0=0.0)
            g2 = GaussianEnvelope(omega0=25.0, sigma=0.35, tau=0.20, phi0=0.3)
            g3 = GaussianEnvelope(omega0=25.0, sigma=0.35, tau=-0.15, phi0=-0.2)

            fig, axs = BaseEnvelope.plot_many(
                [g1, g2, g3],
                labels=["ref", "delayed+phase", "advanced"],
                show_real_imag=True,
                show_phase=False,
                show_formula=True,
                normalize_envelope=True,
                title="Three Gaussian envelopes on a shared grid"
            )

        """
        if not envelopes:
            raise ValueError("No envelopes provided to plot_many().")

        if t is None and share_window:
            centers, scales = zip(
                *(e.center_and_scale() for e in envelopes), strict=False
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
