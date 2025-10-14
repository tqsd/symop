from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

from symop_proto.core.protocols import (
    EnvelopeProto,
    SignatureProto,
    SupportsOverlapWIthGeneric,
    TimeEvaluable,
)

FloatArray = NDArray[np.floating]
RCArray = NDArray[np.floating] | NDArray[np.complexfloating]
TimeFunc = Callable[[FloatArray], RCArray]


def _overlap_numeric(
    f1: TimeFunc,
    f2: TimeFunc,
    *,
    tmin: float,
    tmax: float,
    n: int = 2**16,
) -> complex:
    r"""
    Numerically compute the overlap between two time-domain envelope functions.

    Parameters:

        f1, f2: Time-dependent envelope functions.
        tmin, tmax: Integration limits.
        n: int: Optional, Number sampling points (default: :math:`2^{16}`)

    Returns:
        Numerical estimate of the overlap integral.

    Mathematics:
    ------------
        This function approximates computes the integral:

        .. math::

            \int_{\text{tmin}}^{\text{tmax}}f_1^*(t) f_2(t) dt.

        The integral is approximated using the :func:`np.trapezoid()` function,
        which computes:

        .. math::

            \int_{t_{\min}}^{t_{\max}} y(t)\,\mathrm{d}t
            \;\approx\;
            \sum_{i=0}^{n-2} \frac{t_{i+1}-t_i}{2}\,\bigl(y_i + y_{i+1}\bigr)
            \;=\;
            \Delta t\left[\tfrac{1}{2}y_0 + \sum_{i=1}^{n-2} y_i +
            \tfrac{1}{2}y_{n-1}\right].

    Examples:
    ---------

    .. jupyter-execute::

        import numpy as np
        from IPython.display import display, HTML

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.envelopes.base import _overlap_numeric

        # Two similar envelopes (second is delayed and phase-shifted)
        e1 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        e2 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.30, phi0=0.20)

        # Same window heuristic used by BaseEnvelope.overlap:
        (c1, s1) = e1.center_and_scale()
        (c2, s2) = e2.center_and_scale()
        c  = 0.5 * (c1 + c2)
        S  = max(s1, s2)
        T  = 8.0 * S
        tmin, tmax = c - T, c + T

        ov = _overlap_numeric(
            e1.time_eval,
            e2.time_eval,
            tmin=tmin,
            tmax=tmax)
        display(HTML(f"<p>overlap = <code>{ov}</code> "
                f"( |overlap| = {abs(ov):.6g} )</p>"))

    """
    t: FloatArray = np.linspace(tmin, tmax, n, dtype=float)
    y: RCArray = np.conjugate(f1(t)) * f2(t)
    return complex(np.trapezoid(y, t))


@dataclass(frozen=True)
class BaseEnvelope(EnvelopeProto, ABC):
    """AbestractBaseClass to model arbitrary envelope"""

    @abstractmethod
    def time_eval(self, t: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def delayed(self, dt: float) -> BaseEnvelope: ...

    @abstractmethod
    def phased(self, dphi: float) -> BaseEnvelope: ...

    @property
    @abstractmethod
    def signature(self) -> SignatureProto: ...

    @abstractmethod
    def approx_signature(self, **kw: Any) -> Tuple[Any, ...]: ...

    def center_and_scale(self) -> Tuple[float, float]:
        """Default Center and scale

        Returns
        -------
        tuple
            Two values (``center`` and ``scale``).

        """
        return 0.0, 1.0

    def overlap(self, other: EnvelopeProto) -> complex:
        r"""Overlap with another envelope

        This computes the inner products :math:`\langle\zeta_1,\zeta_2\rangle`
        between this envelope and ``other``. The method tries the following in
        order:

        1) **Cross-family hook** - If ``other`` implements
           :class:`SupportsOverlapWIthGeneric`, delegate via
           ``other.overlap_with_generic``.

        2) **Numeric fallback** - If ``other`` is :class:`TimeEvaluable`,
           evaluate both envelopes on a shared uniform grid and integrate
           using the trapezoidal rule. Thegrid is chose from the envelopes'
           heuristic center/scale:

           .. math::

               c = \frac{1}{2} (c_1+c_2), \qquad
               S = \text{max}(s_1, s_2), \qquad
               T = 8S

           and the integral is taken over :math:`[c-T, c+T]`.

        Parameters
        ----------
        other: EnvelopeProto
            The envelope to overlap with.

        Returns
        -------
        complex:
            The overlap value

        Raises
        ------
        TypeError
            If no implementation is available.

        See Also
        --------
        :py:func:`~symop_proto.envelopes.base._overlap_numeric` :
            Numerical helper used for the fallback.
        """
        if isinstance(other, SupportsOverlapWIthGeneric):
            return other.overlap_with_generic(self)

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
            "(other is not TimeEvaluable and prodives no cross-family hook)"
        )

    def latex_expression(self) -> str | None:
        """
        Return a LaTeX (mathtext) string for this envelope, or None if
        no closed-form is available. The string must be valid inside
        ``$ ... $`` (no surrounding dollar signs).
        """
        return None

    def plot(
        self,
        *,
        t: Optional[FloatArray] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        n: int = 2000,
        show_real_imag: bool = True,
        show_phase: bool = False,
        show_formula: bool = True,
        title: Optional[str] = None,
    ) -> Tuple["plt.Figure", np.ndarray]:  # type: ignore[name-defined]
        """
        Quick visualization of this envelope.

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
        ---------

        .. jupyter-execute::

            import matplotlib.pyplot as plt
            from symop_proto.envelopes.gaussian_envelope import (
                GaussianEnvelope,
            )

            env = GaussianEnvelope(omega0=25.0, sigma=0.35, tau=0.10, phi0=0.4)
            fig, axs = env.plot(show_formula=True, title="Gaussian envelope")

        """
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for BaseEnvelope.plot()."
            )

        if t is None:
            if tmin is None or tmax is None:
                c, S = self.center_and_scale()
                T = 6.0 * S
                tmin = c - T if tmin is None else tmin
                tmax = c + T if tmax is None else tmax
            t = np.linspace(float(tmin), float(tmax), int(n))

        y = self.time_eval(t)

        n_rows = 1 + int(show_real_imag) + int(show_phase)
        header_rows = 1 if show_formula else 0

        fig = plt.figure(figsize=(8, 2.2 * (n_rows + 0.6 * header_rows)))
        gs = fig.add_gridspec(
            nrows=n_rows + header_rows,
            ncols=1,
            height_ratios=([0.22] if show_formula else []) + [1] * n_rows,
        )

        if show_formula:
            tex = self.latex_expression()
            ax_head = fig.add_subplot(gs[0, 0])
            ax_head.axis("off")
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

        data_axes = []
        start = header_rows
        for i in range(n_rows):
            data_axes.append(fig.add_subplot(gs[start + i, 0]))
        axs = np.asarray(data_axes)

        r = 0
        axs[r].plot(t, np.abs(y), label=r"$|\zeta(t)|$")
        axs[r].set_ylabel("envelope")
        axs[r].legend()
        r += 1

        if show_real_imag:
            axs[r].plot(t, y.real, label=r"$\Re\,\zeta(t)$")
            axs[r].plot(t, y.imag, label=r"$\Im\,\zeta(t)$", linestyle="--")
            axs[r].set_ylabel("field")
            axs[r].legend()
            r += 1

        if show_phase:
            axs[r].plot(t, np.unwrap(np.angle(y)), label="phase")
            axs[r].set_ylabel("[rad]")
            axs[r].legend()
            # Put the x-label on the bottom axis only
            axs[r].set_xlabel("time t")

        if title:
            fig.suptitle(title, y=0.995)

        fig.tight_layout(rect=(0, 0, 1, 0.98))
        return fig, axs
