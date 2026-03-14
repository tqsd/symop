r"""Gaussian mixtures of closed-form Gaussian time-domain envelopes.

This module defines
:class:`~symop.modes.envelopes.gaussian_mixture.GaussianMixtureEnvelope`,
a normalized *mode descriptor* formed as a linear combination of normalized
closed-form Gaussian components
(:class:`~symop.modes.envelopes.gaussian.GaussianEnvelope`).

Motivation
----------
A single Gaussian envelope is often too restrictive to represent practical mode
shapes (e.g., mildly asymmetric pulses, superpositions of separated lobes, or
simple approximations to filtered / shaped wavepackets). A mixture of Gaussians
provides a simple, numerically stable, and differentiable representation that
still supports analytic overlaps when all components are in the same
closed-form family.

Key properties
--------------
- Each component is a normalized :class:`GaussianEnvelope` in the
  ``"gaussian_closed"`` formalism.
- The mixture itself is normalized such that :math:`\langle\zeta,\zeta\rangle=1`
  (computed from closed-form pairwise overlaps).
- Overlaps with another closed-form Gaussian envelope (single Gaussian or another
  mixture) are computed in closed form via finite sums.
- Time and frequency evaluation are computed by linear superposition of the
  component evaluations.

Performance notes
-----------------
- Normalization and overlap between two mixtures both require pairwise overlaps.
  If the mixture has :math:`K` components, normalization is :math:`O(K^2)` and
  mixture-vs-mixture overlap is :math:`O(K^2)`. For small K this is typically
  negligible; for large K consider caching Gram matrices keyed by component
  signatures.

Fourier convention
------------------
This module follows the same Fourier convention as :class:`GaussianEnvelope`:

.. math::

    Z(\omega) = \int_{-\infty}^{\infty} \zeta(t)\,e^{+i\omega t}\,dt.

The absolute overall scale of :meth:`freq_eval` is intended to be consistent
within the package, but should not be treated as a universal physical
normalization unless the convention is standardized across all backends.

Examples
--------
See ``examples/gaussian_mixture_overlap.py`` for:
- plotting the time-domain envelope of a mixture,
- overlap magnitude vs delay,
- and the spectral intensity profile.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from symop.core.types import ComplexArray, FloatArray, RCArray, Signature
from symop.core.types.funcs import TimeFunc
from symop.modes.envelopes.base import BaseEnvelope
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.protocols.envelope import (
    EnvelopeFormalism,
    SupportsGaussianClosedOverlap,
)


@dataclass(frozen=True)
class FitReport:
    """Diagnostics for a function-to-mixture fit.

    Attributes
    ----------
    tmin, tmax:
        Fit interval.
    n_samples:
        Number of samples used in the fit.
    l2_rel:
        Relative L2 error on the fit grid:
        ||f - f_hat||_2 / ||f||_2.
    linf_rel:
        Relative max error on the fit grid:
        max|f - f_hat| / max|f|.
    k_used:
        Number of Gaussian components used.

    Notes
    -----
    Errors are computed on the sampling grid used during fitting and are
    therefore *fit-window dependent*.

    """

    tmin: float
    tmax: float
    n_samples: int
    l2_rel: float
    linf_rel: float
    k_used: int


@dataclass(frozen=True)
class GaussianMixtureEnvelope(BaseEnvelope):
    r"""Linear combination of normalized closed-form Gaussian envelopes.

    A Gaussian mixture envelope is defined as a finite superposition

    .. math::

        \zeta(t) = \sum_{k=1}^{K} c_k\,g_k(t),

    where each :math:`g_k(t)` is a normalized
    :class:`~symop.modes.envelopes.gaussian.GaussianEnvelope` (same closed-form
    Gaussian family), and the complex weights :math:`c_k` are stored in
    :attr:`weights`.

    Normalization
    -------------
    This class enforces that the mixture is a *mode descriptor*:

    .. math::

        \langle \zeta, \zeta \rangle = 1.

    Since :math:`\zeta` is a superposition, its norm depends on pairwise overlaps:

    .. math::

        \|\zeta\|^2
        =
        \sum_{i=1}^{K}\sum_{j=1}^{K}
        \overline{c_i}\,c_j\,\langle g_i, g_j\rangle.

    In :meth:`__post_init__`, the provided weights are rescaled by

    .. math::

        c_k \leftarrow \frac{c_k}{\sqrt{\|\zeta\|^2}}

    using closed-form overlaps :math:`\langle g_i, g_j\rangle`.

    Overlaps
    --------
    If the other envelope is in the same closed-form Gaussian family, overlaps are
    computed without numerical quadrature.

    1) Mixture vs single Gaussian:

    .. math::

        \langle \zeta, h\rangle
        =
        \sum_{i=1}^{K} \overline{c_i}\,\langle g_i, h\rangle.

    2) Mixture vs mixture:

    .. math::

        \left\langle \sum_i c_i g_i,\ \sum_j d_j h_j \right\rangle
        =
        \sum_{i=1}^{K}\sum_{j=1}^{L}
        \overline{c_i}\,d_j\,\langle g_i, h_j\rangle.

    Here :math:`\langle g_i, h_j\rangle` is evaluated by
    :meth:`GaussianEnvelope.overlap_gaussian_closed`.

    Time and frequency evaluation
    -----------------------------
    Evaluation is performed by linear superposition:

    .. math::

        \zeta(t) = \sum_k c_k g_k(t), \qquad
        Z(\omega) = \sum_k c_k G_k(\omega).

    Heuristics
    ----------
    :meth:`center_and_scale` returns a plotting/overlap heuristic. The center is a
    power-weighted average of component centers (:math:`|c_k|^2` weights) and the
    scale is the maximum component :math:`\sigma_t` (conservative window choice).

    Parameters
    ----------
    components:
        Tuple of component :class:`GaussianEnvelope` objects. Must be non-empty.
    weights:
        Complex 1D numpy array of shape ``(K,)``. Length must match
        ``len(components)``. The array is copied/coerced to complex and then
        rescaled in :meth:`__post_init__` to enforce unit norm.
    report:
        Optional :class:`FitReport` describing how the mixture was obtained
        (for example from a fit of a target function on a chosen time interval).

    Raises
    ------
    ValueError
        If ``weights`` is not 1D, if lengths mismatch, if components are empty,
        or if the computed norm is non-positive / non-finite.

    """

    formalism: ClassVar[EnvelopeFormalism] = "gaussian_closed"

    components: tuple[GaussianEnvelope, ...]
    weights: ComplexArray
    report: FitReport | None = None

    def __post_init__(self) -> None:
        r"""Validate inputs and normalize the mixture weights.

        The provided weights are coerced to a 1D complex array and checked
        against the component list. The mixture is then normalized so that it
        represents a unit-norm mode descriptor:

        .. math::

            \langle \zeta, \zeta \rangle = 1.

        Normalization is computed from closed-form pairwise overlaps between
        component Gaussians.

        Raises
        ------
        ValueError
            If ``weights`` is not 1D, if the number of weights does not match
            the number of components, if ``components`` is empty, or if the
            computed norm is non-positive or non-finite.

        """
        w = np.asarray(self.weights, dtype=complex)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if len(self.components) != w.shape[0]:
            raise ValueError("weights length must match number of components")
        if len(self.components) == 0:
            raise ValueError("components must be non-empty")

        # Enforce normalization as a mode descriptor.
        n2 = self._norm2_closed(w, self.components)
        if not (n2 > 0.0) or not np.isfinite(n2):
            raise ValueError(f"invalid mixture norm2: {n2!r}")
        w = w / np.sqrt(n2)

        object.__setattr__(self, "weights", w)

    @classmethod
    def from_callable(
        cls,
        func: TimeFunc,
        *,
        tmin: float,
        tmax: float,
        k: int,
        omega0: float = 0.0,
        n_samples: int = 2000,
        sigma: float | None = None,
    ) -> GaussianMixtureEnvelope:
        r"""Fit a Gaussian-mixture envelope to a time-domain target field.

        This constructor fits a complex-valued time-domain amplitude
        :math:`f(t)` on a uniform grid over ``[tmin, tmax]`` using a linear
        combination of :class:`GaussianEnvelope` basis functions:

        .. math::

            f(t) \approx \sum_{j=1}^{K} c_j g_j(t).

        The component centers are chosen uniformly across the fit interval,
        all components share the same carrier frequency ``omega0``, and a
        common width ``sigma`` is used. If ``sigma`` is not provided, a simple
        window-based heuristic is used.

        Parameters
        ----------
        func:
            Target time-domain complex field.
        tmin, tmax:
            Fit interval.
        k:
            Number of Gaussian basis components.
        omega0:
            Shared carrier angular frequency for the basis components.
        n_samples:
            Number of uniform time samples used for the fit.
        sigma:
            Common temporal width of the basis Gaussians. If omitted, a
            heuristic based on the fit interval and ``k`` is used.

        Returns
        -------
        GaussianMixtureEnvelope
            Normalized Gaussian mixture with an attached fit report.

        Raises
        ------
        ValueError
            If the fit configuration is invalid.

        """
        tmin = float(tmin)
        tmax = float(tmax)
        omega0 = float(omega0)
        n_samples = int(n_samples)
        k = int(k)

        if not (tmax > tmin):
            raise ValueError("tmax must be greater than tmin")
        if k < 1:
            raise ValueError("k must be at least 1")
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")

        if sigma is None:
            sigma = (tmax - tmin) / max(2.0 * k, 1.0)
        sigma = float(sigma)
        if not (sigma > 0.0) or not np.isfinite(sigma):
            raise ValueError("sigma must be positive and finite")

        t = np.linspace(tmin, tmax, n_samples, dtype=float)
        y = np.asarray(func(t), dtype=complex)
        if y.shape != t.shape:
            raise ValueError("func(t) must return an array with the same shape as t")

        centers = np.linspace(tmin, tmax, k, dtype=float)
        components = tuple(
            GaussianEnvelope(
                omega0=omega0,
                sigma=sigma,
                tau=float(tc),
                phi0=0.0,
            )
            for tc in centers
        )

        A = np.column_stack([g.time_eval(t) for g in components])
        weights, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        mix = cls(
            components=components,
            weights=np.asarray(weights, dtype=complex),
        )

        y_hat = np.asarray(mix.time_eval(t), dtype=complex)

        denom_l2 = np.linalg.norm(y)
        l2_rel = float(np.linalg.norm(y - y_hat) / denom_l2) if denom_l2 > 0.0 else 0.0

        denom_linf = float(np.max(np.abs(y)))
        linf_rel = (
            float(np.max(np.abs(y - y_hat)) / denom_linf) if denom_linf > 0.0 else 0.0
        )

        report = FitReport(
            tmin=tmin,
            tmax=tmax,
            n_samples=n_samples,
            l2_rel=l2_rel,
            linf_rel=linf_rel,
            k_used=k,
        )
        return replace(mix, report=report)

    @classmethod
    def from_lorentzian(
        cls,
        *,
        gamma: float,
        tau: float = 0.0,
        omega0: float = 0.0,
        phi0: float = 0.0,
        k: int = 8,
        n_samples: int = 2000,
        window_multiple: float = 8.0,
    ) -> GaussianMixtureEnvelope:
        r"""Construct a Gaussian mixture approximation to a Lorentzian pulse.

        This convenience constructor fits a Gaussian mixture to the
        time-domain target field

        .. math::

            f(t)
            =
            e^{i\phi_0}
            \frac{1}{1 + \left(\frac{t-\tau}{\gamma}\right)^2}
            e^{i\omega_0 (t-\tau)}.

        The fit is performed on a symmetric time window around :math:`\tau`:

        .. math::

            [\,\tau - m\gamma,\ \tau + m\gamma\,],

        where :math:`m` is ``window_multiple``.

        Parameters
        ----------
        gamma:
            Lorentzian width parameter. Must be positive.
        tau:
            Center time of the pulse.
        omega0:
            Carrier angular frequency.
        phi0:
            Global phase offset.
        k:
            Number of Gaussian basis components used in the fit.
        n_samples:
            Number of time samples used for the fitting procedure.
        window_multiple:
            Half-window size expressed in multiples of ``gamma``.

        Returns
        -------
        GaussianMixtureEnvelope
            Normalized Gaussian mixture approximating the Lorentzian target,
            with an attached fit report.

        Raises
        ------
        ValueError
            If ``gamma`` is not positive.

        Notes
        -----
        This is a numerical approximation utility built on top of
        :meth:`from_callable`. The resulting mixture is normalized as a mode
        descriptor after fitting.

        """
        gamma = float(gamma)
        tau = float(tau)
        omega0 = float(omega0)
        phi0 = float(phi0)

        if not (gamma > 0.0):
            raise ValueError("gamma must be positive")

        tmin = tau - window_multiple * gamma
        tmax = tau + window_multiple * gamma

        phase = complex(np.cos(phi0), np.sin(phi0))

        def target(t: FloatArray) -> RCArray:
            x = (t - tau) / gamma
            out = phase * (1.0 / (1.0 + x * x)) * np.exp(1j * omega0 * (t - tau))
            return np.asarray(out, dtype=np.complex128)

        return cls.from_callable(
            target,
            tmin=tmin,
            tmax=tmax,
            k=k,
            omega0=omega0,
            n_samples=n_samples,
        )

    @staticmethod
    def _norm2_closed(
        weights: ComplexArray, comps: tuple[GaussianEnvelope, ...]
    ) -> float:
        r"""Compute the squared norm of a Gaussian mixture in closed form.

        For a mixture

        .. math::

            \zeta(t) = \sum_{k=1}^{K} c_k g_k(t),

        the squared norm is

        .. math::

            \|\zeta\|^2
            =
            \sum_{i=1}^{K}\sum_{j=1}^{K}
            \overline{c_i}\,c_j\,\langle g_i, g_j\rangle.

        This helper evaluates that quantity using analytic pairwise overlaps
        between the component :class:`GaussianEnvelope` objects.

        Parameters
        ----------
        weights:
            Complex mixture coefficients :math:`c_k`.
        comps:
            Gaussian components :math:`g_k`.

        Returns
        -------
        float
            Real-valued squared norm of the mixture.

        Notes
        -----
        The return value is the real part of the accumulated overlap sum.
        For physically consistent inputs this should be non-negative, up to
        small numerical error.

        """
        K = len(comps)
        s = 0.0 + 0.0j
        for i in range(K):
            gi = comps[i]
            ci = weights[i]
            for j in range(K):
                gj = comps[j]
                cj = weights[j]
                s += np.conjugate(ci) * cj * gi.overlap_gaussian_closed(gj)
        return float(np.real(s))

    def time_eval(self, t: FloatArray) -> RCArray:
        r"""Evaluate the time-domain complex field :math:`\zeta(t)`.

        Parameters
        ----------
        t:
            Time grid.

        Returns
        -------
        RCArray
            Complex samples of the field.

        """
        t_arr = np.asarray(t, dtype=float)
        out = np.zeros_like(t_arr, dtype=complex)
        for c, g in zip(self.weights, self.components, strict=True):
            out += c * np.asarray(g.time_eval(t_arr), dtype=complex)
        return out

    def freq_eval(self, w: FloatArray) -> RCArray:
        r"""Evaluate the frequency-domain spectrum :math:`Z(\omega)`.

        This returns a GaussianMixture spectrum consistent with the time-domain
        definition used in :meth:`time_eval`, up to an overall real scale factor.

        Parameters
        ----------
        w:
            Frequency grid (angular frequency :math:`\omega`).

        Returns
        -------
        RCArray
            Complex samples of the spectrum.

        """
        w_arr = np.asarray(w, dtype=float)
        out = np.zeros_like(w_arr, dtype=complex)
        for c, g in zip(self.weights, self.components, strict=True):
            out += c * np.asarray(g.freq_eval(w_arr), dtype=complex)
        return out

    def delayed(self, dt: float) -> GaussianMixtureEnvelope:
        r"""Return a copy delayed by dt.

        Parameters
        ----------
        dt:
            Time shift to add to :math:`\tau`.

        Returns
        -------
        GaussianMixtureEnvelope
            Delayed envelope.

        """
        comps = tuple(g.delayed(dt) for g in self.components)
        return replace(self, components=comps)

    def phased(self, dphi: float) -> GaussianMixtureEnvelope:
        r"""Return a copy with an added global phase.

        Parameters
        ----------
        dphi:
            Phase increment to add to :math:`\phi_0`.

        Returns
        -------
        GaussianMixtureEnvelope
            Phased envelope.

        """
        phase = complex(np.cos(float(dphi)), np.sin(float(dphi)))
        return replace(self, weights=self.weights * phase)

    def center_and_scale(self) -> tuple[float, float]:
        r"""Return plotting/overlap heuristics.

        Returns
        -------
        center:
            Center time (:math:`\tau`).
        scale:
            Characteristic scale (:math:`\sigma_t`).

        """
        # Heuristic: use weighted average of component centers/scales by |c|^2
        p = np.abs(self.weights) ** 2
        p_sum = float(np.sum(p))
        if not (p_sum > 0.0) or not np.isfinite(p_sum):
            return 0.0, 1.0

        centers = np.array([g.tau for g in self.components], dtype=float)
        scales = np.array([g.sigma for g in self.components], dtype=float)

        c = float(np.sum(p * centers) / p_sum)
        s = float(np.max(scales))
        s = max(s, 1e-12)
        return c, s

    @property
    def signature(self) -> Signature:
        """Stable signature for caching/comparison."""
        w = tuple((float(np.real(x)), float(np.imag(x))) for x in self.weights.tolist())
        sigs = tuple(g.signature for g in self.components)
        return ("gauss_mix", sigs, w)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature:
        r"""Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimals to round to.
        ignore_global_phase:
            If True, treat :math:`\phi_0` as zero for grouping.

        Returns
        -------
        Signature
            Rounded/approximate signature tuple.

        """
        r = round
        w = np.asarray(self.weights, dtype=complex)
        if ignore_global_phase and w.size > 0 and abs(w[0]) > 0:
            w = w * np.conjugate(w[0]) / abs(w[0])

        w_approx = tuple(
            (r(float(np.real(x)), decimals), r(float(np.imag(x)), decimals))
            for x in w.tolist()
        )
        sigs = tuple(
            g.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            )
            for g in self.components
        )
        return ("gauss_mix_approx", sigs, w_approx)

    def overlap_gaussian_closed(self, other: SupportsGaussianClosedOverlap) -> complex:
        r"""Compute the closed-form overlap with another Gaussian-closed envelope.

        Supported cases are:

        1. Mixture vs single Gaussian:

        .. math::

            \langle \zeta, h\rangle
            =
            \sum_i \overline{c_i}\,\langle g_i, h\rangle.

        2. Mixture vs mixture:

        .. math::

            \left\langle \sum_i c_i g_i,\ \sum_j d_j h_j \right\rangle
            =
            \sum_i \sum_j
            \overline{c_i}\,d_j\,\langle g_i, h_j\rangle.

        In both cases, pairwise overlaps are evaluated analytically using
        :meth:`GaussianEnvelope.overlap_gaussian_closed`.

        Parameters
        ----------
        other:
            Another envelope in the same closed Gaussian formalism.

        Returns
        -------
        complex
            Complex overlap :math:`\langle \text{self}, \text{other} \rangle`.

        Raises
        ------
        TypeError
            If ``other`` is not a supported Gaussian-closed envelope type.

        Notes
        -----
        This method supports overlaps with :class:`GaussianEnvelope` and
        :class:`GaussianMixtureEnvelope`.

        """
        if isinstance(other, GaussianEnvelope):
            s = 0.0 + 0.0j
            for c, g in zip(self.weights, self.components, strict=True):
                s += np.conjugate(c) * g.overlap_gaussian_closed(other)
            return complex(s)

        if isinstance(other, GaussianMixtureEnvelope):
            s = 0.0 + 0.0j
            for ci, gi in zip(self.weights, self.components, strict=True):
                for dj, hj in zip(other.weights, other.components, strict=True):
                    s += np.conjugate(ci) * dj * gi.overlap_gaussian_closed(hj)
            return complex(s)

        raise TypeError(
            "gaussian_closed overlap requires GaussianEnvelope or GaussianMixtureEnvelope"
        )


if TYPE_CHECKING:

    def _accept_gaussian_closed(
        env: SupportsGaussianClosedOverlap,
    ) -> None:
        pass

    _accept_gaussian_closed(
        GaussianMixtureEnvelope(
            components=(
                GaussianEnvelope(
                    omega0=0.0,
                    sigma=1.0,
                    tau=0.0,
                    phi0=0.0,
                ),
            ),
            weights=np.array([1.0 + 0.0j], dtype=np.complex128),
        )
    )
