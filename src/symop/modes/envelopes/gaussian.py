"""Closed-form Gaussian time-domain envelopes.

This module defines :class:`GaussianEnvelope`, a canonical Gaussian
mode envelope with analytic (closed-form) expressions in both the
time and frequency domains.

The time-domain field is a Gaussian-modulated carrier with parameters
(omega0, sigma, tau, phi0). Under the Fourier convention used in this
package, its spectrum is also Gaussian and can be evaluated in closed
form. This allows overlaps and transformations to be computed without
numerical FFT-based reconstruction.

The implementation is intended to provide a consistent reference
envelope for analytic calculations and as a well-behaved building
block for filtering and mode construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np

from symop.core.types import FloatArray, RCArray, Signature
from symop.modes.envelopes.base import BaseEnvelope
from symop.modes.protocols.envelope import (
    EnvelopeFormalism,
    SupportsGaussianClosedOverlap,
)


@dataclass(frozen=True)
class GaussianEnvelope(BaseEnvelope):
    r"""Canonical Gaussian time-domain envelope.

    Parameters
    ----------
    omega0:
        Carrier (angular) frequency :math:`\omega_0`.
    sigma:
        Temporal width :math:`\sigma_t` (standard deviation in time).
    tau:
        Time shift (center) :math:`\tau`.
    phi0:
        Global phase :math:`\phi_0`.

    Definition
    ----------
    We define the complex field (not a rotating-frame envelope) as

    .. math::

        \zeta(t)
        =
        \left(\frac{1}{2\pi\sigma_t^{2}}\right)^{1/4}
        \exp\!\left[-\frac{(t-\tau)^2}{4\sigma_t^{2}}\right]
        \exp\!\bigl(i(\omega_0(t-\tau)+\phi_0)\bigr).

    Frequency domain
    ----------------
    With the Fourier convention used in this package:

    .. math::

        Z(\omega) = \int_{-\infty}^{\infty} \zeta(t)\,e^{+i\omega t}\,dt,

    so that

    .. math::

        \zeta(t)=\frac{1}{2\pi}\int_{-\infty}^{\infty} Z(\omega)\,e^{-i\omega t}\,d\omega.

    Under this convention, the spectrum is also Gaussian:

    .. math::

        Z(\omega) =
        \mathcal{A}\,
        \exp\!\left[-\sigma_t^2(\omega-\omega_0)^2\right]\,
        \exp\!\bigl(-i\omega\tau + i\phi_0\bigr),

    where :math:`\mathcal{A}` is a real constant (depends on normalization and
    Fourier convention). For overlaps and CCR-consistent mode construction,
    the absolute constant does not matter as long as it is used consistently.

    Notes
    -----
    This implementation chooses a consistent normalization constant for `freq_eval`,
    but the exact overall scale is not intended to be relied on unless you have
    explicitly standardized a Fourier convention across the codebase.

    """

    formalism: ClassVar[EnvelopeFormalism] = "gaussian_closed"

    omega0: float
    sigma: float
    tau: float
    phi0: float = 0.0

    @property
    def omega_sigma(self) -> float:
        r"""Heuristic spectral width hint (rad/s).

        For the envelope definition used here, a reasonable scaling for the
        intensity spectral standard deviation is approximately :math:`1/\sigma_t`.
        """
        s = float(self.sigma)
        return 1.0 / max(s, 1e-12)

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
        t = np.asarray(t, dtype=float)
        s = float(self.sigma)
        if not (s > 0.0) or not np.isfinite(s):
            raise ValueError(f"sigma must be positive finite, got {self.sigma!r}")

        norm = (1.0 / (2.0 * np.pi * s * s)) ** 0.25
        x = t - float(self.tau)
        return cast(
            RCArray,
            (
                norm
                * np.exp(-(x * x) / (4.0 * s * s))
                * np.exp(1j * (float(self.omega0) * x + float(self.phi0)))
            ),
        )

    def freq_eval(self, w: FloatArray) -> RCArray:
        r"""Evaluate the frequency-domain spectrum :math:`Z(\omega)`.

        This returns a Gaussian spectrum consistent with the time-domain
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
        w = np.asarray(w, dtype=float)
        s = float(self.sigma)
        if not (s > 0.0) or not np.isfinite(s):
            raise ValueError(f"sigma must be positive finite, got {self.sigma!r}")

        # Consistent (but not necessarily canonical) overall normalization.
        N = (1.0 / (2.0 * np.pi * s * s)) ** 0.25
        A = N * (2.0 * s * np.sqrt(np.pi))

        k = w - float(self.omega0)
        return cast(
            RCArray,
            (
                A
                * np.exp(-(s * s) * (k * k))
                * np.exp(-1j * (w * float(self.tau)))
                * np.exp(1j * float(self.phi0))
            ).astype(complex),
        )

    def center_and_scale(self) -> tuple[float, float]:
        r"""Return plotting/overlap heuristics.

        Returns
        -------
        center:
            Center time (:math:`\tau`).
        scale:
            Characteristic scale (:math:`\sigma_t`).

        """
        return float(self.tau), float(self.sigma)

    def delayed(self, dt: float) -> GaussianEnvelope:
        r"""Return a copy delayed by dt.

        Parameters
        ----------
        dt:
            Time shift to add to :math:`\tau`.

        Returns
        -------
        GaussianEnvelope
            Delayed envelope.

        """
        return replace(self, tau=float(self.tau) + float(dt))

    def phased(self, dphi: float) -> GaussianEnvelope:
        r"""Return a copy with an added global phase.

        Parameters
        ----------
        dphi:
            Phase increment to add to :math:`\phi_0`.

        Returns
        -------
        GaussianEnvelope
            Phased envelope.

        """
        return replace(self, phi0=float(self.phi0) + float(dphi))

    @property
    def signature(self) -> Signature:
        """Stable signature for caching/comparison."""
        return (
            "gauss",
            float(self.omega0),
            float(self.sigma),
            float(self.tau),
            float(self.phi0),
        )

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
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
        phi = 0.0 if ignore_global_phase else float(self.phi0)
        return (
            "gauss_approx",
            r(float(self.omega0), decimals),
            r(float(self.sigma), decimals),
            r(float(self.tau), decimals),
            r(float(phi), decimals),
        )

    def overlap_gaussian_closed(self, other: SupportsGaussianClosedOverlap) -> complex:
        r"""Closed-form overlap with another closed-form Gaussian envelope.

        This computes the inner product

        .. math::

            \langle \zeta_1, \zeta_2 \rangle
            =
            \int_{-\infty}^{\infty} \overline{\zeta_1(t)}\,\zeta_2(t)\,dt,

        specialized to the normalized Gaussian family defined by :meth:`time_eval`.

        Let :math:`\sigma_1,\tau_1,\omega_1,\phi_1` be the parameters of ``self`` and
        :math:`\sigma_2,\tau_2,\omega_2,\phi_2` those of ``other``. Define

        .. math::

            S &= \sigma_1^2 + \sigma_2^2, \\
            \Delta\tau &= \tau_2 - \tau_1, \\
            \Delta\omega &= \omega_2 - \omega_1, \\
            \omega_w &= \frac{\omega_1\sigma_1^2 + \omega_2\sigma_2^2}{S}.

        Then the overlap is

        .. math::

            \langle \zeta_1, \zeta_2 \rangle
            =
            \sqrt{\frac{2\sigma_1\sigma_2}{S}}
            \exp\!\left(-\frac{\Delta\tau^2}{4S}\right)
            \exp\!\left(-\frac{\sigma_1^2\sigma_2^2}{S}\,\Delta\omega^2\right)
            \exp\!\left(i(\phi_2-\phi_1-\omega_w\Delta\tau)\right).

        Notes
        -----
        - This formula assumes both envelopes are normalized such that
        :math:`\langle \zeta, \zeta \rangle = 1`. The normalization used in
        :meth:`time_eval` satisfies this.
        - The result is consistent with conjugate symmetry:
        :math:`\langle \zeta_1,\zeta_2\rangle = \overline{\langle \zeta_2,\zeta_1\rangle}`.

        Parameters
        ----------
        other:
            Another envelope in the same closed-form Gaussian family.

        Returns
        -------
        complex
            The overlap :math:`\langle \zeta_1, \zeta_2 \rangle`.

        Raises
        ------
        TypeError
            If ``other`` is not a :class:`GaussianEnvelope` (current implementation).
        ValueError
            If either temporal width :math:`\sigma` is non-positive.

        """
        if not isinstance(other, GaussianEnvelope):
            raise TypeError(
                "Gaussian closed-form overlap currently requires GaussianEnvelope."
            )

        s1 = float(self.sigma)
        s2 = float(other.sigma)
        if not (s1 > 0.0 and s2 > 0.0):
            raise ValueError("sigma must be positive")

        S = (s1 * s1) + (s2 * s2)
        dt = float(other.tau) - float(self.tau)
        dw = float(other.omega0) - float(self.omega0)

        amp = math.sqrt((2.0 * s1 * s2) / S)
        env_term = math.exp(-(dt * dt) / (4.0 * S))
        spec_term = math.exp(-((s1 * s1) * (s2 * s2) / S) * (dw * dw))

        omega_w = (float(self.omega0) * (s1 * s1) + float(other.omega0) * (s2 * s2)) / S

        phase = (float(other.phi0) - float(self.phi0)) - (omega_w * dt)

        return complex(amp * env_term * spec_term) * complex(
            math.cos(phase), math.sin(phase)
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes.labels import Envelope as EnvelopeProtocol
    from symop.modes.protocols.envelope import (
        GaussianClosedEnvelope,
        HasSpectralHints,
        TimeFrequencyEnvelope,
    )

    _env = GaussianEnvelope(
        omega0=1.0,
        sigma=1.0,
        tau=0.0,
        phi0=0.0,
    )

    _env_check: EnvelopeProtocol = _env
    _tf_env_check: TimeFrequencyEnvelope = _env
    _gaussian_closed_check: GaussianClosedEnvelope = _env
    _spectral_hints_check: HasSpectralHints = _env
