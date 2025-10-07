from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Envelope:
    """
    Represents a normalized Gaussian spectral envelope

    Parameters
    ----------
    omega0: float
        Central frequency of the Gaussian (rad/s)
    sigma: float
        Standard deviation(spectral width)
    tau: float, optional
        Temporal delay. Default is 0
    phi0: float, optional
        Constant phase offset. Default is 0
    """

    omega0: float
    sigma: float
    tau: float = 0.0
    phi0: float = 0.0

    def overlap(self, other: "Envelope") -> complex:
        r"""
        Compute the analytic overlap <self|other> between two Gaussian
        envelopes.

        The envelopes are of the form

        .. math::

            \zeta(\omega) =
            \exp\!\left(-\frac{(\omega - \omega_0)^2}{2\sigma^2}\right)
            \exp\!\big(-i\omega\tau + i\phi_0\big),

        normalized such that :math:`\langle \zeta|\zeta \rangle = 1`.

        The analytic overlap is

        .. math::

            \langle \zeta_1 | \zeta_2 \rangle =
            \sqrt{\frac{2\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2}}
            \exp\!\left(-\tfrac{1}{2}(\omega_1-\omega_2)^2
            \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2}\right)
            \exp\!\left(-\tfrac{1}{2}\frac{(\tau_1-\tau_2)^2}{\sigma_1^2+\sigma_2^2}\right)
            e^{i(\phi_2 - \phi_1)}.

        Returns
        -------
        complex
            Overlap integral value.
        """
        s1, s2 = self.sigma, other.sigma
        w1, w2 = self.omega0, other.omega0
        t1, t2 = self.tau, other.tau
        p1, p2 = self.phi0, other.phi0

        S = s1**2 + s2**2  # width sum
        dw = w1 - w2  # frequency mismatch
        dt = t1 - t2  # delay mismatch

        # magnitude parts
        pref = np.sqrt(2.0 * s1 * s2 / S)
        exp_freq = np.exp(-(dw**2) / (2.0 * S))
        exp_time = np.exp(-(dt**2) * (s1**2 * s2**2) / (2.0 * S))

        # phase: constant phase difference + weighted frequency times dt
        w_bar = (w1 * s2**2 + w2 * s1**2) / S
        phase = np.exp(1j * ((p2 - p1) + dt * w_bar))

        return pref * exp_freq * exp_time * phase

    def delayed(self, dt: float) -> "Envelope":
        """
        Return a new envelope delayed by dt.
        """
        return Envelope(self.omega0, self.sigma, self.tau + dt, self.phi0)

    def phased(self, dphi: float) -> "Envelope":
        """
        Return a new envelope with additional phase shift dphi.
        """
        return Envelope(self.omega0, self.sigma, self.tau, self.phi0 + dphi)

    def signature(self) -> tuple:
        """Exact, content-based identity."""
        return (
            "gauss",
            float(self.omega0),
            float(self.sigma),
            float(self.tau),
            float(self.phi0),
        )

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> tuple:
        """Optional: tolerance-based identity."""
        r = round
        phi = 0.0 if ignore_global_phase else self.phi0
        return (
            "gauss approx",
            r(float(self.omega0), decimals),
            r(float(self.sigma), decimals),
            r(float(self.tau), decimals),
            r(float(phi), decimals),
        )
