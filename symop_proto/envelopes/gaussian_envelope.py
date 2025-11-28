from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Tuple
import numpy as np

from symop_proto.core.protocols import SignatureProto
from symop_proto.envelopes.base import BaseEnvelope
from symop_proto.envelopes.protocols import FloatArray, RCArray, EnvelopeProto


@dataclass(frozen=True)
class GaussianEnvelope(BaseEnvelope):
    r"""
    **Canonical** Gaussian time-domain envelope

    Parameters:
    -----------
    omega0: float
        Carrier (angular) frequency :math:`\omega_0`.
    sigma: float
        Temporal width :math:`\sigma` (standard-deviation).
    tau: float
        Time shift (center) :math:`\tau`.
    phi0: float, default 0.0
        Global phase :math:`\phi_0`.

    Mathematics:
    ------------

        **Envelope.**  The time-domain envelope is

        .. math::

            \zeta(t;\,\omega_0,\sigma,\tau,\phi_0)
            \;=\;
            \left(\frac{1}{2\pi\sigma^{2}}\right)^{\!1/4}
            \exp\!\left[-\frac{(t-\tau)^2}{4\sigma^{2}}\right]
            \exp\!\bigl(i(\omega_0(t-\tau)+\phi_0)\bigr).

    """

    omega0: float
    sigma: float
    tau: float
    phi0: float = 0.0

    def time_eval(self, t: FloatArray) -> RCArray:
        norm = (1.0 / (2.0 * np.pi * self.sigma**2)) ** 0.25
        x = t - self.tau
        return (
            norm
            * np.exp(-(x**2) / (4.0 * self.sigma**2))
            * np.exp(1j * (self.omega0 * x + self.phi0))
        )

    def freq_eval(self, w: FloatArray) -> RCArray:
        norm = (1.0 / (2.0 * np.pi * self.sigma**2)) ** 0.25
        x = w - self.tau
        return (
            norm
            * np.exp(-(x**2) / (4.0 * self.sigma**2))
            * np.exp(1j * (self.omega0 * x + self.phi0))
        )

    def center_and_scale(self) -> Tuple[float, float]:
        return self.tau, self.sigma

    def delayed(self, dt: float) -> GaussianEnvelope:
        return replace(self, tau=self.tau + dt)

    def phased(self, dphi: float) -> GaussianEnvelope:
        return replace(self, phi0=self.phi0 + dphi)

    @property
    def signature(self) -> SignatureProto:
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
        **kw: Any,
    ) -> SignatureProto:
        r = round
        phi = 0.0 if ignore_global_phase else self.phi0
        return (
            "gauss_approx",
            r(float(self.omega0), decimals),
            r(float(self.sigma), decimals),
            r(float(self.tau), decimals),
            r(float(phi), decimals),
        )

    def overlap(self, other: EnvelopeProto) -> complex:
        r"""
        Closed-form overlap with another envelope.

        For two Gaussians
        :math:`\zeta_1(t)=\zeta(t;\omega_1,\sigma_1,\tau_1,\phi_1)` and
        :math:`\zeta_2(t)=\zeta(t;\omega_2,\sigma_2,\tau_2,\phi_2)` with

        .. math::

            \zeta(t;\omega,\sigma,\tau,\phi)
            = \left(\frac{1}{2\pi\sigma^2}\right)^{1/4}
            e^{-\frac{(t-\tau)^2}{4\sigma^2}}
            e^{i(\omega(t-\tau)+\phi)},

        the overlap :math:`\langle \zeta_1,\zeta_2\rangle
        = \int_{-\infty}^{\infty}\zeta_1(t)^*\,\zeta_2(t)\,dt`
        evaluates to

        .. math::
            \boxed{
            \begin{aligned}
            \langle \zeta_1,\zeta_2\rangle
            &= \sqrt{\frac{2\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2}}
            \exp\!\left[-\frac{(\tau_1-\tau_2)^2}{4(\sigma_1^2+\sigma_2^2)}\right]
            \exp\!\left[-\frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2}
                        (\omega_1-\omega_2)^2\right] \\
            &\quad\times
            \exp\!\left\{ i\Big[(\phi_2-\phi_1)
                + \frac{\sigma_1^2\omega_1+\sigma_2^2\omega_2}{\sigma_1^2+\sigma_2^2}
                    (\tau_1-\tau_2)\Big]\right\}.
            \end{aligned}
            }


        Mathematics:
        ------------

            Let
            :math:`\displaystyle
            \zeta_k(t)=(2\pi\sigma_k^2)^{-1/4}
            \exp\!\!\left[-\frac{(t-\tau_k)^2}{4\sigma_k^2}\right]
            \exp\!\big(i(\omega_k(t-\tau_k)+\phi_k)\big),\quad k\in\{1,2\}.`
            Set :math:`S_k=\tfrac{1}{4\sigma_k^2}`. Then

            .. math::

                \begin{aligned}
                \zeta_1^*(t)\,\zeta_2(t)
                &= C \exp\!\Big(
                        -S_1(t-\tau_1)^2 - S_2(t-\tau_2)^2
                        + i(\omega_2-\omega_1)t \\
                &\qquad\qquad
                        + i(\phi_2-\phi_1-\omega_2\tau_2+\omega_1\tau_1)
                    \Big),
                \end{aligned}

            with :math:`C=(2\pi\sigma_1^2)^{-1/4}(2\pi\sigma_2^2)^{-1/4}`.
            Collecting terms gives a quadratic in :math:`t`,

            .. math::

                \begin{aligned}
                \zeta_1^*(t)\,\zeta_2(t)
                &= C\,\exp(-A t^2 + B t + D),\\
                A&=S_1+S_2,\\
                B&=2S_1\tau_1+2S_2\tau_2+i(\omega_2-\omega_1),\\
                D&=-S_1\tau_1^2-S_2\tau_2^2+i(\phi_2-\phi_1-\omega_2\tau_2+\omega_1\tau_1).
                \end{aligned}

            Complete the square,

            .. math::

                \begin{aligned}
                -A t^2 + B t
                = -A\!\left(t-\frac{B}{2A}\right)^{\!2} + \frac{B^2}{4A},
                \end{aligned}

            and use
            :math:`\displaystyle \int_{-\infty}^{\infty} e^{-A(t-\mu)^2}\,dt=\sqrt{\pi/A}`
            for :math:`\Re(A)>0` to obtain

            .. math::

                \begin{aligned}
                \int_{-\infty}^{\infty}\zeta_1^*(t)\,\zeta_2(t)\,dt
                &= C\,\sqrt{\frac{\pi}{A}}\,
                    \exp\!\left(\frac{B^2}{4A}+D\right).
                \end{aligned}

            Substituting :math:`A=S_1+S_2=\dfrac{\sigma_1^2+\sigma_2^2}{4\sigma_1^2\sigma_2^2}`
            and simplifying gives

            .. math::

                \boxed{
                \begin{aligned}
                \langle \zeta_1,\zeta_2\rangle
                &= \sqrt{\frac{2\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2}}\;
                    \exp\!\left[-\frac{(\tau_1-\tau_2)^2}{4(\sigma_1^2+\sigma_2^2)}\right] \\
                &\quad\times
                    \exp\!\left[-\frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2}
                                (\omega_1-\omega_2)^2\right] \\
                &\quad\times
                    \exp\!\left\{ i\left[(\phi_2-\phi_1) +
                        \frac{\sigma_1^2\omega_1+\sigma_2^2\omega_2}{\sigma_1^2+\sigma_2^2}
                        (\tau_1-\tau_2)\right]\right\}.
                \end{aligned}
                }
        Special cases
        ------------
        * If :math:`\sigma_1=\sigma_2=\sigma` and :math:`\omega_1=\omega_2=\omega_0`,

        .. math::
            \langle \zeta_1,\zeta_2\rangle
            = \exp\!\Big[-\frac{(\tau_1-\tau_2)^2}{8\sigma^2}\Big]\,
            e^{\,i\big[(\phi_2-\phi_1)+\omega_0(\tau_1-\tau_2)\big]}.

        Notes
        -----
        If ``other`` is not a :class:`GaussianEnvelope`, this method falls back to
        :py:meth:`BaseEnvelope.overlap`, which may use numeric integration.

        Returns
        -------
        complex
            The inner product :math:`\langle \zeta_1,\zeta_2\rangle`.
        """
        if isinstance(other, GaussianEnvelope):
            s1, s2 = self.sigma, other.sigma
            w1, w2 = self.omega0, other.omega0
            t1, t2 = self.tau, other.tau
            p1, p2 = self.phi0, other.phi0
            S2 = s1**2 + s2**2
            dtau, domega, dphi = (t1 - t2), (w1 - w2), (p2 - p1)
            pref = np.sqrt(2.0 * s1 * s2 / S2)
            exp_time = np.exp(-(dtau**2) / (4.0 * S2))
            exp_freq = np.exp(-(s1**2 * s2**2 / S2) * (domega**2))
            omega_bar = (s1**2 * w1 + s2**2 * w2) / S2
            phase = np.exp(1j * (dphi + dtau * omega_bar))
            return pref * exp_time * exp_freq * phase
        return super().overlap(other)

    def latex_expression(self) -> str | None:
        return (
            r"\zeta(t;\omega_0,\sigma,\tau,\phi_0)"
            r"=\left(\frac{1}{2\pi\sigma^{2}}\right)^{1/4}"
            r"\exp\left(-\frac{(t-\tau)^2}{4\sigma^{2}}\right)"
            r"\exp\left(i(\omega_0(t-\tau)+\phi_0)\right)"
        )
