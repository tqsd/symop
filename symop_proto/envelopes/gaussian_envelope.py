from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Tuple
import numpy as np

from symop_proto.core.protocols import EnvelopeProto, SignatureProto
from symop_proto.envelopes.base import BaseEnvelope


@dataclass(frozen=True)
class GaussianEnvelope(BaseEnvelope):
    omega0: float
    sigma: float
    tau: float
    phi0: float = 0.0

    def time_eval(self, t: np.ndarray) -> np.ndarray:
        norm = (1.0 / (2.0 * np.pi * self.sigma**2)) ** 0.25
        x = t - self.tau
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

    def overlap(self, other: EnvelopeProto) -> complex:
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
