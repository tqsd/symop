from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from symop_proto.core.protocols import SignatureProto
from symop_proto.labels.protocols import PolraizationLabelProto


@dataclass(frozen=True)
class PolarizationLabel(PolraizationLabelProto):
    jones: tuple[complex, complex]

    def __post_init__(self):
        a, b = self.jones
        v = np.array([a, b], dtype=complex)

        n = np.linalg.norm(v)
        if n == 0.0:
            raise ValueError("PolarizationLabel: Jones vector cannot be zero")

        v = v / n

        eps = 1e-15
        idx = None
        if abs(v[0]) > eps:
            idx = 0
        elif abs(v[1]) > eps:
            idx = 1

        if idx is not None:
            phase = np.exp(-1j * np.angle(v[idx]))
            v = v * phase

            v[idx] = v[idx].real + 0.0j
            if v[idx].real < 0:
                v = -v

        def _chopc(z: complex, eps: float = 1e-15) -> complex:
            # To cut the numerical noise in signature
            r = 0.0 if abs(z.real) < eps else float(z.real)
            i = 0.0 if abs(z.imag) < eps else float(z.imag)
            return complex(r, i)

        object.__setattr__(
            self, "jones", (_chopc(complex(v[0])), _chopc(complex(v[1])))
        )

    def overlap(self, other: PolarizationLabel) -> complex:
        v1 = np.array(self.jones, dtype=complex)
        v2 = np.array(other.jones, dtype=complex)
        return complex(np.vdot(v1, v2))

    # presets
    @classmethod
    def H(cls) -> PolarizationLabel:
        return cls((1 + 0j, 0 + 0j))

    @classmethod
    def V(cls) -> PolarizationLabel:
        return cls((0 + 0j, 1 + 0j))

    @classmethod
    def D(cls) -> PolarizationLabel:
        s = 2**-0.5
        return cls((s + 0j, s + 0j))

    @classmethod
    def A(cls) -> PolarizationLabel:
        s = 2**-0.5
        return cls((s + 0j, -s + 0j))

    @classmethod
    def R(cls) -> PolarizationLabel:
        s = 2**-0.5
        return cls((s + 0j, -1j * s))

    @classmethod
    def L(cls) -> PolarizationLabel:
        s = 2**-0.5
        return cls((s + 0j, 1j * s))

    @classmethod
    def linear(cls, theta: float) -> PolarizationLabel:
        return cls((np.cos(theta), np.sin(theta)))

    def rotated(self, theta: float) -> PolarizationLabel:
        a, b = self.jones
        c, s = np.cos(theta), np.sin(theta)
        return PolarizationLabel((c * a + s * b, -s * a + c * b))

    @property
    def signature(self) -> SignatureProto:
        a, b = self.jones
        return (
            "pol",
            float(a.real),
            float(a.imag),
            float(b.real),
            float(b.imag),
        )

    def approx_signature(self, *, decimals: int = 12, **kw: Any) -> SignatureProto:
        def r(z):
            return (round(z.real, decimals), round(z.imag, decimals))

        a, b = self.jones
        ra = r(a)
        rb = r(b)
        return ("pol_approx", *ra, *rb)
