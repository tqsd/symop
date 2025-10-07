from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np


@dataclass(frozen=True)
class PolarizationLabel:
    jones: tuple[complex, complex]

    def overlap(self, other: PolarizationLabel) -> complex:
        v1 = np.array(self.jones, dtype=complex)
        v2 = np.array(other.jones, dtype=complex)
        return complex(np.vdot(v1, v2))

    # presets
    @classmethod
    def H(cls) -> "PolarizationLabel":
        return cls((1 + 0j, 0 + 0j))

    @classmethod
    def V(cls) -> "PolarizationLabel":
        return cls((0 + 0j, 1 + 0j))

    @classmethod
    def D(cls) -> "PolarizationLabel":
        return cls((1 + 0j, 1 + 0j))

    @classmethod
    def A(cls) -> "PolarizationLabel":
        return cls((1 + 0j, -1 + 0j))

    @classmethod
    def R(cls) -> "PolarizationLabel":
        return cls((1 + 0j, -1j))

    @classmethod
    def L(cls) -> "PolarizationLabel":
        return cls((1 + 0j, 1j))

    @classmethod
    def linear(cls, theta: float) -> "PolarizationLabel":
        return cls((np.cos(theta), np.sin(theta)))

    def rotated(self, theta: float) -> "PolarizationLabel":
        a, b = self.jones
        c, s = np.cos(theta), np.sin(theta)
        return PolarizationLabel((c * a + s * b, -s * a + c * b))

    def signature(self) -> Tuple[Any, ...]:
        a, b = self.jones
        return (
            "pol",
            float(a.real),
            float(a.imag),
            float(b.real),
            float(b.imag),
        )

    def approx_signature(
        self, *, decimals: int = 12, **kw: Any
    ) -> Tuple[Any, ...]:
        def r(z):
            return (round(z.real, decimals), round(z.imag, decimals))

        a, b = self.jones
        ra = r(a)
        rb = r(b)
        return ("pol_approx", *ra, *rb)
