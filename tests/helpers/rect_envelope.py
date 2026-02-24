from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Any, Tuple

from symop_proto.envelopes.base import BaseEnvelope


@dataclass(frozen=True)
class RectEnvelope(BaseEnvelope):
    center: float = 0.0
    width: float = 1.0
    phase: float = 0.0

    def time_eval(self, t: np.ndarray) -> np.ndarray:
        inside = np.abs(t - self.center) <= (self.width * 0.5 + 1e-15)
        amp = np.exp(1j * self.phase) / np.sqrt(self.width)
        out = np.zeros_like(t, dtype=complex)
        out[inside] = amp
        return out

    def freq_eval(self, w: np.ndarray) -> np.ndarray:
        return NotImplemented

    def delayed(self, dt: float) -> "RectEnvelope":
        return RectEnvelope(center=self.center + dt, width=self.width, phase=self.phase)

    def phased(self, dphi: float) -> "RectEnvelope":
        return RectEnvelope(
            center=self.center, width=self.width, phase=self.phase + dphi
        )

    def signature(self) -> Tuple[Any, ...]:
        return (
            "rect",
            float(self.center),
            float(self.width),
            float(self.phase),
        )

    def approx_signature(self, **kw: Any) -> Tuple[Any, ...]:
        dec = kw.get("decimals", 12)

        def r(x):
            return round(float(x), dec)

        return ("rect_approx", r(self.center), r(self.width), r(self.phase))

    def center_and_scale(self) -> Tuple[float, float]:
        return (self.center, self.width)
