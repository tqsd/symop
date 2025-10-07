from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np

from symop_proto.core.envelope_protocol import (
    EnvelopeProto,
    SupportsOverlapWIthGeneric,
    TimeEvaluable,
)


def _overlap_numeric(
    f1, f2, *, tmin: float, tmax: float, n: int = 2**16
) -> complex:
    t = np.linspace(tmin, tmax, n, dtype=float)
    y = np.conjugate(f1(t)) * f2(t)
    return np.trapezoid(y, t)


@dataclass(frozen=True)
class BaseEnvelope(EnvelopeProto, ABC):

    @abstractmethod
    def time_eval(self, t: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def delayed(self, dt: float) -> BaseEnvelope: ...

    @abstractmethod
    def phased(self, dphi: float) -> BaseEnvelope: ...

    @abstractmethod
    def signature(self) -> Tuple[Any, ...]: ...

    @abstractmethod
    def approx_signature(self, **kw: Any) -> Tuple[Any, ...]: ...

    def center_and_scale(self) -> Tuple[float, float]:
        return 0.0, 1.0

    def overlap(self, other: EnvelopeProto) -> complex:
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
