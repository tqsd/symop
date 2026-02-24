from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .signature import HasSignature


@runtime_checkable
class EnvelopeLike(HasSignature, Protocol):
    def overlap(self, other: EnvelopeLike) -> complex: ...
    def delayed(self, dt: float) -> EnvelopeLike: ...
    def phased(self, dphi: float) -> EnvelopeLike: ...


@runtime_checkable
class TimeEvaluable(Protocol):
    def time_eval(self, t: np.ndarray) -> np.ndarray: ...
    def center_and_scale(self) -> tuple[float, float]: ...


@runtime_checkable
class SupportsOverlapWithGeneric(Protocol):
    def overlap_with_generic(self, other: EnvelopeLike) -> complex: ...
