from __future__ import annotations

from typing import ClassVar, Literal, Protocol, Self, runtime_checkable

from symop.core.protocols.modes.labels import Envelope
from symop.core.types.arrays import FloatArray, RCArray

EnvelopeFormalism = Literal["generic", "gaussian_closed"]


@runtime_checkable
class HasEnvelopeFormalism(Protocol):
    formalism: ClassVar[EnvelopeFormalism]


@runtime_checkable
class TimeFrequencyEnvelope(Envelope, HasEnvelopeFormalism, Protocol):
    """
    Envelope with explicit time/frequency evaluation capability.
    """

    def time_eval(self, t: FloatArray) -> RCArray: ...
    def freq_eval(self, w: FloatArray) -> RCArray: ...

    def delayed(self, dt: float) -> Self: ...
    def phased(self, dphi: float) -> Self: ...

    def center_and_scale(self) -> tuple[float, float]: ...


@runtime_checkable
class SupportsGaussianClosedOverlap(TimeFrequencyEnvelope, Protocol):
    def overlap_gaussian_closed(
        self,
        other: SupportsGaussianClosedOverlap,
    ) -> complex: ...


@runtime_checkable
class GaussianClosedEnvelope(
    SupportsGaussianClosedOverlap,
    Protocol,
): ...


@runtime_checkable
class SupportsOverlapWithGeneric(Protocol):
    def overlap_with_generic(self, other: TimeFrequencyEnvelope) -> complex: ...


@runtime_checkable
class HasSpectralHints(Protocol):
    @property
    def omega0(self) -> float: ...

    @property
    def omega_sigma(self) -> float: ...
