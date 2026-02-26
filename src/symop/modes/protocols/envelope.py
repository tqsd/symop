from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols import HasSignature
from symop.modes.types import FloatArray, RCArray


@runtime_checkable
class EnvelopeProto(HasSignature, Protocol):
    """Structural interface for time/frequency envelopes."""

    def time_eval(self, t: FloatArray) -> RCArray: ...
    def freq_eval(self, w: FloatArray) -> RCArray: ...

    def delayed(self, dt: float) -> EnvelopeProto: ...
    def phased(self, dphi: float) -> EnvelopeProto: ...

    def center_and_scale(self) -> tuple[float, float]: ...
    def overlap(self, other: EnvelopeProto) -> complex: ...


@runtime_checkable
class SupportsOverlapWithGeneric(Protocol):
    def overlap_with_generic(self, other: EnvelopeProto) -> complex: ...


@runtime_checkable
class HasSpectralHints(Protocol):
    @property
    def omega0(self) -> float: ...

    @property
    def omega_sigma(self) -> float: ...


@runtime_checkable
class HasLatex(Protocol):
    @property
    def latex(self) -> str | None: ...
