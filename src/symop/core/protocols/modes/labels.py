from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.base.overlap import HasOverlap
from symop.core.protocols.base.signature import HasSignature


@runtime_checkable
class ModeComponent(HasSignature, HasOverlap, Protocol): ...


@runtime_checkable
class Path(ModeComponent, Protocol): ...


@runtime_checkable
class Polarization(ModeComponent, Protocol): ...


@runtime_checkable
class Envelope(ModeComponent, Protocol): ...


@runtime_checkable
class ModeLabel(HasSignature, HasOverlap, Protocol):
    @property
    def path(self) -> Path: ...
    @property
    def polarization(self) -> Polarization: ...
    @property
    def envelope(self) -> Envelope: ...

    def with_path(self, path: Path) -> Self: ...
    def with_polarization(self, polarization: Polarization) -> Self: ...
    def with_envelope(self, envelope: Envelope) -> Self: ...
