from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.base.signature import HasSignature
from symop.core.protocols.modes.labels import (
    Envelope,
    ModeLabel,
    Path,
    Polarization,
)
from symop.core.types import OperatorKind


@runtime_checkable
class ModeOp(HasSignature, Protocol):
    label: ModeLabel
    user_label: str | None
    display_index: int | None

    @property
    def ann(self) -> LadderOp: ...
    @property
    def cre(self) -> LadderOp: ...

    @property
    def create(self) -> LadderOp: ...
    @property
    def annihilate(self) -> LadderOp: ...

    def with_user_label(self, tag: str) -> Self: ...
    def with_index(self, idx: int) -> Self: ...
    def with_label(self, label: ModeLabel) -> Self: ...

    def with_envelope(self, envelope: Envelope) -> Self: ...
    def with_polarization(self, polarization: Polarization) -> Self: ...
    def with_path(self, path: Path) -> Self: ...


@runtime_checkable
class LadderOp(HasSignature, Protocol):
    kind: OperatorKind
    mode: ModeOp

    @property
    def is_annihilation(self) -> bool: ...

    @property
    def is_creation(self) -> bool: ...

    def dagger(self) -> Self: ...
    def commutator(self, other: LadderOp) -> complex: ...
