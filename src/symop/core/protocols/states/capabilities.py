from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.base.signature import Signature
from symop.core.protocols.modes.labels import ModeLabel, Path
from symop.core.protocols.ops.operators import ModeOp
from symop.core.protocols.states.state_kind import DensityState


@runtime_checkable
class SupportsTrace(Protocol):
    def trace(self) -> complex: ...


@runtime_checkable
class SupportsNormalizeTrace(SupportsTrace, Protocol):
    def normalize_trace(self, *, eps: float = 1e-14) -> Self: ...


@runtime_checkable
class SupportsToDensity(Protocol):
    def to_density(self) -> DensityState: ...


@runtime_checkable
class SupportsModeLabels(Protocol):
    @property
    def mode_labels(self) -> dict[Signature, ModeLabel]: ...

    def label_for_mode(self, mode_sig: Signature) -> ModeLabel: ...

    def apply_label_edits(self, edits: tuple[object, ...]) -> Self: ...

    @property
    def modes(self) -> tuple[ModeOp, ...]: ...

    def modes_on_path(self, path: Path) -> tuple[ModeOp, ...]: ...
