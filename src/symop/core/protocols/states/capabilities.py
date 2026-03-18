from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.base.signature import Signature
from symop.core.protocols.devices.label_edit import LabelEdit
from symop.core.protocols.modes.labels import (
    ModeLabel as ModeLabelProtocol,
)
from symop.core.protocols.modes.labels import (
    Path as PathProtocol,
)
from symop.core.protocols.ops.operators import ModeOp as ModeOpProtocol
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
    def mode_labels(self) -> Mapping[Signature, ModeLabelProtocol]: ...

    def label_for_mode(self, mode_sig: Signature) -> ModeLabelProtocol: ...

    def apply_label_edits(self, edits: Sequence[LabelEdit]) -> Self: ...

    @property
    def modes(self) -> tuple[ModeOpProtocol, ...]: ...

    def modes_on_path(self, path: PathProtocol) -> tuple[ModeOpProtocol, ...]: ...
