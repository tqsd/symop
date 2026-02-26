from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols import LabelProto


@runtime_checkable
class PolarizationLabelProto(LabelProto, Protocol):
    pass


@runtime_checkable
class PathLabelProto(LabelProto, Protocol):
    pass


@runtime_checkable
class ModeLabelProto(LabelProto, Protocol):
    def with_pol(self, pol: PolarizationLabelProto) -> ModeLabelProto: ...
    def with_path(self, path: PathLabelProto) -> ModeLabelProto: ...
