from __future__ import annotations

from typing import Protocol

from symop_proto.core.protocols import LabelProto


class PolraizationLabelProto(LabelProto, Protocol): ...


class PathLabelProto(LabelProto, Protocol): ...


class ModeLabelProto(LabelProto, Protocol):
    def with_pol(self, pol: PolraizationLabelProto) -> ModeLabelProto: ...
    def with_path(self, path: PathLabelProto) -> ModeLabelProto: ...
