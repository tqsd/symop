from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from symop_proto.labels.protocols import PathLabelProto


@dataclass(frozen=True)
class PathLabel(PathLabelProto):
    name: str

    def overlap(self, other: PathLabel) -> complex:
        return 1.0 + 0.0j if self.name == other.name else 0.0 + 0.0j

    @property
    def signature(self) -> tuple[Any, ...]:
        return ("path", self.name)

    def approx_signature(self, **kw: Any) -> tuple[Any, ...]:
        return self.signature
