from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class PathLabel:
    name: str

    def overlap(self, other: PathLabel) -> complex:
        return 1.0 + 0.0j if self.name == other.name else 0.0 + 0.0j

    def signature(self) -> Tuple[Any, ...]:
        return ("path", self.name)

    def approx_signature(self, **kw: Any) -> Tuple[Any, ...]:
        return self.signature()
