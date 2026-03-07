from __future__ import annotations

from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class HasOverlap(Protocol):
    def overlap(self, other: Self) -> complex: ...
