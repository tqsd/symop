from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.ops.operators import LadderOp


@runtime_checkable
class SupportsLeftWordAction(Protocol):
    @classmethod
    def zero(cls) -> Self: ...
    def apply_left(self, word: Iterable[LadderOp]) -> Self: ...


@runtime_checkable
class SupportsRightWordAction(Protocol):
    @classmethod
    def zero(cls) -> Self: ...
    def apply_right(self, word: Iterable[LadderOp]) -> Self: ...
