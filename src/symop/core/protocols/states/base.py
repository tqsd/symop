from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.types import RepKind, StateKind


@runtime_checkable
class State(Protocol):
    @property
    def rep_kind(self) -> RepKind: ...
    @property
    def state_kind(self) -> StateKind: ...
