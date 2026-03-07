from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from symop.core.protocols.states.base import State


@runtime_checkable
class PolyState(State, Protocol):
    @property
    def rep_kind(self) -> Literal["poly"]: ...


@runtime_checkable
class GaussianState(State, Protocol):
    @property
    def rep_kind(self) -> Literal["gaussian"]: ...
