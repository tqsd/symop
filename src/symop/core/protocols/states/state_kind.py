from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from symop.core.protocols.states.base import State


@runtime_checkable
class KetState(State, Protocol):
    @property
    def state_kind(self) -> Literal["ket"]: ...


@runtime_checkable
class DensityState(State, Protocol):
    @property
    def state_kind(self) -> Literal["density"]: ...
