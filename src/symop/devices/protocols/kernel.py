from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from symop.core.protocols.states.base import State
from symop.devices.protocols.action import DeviceAction
from symop.devices.protocols.apply_context import ApplyContext

InStateT = TypeVar("InStateT", bound=State, contravariant=True)
OutStateT = TypeVar("OutStateT", bound=State, covariant=True)


@runtime_checkable
class DeviceKernel(Protocol[InStateT, OutStateT]):
    def __call__(
        self, *, state: InStateT, action: DeviceAction, ctx: ApplyContext
    ) -> OutStateT: ...


@runtime_checkable
class KernelFn(Protocol):
    def __call__(
        self,
        *,
        state: State,
        action: DeviceAction,
        ctx: ApplyContext,
    ) -> State: ...
