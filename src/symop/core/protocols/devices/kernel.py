from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from symop.core.protocols.devices.action import DeviceAction
from symop.core.protocols.devices.apply_context import ApplyContext
from symop.core.protocols.states.base import State
from symop.core.protocols.states.capabilities import SupportsModeLabels

StateT = TypeVar("StateT", bound=SupportsModeLabels)


class ActionableState(SupportsModeLabels, State, Protocol): ...


@runtime_checkable
class DeviceKernel(Protocol[StateT]):
    def __call__(
        self,
        *,
        state: StateT,
        action: DeviceAction,
        ctx: ApplyContext,
    ) -> StateT: ...


KernelFn = DeviceKernel[ActionableState]
