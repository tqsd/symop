from __future__ import annotations

from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

from symop.core.protocols.devices.action import DeviceAction
from symop.core.protocols.devices.apply_context import ApplyContext
from symop.core.protocols.states.capabilities import SupportsModeLabels

StateT = TypeVar("StateT", bound=SupportsModeLabels)


@runtime_checkable
class DeviceKernel(Protocol[StateT]):
    def __call__(
        self,
        *,
        state: StateT,
        action: DeviceAction,
        ctx: ApplyContext,
    ) -> StateT: ...


KernelFn: TypeAlias = DeviceKernel
