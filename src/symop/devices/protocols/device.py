from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from symop.core.protocols.modes.labels import Path
from symop.core.protocols.states.base import State
from symop.devices.protocols.action import DeviceAction
from symop.devices.protocols.apply_context import ApplyContext
from symop.devices.protocols.port_spec import PortSpec
from symop.devices.types.device_kind import DeviceKind


@runtime_checkable
class Device(Protocol):
    @property
    def kind(self) -> DeviceKind: ...

    @property
    def port_specs(self) -> Sequence[PortSpec]: ...

    def plan(
        self,
        *,
        state: State,
        ports: Mapping[str, Path],
        selection: object | None,
        ctx: ApplyContext,
    ) -> DeviceAction: ...
