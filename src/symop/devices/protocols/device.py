from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from symop.core.protocols.modes.labels import Path
from symop.core.protocols.states.base import State
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.protocols.action import (
    DetectAction,
    DeviceAction,
    ObserveAction,
    PostselectAction,
)
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


@runtime_checkable
class MeasurementDevice(Protocol):
    @property
    def kind(self) -> DeviceKind: ...

    @property
    def port_specs(self) -> Sequence[PortSpec]: ...

    def plan_observe(
        self,
        *,
        state: State,
        ports: Mapping[str, Path],
        selection: object | None = None,
        ctx: ApplyContext | None = None,
    ) -> ObserveAction: ...

    def plan_detect(
        self,
        *,
        state: State,
        ports: Mapping[str, Path],
        selection: object | None = None,
        ctx: ApplyContext | None = None,
    ) -> DetectAction: ...

    def plan_postselect(
        self,
        *,
        state: State,
        outcome: MeasurementOutcome,
        ports: Mapping[str, Path],
        selection: object | None = None,
        ctx: ApplyContext | None = None,
    ) -> PostselectAction: ...
