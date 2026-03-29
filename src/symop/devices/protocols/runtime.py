from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State
from symop.core.types.state_kind import StateKind
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.protocols.apply_context import ApplyContext
from symop.devices.protocols.device import Device, MeasurementDevice
from symop.devices.protocols.registry import (
    KernelRegistry as KernelRegistryProtocol,
)
from symop.devices.protocols.registry import (
    MeasurementKernelRegistry as MeasurementKernelRegistryProtocol,
)
from symop.devices.protocols.result import (
    DetectionResult as DetectionResultProtocol,
)
from symop.devices.protocols.result import (
    ObservationResult as ObservationResultProtocol,
)
from symop.devices.protocols.result import (
    PostselectionResult as PostselectionResultProtocol,
)


@runtime_checkable
class DeviceRuntime(Protocol):
    """
    Protocol for device runtime.

    A device runtime orchestrates semantic planning and
    representation-specific execution for a device application.
    """

    @property
    def device_registry(self) -> KernelRegistryProtocol: ...

    @property
    def measurement_registry(self) -> MeasurementKernelRegistryProtocol: ...
    def apply(
        self,
        *,
        device: Device,
        state: State,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContext | None = None,
        out_kind: StateKind | None = None,
    ) -> State:
        """
        Apply a semantic device to a state.

        Parameters
        ----------
        device: Device
            (DeviceProtocol) to apply.
        state: State
            (StateProtocol) Input state.
        ports: Mapping[str, Path]
            Binding from semantic device port names to paths (PathProtocol)
        selection: object | None, optional
            Optional selection objectc used by the device planner.
        ctx: ApplyContext | None, optional
            Optional planning/execution context.
        out_kind: StateKind | None, optional
            Desired output state kind. If omitted, the runtime
            preserves the input state kind unless the selected kernel
            implement a different default behavior.

        Return
        ------
        State
            Output state (StateProtocol) after device application.

        """
        ...

    def observe(
        self,
        *,
        device: MeasurementDevice,
        state: State,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContext | None = None,
    ) -> ObservationResultProtocol:
        """Evaluate an observation query for a measurement device."""
        ...

    def detect(
        self,
        *,
        device: MeasurementDevice,
        state: State,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContext | None = None,
    ) -> DetectionResultProtocol:
        """Evaluate an observation query for a measurement device."""
        ...

    def postselect(
        self,
        *,
        device: MeasurementDevice,
        state: State,
        outcome: MeasurementOutcome,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContext | None = None,
    ) -> PostselectionResultProtocol:
        """Evaluate a postelectin query for a measurement device."""
        ...
