from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State
from symop.core.types.state_kind import StateKind
from symop.devices.protocols.apply_context import ApplyContext
from symop.devices.protocols.device import Device
from symop.devices.protocols.registry import (
    KernelRegistry as KernelRegistryProtocol,
)


@runtime_checkable
class DeviceRuntime(Protocol):
    """
    Protocol for device runtime.

    A device runtime orchestrates semantic planning and
    representation-specific execution for a device application.
    """

    @property
    def registry(self) -> KernelRegistryProtocol: ...

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
