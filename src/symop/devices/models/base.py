r"""Abstract base class for semantic devices.

Defines the common interface for device objects that participate in the
device planning and runtime-application pipeline.

A device declares its kind and port interface, provides a semantic
planning step via :meth:`plan`, and can be applied to a state through a
runtime that resolves the appropriate representation-specific kernel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State as StateProtocol
from symop.core.types.state_kind import StateKind
from symop.devices.action import DeviceAction
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.protocols.port_spec import PortSpec as PortSpecProtocol
from symop.devices.protocols.runtime import (
    DeviceRuntime as DeviceRuntimeProtocol,
)
from symop.devices.runtime import get_default_runtime
from symop.devices.types.device_kind import DeviceKind


class DeviceBase(ABC):
    r"""Abstract base class for semantic devices.

    Subclasses define a stable device kind, a declared port interface,
    and a planning method that converts a device invocation into a
    semantic :class:`DeviceAction`.

    Notes
    -----
    Device application is performed via a runtime. If no runtime is
    provided explicitly, the default runtime is used.

    """

    @property
    @abstractmethod
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier.

        Returns
        -------
        DeviceKind
            Stable kind identifier for this device.

        """
        """Device kind identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def port_specs(self) -> Sequence[PortSpecProtocol]:
        r"""Return the declared port specifications for the device.

        Returns
        -------
        Sequence[PortSpecProtocol]
            Ordered collection of port specifications describing the
            device interface.

        """
        raise NotImplementedError

    def apply(
        self,
        state: StateProtocol,
        *,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        runtime: DeviceRuntimeProtocol | None,
        ctx: ApplyContextProtocol | None = None,
        out_kind: StateKind | None = None,
    ) -> StateProtocol:
        r"""Apply the device to a state through a runtime.

        Parameters
        ----------
        state:
            Input state to which the device is applied.
        ports:
            Mapping from logical port names to path labels.
        selection:
            Optional device-specific selection or configuration object.
        runtime:
            Runtime used to execute the device application. If ``None``,
            the default runtime is used.
        ctx:
            Optional apply context shared across planning and execution.
        out_kind:
            Optional requested output state kind.

        Returns
        -------
        StateProtocol
            Output state returned by the runtime.

        Notes
        -----
        This method first selects a runtime and then delegates execution
        to ``runtime.apply(...)``.

        """
        rt = get_default_runtime() if runtime is None else runtime
        return rt.apply(
            device=self,
            state=state,
            ports=ports,
            selection=selection,
            ctx=ctx,
            out_kind=out_kind,
        )

    @abstractmethod
    def plan(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DeviceAction:
        r"""Plan a semantic device action for the given invocation.

        Parameters
        ----------
        state:
            Input state for which the device action is planned.
        ports:
            Mapping from logical port names to path labels.
        selection:
            Optional device-specific selection or configuration object.
        ctx:
            Optional apply context that may be used during planning.

        Returns
        -------
        DeviceAction
            Semantic action describing how the device should be applied.

        Notes
        -----
        Subclasses implement this method to translate high-level device
        invocation into a representation-independent action.

        """
        pass

    def __call__(
        self,
        state: StateProtocol,
        *,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        runtime: DeviceRuntimeProtocol | None = None,
        ctx: ApplyContextProtocol | None = None,
        out_kind: StateKind | None = None,
    ) -> StateProtocol:
        r"""Apply the device using call syntax.

        Parameters
        ----------
        state:
            Input state to which the device is applied.
        ports:
            Mapping from logical port names to path labels.
        selection:
            Optional device-specific selection or configuration object.
        runtime:
            Runtime used to execute the device application. If ``None``,
            the default runtime is used.
        ctx:
            Optional apply context shared across planning and execution.
        out_kind:
            Optional requested output state kind.

        Returns
        -------
        StateProtocol
            Output state returned by :meth:`apply`.

        Notes
        -----
        This is a convenience wrapper around :meth:`apply`.

        """
        return self.apply(
            state,
            ports=ports,
            selection=selection,
            runtime=runtime,
            ctx=ctx,
            out_kind=out_kind,
        )
