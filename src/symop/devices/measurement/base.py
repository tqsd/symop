"""Abstract base class for semantic measurement devices.

A measurement device defines a stable semantic interface for three
measurement-style operations:

- observe: return an outcome distribution
- detect: return a detector-style classical record
- postselect: condition on a chosen outcome and return the surviving
  branch

Concrete subclasses implement semantic planning methods, while runtime
evaluation is delegated to a measurement runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State as StateProtocol
from symop.devices.measurement.action import (
    DetectAction,
    ObserveAction,
    PostselectAction,
)
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.protocols.port_spec import PortSpec as PortSpecProtocol
from symop.devices.protocols.runtime import (
    DeviceRuntime as DeviceRuntimeProtocol,
)
from symop.devices.runtime import get_default_runtime
from symop.devices.types.device_kind import DeviceKind


class MeasurementDeviceBase(ABC):
    r"""Abstract base class for semantic measurement devices.

    Subclasses define a stable device kind, a declared port interface,
    and semantic planning methods for observation, detection, and
    postselection.

    Notes
    -----
    Runtime evaluation is delegated to a measurement runtime. If no
    runtime is supplied explicitly, the default measurement runtime is
    used.

    """

    destructive: bool = True

    @property
    @abstractmethod
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def port_specs(self) -> Sequence[PortSpecProtocol]:
        r"""Return the declared port specifications for the device."""
        raise NotImplementedError

    def observe(
        self,
        state: StateProtocol,
        *,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        runtime: DeviceRuntimeProtocol | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> object:
        r"""Evaluate an observation query on the given state.

        Parameters
        ----------
        state:
            Input quantum state.
        ports:
            Mapping from logical port names to path labels.
        selection:
            Optional device-specific configuration object.
        runtime:
            Measurement runtime used to evaluate the action. If ``None``,
            the default measurement runtime is used.
        ctx:
            Optional context shared across planning and execution.

        Returns
        -------
        object
            Concrete observation result returned by the runtime.

        """
        rt = get_default_runtime() if runtime is None else runtime
        return rt.observe(
            device=self,
            state=state,
            ports=ports,
            selection=selection,
            ctx=ctx,
        )

    def detect(
        self,
        state: StateProtocol,
        *,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        runtime: DeviceRuntimeProtocol | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> object:
        r"""Evaluate a detection query on the given state.

        Parameters
        ----------
        state:
            Input quantum state.
        ports:
            Mapping from logical port names to path labels.
        selection:
            Optional device-specific configuration object.
        runtime:
            Measurement runtime used to evaluate the action. If ``None``,
            the default measurement runtime is used.
        ctx:
            Optional context shared across planning and execution.

        Returns
        -------
        object
            Concrete detection result returned by the runtime.

        """
        rt = get_default_runtime() if runtime is None else runtime
        return rt.detect(
            device=self,
            state=state,
            ports=ports,
            selection=selection,
            ctx=ctx,
        )

    def postselect(
        self,
        state: StateProtocol,
        *,
        outcome: MeasurementOutcome,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        runtime: DeviceRuntimeProtocol | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> object:
        r"""Evaluate a postselection query on the given state.

        Parameters
        ----------
        state:
            Input quantum state.
        outcome:
            Outcome on which the state should be conditioned.
        ports:
            Mapping from logical port names to path labels.
        selection:
            Optional device-specific configuration object.
        runtime:
            Measurement runtime used to evaluate the action. If ``None``,
            the default measurement runtime is used.
        ctx:
            Optional context shared across planning and execution.

        Returns
        -------
        object
            Concrete postselection result returned by the runtime.

        """
        rt = get_default_runtime() if runtime is None else runtime
        return rt.postselect(
            device=self,
            state=state,
            outcome=outcome,
            ports=ports,
            selection=selection,
            ctx=ctx,
        )

    @abstractmethod
    def plan_observe(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> ObserveAction:
        r"""Plan a semantic observation action."""
        raise NotImplementedError

    @abstractmethod
    def plan_detect(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DetectAction:
        r"""Plan a semantic detection action."""
        raise NotImplementedError

    @abstractmethod
    def plan_postselect(
        self,
        *,
        state: StateProtocol,
        outcome: MeasurementOutcome,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> PostselectAction:
        r"""Plan a semantic postselection action."""
        raise NotImplementedError
