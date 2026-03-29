r"""Number-resolving measurement device model.

This module defines :class:`NumberDetector`, a semantic measurement
device that performs projective number measurements on selected modes.

The device constructs a semantic measurement specification describing
a number-resolving measurement over the input port. The resulting
semantic actions are later evaluated by backend-specific measurement
kernels.

Notes
-----
The detector supports three types of measurement queries:

- observation: returns outcome probabilities,
- detection: samples an outcome and optionally applies a state update,
- postselection: conditions the state on a chosen outcome.

The measurement is defined via
:class:`ProjectiveNumberMeasurementSpec` and operates on the
selected input paths.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State as StateProtocol
from symop.devices.measurement.action import (
    DetectAction,
    ObserveAction,
    PostselectAction,
)
from symop.devices.measurement.base import MeasurementDeviceBase
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.measurement.resolution import MeasurementResolution
from symop.devices.measurement.specs import ProjectiveNumberMeasurementSpec
from symop.devices.measurement.target import (
    MeasurementSelection,
    MeasurementTarget,
)
from symop.devices.ports import PortSpec
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.types.device_kind import DeviceKind


@dataclass(frozen=True)
class NumberDetector(MeasurementDeviceBase):
    r"""Semantic number-measurement device.

    This device performs projective number measurements on modes
    associated with its input port.

    Parameters
    ----------
    destructive:
        Whether the measurement should discard the measured subsystem
        after a selective update (for detection and postselection).

    """

    destructive: bool = True

    @property
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier."""
        return DeviceKind.NUMBER_DETECTOR

    @property
    def port_specs(self) -> tuple[PortSpec, ...]:
        r"""Return the declared port interface.

        Returns
        -------
        tuple[PortSpec, ...]
            Single input port named ``"in"``.

        """
        return (PortSpec("in", "in"),)

    def _build_target(
        self,
        *,
        ports: Mapping[str, PathProtocol],
    ) -> MeasurementTarget:
        r"""Construct the measurement target.

        Parameters
        ----------
        ports:
            Mapping from device ports to paths.

        Returns
        -------
        MeasurementTarget
            Target describing the selected input modes.

        """
        return MeasurementTarget(
            selections=(
                MeasurementSelection(
                    port_name="in",
                    paths=(ports["in"],),
                ),
            ),
        )

    def _build_spec(
        self,
        *,
        ports: Mapping[str, PathProtocol],
    ) -> ProjectiveNumberMeasurementSpec:
        r"""Construct the number-measurement specification.

        Parameters
        ----------
        ports:
            Mapping from device ports to paths.

        Returns
        -------
        ProjectiveNumberMeasurementSpec
            Measurement specification defining a number-resolving measurement.

        """
        target = self._build_target(ports=ports)
        return ProjectiveNumberMeasurementSpec(
            target=target,
            resolution=MeasurementResolution(),
        )

    def plan_observe(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> ObserveAction:
        r"""Plan an observation query.

        Parameters
        ----------
        state:
            Input quantum state.
        ports:
            Mapping from device ports to paths.
        selection:
            Optional selection object (unused).
        ctx:
            Optional apply context (unused).

        Returns
        -------
        ObserveAction
            Semantic action requesting outcome probabilities.

        """
        spec = self._build_spec(ports=ports)
        return ObserveAction(
            measurement_spec=spec,
            destructive=False,
        )

    def plan_detect(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DetectAction:
        r"""Plan a detection query.

        Parameters
        ----------
        state:
            Input quantum state.
        ports:
            Mapping from device ports to paths.
        selection:
            Optional selection object (unused).
        ctx:
            Optional apply context (unused).

        Returns
        -------
        DetectAction
            Semantic action requesting a sampled measurement outcome.

        """
        spec = self._build_spec(ports=ports)
        return DetectAction(
            measurement_spec=spec,
            destructive=self.destructive,
        )

    def plan_postselect(
        self,
        *,
        state: StateProtocol,
        outcome: MeasurementOutcome,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> PostselectAction:
        r"""Plan a postselection query.

        Parameters
        ----------
        state:
            Input quantum state.
        outcome:
            Measurement outcome to condition on.
        ports:
            Mapping from device ports to paths.
        selection:
            Optional selection object (unused).
        ctx:
            Optional apply context (unused).

        Returns
        -------
        PostselectAction
            Semantic action conditioning the state on the selected outcome.

        """
        spec = self._build_spec(ports=ports)
        return PostselectAction(
            measurement_spec=spec,
            selected_outcome=outcome,
            destructive=self.destructive,
        )
