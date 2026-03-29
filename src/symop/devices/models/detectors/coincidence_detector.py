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
from dataclasses import dataclass, field

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
class CoincidenceDetector(MeasurementDeviceBase):
    r"""Semantic joint-number coincidence detector.

    This device performs a joint photon-number measurement across
    multiple logical detector ports.

    Parameters
    ----------
    input_ports:
        Names of logical input ports participating in the joint
        measurement.
    destructive:
        Whether selective measurements discard the measured subsystem.

    """

    input_ports: tuple[str, ...] = field(default_factory=lambda: ("in0", "in1"))
    destructive: bool = True

    def __post_init__(self) -> None:
        r"""Validate detector configuration.

        Raises
        ------
        ValueError
            If no input ports are configured or if input port names are not
            unique.

        """
        if not self.input_ports:
            raise ValueError("CoincidenceDetector requires at least one input port.")
        if len(set(self.input_ports)) != len(self.input_ports):
            raise ValueError("CoincidenceDetector input_ports must be unique.")

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
        return tuple(PortSpec(name, "in") for name in self.input_ports)

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
        selections = []

        for port_name in self.input_ports:
            try:
                path = ports[port_name]
            except KeyError as exc:
                raise KeyError(
                    f"Missing required coincidence input port {port_name!r}."
                ) from exc

            selections.append(
                MeasurementSelection(
                    port_name=port_name,
                    paths=(path,),
                )
            )

        return MeasurementTarget(selections=tuple(selections))

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

        resolution = MeasurementResolution(
            axes=("path",),
            grouping="joint_ports",
            readout="number",
        )

        return ProjectiveNumberMeasurementSpec(
            target=target,
            resolution=resolution,
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
        del state
        del selection
        del ctx

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
        del state
        del selection
        del ctx

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
        del state
        del selection
        del ctx

        spec = self._build_spec(ports=ports)
        return PostselectAction(
            measurement_spec=spec,
            selected_outcome=outcome,
            destructive=self.destructive,
        )
