r"""Polarizing filter device.

This module defines :class:`PolarizingFilter`, a path-based device that
projects mode polarizations onto a selected transmitted polarization and
records the corresponding attenuation parameters for backend execution.

For each mode on the selected input path, the planning stage computes an
effective transmissivity from the overlap between the input polarization
label and the configured transmitted polarization. The physical attenuation
is not applied during planning. Instead, the planner stores per-mode
transmissivities in ``action.params["eta_by_mode"]`` for the backend kernel
and emits label edits that update output mode descriptors.

Notes
-----
For an ideal polarizing filter with transmitted polarization
:math:`|\psi\rangle` and input polarization :math:`|\phi\rangle`, the
effective transmissivity is

.. math::

    \eta = |\langle \psi \mid \phi \rangle|^2.

The backend kernel is expected to realize the attenuation channel, for
example through a pure-loss dilation followed by tracing out the
environment mode.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symop.core.protocols.devices.label_edit import SetModeLabel
from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.modes.labels import (
    Polarization as PolarizationProtocol,
)
from symop.core.protocols.states.base import State as StateProtocol
from symop.core.types.signature import Signature
from symop.core.types.state_kind import DENSITY, StateKind
from symop.devices.action import DeviceAction
from symop.devices.models.base import DeviceBase
from symop.devices.ports import PortSpec
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.protocols.runtime import (
    DeviceRuntime as DeviceRuntimeProtocol,
)
from symop.devices.protocols.state import LabelEditableState
from symop.devices.runtime import get_default_runtime
from symop.devices.types.device_kind import DeviceKind


@dataclass(frozen=True)
class PolarizingFilter(DeviceBase):
    r"""Ideal polarizing filter device.

    A polarizing filter acts on all modes on a selected input path by
    projecting their polarization content onto a configured transmitted
    polarization. The updated mode labels are routed to the output path,
    while the corresponding attenuation strengths are recorded for the
    backend kernel.

    Parameters
    ----------
    passed_polarization:
        Polarization transmitted by the filter.

    Notes
    -----
    Planning performs descriptor updates only. The actual physical
    attenuation is delegated to the runtime kernel through
    ``action.params["eta_by_mode"]``.

    """

    passed_polarization: PolarizationProtocol

    @property
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier.

        Returns
        -------
        DeviceKind
            The polarizing-filter device kind.

        """
        return DeviceKind.POLARIZING_FILTER

    @property
    def port_specs(self) -> tuple[PortSpec, ...]:
        r"""Return the port specification of the device.

        Returns
        -------
        tuple[PortSpec, ...]
            Two-port specification consisting of one input port named
            ``"in"`` and one output port named ``"out"``.

        """
        return (PortSpec("in", "in"), PortSpec("out", "out"))

    def apply(
        self,
        state: StateProtocol,
        *,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        runtime: DeviceRuntimeProtocol | None = None,
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
        del out_kind
        rt = get_default_runtime() if runtime is None else runtime
        dense_out_kind: StateKind = DENSITY
        return rt.apply(
            device=self,
            state=state,
            ports=ports,
            selection=selection,
            ctx=ctx,
            out_kind=dense_out_kind,
        )

    def plan(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DeviceAction:
        r"""Plan ideal polarization filtering on all modes of the input path.

        For every mode on ``ports["in"]``, this method:

        1. computes the overlap with the transmitted polarization,
        2. converts that overlap into an effective transmissivity,
        3. emits a label edit redirecting the mode to ``ports["out"]``
           with the transmitted polarization label.

        Parameters
        ----------
        state:
            Input state whose mode labels are inspected.
        ports:
            Mapping from device-port names to physical or logical paths.
            Must contain the keys ``"in"`` and ``"out"``.
        selection:
            Optional selection object forwarded by the runtime. It is not used
            by this device.
        ctx:
            Optional apply context forwarded by the runtime. It is not used by
            this device.

        Returns
        -------
        DeviceAction
            Planned action containing:

            - ``params["eta_by_mode"]``: transmissivity per affected mode
            - ``edits``: label updates for all transformed modes

        Raises
        ------
        TypeError
            If the state does not support label editing and path-based mode
            lookup.

        Notes
        -----
        The attenuation channel itself is not applied here. This method only
        prepares the semantic and kernel parameters required for execution.

        """
        del ctx
        del selection

        if not isinstance(state, LabelEditableState):
            raise TypeError(
                "Cannot modify labels on this state implementation. "
                "Expected LabelEditableState"
            )

        in_path = ports["in"]
        out_path = ports["out"]

        eta_by_mode: dict[Signature, float] = {}
        edits: list[SetModeLabel] = []

        for mode in state.modes_on_path(in_path):
            pol_in = mode.label.polarization
            eta = abs(self.passed_polarization.overlap(pol_in)) ** 2

            new_label = mode.label.with_polarization(
                self.passed_polarization
            ).with_path(out_path)
            edits.append(SetModeLabel(mode_sig=mode.signature, label=new_label))
            eta_by_mode[mode.signature] = float(eta)

        return DeviceAction(
            ports=ports,
            kind=self.kind,
            selection=None,
            params={"eta_by_mode": eta_by_mode},
            edits=tuple(edits),
        )
