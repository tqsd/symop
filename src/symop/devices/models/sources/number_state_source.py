r"""Number-state source device.

This module defines :class:`NumberStateSource`, a source device that prepares
a fixed photon-number state in a designated output mode.

The planning stage constructs the emitted mode label on the selected output
path and records the requested excitation count for backend kernels.

Notes
-----
This first implementation models a pure number-state source. It does not
sample, mix, or condition on measurement outcomes.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symop.core.operators import ModeOp
from symop.core.protocols.modes.labels import (
    Envelope as EnvelopeProtocol,
)
from symop.core.protocols.modes.labels import (
    Path as PathProtocol,
)
from symop.core.protocols.modes.labels import (
    Polarization as PolarizationProtocol,
)
from symop.core.protocols.states.base import State as StateProtocol
from symop.devices.action import DeviceAction
from symop.devices.models.base import DeviceBase
from symop.devices.ports import PortSpec
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.types.device_kind import DeviceKind
from symop.modes.labels.mode import ModeLabel


@dataclass(frozen=True)
class NumberStateSource(DeviceBase):
    r"""Photon number-state source.

    This device prepares a pure number state on a single emitted output mode.
    The mode is defined by an envelope and a polarization label and is placed
    on the path connected to the ``"out"`` port.

    Parameters
    ----------
    envelope:
        Envelope assigned to the emitted output mode.
    polarization:
        Polarization assigned to the emitted output mode.
    n:
        Number of excitations emitted into the output mode.

    Raises
    ------
    ValueError
        If ``n < 0``.

    """

    envelope: EnvelopeProtocol
    polarization: PolarizationProtocol
    n: int

    def __post_init__(self) -> None:
        r"""Validate source parameters.

        Raises
        ------
        ValueError
            If the requested photon number is negative.

        """
        if self.n < 0:
            raise ValueError("NumberStateSource requires n >= 0.")

    @property
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier.

        Returns
        -------
        DeviceKind
            The number-state source device kind.

        """
        return DeviceKind.NUMBER_STATE_SOURCE

    @property
    def port_specs(self) -> tuple[PortSpec, ...]:
        r"""Return the port specification of the device.

        Returns
        -------
        tuple[PortSpec, ...]
            Single output port named ``"out"``.

        """
        return (PortSpec("out", "out"),)

    def plan(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DeviceAction:
        r"""Plan emission of a fixed number state.

        Parameters
        ----------
        state:
            Input state forwarded by the runtime. It is not used by this
            source planner.
        ports:
            Mapping from device-port names to paths. Must contain ``"out"``.
        selection:
            Optional selection object forwarded by the runtime. It is stored
            unchanged in the returned action.
        ctx:
            Optional apply context forwarded by the runtime. It is not used by
            this source planner.

        Returns
        -------
        DeviceAction
            Planned action containing emitted source modes and excitation
            counts for backend kernels.

            The returned action stores:

            - ``params["source_modes"]``:
              tuple containing the emitted mode
            - ``params["excitations_by_mode"]``:
              mapping from emitted mode signature to photon number

        Notes
        -----
        This planner creates the emitted mode directly on the output path.
        No label edits are required, because the source introduces a fresh mode
        rather than relabeling an existing one.

        """
        del state
        del ctx

        out_path = ports["out"]

        label = ModeLabel(
            path=out_path,
            polarization=self.polarization,
            envelope=self.envelope,
        )
        mode = ModeOp(label=label)

        return DeviceAction(
            ports=ports,
            kind=self.kind,
            selection=selection,
            params={
                "source_modes": (mode,),
                "excitations_by_mode": {
                    mode.signature: self.n,
                },
            },
            edits=(),
        )
