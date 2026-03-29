r"""Path delay device model.

This module defines :class:`Delay`, a semantic device that applies a
temporal shift to all modes on a given input path.

The delay is implemented at the semantic level by modifying the temporal
envelope associated with each mode. Specifically, the device produces
label edits that replace each mode's envelope with a delayed version and
optionally redirect the mode to a new output path.

Notes
-----
The delay operation requires envelopes that support temporal shifting.
Therefore, only envelopes implementing
:class:`~symop.modes.protocols.envelope.TimeFrequencyEnvelope` are
supported. A :class:`TypeError` is raised otherwise.

This device does not require a backend kernel. The transformation is
fully expressed via label edits during the planning stage.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symop.core.protocols.devices.label_edit import SetModeLabel
from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State as StateProtocol
from symop.devices.action import DeviceAction
from symop.devices.models.base import DeviceBase
from symop.devices.ports import PortSpec
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.protocols.state import LabelEditableState
from symop.devices.types.device_kind import DeviceKind
from symop.modes.protocols.envelope import TimeFrequencyEnvelope


@dataclass(frozen=True)
class Delay(DeviceBase):
    r"""Path delay device.

    Applies a temporal delay to all modes on the input path by shifting
    their associated envelopes in time.

    Parameters
    ----------
    dt:
        Time delay applied to each mode envelope.

    """

    dt: float

    @property
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier."""
        return DeviceKind.DELAY

    @property
    def port_specs(self) -> tuple[PortSpec, ...]:
        """Return the declared port interface.

        Returns
        -------
        tuple[PortSpec, ...]
            Two ports: ``"in"`` and ``"out"``i

        """
        return (PortSpec("in", "in"), PortSpec("out", "out"))

    def plan(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DeviceAction:
        r"""Plan a delay action as label edits.

        Parameters
        ----------
        state:
            Input quantum state.
        ports:
            Mapping from device ports to paths.
        selection:
            Unused.
        ctx:
            Unused.

        Returns
        -------
        DeviceAction
            Action containing label edits that apply the delay.

        """
        del selection
        del ctx

        if not isinstance(state, LabelEditableState):
            raise TypeError(
                "Cannot modify labels on this state implementation. "
                "Expected LabelEditableState"
            )

        in_path = ports["in"]
        out_path = ports["out"]

        edits: list[SetModeLabel] = []
        for mode in state.modes_on_path(in_path):
            envelope = mode.label.envelope
            if not isinstance(envelope, TimeFrequencyEnvelope):
                raise TypeError("Delay requires envelopes supporting temporal delay.")
            env_out = envelope.delayed(self.dt)
            new_label = mode.label.with_envelope(env_out).with_path(out_path)
            edits.append(SetModeLabel(mode_sig=mode.signature, label=new_label))

        return DeviceAction(
            ports=ports,
            kind=self.kind,
            params={"dt": float(self.dt)},
            edits=tuple(edits),
            selection=None,
            requires_kernel=False,
        )
