r"""Path beamsplitter device.

This module defines :class:`BeamSplitter`, a path-based two-port mixing
device that prepares mode-pair information for backend execution.

For each matched pair of modes on the two input paths, the planning stage
records a pair specification in ``action.params["pairs"]`` for the backend
kernel. The physical two-mode unitary is not applied during planning.

The high-level device parameter ``theta`` is interpreted through
``t = cos(theta)`` and ``r = sin(theta)``. Therefore ``theta = pi / 4``
corresponds to a balanced 50/50 beamsplitter.

Notes
-----
The backend kernel is expected to realize the full beamsplitter rewrite,
including creation of output-path modes. Planning is purely semantic and
does not modify amplitudes or apply label edits.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State as StateProtocol
from symop.core.types.state_kind import StateKind
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
class BeamSplitter(DeviceBase):
    r"""Ideal path beamsplitter device.

    A beamsplitter acts on matched mode pairs drawn from two input paths.
    Planning records the participating input modes, output paths, and
    beamsplitter parameters for backend execution.

    The device uses the package Heisenberg convention, where creation
    operators transform as

    .. math::

        \hat a^\dagger_{\mathrm{out},k}
        =
        \sum_j U_{k j}\,\hat a^\dagger_{\mathrm{in},j}.

    The angle ``theta`` determines the transmission and reflection
    amplitudes by

    .. math::

        t = \cos(\theta), \qquad r = \sin(\theta).

    With phases ``phi_t`` and ``phi_r``, the two-mode unitary is

    .. math::

        U =
        \begin{pmatrix}
            t e^{i\phi_t} & r e^{i\phi_r} \\
            -r e^{-i\phi_r} & t e^{-i\phi_t}
        \end{pmatrix}.

    Thus ``theta = pi / 4`` gives a balanced 50/50 beamsplitter.

    In this implementation, transmission corresponds to remaining on the same
    path index, while reflection corresponds to switching to the opposite path:

    - ``in0 -> out0``: transmission
    - ``in0 -> out1``: reflection
    - ``in1 -> out1``: transmission
    - ``in1 -> out0``: reflection, with the phase determined by the lower-left
    unitary element

    Thus a fully transmitting beamsplitter (theta = 0) leaves paths unchanged,
    while a fully reflecting beamsplitter (theta = pi/2) swaps the two paths.

    Parameters
    ----------
    theta:
        Mixing angle of the beamsplitter. The transmission and reflection
        amplitudes are ``cos(theta)`` and ``sin(theta)``, respectively.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.

    Examples
    --------
    Create a balanced (50/50) beamsplitter and apply it to two paths:

    >>> import numpy as np
    >>> bs = BeamSplitter(theta=np.pi / 4)
    >>> state_out = bs(
    ...     state_in,
    ...     ports={
    ...         "in0": Path("a"),
    ...         "in1": Path("b"),
    ...         "out0": Path("c"),
    ...         "out1": Path("d"),
    ...     },
    ... )

    Notes
    -----
    Planning does not relabel existing modes. The actual two-mode rewrite,
    including construction of output-path modes, is delegated to the runtime
    kernel through ``action.params["pairs"]``.

    """

    theta: float
    phi_t: float = 0.0
    phi_r: float = 0.0

    @property
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier."""
        return DeviceKind.BEAMSPLITTER

    @property
    def port_specs(self) -> tuple[PortSpec, ...]:
        r"""Return the port specification of the device."""
        return (
            PortSpec("in0", "in"),
            PortSpec("in1", "in"),
            PortSpec("out0", "out"),
            PortSpec("out1", "out"),
        )

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
        r"""Apply the device to a state through a runtime."""
        del out_kind
        rt = get_default_runtime() if runtime is None else runtime
        return rt.apply(
            device=self,
            state=state,
            ports=ports,
            selection=selection,
            ctx=ctx,
            out_kind=None,
        )

    def plan(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DeviceAction:
        r"""Plan beamsplitter mixing on matched mode pairs.

        Parameters
        ----------
        state:
            Input state whose mode labels are inspected.
        ports:
            Mapping from device-port names to path labels.
            Must contain ``"in0"``, ``"in1"``, ``"out0"``, and ``"out1"``.
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

            - ``params["pairs"]``: tuple of beamsplitter pair specifications
            - ``edits``: always empty for this device

        Raises
        ------
        TypeError
            If the state does not support path-based mode lookup.
        ValueError
            If both inputs are populated but contain incompatible numbers of
            modes.

        Notes
        -----
        The unitary itself is not applied here. This method only prepares
        semantic pairing and kernel parameters. If one side is missing, the
        backend kernel is expected to synthesize a matching vacuum partner.

        """
        del selection
        del ctx

        if not isinstance(state, LabelEditableState):
            raise TypeError(
                "Cannot inspect path-bound modes on this state implementation. "
                "Expected LabelEditableState"
            )

        in0 = ports["in0"]
        in1 = ports["in1"]
        out0 = ports["out0"]
        out1 = ports["out1"]

        modes0 = tuple(
            sorted(state.modes_on_path(in0), key=lambda m: repr(m.signature))
        )
        modes1 = tuple(
            sorted(state.modes_on_path(in1), key=lambda m: repr(m.signature))
        )

        pairs: list[dict[str, object | None]] = []

        if not modes0 and not modes1:
            return DeviceAction(
                ports=ports,
                kind=self.kind,
                selection=None,
                params={"pairs": ()},
                edits=(),
            )

        if not modes0:
            for mode1 in modes1:
                pairs.append(
                    {
                        "mode0": None,
                        "mode1": mode1.signature,
                        "in0": in0,
                        "in1": in1,
                        "out0": out0,
                        "out1": out1,
                        "theta": float(self.theta),
                        "phi_t": float(self.phi_t),
                        "phi_r": float(self.phi_r),
                    }
                )

        elif not modes1:
            for mode0 in modes0:
                pairs.append(
                    {
                        "mode0": mode0.signature,
                        "mode1": None,
                        "in0": in0,
                        "in1": in1,
                        "out0": out0,
                        "out1": out1,
                        "theta": float(self.theta),
                        "phi_t": float(self.phi_t),
                        "phi_r": float(self.phi_r),
                    }
                )

        else:
            if len(modes0) != len(modes1):
                raise ValueError(
                    "Beamsplitter planning requires matching mode counts when "
                    "both input paths are populated"
                )

            for mode0, mode1 in zip(modes0, modes1, strict=True):
                pairs.append(
                    {
                        "mode0": mode0.signature,
                        "mode1": mode1.signature,
                        "in0": in0,
                        "in1": in1,
                        "out0": out0,
                        "out1": out1,
                        "theta": float(self.theta),
                        "phi_t": float(self.phi_t),
                        "phi_r": float(self.phi_r),
                    }
                )

        return DeviceAction(
            ports=ports,
            kind=self.kind,
            selection=None,
            params={"pairs": tuple(pairs)},
            edits=(),
        )
