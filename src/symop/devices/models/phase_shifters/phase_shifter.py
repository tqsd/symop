r"""Path phase shifter device.

This module defines :class:`PhaseShifter`, a single-path device that applies
a constant phase rotation to all modes on a selected path.

The planning stage records the target path and phase parameter for backend
execution. No operator-level transformation is performed during planning.

Notes
-----
The backend kernel is expected to realize the phase transformation at the
operator level, typically corresponding to the unitary

.. math::

    U(\phi) = e^{i \phi \hat{n}},

which induces the mapping

.. math::

    \hat{a}^\dagger \rightarrow e^{i\phi} \hat{a}^\dagger,
    \quad
    \hat{a} \rightarrow e^{-i\phi} \hat{a}.

Planning is purely semantic and does not modify amplitudes or apply
label edits.

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
from symop.devices.runtime import get_default_runtime
from symop.devices.types.device_kind import DeviceKind


@dataclass(frozen=True)
class PhaseShifter(DeviceBase):
    r"""Ideal path phase shifter.

    A phase shifter applies a constant phase rotation to all modes on a
    selected path. The transformation is represented at the operator level
    and affects the phase of creation and annihilation operators associated
    with the path.

    Parameters
    ----------
    phi:
        Phase angle (in radians) applied to the selected path.

    Notes
    -----
    - The phase is applied uniformly across all modes on the path.
    - Planning does not inspect the state or enumerate modes.
    - The actual operator-level rewrite is performed by the backend kernel.

    """

    phi: float

    @property
    def kind(self) -> DeviceKind:
        r"""Return the device kind identifier."""
        return DeviceKind.PHASE_SHIFTER

    @property
    def port_specs(self) -> tuple[PortSpec, ...]:
        r"""Return the port specification of the device.

        Returns
        -------
        tuple of PortSpec
            A single in-place port named ``"path"`` with direction ``"inout"``.

        Notes
        -----
        The phase shifter acts on a single path and does not create or
        destroy paths.

        """
        return (PortSpec("path", "inout"),)

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
            Input quantum state.
        ports:
            Mapping from device port names to path labels. Must contain
            the key ``"path"``.
        selection:
            Optional device-specific selection object. Not used by this device.
        runtime:
            Optional runtime instance. If not provided, the default runtime
            is used.
        ctx:
            Optional apply context forwarded to the runtime.
        out_kind:
            Optional requested output state kind.

        Returns
        -------
        StateProtocol
            Output state after applying the phase shifter.

        Notes
        -----
        This method delegates execution to the device runtime. The actual
        phase transformation is performed by a representation-specific kernel.

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

    def plan(
        self,
        *,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
    ) -> DeviceAction:
        r"""Plan a phase shift on a selected path.

        Parameters
        ----------
        state:
            Input state. Not inspected by this device.
        ports:
            Mapping from device-port names to path labels. Must contain
            the key ``"path"``.
        selection:
            Optional selection object forwarded by the runtime. Not used.
        ctx:
            Optional apply context forwarded by the runtime. Not used.

        Returns
        -------
        DeviceAction
            Planned action containing:

            - ``params["path"]``: target path label
            - ``params["phi"]``: phase angle
            - ``edits``: always empty

        Notes
        -----
        Planning is purely semantic and does not modify the state or labels.
        The backend kernel is responsible for applying the phase rotation
        to all modes associated with the selected path.

        """
        del state
        del selection
        del ctx

        return DeviceAction(
            ports=ports,
            kind=self.kind,
            selection=None,
            params={
                "path": ports["path"],
                "phi": float(self.phi),
            },
            edits=(),
            requires_kernel=True,
        )
