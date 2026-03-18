r"""Device runtime for applying planned device actions to states.

This module provides the concrete runtime responsible for validating
device port assignments, planning device actions, dispatching to the
appropriate representation-specific kernel, and applying any resulting
label edits to the output state.

It also exposes a lazily initialized default runtime with automatic
kernel registration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.states.base import State as StateProtocol
from symop.core.types.state_kind import StateKind
from symop.devices.apply_context import SimpleApplyContext
from symop.devices.dispatch import dispatch_apply
from symop.devices.protocols.action import DeviceAction as DeviceActionProtocol
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.protocols.device import Device as DeviceProtocol
from symop.devices.protocols.registry import (
    KernelRegistry as KernelRegistryProtocol,
)
from symop.devices.protocols.runtime import (
    DeviceRuntime as DeviceRuntimeProtocol,
)
from symop.devices.protocols.state import LabelEditableState
from symop.devices.registry import KernelRegistry

_default_runtime: DeviceRuntimeProtocol | None = None
_registered: bool = False


def _validate_ports(
    *,
    device: DeviceProtocol,
    ports: Mapping[str, PathProtocol],
) -> None:
    r"""Validate the provided port mapping against device port specs.

    Parameters
    ----------
    device:
        Device whose declared port specifications are used for validation.
    ports:
        Mapping from port name to bound path label.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If an unknown port name is supplied or if a required port is missing.

    Notes
    -----
    Validation checks that:

    - all provided port names are declared by the device
    - all required device ports are present in ``ports``

    """
    specs = {spec.name: spec for spec in device.port_specs}

    unknown = set(ports) - set(specs)
    if unknown:
        raise KeyError(f"Unknown ports for device {device.kind!r}: {sorted(unknown)!r}")
    missing = [
        spec.name for spec in specs.values() if spec.required and spec.name not in ports
    ]
    if missing:
        raise KeyError(
            f"Missing required ports for device {device.kind!r}: {missing!r}"
        )


@dataclass(frozen=True)
class DeviceRuntime:
    r"""Concrete runtime for executing device actions on states.

    Parameters
    ----------
    registry:
        Kernel registry used to resolve representation- and device-specific
        apply kernels.

    Notes
    -----
    The runtime performs four main steps:

    - validate device ports
    - construct or reuse an apply context
    - request a semantic action from the device planner
    - dispatch to the kernel layer and apply any label edits

    """

    registry: KernelRegistryProtocol

    def apply(
        self,
        *,
        device: DeviceProtocol,
        state: StateProtocol,
        ports: Mapping[str, PathProtocol],
        selection: object | None = None,
        ctx: ApplyContextProtocol | None = None,
        out_kind: StateKind | None = None,
    ) -> StateProtocol:
        r"""Apply a device to a state through planning and kernel dispatch.

        Parameters
        ----------
        device:
            Device to apply.
        state:
            Input quantum state.
        ports:
            Mapping from device port names to path labels.
        selection:
            Optional device-specific selection or configuration object forwarded
            to the planning stage.
        ctx:
            Optional apply context. If omitted, a fresh
            :class:`SimpleApplyContext` is created.
        out_kind:
            Optional requested output state kind.

        Returns
        -------
        StateProtocol
            Output state produced by the dispatched kernel, after any label
            edits have been applied.

        Raises
        ------
        KeyError
            If the supplied port mapping is invalid.
        TypeError
            If label edits are requested for a state type that does not support
            them.

        Notes
        -----
        The device is first asked to produce a semantic action via
        ``device.plan(...)``. That action is then executed by the kernel
        dispatch layer.

        """
        _validate_ports(device=device, ports=ports)

        ctx2 = SimpleApplyContext() if ctx is None else ctx

        action = device.plan(state=state, ports=ports, selection=selection, ctx=ctx2)

        state2 = dispatch_apply(
            registry=self.registry,
            device=device,
            state=state,
            action=action,
            ctx=ctx2,
            out_kind=out_kind,
        )

        return self._apply_label_edits(state2, action)

    @staticmethod
    def _apply_label_edits(
        state: StateProtocol, action: DeviceActionProtocol
    ) -> StateProtocol:
        r"""Apply label edits from a device action to a state.

        Parameters
        ----------
        state:
            State to update.
        action:
            Device action carrying label edits.

        Returns
        -------
        StateProtocol
            Either the unchanged input state, or an updated state with label
            edits applied.

        Raises
        ------
        TypeError
            If label edits are present but the state does not implement the
            required label-editing interface.

        Notes
        -----
        If ``action.edits`` is empty, the input state is returned unchanged.

        """
        if not action.edits:
            return state
        if not isinstance(state, LabelEditableState):
            raise TypeError(
                f"State type {type(state).__name__} does not support label edits."
            )
        state_editable = state
        return state_editable.apply_label_edits(action.edits)


def _register_all_kernels(registry: KernelRegistryProtocol) -> None:
    r"""Register all available device kernels into a registry.

    Parameters
    ----------
    registry:
        Kernel registry to populate.

    Returns
    -------
    None

    Notes
    -----
    Registration currently delegates to the polynomial kernel registry.

    """
    from symop.polynomial.device_kernels.registry import (
        register_polynomial_kernels,
    )

    register_polynomial_kernels(registry)


def get_default_runtime() -> DeviceRuntimeProtocol:
    r"""Return the lazily initialized default device runtime.

    Returns
    -------
    DeviceRuntimeProtocol
        Shared default runtime instance.

    Notes
    -----
    The default runtime is created on first access and its kernels are
    registered exactly once.

    """
    global _default_runtime, _registered

    if _default_runtime is None:
        _default_runtime = DeviceRuntime(registry=KernelRegistry())

    if not _registered and _default_runtime is not None:
        _register_all_kernels(_default_runtime.registry)
        _registered = True

    return _default_runtime


if TYPE_CHECKING:
    _runtime_check: DeviceRuntimeProtocol = DeviceRuntime(registry=KernelRegistry())
