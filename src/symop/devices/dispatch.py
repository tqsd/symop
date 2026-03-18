r"""Kernel dispatch for device actions.

Provides utilities to resolve and execute representation-specific
device kernels based on the current state, device kind, and desired
input/output state kinds.

The dispatch mechanism uses a kernel registry to select the appropriate
implementation and applies it to the input state using the semantic
action produced during device planning.
"""

from __future__ import annotations

from symop.core.protocols.states.base import State as StateProtocol
from symop.core.types.state_kind import StateKind
from symop.devices.protocols.action import DeviceAction as DeviceActionProtocol
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.protocols.device import Device as DeviceProtocol
from symop.devices.protocols.registry import (
    KernelRegistry as KernelRegistryProtocol,
)


def dispatch_apply(
    *,
    registry: KernelRegistryProtocol,
    device: DeviceProtocol,
    state: StateProtocol,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
    out_kind: StateKind | None = None,
) -> StateProtocol:
    """Resolve and execute the representation-specific kernel for a device action.

    Parameters
    ----------
    registry : KernelRegistryProtocol
        Kernel registry used to resolve the representation-specific executor.
    device : DeviceProtocol
        Semantic device that produced the action.
    state : StateProtocol
        Input state.
    action : DeviceActionProtocol
        Semantic action produced by device planning.
    ctx : ApplyContextProtocol
        Shared apply context used during planning and execution.
    out_kind : StateKind | None, optional
        Desired output state kind. If omitted, the input state kind is used.

    Returns
    -------
    StateProtocol
        Result of applying the resolved kernel to the input state.

    """
    in_kind = state.state_kind
    resolved_out_kind = in_kind if out_kind is None else out_kind

    fn = registry.resolve(
        device_kind=device.kind,
        rep=state.rep_kind,
        in_kind=in_kind,
        out_kind=resolved_out_kind,
    )

    return fn(state=state, action=action, ctx=ctx)
