r"""Kernel dispatch for device actions.

Provides utilities to resolve and execute representation-specific
device kernels based on the current state, device kind, and desired
input/output state kinds.

The dispatch mechanism uses a kernel registry to select the appropriate
implementation and applies it to the input state using the semantic
action produced during device planning.
"""

from __future__ import annotations

from typing import cast

from symop.core.protocols.states.base import State as StateProtocol
from symop.core.protocols.states.capabilities import SupportsToDensity
from symop.core.types.state_kind import DENSITY, StateKind
from symop.devices.measurement.result import (
    DetectionResult,
    MeasurementResult,
    ObservationResult,
    PostselectionResult,
)
from symop.devices.protocols.action import (
    DetectAction as DetectActionProtocol,
)
from symop.devices.protocols.action import (
    DeviceAction as DeviceActionProtocol,
)
from symop.devices.protocols.action import (
    MeasurementAction as MeasurementActionProtocol,
)
from symop.devices.protocols.action import (
    ObserveAction as ObserveActionProtocol,
)
from symop.devices.protocols.action import (
    PostselectAction as PostselectActionProtocol,
)
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.devices.protocols.device import (
    Device as DeviceProtocol,
)
from symop.devices.protocols.device import (
    MeasurementDevice as MeasurementDeviceProtocol,
)
from symop.devices.protocols.registry import (
    KernelRegistry as KernelRegistryProtocol,
)
from symop.devices.protocols.registry import (
    MeasurementKernelRegistry as MeasurementKernelRegistryProtocol,
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


def dispatch_measurement(
    *,
    registry: MeasurementKernelRegistryProtocol,
    device: MeasurementDeviceProtocol,
    state: StateProtocol,
    action: MeasurementActionProtocol,
    ctx: ApplyContextProtocol,
) -> MeasurementResult:
    """Resolve and execute the representation-specific measurement kernel.

    Parameters
    ----------
    registry:
        Measurement kernel registry used to resolve the
        representation-specific executor.
    device:
        Semantic measurement device that produced the action.
    state:
        Input state.
    action:
        Semantic measurement action produced by device planning.
    ctx:
        Shared context used during planning and execution.

    Returns
    -------
    MeasurementResultProtocol
        Result of applying the resolved measurement kernel to the input
        state.

    """
    effective_state: StateProtocol = state
    try:
        fn = registry.resolve(
            device_kind=device.kind,
            intent=action.intent,
            rep=state.rep_kind,
            in_kind=state.state_kind,
        )
    except KeyError:
        if state.state_kind != DENSITY and isinstance(state, SupportsToDensity):
            effective_state = state.to_density()
            fn = registry.resolve(
                device_kind=device.kind,
                intent=action.intent,
                rep=effective_state.rep_kind,
                in_kind=effective_state.state_kind,
            )
        else:
            raise
    return fn(state=effective_state, action=action, ctx=ctx)


def dispatch_observe(
    *,
    registry: MeasurementKernelRegistryProtocol,
    device: MeasurementDeviceProtocol,
    state: StateProtocol,
    action: ObserveActionProtocol,
    ctx: ApplyContextProtocol,
) -> ObservationResult:
    """Resolve and execute an observation kernel."""
    result = dispatch_measurement(
        registry=registry,
        device=device,
        state=state,
        action=action,
        ctx=ctx,
    )
    return cast(ObservationResult, result)


def dispatch_detect(
    *,
    registry: MeasurementKernelRegistryProtocol,
    device: MeasurementDeviceProtocol,
    state: StateProtocol,
    action: DetectActionProtocol,
    ctx: ApplyContextProtocol,
) -> DetectionResult:
    """Resolve and execute a detection kernel."""
    result = dispatch_measurement(
        registry=registry,
        device=device,
        state=state,
        action=action,
        ctx=ctx,
    )
    return cast(DetectionResult, result)


def dispatch_postselect(
    *,
    registry: MeasurementKernelRegistryProtocol,
    device: MeasurementDeviceProtocol,
    state: StateProtocol,
    action: PostselectActionProtocol,
    ctx: ApplyContextProtocol,
) -> PostselectionResult:
    """Resolve and execute a postselection kernel."""
    result = dispatch_measurement(
        registry=registry,
        device=device,
        state=state,
        action=action,
        ctx=ctx,
    )
    return cast(PostselectionResult, result)
