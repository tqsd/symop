from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.types.rep_kind import RepKind
from symop.core.types.state_kind import StateKind
from symop.devices.protocols.kernel import KernelFn, MeasurementKernelFn
from symop.devices.types.device_kind import DeviceKind
from symop.devices.types.measurement import MeasurementIntent


@runtime_checkable
class KernelRegistry(Protocol):
    def register(
        self,
        *,
        device_kind: DeviceKind,
        rep: RepKind,
        in_kind: StateKind,
        out_kind: StateKind,
        fn: KernelFn,
    ) -> None:
        """
        Register a kernel for the given device and state representation/kind
        """

    def resolve(
        self,
        *,
        device_kind: DeviceKind,
        rep: RepKind,
        in_kind: StateKind,
        out_kind: StateKind,
    ) -> KernelFn:
        """
        Resolve the kernel function for the given device and state.
        """
        ...


@runtime_checkable
class MeasurementKernelRegistry(Protocol):
    def register(
        self,
        *,
        device_kind: DeviceKind,
        intent: MeasurementIntent,
        rep: RepKind,
        in_kind: StateKind,
        fn: MeasurementKernelFn,
    ) -> None:
        """
        Regiester a measurement kernel for the given device, intent,
        representation, and input state kind.
        """
        ...

    def resolve(
        self,
        *,
        device_kind: DeviceKind,
        intent: MeasurementIntent,
        rep: RepKind,
        in_kind: StateKind,
    ) -> MeasurementKernelFn:
        """
        Resolve the measurement kernel for the given device, intent,
        representation, and input state kind.
        """
        ...
