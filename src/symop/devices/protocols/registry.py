from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.types.rep_kind import RepKind
from symop.core.types.state_kind import StateKind
from symop.devices.protocols.kernel import KernelFn
from symop.devices.types.device_kind import DeviceKind


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
