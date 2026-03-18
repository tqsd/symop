"""Kernel registry implementation.

This module provides the concrete implementation of the device kernel
registry used by the device runtime. The registry maps semantic device
operations to representation-specific kernel functions.

Kernels are resolved using the following key:

    (device_kind, rep, in_kind, out_kind)

This allows a single semantic device to support multiple state
representations and state kinds. For example, the same device may
operate on both polynomial and Gaussian states, or transform a ket
state into a density state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from symop.core.types.rep_kind import RepKind
from symop.core.types.state_kind import StateKind
from symop.devices.protocols.kernel import KernelFn
from symop.devices.protocols.registry import (
    KernelRegistry as KernelRegistryProtocol,
)
from symop.devices.types.device_kind import DeviceKind

KernelKey: TypeAlias = tuple[DeviceKind, RepKind, StateKind, StateKind]


@dataclass
class KernelRegistry(KernelRegistryProtocol):
    """Concrete registry for representation-specific device kernels.

    The registry stores kernel functions that implement device actions
    for a specific combination of:

    - device kind
    - state representation
    - input state kind
    - output state kind

    Notes
    -----
    A single semantic device may have multiple kernels registered for
    different representations or state transformations. For example:

    - Polynomial ket → polynomial ket
    - Polynomial ket → polynomial density
    - Gaussian ket → Gaussian ket

    The registry ensures that a unique kernel exists for each
    `(device_kind, rep, in_kind, out_kind)` combination.

    Attributes
    ----------
    _table : dict[KernelKey, KernelFn]
        Internal lookup table mapping kernel keys to kernel functions.

    """

    _table: dict[KernelKey, KernelFn] = field(default_factory=dict)

    def register(
        self,
        *,
        device_kind: DeviceKind,
        rep: RepKind,
        in_kind: StateKind,
        out_kind: StateKind,
        fn: KernelFn,
    ) -> None:
        """Register a kernel function.

        Parameters
        ----------
        device_kind : DeviceKind
            Semantic device identifier.
        rep : RepKind
            State representation (e.g., polynomial, Gaussian).
        in_kind : StateKind
            Kind of the input state.
        out_kind : StateKind
            Kind of the output state.
        fn : KernelFn
            Kernel function implementing the device action.

        Raises
        ------
        KeyError
            If a kernel is already registered for the given key.

        Notes
        -----
        Each `(device_kind, rep, in_kind, out_kind)` combination may
        only have one registered kernel.

        """
        key = (device_kind, rep, in_kind, out_kind)
        if key in self._table:
            raise KeyError(f"Kernel already registered for key={key!r}")
        self._table[key] = fn

    def resolve(
        self,
        *,
        device_kind: DeviceKind,
        rep: RepKind,
        in_kind: StateKind,
        out_kind: StateKind,
    ) -> KernelFn:
        """Resolve a kernel function.

        Parameters
        ----------
        device_kind : DeviceKind
            Semantic device identifier.
        rep : RepKind
            State representation of the input state.
        in_kind : StateKind
            Kind of the input state.
        out_kind : StateKind
            Desired kind of the output state.

        Returns
        -------
        KernelFn
            The registered kernel function implementing the requested
            device action.

        Raises
        ------
        KeyError
            If no kernel is registered for the requested signature.

        Notes
        -----
        The error message includes a list of available kernel keys
        for the requested device kind to aid debugging.

        """
        key = (device_kind, rep, in_kind, out_kind)
        try:
            return self._table[key]
        except KeyError as e:
            available = sorted((k for k in self._table if k[0] == device_kind), key=str)
            raise KeyError(
                "No kernel registered for "
                f"device_kind={device_kind!r}, rep={rep!r}, "
                f"in_kind={in_kind!r}, out_kind={out_kind!r}. "
                f"Available for this device: {available!r}"
            ) from e


if TYPE_CHECKING:
    _registry_check: KernelRegistryProtocol = KernelRegistry()
