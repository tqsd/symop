from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Generic,
    Protocol,
    TypeVar,
)

from symop_proto.core.protocols import ModeOpProto
from symop_proto.devices.io import DeviceIO, DeviceResult, DeviceReturnMode


class BasisLike(Protocol):
    """Minimal basis interface needed by devices.

    This is intentionally small to avoid importing gaussian/hybrid modules.
    """

    @property
    def modes(self) -> tuple[ModeOpProto, ...]: ...

    def require_index_of(self, mode: ModeOpProto) -> int: ...


class HasBasis(Protocol):
    """Minimal state interface: the state must expose a basis."""

    @property
    def basis(self) -> BasisLike: ...


TState = TypeVar("TState", bound=HasBasis)
ModeSelector = Callable[[ModeOpProto], bool]


@dataclass(frozen=True)
class DeviceApplyOptions:
    """Per-application policies that are state-agnostic.

    return_mode:
        KEEP_ALL: keep the full basis after applying the device.
        KEEP_OUTPUTS: reduce to output_modes after apply (and optional env trace).

    trace_env:
        If True, trace out io.env_modes after the core apply step.
    """

    return_mode: DeviceReturnMode = DeviceReturnMode.KEEP_ALL
    trace_env: bool = True


class BaseDevice(ABC, Generic[TState]):
    """State-agnostic device base class.

    Structure
    ---------
    A device application is split into:
      1) resolve_io(state): decide concrete input/output/env modes
      2) do_apply(state, io): apply device dynamics
      3) optional post-processing: trace env, keep outputs

    Subclasses implement:
      - resolve_io
      - do_apply
      - _trace_out
      - _keep

    Notes
    -----
    This file intentionally avoids importing GaussianCore or HybridState to
    prevent recursive imports. Concrete device implementations live in
    state-specific packages and implement the hooks.

    """

    def __init__(
        self,
        *,
        modes: tuple[ModeOpProto, ...] | None = None,
        selector: ModeSelector | None = None,
        default_options: DeviceApplyOptions | None = None,
    ) -> None:
        if modes is not None and selector is not None:
            raise ValueError("Provide either modes or selector, not both.")
        self._modes = modes
        self._selector = selector
        self._default_options = default_options or DeviceApplyOptions()

    def _init_base(
        self,
        *,
        modes: tuple[ModeOpProto, ...] | None = None,
        selector: ModeSelector | None = None,
        default_options: DeviceApplyOptions | None = None,
    ) -> None:
        if modes is not None and selector is not None:
            raise ValueError("Provide either modes or selector, not both.")

        object.__setattr__(self, "_modes", modes)
        object.__setattr__(self, "_selector", selector)
        object.__setattr__(
            self,
            "_default_options",
            (default_options if default_options is not None else DeviceApplyOptions()),
        )

    def resolve_inputs(self, state: TState) -> tuple[ModeOpProto, ...]:
        """Default input resolution helper.

        - If modes were provided, return them (in that order).
        - If selector was provided, pick matching modes in basis order.
        - Otherwise raise (subclass should override resolve_io).
        """
        if self._modes is not None:
            for m in self._modes:
                state.basis.require_index_of(m)
            return tuple(self._modes)

        if self._selector is not None:
            chosen = tuple(m for m in state.basis.modes if self._selector(m))
            if len(chosen) == 0:
                raise ValueError("Selector matched no modes.")
            return chosen

        raise ValueError(
            "No input modes specified. Provide modes=... or selector=..., "
            "or implement resolve_io()."
        )

    _default_options: DeviceApplyOptions = field(
        default_factory=lambda: DeviceApplyOptions(),
        init=False,
        repr=False,
        compare=False,
    )

    def apply(
        self,
        state: TState,
        *,
        options: DeviceApplyOptions | None = None,
    ) -> DeviceResult[TState]:
        """Apply the device.

        Parameters
        ----------
        state:
            Any state object that has a .basis with require_index_of(...) and modes.
        options:
            Post-processing policies (trace env, return mode). If omitted,
            the device default is used.

        """
        opts = options or self._default_options

        io = self.resolve_io(state)
        out = self.do_apply(state, io)

        if len(io.mode_map) > 0:
            out = self._relabel(out, io.mode_map)

        if opts.trace_env and len(io.env_modes) > 0:
            out = self._trace_out(out, io.env_modes)

        if opts.return_mode == DeviceReturnMode.KEEP_OUTPUTS:
            out = self._keep(out, io.output_modes)

        return DeviceResult(state=out, io=io)

    @abstractmethod
    def resolve_io(self, state: TState) -> DeviceIO:
        """Decide concrete input/output/env mode bindings for this application.
        Must validate that input modes exist in the incoming basis.
        """

    @abstractmethod
    def do_apply(self, state: TState, io: DeviceIO) -> TState:
        """Apply device dynamics to the state. No tracing/keeping here."""

    @abstractmethod
    def _trace_out(self, state: TState, modes: Sequence[ModeOpProto]) -> TState:
        """Trace out (discard) modes from the state.
        Implemented by GaussianCore/HybridState adapters.
        """

    @abstractmethod
    def _keep(self, state: TState, modes: Sequence[ModeOpProto]) -> TState:
        """Keep only selected modes (reduce state).
        Implemented by GaussianCore/HybridState adapters.
        """

    @abstractmethod
    def _relabel(
        self,
        state: TState,
        mode_map: Sequence[tuple[ModeOpProto, ModeOpProto]],
    ) -> TState:
        """Relabel the modes and updates the GaussianCore basis"""
