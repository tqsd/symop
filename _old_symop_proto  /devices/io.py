from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generic, Optional, Tuple, TypeVar

from symop_proto.core.protocols import ModeOpProto

TState = TypeVar("TState")


class DeviceReturnMode(str, Enum):
    """
    Policy for what subsystem is returned after applying a device.
    """

    KEEP_ALL = "keep_all"
    KEEP_OUTPUTS = "keep_outputs"


@dataclass(frozen=True)
class DeviceIO:
    """
    Concrete mode bindings for one device application.

    input_modes:
        The modes the device reads/consumes as its inputs (ports).
        These must exist in the incoming state's basis.

    output_modes:
        The logical output modes produced by the device.
        These may be the same objects as inputs (in-place), or newly constructed
        ModeOp objects (reroute/relabel).

    env_modes:
        Any environment/dump modes introduced by the device. These are often
        traced out by default.

    meta:
        Extra info about the realized application (optional).
        Typical entries: matrices, parameters, port mapping, etc.
    """

    input_modes: Tuple[ModeOpProto, ...]
    output_modes: Tuple[ModeOpProto, ...]
    env_modes: Tuple[ModeOpProto, ...] = ()
    meta: Dict[str, object] = field(default_factory=dict)

    mode_map: Tuple[Tuple[ModeOpProto, ModeOpProto], ...] = ()


@dataclass(frozen=True)
class DeviceResult(Generic[TState]):
    """
    Result of applying a device.

    state:
        The updated state.

    io:
        The concrete input/output/env mode binding used.
    """

    state: TState
    io: DeviceIO
