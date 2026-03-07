from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols.base.signature import Signature
from symop.core.protocols.ops.operators import ModeOp


@runtime_checkable
class DeviceAction(Protocol):
    ports: dict[str, object]
    selection: object | None
    mode_map: dict[Signature, ModeOp]
    params: object | None
