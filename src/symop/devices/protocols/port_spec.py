from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.devices.types.ports import PORT_DIRECTION


@runtime_checkable
class PortSpec(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def direction(self) -> PORT_DIRECTION: ...
    @property
    def required(self) -> bool: ...
