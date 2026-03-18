from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from symop.core.protocols.devices.label_edit import LabelEdit
from symop.devices.types.device_kind import DeviceKind


@runtime_checkable
class DeviceAction(Protocol):
    @property
    def kind(self) -> DeviceKind: ...
    @property
    def params(self) -> Mapping[str, object]: ...
    @property
    def edits(self) -> Sequence[LabelEdit]: ...
