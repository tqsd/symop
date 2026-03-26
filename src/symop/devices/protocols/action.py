from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from symop.core.protocols.devices.label_edit import LabelEdit
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.types.device_kind import DeviceKind
from symop.devices.types.measurement import MeasurementIntent


@runtime_checkable
class DeviceAction(Protocol):
    @property
    def kind(self) -> DeviceKind: ...
    @property
    def params(self) -> Mapping[str, object]: ...
    @property
    def edits(self) -> Sequence[LabelEdit]: ...
    @property
    def requires_kernel(self) -> bool: ...


@runtime_checkable
class MeasurementAction(Protocol):
    r"""Base for semantic measurement actions."""

    @property
    def intent(self) -> MeasurementIntent:
        r"""Return the semantic measurement intent."""
        ...

    @property
    def target(self) -> object:
        r"""Return the semantic target of the action."""
        ...

    @property
    def outcome(self) -> object | None:
        r"""Return the selected outcome, if any."""
        ...


@runtime_checkable
class ObserveAction(MeasurementAction, Protocol):
    r"""Protocol for observation actions."""

    @property
    def intent(self) -> MeasurementIntent:
        r"""Return ``"observe"``."""
        ...


@runtime_checkable
class DetectAction(MeasurementAction, Protocol):
    r"""Protocol for detection actions."""

    @property
    def destructive(self) -> bool: ...
    @property
    def intent(self) -> MeasurementIntent:
        r"""Return ``"detect"``."""
        ...


@runtime_checkable
class PostselectAction(MeasurementAction, Protocol):
    r"""Protocol for postselection actions."""

    @property
    def destructive(self) -> bool: ...

    @property
    def intent(self) -> MeasurementIntent:
        r"""Return ``"postselect"``."""
        ...

    @property
    def outcome(self) -> MeasurementOutcome:
        r"""Return the selected outcome."""
        ...
