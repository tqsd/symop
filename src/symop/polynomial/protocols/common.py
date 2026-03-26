from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols.ops.operators import ModeOp as ModeOpProtocol
from symop.core.protocols.states.capabilities import SupportsModeLabels
from symop.core.protocols.states.state_rep import PolyState
from symop.devices.measurement.target import MeasurementTarget


@runtime_checkable
class SupportsModeResolution(Protocol):
    def resolve_target_modes(
        self,
        target: MeasurementTarget,
    ) -> tuple[ModeOpProtocol, ...]: ...


@runtime_checkable
class PolyStateCommon(PolyState, SupportsModeLabels, SupportsModeResolution, Protocol):
    pass
