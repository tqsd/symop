from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.states.capabilities import SupportsTrace
from symop.core.protocols.states.state_kind import DensityState
from symop.polynomial.protocols.common import PolyStateCommon


@runtime_checkable
class SupportsNormalizeTrace(Protocol):
    def normalize_trace(self, *, eps: float = 1e-14) -> Self: ...


@runtime_checkable
class DensityPolyState(
    PolyStateCommon,
    DensityState,
    SupportsTrace,
    SupportsNormalizeTrace,
    Protocol,
):
    pass
