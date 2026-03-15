from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.ccr.protocols.ket import KetPoly as KetPolyProtocol
from symop.core.protocols.states.capabilities import SupportsToDensity
from symop.core.protocols.states.state_kind import KetState
from symop.polynomial.protocols.common import PolyStateCommon


@runtime_checkable
class SupportsNormalizeKet(Protocol):
    def normalized(self, *, eps: float = 1e-14) -> Self: ...
    def is_normalized(self, *, eps: float = 1e-14) -> bool: ...

    @property
    def norm2(self) -> float: ...


@runtime_checkable
class KetPolyState(
    PolyStateCommon,
    KetState,
    SupportsToDensity,
    SupportsNormalizeKet,
    Protocol,
):
    @property
    def ket(self) -> KetPolyProtocol: ...
