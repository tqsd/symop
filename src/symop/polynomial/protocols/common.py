from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols.states.capabilities import SupportsModeLabels
from symop.core.protocols.states.state_rep import PolyState


@runtime_checkable
class PolyStateCommon(PolyState, SupportsModeLabels, Protocol):
    pass
