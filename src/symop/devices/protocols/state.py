from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols.states.base import State as StateProtocol
from symop.core.protocols.states.capabilities import SupportsModeLabels


@runtime_checkable
class LabelEditableState(StateProtocol, SupportsModeLabels, Protocol):
    pass
