from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols.base.signature import HasSignature
from symop.core.protocols.modes.labels import ModeLabel
from symop.core.protocols.modes.transfer import TransferFunction
from symop.modes.protocols.envelope import TimeFrequencyEnvelope


@runtime_checkable
class ModeDescriptorProto(HasSignature, Protocol):
    @property
    def label(self) -> ModeLabel: ...

    @property
    def envelope(self) -> TimeFrequencyEnvelope: ...


@runtime_checkable
class HasTransferChain(Protocol):
    @property
    def transfers(self) -> tuple[TransferFunction, ...]: ...
