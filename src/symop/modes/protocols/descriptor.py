from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols import HasSignature
from symop.modes.protocols.envelope import EnvelopeProto
from symop.modes.protocols.labels import ModeLabelProto
from symop.modes.protocols.transfer import TransferFunctionProto


@runtime_checkable
class ModeDescriptorProto(HasSignature, Protocol):
    @property
    def label(self) -> ModeLabelProto: ...

    @property
    def envelope(self) -> EnvelopeProto: ...


@runtime_checkable
class HasTransferChain(Protocol):
    @property
    def transfers(self) -> tuple[TransferFunctionProto, ...]: ...
