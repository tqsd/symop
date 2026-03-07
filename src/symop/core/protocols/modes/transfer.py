from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols.base.signature import HasSignature
from symop.core.types.arrays import FloatArray, RCArray


@runtime_checkable
class TransferFunction(HasSignature, Protocol):
    def __call__(self, w: FloatArray) -> RCArray: ...
