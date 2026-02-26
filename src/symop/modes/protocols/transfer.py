from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols import HasSignature
from symop.modes.types import FloatArray, RCArray


@runtime_checkable
class TransferFunctionProto(HasSignature, Protocol):
    """Frequency-domain transfer function H(w)."""

    def __call__(self, w: FloatArray) -> RCArray: ...
