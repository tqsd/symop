from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.types.signature import Signature


@runtime_checkable
class HasSignature(Protocol):
    @property
    def signature(self) -> Signature: ...

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature: ...
