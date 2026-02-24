from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

SignatureProto = tuple[Any, ...]


@runtime_checkable
class HasSignature(Protocol):
    @property
    def signature(self) -> SignatureProto: ...

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> SignatureProto: ...
