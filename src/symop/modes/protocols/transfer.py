from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.modes.protocols.envelope import GaussianClosedEnvelope


@runtime_checkable
class SupportsGaussianClosedTransfer(Protocol):
    def apply_to_gaussian(
        self,
        env: GaussianClosedEnvelope,
    ) -> tuple[GaussianClosedEnvelope, float]: ...
