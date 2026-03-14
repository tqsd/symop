from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.ccr.protocols.common import (
    Additive,
    Canonical,
    HasModes,
    HasTerms,
    ScalarMultipliable,
    Scaled,
    SupportsAdjoint,
)
from symop.core.protocols.terms.op_term import OpTerm


@runtime_checkable
class OpPoly(
    HasTerms[OpTerm],
    HasModes,
    Scaled,
    Additive,
    ScalarMultipliable,
    SupportsAdjoint,
    Canonical,
    Protocol,
):
    def normalize(
        self,
        *,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Self: ...
    @property
    def is_zero(self) -> bool: ...

    @property
    def is_identity(self) -> bool: ...

    def __mul__(self, other: Self | complex) -> Self: ...
    def __matmul__(self, other: object) -> object: ...
    def __rmatmul__(self, other: object) -> object: ...
