from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, Self, runtime_checkable

from symop.ccr.protocols.actions import (
    SupportsLeftWordAction,
    SupportsRightWordAction,
)
from symop.ccr.protocols.common import (
    Additive,
    Canonical,
    HasModes,
    HasTerms,
    ScalarMultipliable,
    Scaled,
)
from symop.core.protocols.ops.operators import ModeOp
from symop.core.protocols.terms.density_term import DensityTerm


@runtime_checkable
class DensityPoly(
    HasTerms[DensityTerm],
    Additive,
    ScalarMultipliable,
    HasModes,
    Scaled,
    Canonical,
    SupportsLeftWordAction,
    SupportsRightWordAction,
    Protocol,
):
    def trace(self) -> complex: ...

    def partial_trace(self, trace_over_modes: Iterable[ModeOp]) -> Self: ...

    def inner(self, other: Self) -> complex: ...
    def purity(self) -> float: ...
    def normalize_trace(self, *, eps: float = ...) -> Self: ...
    def hs_norm2(self) -> float: ...
    def hs_norm(self) -> float: ...

    # ---- Properties / queries ----------------------------------------------
    @property
    def is_creator_only_left(self) -> bool: ...

    @property
    def is_creator_only_right(self) -> bool: ...

    @property
    def is_creator_only(self) -> bool: ...

    @property
    def is_identity_left(self) -> bool: ...

    @property
    def is_identity_right(self) -> bool: ...

    @property
    def is_diagonal_in_monomials(self) -> bool: ...

    # ---- Physical sanity checks --------------------------------------------
    def is_trace_normalized(self, eps: float = ...) -> bool: ...
    def is_pure(self, eps: float = ...) -> bool: ...
    def require_trace_normalized(self, eps: float = ...) -> None: ...
    def is_block_diagonal_by_modes(self) -> bool: ...

    def __mul__(self, other: complex) -> Self: ...
    def __matmul__(self, other: object) -> object: ...
    def __rmatmul__(self, other: object) -> object: ...
