from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from symop.ccr.protocols.ket import KetPolyProto
from symop.core.protocols import (
    DensityTermProto,
    HasSignature,
    LadderOpProto,
    ModeOpProto,
)


@runtime_checkable
class SupportsLeftActionDensity(Protocol):
    @staticmethod
    def zero() -> SupportsLeftActionDensity: ...

    def apply_left(
        self, word: Iterable[LadderOpProto]
    ) -> SupportsLeftActionDensity: ...

    def scaled(self, c: complex) -> SupportsLeftActionDensity: ...

    def __add__(
        self, other: SupportsLeftActionDensity
    ) -> SupportsLeftActionDensity: ...


@runtime_checkable
class SupportsRightActionDensity(Protocol):
    @staticmethod
    def zero() -> SupportsRightActionDensity: ...

    def apply_right(
        self, word: Iterable[LadderOpProto]
    ) -> SupportsRightActionDensity: ...

    def scaled(self, c: complex) -> SupportsRightActionDensity: ...

    def __add__(
        self, other: SupportsRightActionDensity
    ) -> SupportsRightActionDensity: ...


@runtime_checkable
class DensityPolyProto(Protocol):
    terms: tuple[DensityTermProto, ...]

    # ---- Constructors -------------------------------------------------------
    @staticmethod
    def zero() -> DensityPolyProto: ...
    @staticmethod
    def pure(psi: KetPolyProto) -> DensityPolyProto: ...

    # ---- Algebra ------------------------------------------------------------
    def scaled(self, c: complex) -> DensityPolyProto: ...

    def combine_like_terms(
        self,
        *,
        eps: float = 1e-12,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> DensityPolyProto: ...

    def apply_left(self, word: Iterable[LadderOpProto]) -> DensityPolyProto: ...

    def apply_right(self, word: Iterable[LadderOpProto]) -> DensityPolyProto: ...

    def trace(self) -> complex: ...

    def partial_trace(
        self, trace_over_modes: Iterable[HasSignature]
    ) -> DensityPolyProto: ...

    def inner(self, other: DensityPolyProto) -> complex: ...
    def purity(self) -> float: ...
    def normalize_trace(self, *, eps: float = ...) -> DensityPolyProto: ...
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

    @property
    def unique_modes(self) -> tuple[ModeOpProto, ...]: ...

    @property
    def mode_count(self) -> int: ...

    # ---- Physical sanity checks --------------------------------------------
    def is_trace_normalized(self, eps: float = ...) -> bool: ...
    def is_pure(self, eps: float = ...) -> bool: ...
    def require_trace_normalized(self, eps: float = ...) -> None: ...
    def is_block_diagonal_by_modes(self) -> bool: ...

    def __add__(self, other: DensityPolyProto) -> DensityPolyProto: ...
