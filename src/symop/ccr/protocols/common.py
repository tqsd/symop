from __future__ import annotations

from typing import Protocol, Self, TypeVar, runtime_checkable

from symop.core.protocols.ops.operators import ModeOp

TermT = TypeVar("TermT", covariant=True)


@runtime_checkable
class HasTerms(Protocol[TermT]):
    @property
    def terms(self) -> tuple[TermT, ...]: ...


@runtime_checkable
class Scaled(Protocol):
    def scaled(self, c: complex) -> Self: ...


@runtime_checkable
class Additive(Protocol):
    def __add__(self, other: Self) -> Self: ...


@runtime_checkable
class ScalarMultipliable(Protocol):
    def __rmul__(self, other: complex) -> Self: ...


@runtime_checkable
class Canonical(Protocol):
    def combine_like_terms(
        self,
        *,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Self: ...


@runtime_checkable
class HasModes(Protocol):
    @property
    def unique_modes(self) -> tuple[ModeOp, ...]: ...

    @property
    def mode_count(self) -> int: ...


@runtime_checkable
class SupportsAdjoint(Protocol):
    def adjoint(self) -> Self: ...
