from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Iterable as IterableABC
from typing import (
    Any,
    Protocol,
    overload,
    runtime_checkable,
)

from symop_proto.algebra.protocols import (
    DensityPolyProto,
    KetPolyProto,
    OpPolyProto,
)
from symop_proto.core.protocols import LadderOpProto, ModeOpProto


@runtime_checkable
class PrettyPrintable(Protocol):
    def __repr__(self) -> str: ...
    def _repr_latex_(self) -> str: ...


@runtime_checkable
class Normalizable(Protocol):
    def is_normalized(self, *, eps: float = 1e-14) -> bool: ...
    def normalized(self, *, eps: float = 1e-14) -> Any: ...


@runtime_checkable
class ExpectationOnOpPoly(Protocol):
    def expect(
        self, op: OpPolyProto, *, normalize: bool = True, eps: float = 1e-14
    ) -> complex: ...


@runtime_checkable
class TraceLike(Protocol):
    def trace(self) -> complex: ...
    def is_trace_normalized(self, eps: float = 1e-12) -> bool: ...


@runtime_checkable
class PurityLike(Protocol):
    def purity(self) -> float: ...
    def is_pure(self, eps: float = 1e-12) -> bool: ...


@runtime_checkable
class PartialTraceable(Protocol):
    def partial_trace(self, trace_over_modes: set) -> DensityPolyStateProto: ...


@runtime_checkable
class HasUniqueModes(Protocol):
    @property
    def unique_modes(self) -> tuple[ModeOpProto, ...]: ...


# --- Operator application protocol (left-action) --------------
@runtime_checkable
class LeftActionable(Protocol):
    """Supports left actions:
    OpPoly   @ state
    KetPoly  @ state
    LadderOp @ state
    Iterable[LadderOp] @ state
    ModeOp   @ state (as ModeOp.create by convention)
    """

    @overload
    def __rmatmul__(self, other: OpPolyProto): ...
    @overload
    def __rmatmul__(self, other: KetPolyProto): ...
    @overload
    def __rmatmul__(self, other: LadderOpProto): ...
    @overload
    def __rmatmul__(self, other: ModeOpProto): ...
    @overload
    def __rmatmul__(self, other: IterableABC[LadderOpProto]): ...
    def __rmatmul__(self, other): ...


# --- Concrete state protocols you can type against ------------


@runtime_checkable
class KetPolyStateProto(
    Normalizable,
    ExpectationOnOpPoly,
    HasUniqueModes,
    LeftActionable,
    PrettyPrintable,
    Protocol,
):
    """Protocol for your KetPolyState-like objects.
    Designed to be satisfied by your current KetPolyState and any wrappers.
    """

    ket: KetPolyProto
    label: str | None
    index: int | None

    @staticmethod
    def vacuum() -> KetPolyStateProto: ...

    @staticmethod
    def from_creators(
        creators: Iterable[LadderOpProto], coeff: complex = 1.0
    ) -> KetPolyStateProto: ...
    @staticmethod
    def from_ketpoly(ket: KetPolyProto) -> KetPolyStateProto: ...

    def with_label(self, label: str | None) -> KetPolyStateProto: ...
    def with_index(self, index: int | None) -> KetPolyStateProto: ...

    @property
    def norm2(self) -> float: ...

    def to_density(self) -> DensityPolyStateProto: ...


@runtime_checkable
class DensityPolyStateProto(
    Normalizable,
    ExpectationOnOpPoly,
    TraceLike,
    PurityLike,
    PartialTraceable,
    PrettyPrintable,
    Protocol,
):
    """Protocol for your DensityPolyState-like objects."""

    rho: DensityPolyProto

    @staticmethod
    def pure(
        psi: KetPolyProto | KetPolyStateProto,
    ) -> DensityPolyStateProto: ...

    @staticmethod
    def from_densitypoly(
        rho: DensityPolyProto,
        *,
        normalize_trace: bool = False,
        eps: float = 1e-14,
    ) -> DensityPolyStateProto: ...

    @staticmethod
    def mix(
        states: Sequence[DensityPolyStateProto],
        weights: Sequence[float],
        *,
        normalize_weights: bool = True,
    ) -> DensityPolyStateProto: ...

    def normalized(self, *, eps: float = 1e-14) -> DensityPolyStateProto: ...
