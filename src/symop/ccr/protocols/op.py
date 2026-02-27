from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from symop.core.protocols import HasSignature, LadderOpProto, ModeOpProto


@runtime_checkable
class OpTermProto(HasSignature, Protocol):
    ops: tuple[LadderOpProto, ...]
    coeff: complex

    @staticmethod
    def identity(c: complex = ...) -> OpTermProto: ...

    def scaled(self, c: complex) -> OpTermProto: ...
    def adjoint(self) -> OpTermProto: ...


@runtime_checkable
class OpPolyProto(Protocol):
    terms: tuple[OpTermProto, ...]

    @staticmethod
    def from_words(
        words: Iterable[Iterable[LadderOpProto]],
        coeffs: Iterable[complex] | None = ...,
    ) -> OpPolyProto: ...

    @staticmethod
    def identity(c: complex = ...) -> OpPolyProto: ...

    @staticmethod
    def zero() -> OpPolyProto: ...

    @staticmethod
    def a(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def adag(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def n(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def q(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def x(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def p(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def X_theta(mode: ModeOpProto, theta: float) -> OpPolyProto: ...

    @staticmethod
    def q2(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def p2(mode: ModeOpProto) -> OpPolyProto: ...

    @staticmethod
    def n2(mode: ModeOpProto) -> OpPolyProto: ...

    def scaled(self, c: complex) -> OpPolyProto: ...
    def adjoint(self) -> OpPolyProto: ...

    def combine_like_terms(
        self,
        *,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> OpPolyProto: ...

    def normalize(
        self,
        *,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> OpPolyProto: ...
    @property
    def is_zero(self) -> bool: ...

    @property
    def is_identity(self) -> bool: ...

    def __add__(self, other: OpPolyProto) -> OpPolyProto: ...
    def __mul__(self, other: OpPolyProto | complex) -> OpPolyProto: ...
    def __rmul__(self, other: complex) -> OpPolyProto: ...
    def __matmul__(self, other: object) -> object: ...
    def __rmatmul__(self, other: object) -> object: ...
