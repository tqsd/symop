from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from symop.core.protocols import KetTermProto, LadderOpProto, ModeOpProto


@runtime_checkable
class KetPolyProto(Protocol):
    terms: tuple[KetTermProto, ...]

    # ---- Constructors -------------------------------------------------------
    @staticmethod
    def from_ops(
        *,
        creators: Iterable[LadderOpProto] = ...,
        annihilators: Iterable[LadderOpProto] = ...,
        coeff: complex = ...,
    ) -> KetPolyProto: ...

    @staticmethod
    def from_word(*, ops: Iterable[LadderOpProto]) -> KetPolyProto: ...

    # ---- Algebra ------------------------------------------------------------
    def combine_like_terms(
        self,
        *,
        eps: float = 1e-12,
        approx: bool = False,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> KetPolyProto: ...

    def scaled(self, c: complex) -> KetPolyProto: ...
    def multiply(self, other: KetPolyProto) -> KetPolyProto: ...
    def inner(self, other: KetPolyProto) -> complex: ...
    def norm2(self) -> float: ...
    def normalize(self, *, eps: float = ...) -> KetPolyProto: ...

    def apply_word(self, word: Iterable[LadderOpProto]) -> KetPolyProto: ...

    def apply_words(
        self, terms: Iterable[tuple[complex, Iterable[LadderOpProto]]]
    ) -> KetPolyProto: ...

    # ---- Operators ----------------------------------------------------------
    def __add__(self, other: KetPolyProto) -> KetPolyProto: ...

    def __mul__(self, other: KetPolyProto | complex) -> KetPolyProto: ...

    def __rmul__(self, other: complex) -> KetPolyProto: ...

    # ---- Queries / properties ----------------------------------------------
    def is_normalized(self, eps: float = ...) -> bool: ...

    @property
    def is_creator_only(self) -> bool: ...

    @property
    def is_annihilator_only(self) -> bool: ...

    @property
    def is_identity(self) -> bool: ...

    @property
    def creation_count(self) -> int: ...

    @property
    def annihilation_count(self) -> int: ...

    @property
    def total_degree(self) -> int: ...

    @property
    def unique_modes(self) -> tuple[ModeOpProto, ...]: ...

    @property
    def mode_count(self) -> int: ...

    # ---- Guards -------------------------------------------------------------
    def require_creator_only(self) -> None: ...
