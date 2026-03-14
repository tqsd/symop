from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, Self, runtime_checkable

from symop.ccr.protocols.common import (
    Additive,
    Canonical,
    HasModes,
    HasTerms,
    ScalarMultipliable,
    Scaled,
)
from symop.core.protocols.ops.operators import LadderOp
from symop.core.protocols.terms.ket_term import KetTerm


@runtime_checkable
class KetPoly(
    HasTerms[KetTerm],
    Additive,
    Scaled,
    ScalarMultipliable,
    Canonical,
    HasModes,
    Protocol,
):
    def multiply(self, other: Self) -> Self: ...
    def inner(self, other: Self) -> complex: ...
    def norm2(self) -> float: ...
    def normalize(self, *, eps: float = ...) -> Self: ...

    def apply_word(self, word: Iterable[LadderOp]) -> Self: ...

    def apply_words(
        self, terms: Iterable[tuple[complex, Iterable[LadderOp]]]
    ) -> Self: ...

    # ---- Operators ----------------------------------------------------------
    def __mul__(self, other: Self | complex) -> Self: ...
    def __rmatmul__(self, other: object) -> object: ...

    # ---- Queries / properties ----------------------------------------------
    def is_normalized(self, *, eps: float = ...) -> bool: ...

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

    # ---- Guards -------------------------------------------------------------
    def require_creator_only(self) -> None: ...
