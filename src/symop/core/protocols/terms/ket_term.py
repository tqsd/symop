from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.base.signature import HasSignature
from symop.core.protocols.ops.monomial import Monomial
from symop.core.protocols.ops.operators import ModeOp


@runtime_checkable
class KetTerm(HasSignature, Protocol):
    @property
    def coeff(self) -> complex: ...
    @property
    def monomial(self) -> Monomial: ...

    @classmethod
    def identity(cls) -> Self: ...

    def adjoint(self) -> Self: ...
    def scaled(self, s: complex) -> Self: ...

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
    def mode_ops(self) -> tuple[ModeOp, ...]: ...
