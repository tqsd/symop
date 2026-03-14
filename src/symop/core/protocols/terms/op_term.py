from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from symop.core.protocols.base.signature import HasSignature
from symop.core.protocols.ops.operators import LadderOp


@runtime_checkable
class OpTerm(HasSignature, Protocol):
    @property
    def ops(self) -> tuple[LadderOp, ...]: ...

    @property
    def coeff(self) -> complex: ...

    @classmethod
    def identity(cls, c: complex = ...) -> Self: ...

    def scaled(self, c: complex) -> Self: ...

    def adjoint(self) -> Self: ...
