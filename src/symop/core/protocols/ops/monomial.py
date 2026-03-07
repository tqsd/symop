from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.core.protocols.base.signature import HasSignature
from symop.core.protocols.ops.operators import LadderOp, ModeOp


@runtime_checkable
class Monomial(HasSignature, Protocol):
    @property
    def annihilators(self) -> tuple[LadderOp, ...]: ...

    @property
    def creators(self) -> tuple[LadderOp, ...]: ...

    @staticmethod
    def identity() -> Monomial: ...

    @property
    def mode_ops(self) -> tuple[ModeOp, ...]: ...

    def adjoint(self) -> Monomial: ...

    @property
    def is_creator_only(self) -> bool: ...

    @property
    def is_annihilator_only(self) -> bool: ...

    @property
    def is_identity(self) -> bool: ...

    @property
    def has_creators(self) -> bool: ...

    @property
    def has_annihilators(self) -> bool: ...
