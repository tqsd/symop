from __future__ import annotations

from typing import Protocol, runtime_checkable

from .operators import LadderOpProto, ModeOpProto
from .signature import HasSignature


@runtime_checkable
class MonomialProto(HasSignature, Protocol):
    creators: tuple[LadderOpProto, ...]
    annihilators: tuple[LadderOpProto, ...]

    @property
    def mode_ops(self) -> tuple[ModeOpProto, ...]: ...

    def adjoint(self) -> MonomialProto: ...

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
