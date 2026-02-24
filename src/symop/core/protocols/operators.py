from __future__ import annotations

from typing import Protocol, runtime_checkable

from .envelope import EnvelopeLike
from .labels import LabelProto, ModeLabelLike
from .signature import HasSignature


@runtime_checkable
class OperatorKindProto(Protocol):
    @property
    def value(self) -> str: ...


@runtime_checkable
class LadderOpProto(HasSignature, Protocol):
    kind: OperatorKindProto
    mode: ModeOpProto

    @property
    def is_annihilation(self) -> bool: ...

    @property
    def is_creation(self) -> bool: ...

    def dagger(self) -> LadderOpProto: ...
    def commutator(self, other: LadderOpProto) -> complex: ...


@runtime_checkable
class ModeOpProto(HasSignature, Protocol):
    env: EnvelopeLike
    label: ModeLabelLike

    user_label: str | None
    display_index: int | None

    ann: LadderOpProto
    create: LadderOpProto

    def with_user_label(self, tag: str) -> ModeOpProto: ...
    def with_index(self, idx: int) -> ModeOpProto: ...
    def with_env(self, env: EnvelopeLike) -> ModeOpProto: ...
    def with_pol(self, pol: LabelProto) -> ModeOpProto: ...
    def with_label(self, label: ModeLabelLike) -> ModeOpProto: ...
    def with_path(self, path: LabelProto) -> ModeOpProto: ...
