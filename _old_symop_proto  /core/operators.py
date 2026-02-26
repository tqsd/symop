from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import count
from typing import Any

from symop_proto.core.pretty.ladder import ladder_latex, ladder_text
from symop_proto.core.protocols import (
    EnvelopeLike,
    LabelProto,
    LadderOpProto,
    ModeLabelLike,
    ModeOpProto,
    OperatorKindProto,
    SignatureProto,
)

_mode_display_counter = count(1)


class OperatorKind(str, Enum):
    ANN = "a"
    CREATE = "adag"


@dataclass(frozen=True)
class ModeOp(ModeOpProto):
    """Logical Mode -> Envelope + label
    creates ladder-operators
    """

    env: EnvelopeLike
    label: ModeLabelLike

    user_label: str | None = None
    display_index: int | None = field(
        default_factory=lambda: next(_mode_display_counter)
    )

    ann: LadderOpProto = field(init=False, repr=False, compare=False)
    create: LadderOpProto = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "ann", LadderOp(kind=OperatorKind.ANN, mode=self))
        object.__setattr__(
            self, "create", LadderOp(kind=OperatorKind.CREATE, mode=self)
        )

    def with_user_label(self, tag: str) -> ModeOp:
        return replace(self, user_label=tag)

    def with_index(self, idx: int) -> ModeOp:
        return replace(self, display_index=idx)

    def with_env(self, env: EnvelopeLike) -> ModeOp:
        return replace(self, env=env)

    def with_label(self, label: LabelProto) -> ModeOp:
        return replace(self, label=label)

    def with_pol(self, pol: LabelProto) -> ModeOp:
        return replace(self, label=self.label.with_pol(pol))

    def with_path(self, path: LabelProto) -> ModeOp:
        return replace(self, label=self.label.with_path(path))

    @property
    def signature(self) -> SignatureProto:
        return ("mode", self.env.signature, self.label.signature)

    def approx_signature(self, **kw: Any) -> tuple[Any, ...]:
        return (
            "mode_approx",
            self.env.approx_signature(**kw),
            self.label.approx_signature(**kw),
        )


@dataclass(frozen=True)
class LadderOp(LadderOpProto):
    kind: OperatorKindProto
    mode: ModeOpProto

    @property
    def is_annihilation(self) -> bool:
        return self.kind.value == OperatorKind.ANN.value

    @property
    def is_creation(self) -> bool:
        return self.kind.value == OperatorKind.CREATE.value

    def dagger(self) -> LadderOpProto:
        return self.mode.ann if self.is_creation else self.mode.create

    def commutator(self, other: LadderOpProto) -> complex:
        L = self.mode.label.overlap(other.mode.label)
        if abs(L) < 1e-15:
            return 0.0 + 0.0j
        if self.is_annihilation and other.is_creation:
            return self.mode.env.overlap(other.mode.env) * L
        if self.is_creation and other.is_annihilation:
            return -self.mode.env.overlap(other.mode.env) * L
        return 0.0 + 0.0j

    @property
    def signature(self) -> SignatureProto:
        return ("lop", self.kind.value, self.mode.signature)

    def approx_signature(self, **env_kw: Any) -> SignatureProto:
        return ("lop", self.kind.value, self.mode.approx_signature(**env_kw))

    def __repr__(self) -> str:
        return ladder_text(self)

    @property
    def latex(self) -> str:
        return rf"{ladder_latex(self)}"

    def _repr_latex_(self) -> str:
        return rf"${self.latex}$"
