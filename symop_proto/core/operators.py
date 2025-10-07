from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Tuple

from symop_proto.core.envelope_protocol import EnvelopeProto
from symop_proto.core.label_protocol import LabelProto


class OperatorKind(str, Enum):
    ANN = "a"
    CREATE = "adag"


@dataclass(frozen=True)
class ModeOp:
    """
    Logical Mode -> Envelope + label
    creates ladder-operators
    """

    env: EnvelopeProto
    label: LabelProto

    ann: LadderOp = field(init=False, repr=False, compare=False)
    create: LadderOp = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self, "ann", LadderOp(kind=OperatorKind.ANN, mode=self)
        )
        object.__setattr__(
            self, "create", LadderOp(kind=OperatorKind.CREATE, mode=self)
        )

    @property
    def signature(self) -> Tuple[Any, ...]:
        return ("mode", self.env.signature(), self.label.signature())

    def approx_signature(self, **kw: Any) -> Tuple[Any, ...]:
        return (
            "mode_approx",
            self.env.approx_signature(**kw),
            self.label.approx_signature(**kw),
        )


@dataclass(frozen=True)
class LadderOp:
    kind: OperatorKind
    mode: ModeOp

    @property
    def is_annihilation(self) -> bool:
        return self.kind is OperatorKind.ANN

    @property
    def is_creation(self) -> bool:
        return self.kind is OperatorKind.CREATE

    def dagger(self) -> LadderOp:
        return self.mode.ann if self.is_creation else self.mode.create

    def commutator(self, other: LadderOp) -> complex:
        L = self.mode.label.overlap(other.mode.label)
        if abs(L) < 1e-15:
            return 0.0 + 0.0j
        if self.is_annihilation and other.is_creation:
            return self.mode.env.overlap(other.mode.env) * L
        if self.is_creation and other.is_annihilation:
            return -self.mode.env.overlap(other.mode.env) * L
        return 0.0 + 0.0j

    @property
    def signature(self) -> Tuple[Any, ...]:
        return ("lop", self.kind.value, self.mode.signature)

    def approx_signature(self, **env_kw: Any) -> Tuple[Any, ...]:
        return ("lop", self.kind.value, self.mode.approx_signature(**env_kw))
