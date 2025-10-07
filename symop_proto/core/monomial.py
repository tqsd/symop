from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple

from symop_proto.core.operators import LadderOp, ModeOp


@dataclass(frozen=True)
class Monomial:
    creators: Tuple[LadderOp, ...] = ()
    annihilators: Tuple[LadderOp, ...] = ()

    @property
    def mode_ops(self) -> Tuple[ModeOp, ...]:
        seen: Set[Tuple] = set()
        out: List[ModeOp] = []
        for op in (*self.creators, *self.annihilators):
            sig = op.mode.signature
            if sig not in seen:
                seen.add(sig)
                out.append(op.mode)
        return tuple(out)

    def adjoint(self) -> Monomial:
        dag_creators = tuple(op.dagger() for op in self.annihilators)
        dag_annihilators = tuple(op.dagger() for op in self.creators)
        return Monomial(creators=dag_creators, annihilators=dag_annihilators)

    @property
    def signature(self) -> tuple:
        c = tuple(sorted(op.signature for op in self.creators))
        a = tuple(sorted(op.signature for op in self.annihilators))
        return ("cre", c, "ann", a)

    def approx_signature(self, **env_kw) -> tuple:
        c = tuple(
            sorted(op.approx_signature(**env_kw) for op in self.creators)
        )
        a = tuple(
            sorted(op.approx_signature(**env_kw) for op in self.annihilators)
        )
        return ("cre", c, "ann", a)

    def is_creator_only(self) -> bool:
        return len(self.annihilators) == 0
