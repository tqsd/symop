from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from symop_proto.core.pretty.monomial import collect_mode_order
from symop_proto.core.pretty.terms import (
    densityterm_latex,
    densityterm_text,
    ketterm_latex,
    ketterm_text,
)
from symop_proto.core.protocols import (
    DensityTermProto,
    KetTermProto,
    ModeOpProto,
    MonomialProto,
    SignatureProto,
)


@dataclass(frozen=True)
class KetTerm(KetTermProto):
    coeff: complex
    monomial: MonomialProto

    @staticmethod
    def identity() -> KetTermProto:
        from symop_proto.core.monomial import (
            Monomial,
        )

        return KetTerm(1.0, Monomial())

    def adjoint(self) -> KetTermProto:
        return KetTerm(
            coeff=self.coeff.conjugate(), monomial=self.monomial.adjoint()
        )

    @property
    def signature(self) -> SignatureProto:
        return ("KT", self.monomial.signature)

    def approx_signature(self, **env_kw) -> SignatureProto:
        return ("KT_approx", self.monomial.approx_signature(**env_kw))

    @property
    def is_creator_only(self) -> bool:
        return self.monomial.is_creator_only

    @property
    def is_annihilator_only(self) -> bool:
        return self.monomial.is_annihilator_only

    @property
    def is_identity(self) -> bool:
        return self.monomial.is_identity

    @property
    def creation_count(self) -> int:
        return len(self.monomial.creators)

    @property
    def annihilation_count(self) -> int:
        return len(self.monomial.annihilators)

    @property
    def total_degree(self) -> int:
        return self.creation_count + self.annihilation_count

    @property
    def mode_ops(self) -> Tuple[ModeOpProto, ...]:
        return self.monomial.mode_ops

    def __repr__(self) -> str:
        return ketterm_text(self)

    @property
    def latex(self) -> str:
        return ketterm_latex(self)

    def _repr_latex_(self) -> str:
        return rf"${self.latex}$"


@dataclass(frozen=True)
class DensityTerm(DensityTermProto):
    coeff: complex
    left: MonomialProto
    right: MonomialProto

    @staticmethod
    def identity() -> DensityTerm:
        from symop_proto.core.monomial import (
            Monomial,
        )

        return DensityTerm(1.0, Monomial(), Monomial())

    def adjoint(self) -> DensityTerm:
        return DensityTerm(self.coeff.conjugate(), self.right, self.left)

    @property
    def signature(self) -> tuple:
        return ("DT", "L", self.left.signature, "R", self.right.signature)

    def approx_signature(self, **env_kw) -> tuple:
        return (
            "DT_approx",
            "L",
            self.left.approx_signature(**env_kw),
            "R",
            self.right.approx_signature(**env_kw),
        )

    @property
    def is_creator_only_left(self) -> bool:
        return self.left.is_creator_only or self.left.is_identity

    @property
    def is_creator_only_right(self) -> bool:
        return self.right.is_creator_only or self.right.is_identity

    @property
    def is_creator_only(self) -> bool:
        return self.is_creator_only_left and self.is_creator_only_right

    @property
    def is_annihilator_only_left(self) -> bool:
        return self.left.is_annihilator_only or self.left.is_identity

    @property
    def is_annihilator_only_right(self) -> bool:
        return self.right.is_annihilator_only or self.right.is_identity

    @property
    def is_annihilator_only(self) -> bool:
        return self.is_annihilator_only_left and self.is_annihilator_only_right

    @property
    def is_identity_left(self) -> bool:
        return self.left.is_identity

    @property
    def is_identity_right(self) -> bool:
        return self.right.is_identity

    @property
    def is_diagonal_in_monomials(self) -> bool:
        return self.left.signature == self.right.signature

    @property
    def creation_count_left(self) -> int:
        return len(self.left.creators)

    @property
    def creation_count_right(self) -> int:
        return len(self.right.creators)

    @property
    def annihilation_count_left(self) -> int:
        return len(self.left.annihilators)

    @property
    def annihilation_count_right(self) -> int:
        return len(self.right.annihilators)

    @property
    def mode_ops_left(self):
        return self.left.mode_ops

    @property
    def mode_ops_right(self):
        return self.right.mode_ops

    def __repr__(self) -> str:
        return densityterm_text(self)

    @property
    def latex(self) -> str:
        return densityterm_latex(self)

    def _repr_latex_(self) -> str:
        return rf"${self.latex}$"
