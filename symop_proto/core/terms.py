from __future__ import annotations
from dataclasses import dataclass

from symop_proto.core.monomial import Monomial


@dataclass(frozen=True)
class KetTerm:
    coeff: complex
    monomial: Monomial

    def adjoint(self) -> KetTerm:
        return KetTerm(
            coeff=self.coeff.conjugate(), monomial=self.monomial.adjoint()
        )

    @property
    def signature(self) -> tuple:
        return ("KT", self.monomial.signature)

    def approx_signature(self, **env_kw) -> tuple:
        return ("KT_approx", self.monomial.approx_signature(**env_kw))


@dataclass(frozen=True)
class DensityTerm:
    coeff: complex
    left: Monomial
    right: Monomial

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
