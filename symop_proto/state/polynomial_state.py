from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Iterable, Union, Sequence, Optional
from collections.abc import Iterable as IterableABC
from itertools import count

from symop_proto.algebra.operator_polynomial import OpPoly
from symop_proto.algebra.pretty.density import density_latex, density_repr
from symop_proto.algebra.pretty.ket import ket_latex, ket_repr
from symop_proto.core.operators import LadderOp, ModeOp, OperatorKind
from symop_proto.algebra import KetPoly, DensityPoly
from symop_proto.core.terms import KetTerm
from symop_proto.state.pretty.polynomial_state import (
    density_state_latex_from_terms,
    state_latex_from_terms,
)

_state_counter = count(1)


@dataclass(frozen=True)
class KetPolyState:
    """Physical ket state: creators-only polynomial acting on |0>."""

    ket: KetPoly
    label: Optional[str] = None
    index: Optional[int] = field(default_factory=lambda: next(_state_counter))

    def __post_init__(self):
        if not self.ket.is_creator_only:
            raise ValueError(
                "State must be creators-only (plus identity terms)."
            )

    @staticmethod
    def vacuum() -> KetPolyState:
        return KetPolyState(KetPoly((KetTerm.identity(),)))

    @staticmethod
    def from_creators(
        creators: Iterable[LadderOp], coeff: complex = 1.0
    ) -> KetPolyState:
        creators = tuple(creators)
        if any(
            getattr(op, "kind", None) != OperatorKind.CREATE for op in creators
        ):
            raise ValueError("from_creators expects only creation operators.")
        return KetPolyState(
            KetPoly.from_ops(
                creators=creators, annihilators=(), coeff=coeff
            ).combine_like_terms()
        )

    def with_label(self, label: Optional[str]) -> KetPolyState:
        return replace(self, label=label)

    def with_index(self, index: Optional[int]) -> KetPolyState:
        return replace(self, index=index)

    def _with_ket(self, new_ket: KetPoly) -> KetPolyState:
        return replace(self, ket=new_ket)

    @staticmethod
    def from_ketpoly(ket: KetPoly) -> KetPolyState:
        return KetPolyState(ket.combine_like_terms())

    def is_normalized(self, eps: float = 1e-14) -> bool:
        return self.ket.is_normalized(eps=eps)

    def normalized(self, *, eps: float = 1e-14) -> "KetPolyState":
        return self._with_ket(self.ket.normalize(eps=eps))

    def to_density(self) -> DensityPolyState:
        return DensityPolyState.pure(self.ket)

    def expect(
        self, op: OpPoly, *, normalize: bool = True, eps: float = 1e-14
    ) -> complex:
        val = self.to_density().expect(op)
        if not normalize:
            return val
        n2 = self.norm2
        if n2 < eps:
            raise ValueError("Cannot compute normalized expectation")
        return val / n2

    @property
    def norm2(self) -> float:
        return self.ket.norm2()

    @property
    def unique_modes(self):
        return self.ket.unique_modes

    def __repr__(self) -> str:
        return ket_repr(self.ket.terms, is_state=True)

    def _repr_latex_(self) -> str:
        return state_latex_from_terms(
            self.ket.terms, label=self.label, index=self.index
        )

    def __matmul__(self, other):
        # state @ something - not a well-defined right action for kets
        return NotImplemented

    def __rmatmul__(self, other) -> KetPolyState:
        """
        Supports:
          OpPoly   @ KetPolyState  -> KetPolyState
          KetPoly  @ KetPolyState  -> KetPolyState
          LadderOp @ KetPolyState  -> KetPolyState
          Iterable[LadderOp] @ KetPolyState -> KetPolyState
          ModeOp   @ KetPolyState  -> interpreted as ModeOp.create @ state (optional)
        """
        # 1) OpPoly @ state
        if isinstance(other, OpPoly):
            pairs = ((t.coeff, t.ops) for t in other.terms)
            kp = self.ket.apply_words(pairs).combine_like_terms()
            return self._with_ket(kp)

        # 2) KetPoly @ state
        if isinstance(other, KetPoly):
            kp = other.multiply(self.ket).combine_like_terms()
            return self._with_ket(kp)

        # 3a) LadderOp @ state  (single op -> length-1 word)
        if isinstance(other, LadderOp):
            kp = (
                KetPoly.from_word(ops=(other,))
                .multiply(self.ket)
                .combine_like_terms()
            )
            return self._with_ket(kp)

        # 3b) ModeOp @ state (optional sugar: treat as its creation operator)
        if isinstance(other, ModeOp):
            op = other.create  # or .annihilate depending on your intent
            kp = (
                KetPoly.from_word(ops=(op,))
                .multiply(self.ket)
                .combine_like_terms()
            )
            return self._with_ket(kp)

        # 3c) Iterable[LadderOp] @ state (a whole word)
        if isinstance(other, IterableABC):
            ops = tuple(other)
            if all(isinstance(op, LadderOp) for op in ops):
                kp = (
                    KetPoly.from_word(ops=ops)
                    .multiply(self.ket)
                    .combine_like_terms()
                )
                return self._with_ket(kp)

        return NotImplemented


@dataclass(frozen=True)
class DensityPolyState:
    """Physical density operator rho with nice ergonomics."""

    rho: DensityPoly
    label: Optional[str] = None
    index: Optional[int] = field(default_factory=lambda: next(_state_counter))

    def with_label(self, label: Optional[str]) -> DensityPolyState:
        return replace(self, label=label)

    def with_index(self, index: Optional[int]) -> DensityPolyState:
        return replace(self, index=index)

    @staticmethod
    def pure(psi: Union[KetPoly, KetPolyState]) -> DensityPolyState:
        if isinstance(psi, KetPolyState):
            ket = psi.ket
        else:
            ket = psi
        return DensityPolyState(DensityPoly.pure(ket))

    @staticmethod
    def from_densitypoly(
        rho: DensityPoly, *, normalize_trace: bool = False, eps: float = 1e-14
    ) -> DensityPolyState:
        return DensityPolyState(
            rho.normalize_trace(eps=eps) if normalize_trace else rho
        )

    @staticmethod
    def mix(
        states: Sequence[DensityPolyState],
        weights: Sequence[float],
        *,
        normalize_weights: bool = True,
    ) -> DensityPolyState:
        if len(states) != len(weights):
            raise ValueError("states and weights must have same length")
        wsum = float(sum(weights))
        if wsum <= 0:
            raise ValueError("weights must sum to a positive value")
        scale = 1.0 / wsum if normalize_weights else 1.0
        combined = DensityPoly(())  # start empty
        for s, w in zip(states, weights):
            if w == 0:
                continue
            combined = DensityPoly(
                (
                    *combined.terms,
                    *s.rho.scaled(w * scale).terms,
                )
            )
        combined = combined.combine_like_terms()
        return DensityPolyState(combined.normalize_trace())

    # --- measurements / expectations ---
    def expect(self, OP: OpPoly) -> complex:
        out = 0.0 + 0.0j
        for t in OP.terms:
            out += t.coeff * self.rho.apply_right(t.ops).trace()
        return out

    def trace(self) -> complex:
        return self.rho.trace()

    def partial_trace(self, trace_over_modes: set) -> DensityPolyState:
        reduced = self.rho.partial_trace(trace_over_modes)
        return DensityPolyState(reduced.normalize_trace())

    def purity(self) -> float:
        return self.rho.purity()

    def normalized(self, *, eps: float = 1e-14) -> DensityPolyState:
        return DensityPolyState(self.rho.normalize_trace(eps=eps))

    def is_trace_normalized(self, eps: float = 1e-12) -> bool:
        return abs(self.trace() - 1.0) <= eps

    def is_pure(self, eps: float = 1e-12) -> bool:
        return abs(self.purity() - 1.0) <= eps

    # --- pretty-printing ---
    def __repr__(self) -> str:
        return density_repr(self.rho.terms)

    def _repr_latex_(self) -> str:
        return density_state_latex_from_terms(
            self.rho.terms, label=self.label, index=self.index
        )
