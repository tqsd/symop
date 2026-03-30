from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm
from symop.core.terms.ket_term import KetTerm
from symop.core.terms.op_term import OpTerm

from symop.core.types.signature import Signature
from tests.core.support.fakes import (
    FakeComponentLabel,
    FakeModeLabel,
    make_mode,
    make_mode_label,
    set_symmetric_overlap,
    set_hermitian_overlap,
)


def make_monomial(
    *,
    creators: tuple = (),
    annihilators: tuple = (),
) -> Monomial:
    """Construct a real Monomial for CCR tests."""
    return Monomial(
        creators=tuple(creators),
        annihilators=tuple(annihilators),
    )


def make_ket_term(
    *,
    coeff: complex = 1.0,
    creators: tuple = (),
    annihilators: tuple = (),
) -> KetTerm:
    """Construct a real KetTerm from creator/annihilator tuples."""
    return KetTerm(
        coeff=coeff,
        monomial=make_monomial(
            creators=creators,
            annihilators=annihilators,
        ),
    )


def make_density_term(
    *,
    coeff: complex = 1.0,
    left_creators: tuple = (),
    left_annihilators: tuple = (),
    right_creators: tuple = (),
    right_annihilators: tuple = (),
) -> DensityTerm:
    """Construct a real DensityTerm from left/right monomial pieces."""
    return DensityTerm(
        coeff=coeff,
        left=make_monomial(
            creators=left_creators,
            annihilators=left_annihilators,
        ),
        right=make_monomial(
            creators=right_creators,
            annihilators=right_annihilators,
        ),
    )


def make_op_term(
    *,
    ops: tuple = (),
    coeff: complex = 1.0,
) -> OpTerm:
    """Construct a real OpTerm."""
    return OpTerm(
        ops=tuple(ops),
        coeff=coeff,
    )





@dataclass(frozen=True)
class FakeOpPoly:
    """Minimal but protocol-complete OpPoly test double.

    This implements the full OpPoly surface required by the protocol,
    but does NOT perform real algebra. It is only intended for testing
    integration points such as dispatch and operator application.
    """

    terms: tuple[OpTerm, ...] = ()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def is_zero(self) -> bool:
        return len(self.terms) == 0

    @property
    def is_identity(self) -> bool:
        return self.terms == (OpTerm.identity(),)

    @property
    def unique_modes(self) -> tuple:
        seen = {}
        for term in self.terms:
            for op in term.ops:
                seen.setdefault(op.mode.signature, op.mode)
        return tuple(seen.values())

    @property
    def mode_count(self) -> int:
        return len(self.unique_modes)

    # ------------------------------------------------------------------
    # Core operations (minimal behavior)
    # ------------------------------------------------------------------

    def scaled(self, c: complex) -> FakeOpPoly:
        return FakeOpPoly(tuple(term.scaled(c) for term in self.terms))

    def normalize(self) -> FakeOpPoly:
        return self  # no-op for fake

    def combine_like_terms(self) -> FakeOpPoly:
        return self  # no-op for fake

    def adjoint(self) -> FakeOpPoly:
        return FakeOpPoly(tuple(term.adjoint() for term in self.terms))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: FakeOpPoly) -> FakeOpPoly:
        if not isinstance(other, FakeOpPoly):
            return NotImplemented
        return FakeOpPoly(self.terms + other.terms)

    def __radd__(self, other: FakeOpPoly) -> FakeOpPoly:
        return self.__add__(other)

    def __mul__(self, other: FakeOpPoly | complex) -> FakeOpPoly:
        if isinstance(other, (int, float, complex)):
            return self.scaled(other)
        if isinstance(other, FakeOpPoly):
            # naive cartesian product (no CCR logic)
            terms = tuple(
                OpTerm(ops=t1.ops + t2.ops, coeff=t1.coeff * t2.coeff)
                for t1 in self.terms
                for t2 in other.terms
            )
            return FakeOpPoly(terms)
        return NotImplemented

    def __rmul__(self, other: complex) -> FakeOpPoly:
        if isinstance(other, (int, float, complex)):
            return self.scaled(other)
        return NotImplemented

    def __matmul__(self, other: object) -> object:
        # Not needed for most tests
        return NotImplemented

    def __rmatmul__(self, other: object) -> object:
        # Not needed for most tests
        return NotImplemented

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        return not self.is_zero

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FakeOpPoly):
            return NotImplemented
        return self.terms == other.terms

    @property
    def signature(self) -> Signature:
        return ("fake_op_poly", tuple(t.signature for t in self.terms))

def make_fake_op_poly(*terms: OpTerm) -> FakeOpPoly:
    return FakeOpPoly(tuple(terms))

if TYPE_CHECKING:
    from symop.ccr.protocols.op import OpPoly as OpPolyProtocol

    _fake_op_poly_check: OpPolyProtocol = FakeOpPoly()
