from __future__ import annotations

import unittest
from dataclasses import dataclass

from symop.ccr.algebra.density.overlap_right_left import overlap_right_left
from symop.core.monomial import Monomial
from symop.core.operators import ModeOp


@dataclass(frozen=True)
class DummyModeLabel:
    """
    Minimal label for ModeOp tests.

    overlap rules:
      - same key -> 1
      - different key -> 0
    """

    key: str

    def overlap(self, other: DummyModeLabel) -> complex:
        return 1.0 + 0.0j if self.key == other.key else 0.0 + 0.0j

    @property
    def signature(self) -> tuple:
        return ("dummy_label", self.key)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> tuple:
        return ("dummy_label_approx", self.key)


class TestOverlapRightLeft(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_overlap_identity_identity_is_one(self) -> None:
        I = self._identity_monomial()
        out = overlap_right_left(I, I)
        self.assertAlmostEqual(out.real, 1.0)
        self.assertAlmostEqual(out.imag, 0.0)

    def test_overlap_identity_with_single_annihilator_is_zero(self) -> None:
        I = self._identity_monomial()
        m = ModeOp(label=DummyModeLabel("m"))
        L = Monomial(creators=(), annihilators=(m.ann,))
        out = overlap_right_left(I, L)
        self.assertAlmostEqual(out.real, 0.0)
        self.assertAlmostEqual(out.imag, 0.0)

    def test_overlap_identity_with_single_creator_is_zero(self) -> None:
        I = self._identity_monomial()
        m = ModeOp(label=DummyModeLabel("m"))
        L = Monomial(creators=(m.create,), annihilators=())
        out = overlap_right_left(I, L)
        self.assertAlmostEqual(out.real, 0.0)
        self.assertAlmostEqual(out.imag, 0.0)

    def test_overlap_a_with_a_same_mode_is_zero(self) -> None:
        # <a|a> is identity coeff of (a)^dag * a = a^dag a (already normal ordered) -> 0
        m = ModeOp(label=DummyModeLabel("m"))
        R = Monomial(creators=(), annihilators=(m.ann,))
        L = Monomial(creators=(), annihilators=(m.ann,))
        out = overlap_right_left(R, L)
        self.assertAlmostEqual(out.real, 0.0)
        self.assertAlmostEqual(out.imag, 0.0)

    def test_overlap_adag_with_adag_same_mode_is_one(self) -> None:
        # <a^dag|a^dag> is identity coeff of (a^dag)^dag * a^dag = a a^dag = a^dag a + 1 -> 1
        m = ModeOp(label=DummyModeLabel("m"))
        R = Monomial(creators=(m.create,), annihilators=())
        L = Monomial(creators=(m.create,), annihilators=())
        out = overlap_right_left(R, L)
        self.assertAlmostEqual(out.real, 1.0)
        self.assertAlmostEqual(out.imag, 0.0)

    def test_overlap_adag_with_adag_orthogonal_modes_is_zero(self) -> None:
        # a_i a_j^dag has identity coefficient G_ij; for orthogonal modes G_ij=0
        m1 = ModeOp(label=DummyModeLabel("m1"))
        m2 = ModeOp(label=DummyModeLabel("m2"))
        R = Monomial(creators=(m2.create,), annihilators=())
        L = Monomial(creators=(m1.create,), annihilators=())
        out = overlap_right_left(R, L)
        self.assertAlmostEqual(out.real, 0.0)
        self.assertAlmostEqual(out.imag, 0.0)

    def test_overlap_a_with_adag_is_zero(self) -> None:
        # <a|a^dag> is identity coeff of (a)^dag * (a^dag) = a^dag a^dag -> 0
        m = ModeOp(label=DummyModeLabel("m"))
        R = Monomial(creators=(), annihilators=(m.ann,))
        L = Monomial(creators=(m.create,), annihilators=())
        out = overlap_right_left(R, L)
        self.assertAlmostEqual(out.real, 0.0)
        self.assertAlmostEqual(out.imag, 0.0)
