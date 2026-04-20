from __future__ import annotations

import math
import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.op import OpPoly
from symop.core.terms import OpTerm
from symop.polynomial.channels.models.phase import (
    phase_densitypoly,
    phase_ketpoly,
    phase_oppoly,
)

from tests.polynomial.state._builders import make_test_mode


class TestPhaseChannels(unittest.TestCase):
    def test_phase_ketpoly_zero_phase_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=2.0)

        result = phase_ketpoly(poly, mode=mode, phi=0.0)

        self.assertEqual(result, poly.combine_like_terms())

    def test_phase_ketpoly_pi_phase_flips_single_creator_sign(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)

        result = phase_ketpoly(poly, mode=mode, phi=math.pi)

        self.assertEqual(len(result.terms), 1)
        self.assertAlmostEqual(result.terms[0].coeff.real, -1.0, places=12)
        self.assertAlmostEqual(result.terms[0].coeff.imag, 0.0, places=12)

    def test_phase_densitypoly_zero_phase_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = phase_densitypoly(rho, mode=mode, phi=0.0)

        self.assertEqual(result, rho.combine_like_terms())

    def test_phase_densitypoly_preserves_trace_for_identity_case(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = phase_densitypoly(rho, mode=mode, phi=0.7)

        self.assertEqual(result.trace(), 1.0 + 0.0j)

    def test_phase_oppoly_zero_phase_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        op = OpPoly((OpTerm(coeff=1.0, ops=(mode.cre, mode.ann)),)).combine_like_terms()

        result = phase_oppoly(op, mode=mode, phi=0.0)

        self.assertEqual(result, op)
