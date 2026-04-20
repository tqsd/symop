from __future__ import annotations

import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.op import OpPoly
from symop.core.terms import OpTerm
from symop.polynomial.channels.models.mzi import (
    mzi_densitypoly,
    mzi_ketpoly,
    mzi_oppoly,
)

from tests.polynomial.state._builders import make_test_mode


class TestMZIChannels(unittest.TestCase):
    def test_mzi_ketpoly_runs_and_returns_ketpoly(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        poly = KetPoly.from_ops(creators=(mode0.cre,), annihilators=(), coeff=1.0)

        result = mzi_ketpoly(
            poly,
            mode0=mode0,
            mode1=mode1,
            theta1=0.0,
            theta2=0.0,
            phi_internal=0.0,
        )

        self.assertIsInstance(result, KetPoly)

    def test_mzi_densitypoly_runs_and_preserves_trace(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(creators=(mode0.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = mzi_densitypoly(
            rho,
            mode0=mode0,
            mode1=mode1,
            theta1=0.0,
            theta2=0.0,
            phi_internal=0.0,
        )

        self.assertEqual(result.trace(), 1.0 + 0.0j)

    def test_mzi_oppoly_runs(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        op = OpPoly((OpTerm(coeff=1.0, ops=(mode0.cre,)),)).combine_like_terms()

        result = mzi_oppoly(
            op,
            mode0=mode0,
            mode1=mode1,
            theta1=0.0,
            theta2=0.0,
            phi_internal=0.0,
        )

        self.assertIsInstance(result, OpPoly)
