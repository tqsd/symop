from __future__ import annotations

import math
import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.op import OpPoly
from symop.core.monomial import Monomial
from symop.core.terms import OpTerm
from symop.modes.labels import Path
from symop.polynomial.channels.models.beamsplitter import (
    _beamsplitter_substitution,
    beamsplitter_50_50_densitypoly,
    beamsplitter_50_50_ketpoly,
    beamsplitter_50_50_oppoly,
    beamsplitter_densitypoly,
    beamsplitter_ketpoly,
    beamsplitter_oppoly,
)

from tests.polynomial.state._builders import make_test_mode


class TestBeamsplitterSubstitution(unittest.TestCase):
    def test_unknown_mode_is_left_unchanged(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        other = make_test_mode(name="c", path="p2")

        subst = _beamsplitter_substitution(
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
            theta=math.pi / 4.0,
            phi_t=0.0,
            phi_r=0.0,
        )

        result = subst(other.cre)

        self.assertEqual(result, [(1.0 + 0.0j, other.cre)])

    def test_creator_substitution_for_mode0_uses_two_output_paths(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")

        subst = _beamsplitter_substitution(
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
            theta=math.pi / 4.0,
            phi_t=0.0,
            phi_r=0.0,
        )

        result = subst(mode0.cre)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1].mode.label.path, Path("out0"))
        self.assertEqual(result[1][1].mode.label.path, Path("out1"))

    def test_creator_substitution_for_mode1_has_minus_sign_on_second_output(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")

        subst = _beamsplitter_substitution(
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
            theta=0.0,
            phi_t=0.0,
            phi_r=0.0,
        )

        result = subst(mode1.cre)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 0.0 + 0.0j)
        self.assertEqual(result[1][0], -1.0 + 0.0j)

    def test_annihilator_substitution_uses_conjugated_coefficients(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")

        subst = _beamsplitter_substitution(
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
            theta=math.pi / 4.0,
            phi_t=0.3,
            phi_r=-0.2,
        )

        result = subst(mode0.ann)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1].mode.label.path, Path("out0"))
        self.assertEqual(result[1][1].mode.label.path, Path("out1"))


class TestBeamsplitterChannels(unittest.TestCase):
    def test_beamsplitter_ketpoly_theta_zero_routes_to_expected_outputs(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        poly = KetPoly.from_ops(creators=(mode0.cre,), annihilators=(), coeff=1.0)

        result = beamsplitter_ketpoly(
            poly,
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
            theta=0.0,
        )

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].monomial.creators[0].mode.label.path, Path("out0"))

    def test_beamsplitter_ketpoly_50_50_splits_single_photon_into_two_terms(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        poly = KetPoly.from_ops(creators=(mode0.cre,), annihilators=(), coeff=1.0)

        result = beamsplitter_50_50_ketpoly(
            poly,
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
        )

        self.assertEqual(len(result.terms), 2)
        out_paths = {t.monomial.creators[0].mode.label.path for t in result.terms}
        self.assertEqual(out_paths, {Path("out0"), Path("out1")})

    def test_beamsplitter_densitypoly_theta_zero_is_trace_preserving(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(creators=(mode0.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = beamsplitter_densitypoly(
            rho,
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
            theta=0.0,
        )

        self.assertEqual(result.trace(), 1.0 + 0.0j)

    def test_beamsplitter_oppoly_theta_zero_rewrites_mode_paths(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        op = OpPoly((OpTerm(coeff=1.0, ops=(mode0.cre,)),)).combine_like_terms()

        result = beamsplitter_oppoly(
            op,
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
            theta=0.0,
        )

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].ops[0].mode.label.path, Path("out0"))

    def test_beamsplitter_50_50_densitypoly_and_oppoly_run(self):
        mode0 = make_test_mode(name="a", path="p0")
        mode1 = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(creators=(mode0.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)
        op = OpPoly((OpTerm(coeff=1.0, ops=(mode0.cre,)),)).combine_like_terms()

        rho_out = beamsplitter_50_50_densitypoly(
            rho,
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
        )
        op_out = beamsplitter_50_50_oppoly(
            op,
            mode0=mode0,
            mode1=mode1,
            out0=Path("out0"),
            out1=Path("out1"),
        )

        self.assertEqual(rho_out.trace(), 1.0 + 0.0j)
        self.assertGreaterEqual(len(op_out.terms), 1)
