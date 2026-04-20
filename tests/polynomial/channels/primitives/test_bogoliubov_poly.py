from __future__ import annotations

import unittest

import numpy as np

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.op import OpPoly
from symop.core.terms import OpTerm
from symop.core.types.operator_kind import OperatorKind
from symop.polynomial.channels.primitives.bogoliubov import (
    BogoliubovMap,
    apply_to_densitypoly,
    apply_to_ketpoly,
    apply_to_oppoly,
    make_substitution,
)

from tests.polynomial.state._builders import make_test_mode


class TestBogoliubovMap(unittest.TestCase):
    def test_rejects_duplicate_modes(self):
        mode = make_test_mode(name="a", path="p0")
        X = np.eye(2, dtype=np.complex128)
        Y = np.zeros((2, 2), dtype=np.complex128)

        with self.assertRaises(ValueError):
            BogoliubovMap(modes=(mode, mode), X=X, Y=Y)

    def test_rejects_wrong_x_shape(self):
        mode = make_test_mode(name="a", path="p0")
        X = np.eye(2, dtype=np.complex128)
        Y = np.zeros((1, 1), dtype=np.complex128)

        with self.assertRaises(ValueError):
            BogoliubovMap(modes=(mode,), X=X, Y=Y)

    def test_rejects_wrong_y_shape(self):
        mode = make_test_mode(name="a", path="p0")
        X = np.eye(1, dtype=np.complex128)
        Y = np.zeros((2, 2), dtype=np.complex128)

        with self.assertRaises(ValueError):
            BogoliubovMap(modes=(mode,), X=X, Y=Y)

    def test_accepts_valid_ccr_map(self):
        mode = make_test_mode(name="a", path="p0")
        X = np.array([[1.0]], dtype=np.complex128)
        Y = np.array([[0.0]], dtype=np.complex128)

        bmap = BogoliubovMap(modes=(mode,), X=X, Y=Y, check_ccr=True)

        self.assertEqual(bmap.modes, (mode,))

    def test_rejects_invalid_ccr_map(self):
        mode = make_test_mode(name="a", path="p0")
        X = np.array([[2.0]], dtype=np.complex128)
        Y = np.array([[0.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            BogoliubovMap(modes=(mode,), X=X, Y=Y, check_ccr=True)


class TestBogoliubovSubstitution(unittest.TestCase):
    def test_unknown_mode_is_left_unchanged(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        X = np.array([[1.0]], dtype=np.complex128)
        Y = np.array([[0.0]], dtype=np.complex128)
        bmap = BogoliubovMap(modes=(mode_a,), X=X, Y=Y)
        subst = make_substitution(bmap)

        result = subst(mode_b.cre)

        self.assertEqual(result, [(1.0 + 0.0j, mode_b.cre)])

    def test_creation_operator_uses_x_and_y(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        X = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.complex128,
        )
        Y = np.array(
            [
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.complex128,
        )
        bmap = BogoliubovMap(modes=(mode_a, mode_b), X=X, Y=Y)
        subst = make_substitution(bmap)

        result = subst(mode_a.cre)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0][0], 1.0 + 0.0j)
        self.assertEqual(result[1][0], 3.0 + 0.0j)
        self.assertEqual(result[2][0], 2.0 + 0.0j)
        self.assertEqual(result[3][0], 4.0 + 0.0j)

        self.assertEqual(result[0][1].mode, mode_a)
        self.assertEqual(result[0][1].kind, OperatorKind.CRE)
        self.assertEqual(result[1][1].mode, mode_a)
        self.assertEqual(result[1][1].kind, OperatorKind.ANN)
        self.assertEqual(result[2][1].mode, mode_b)
        self.assertEqual(result[2][1].kind, OperatorKind.CRE)
        self.assertEqual(result[3][1].mode, mode_b)
        self.assertEqual(result[3][1].kind, OperatorKind.ANN)

    def test_annihilation_operator_uses_conjugated_y_and_x(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        X = np.array(
            [
                [1.0 + 1.0j, 0.0],
                [2.0 - 1.0j, 0.0],
            ],
            dtype=np.complex128,
        )
        Y = np.array(
            [
                [3.0 + 2.0j, 0.0],
                [4.0 - 5.0j, 0.0],
            ],
            dtype=np.complex128,
        )
        bmap = BogoliubovMap(modes=(mode_a, mode_b), X=X, Y=Y)
        subst = make_substitution(bmap)

        result = subst(mode_a.ann)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0][0], np.conjugate(Y[0, 0]))
        self.assertEqual(result[1][0], np.conjugate(X[0, 0]))
        self.assertEqual(result[2][0], np.conjugate(Y[1, 0]))
        self.assertEqual(result[3][0], np.conjugate(X[1, 0]))

        self.assertEqual(result[0][1].mode, mode_a)
        self.assertEqual(result[0][1].kind, OperatorKind.CRE)
        self.assertEqual(result[1][1].mode, mode_a)
        self.assertEqual(result[1][1].kind, OperatorKind.ANN)
        self.assertEqual(result[2][1].mode, mode_b)
        self.assertEqual(result[2][1].kind, OperatorKind.CRE)
        self.assertEqual(result[3][1].mode, mode_b)
        self.assertEqual(result[3][1].kind, OperatorKind.ANN)


class TestBogoliubovApplyHelpers(unittest.TestCase):
    def test_apply_to_densitypoly_identity_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)
        bmap = BogoliubovMap(
            modes=(mode,),
            X=np.array([[1.0]], dtype=np.complex128),
            Y=np.array([[0.0]], dtype=np.complex128),
        )

        result = apply_to_densitypoly(rho, bmap=bmap)

        self.assertEqual(result, rho.combine_like_terms())

    def test_apply_to_oppoly_identity_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        poly = OpPoly((OpTerm(coeff=1.0, ops=(mode.cre, mode.ann)),)).combine_like_terms()
        bmap = BogoliubovMap(
            modes=(mode,),
            X=np.array([[1.0]], dtype=np.complex128),
            Y=np.array([[0.0]], dtype=np.complex128),
        )

        result = apply_to_oppoly(poly, bmap=bmap)

        self.assertEqual(result, poly)

    def test_apply_to_ketpoly_calls_reducer(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        bmap = BogoliubovMap(
            modes=(mode,),
            X=np.array([[1.0]], dtype=np.complex128),
            Y=np.array([[0.0]], dtype=np.complex128),
        )

        seen = {}

        def reducer(rewritten):
            seen["called"] = True
            seen["poly"] = rewritten
            return rewritten

        result = apply_to_ketpoly(poly, bmap=bmap, reduce_ketpoly=reducer)

        self.assertTrue(seen["called"])
        self.assertEqual(result, seen["poly"])
        self.assertEqual(result, poly.combine_like_terms())

    def test_apply_to_ketpoly_can_generate_annihilators_before_reduction(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        bmap = BogoliubovMap(
            modes=(mode,),
            X=np.array([[0.0]], dtype=np.complex128),
            Y=np.array([[1.0]], dtype=np.complex128),
        )

        seen = {}

        def reducer(rewritten):
            seen["poly"] = rewritten
            return KetPoly.identity()

        result = apply_to_ketpoly(poly, bmap=bmap, reduce_ketpoly=reducer)

        self.assertEqual(result, KetPoly.identity())
        self.assertTrue(
            any(
                term.monomial.annihilators
                for term in seen["poly"].terms
            )
        )
    def test_apply_to_densitypoly_identity_map_preserves_trace_normalization(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=2.0)
        rho = DensityPoly.pure(ket).normalize_trace()

        bmap = BogoliubovMap(
            modes=(mode,),
            X=np.array([[1.0]], dtype=np.complex128),
            Y=np.array([[0.0]], dtype=np.complex128),
        )

        result = apply_to_densitypoly(rho, bmap=bmap)

        self.assertTrue(result.is_trace_normalized())
