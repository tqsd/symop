from __future__ import annotations

import unittest

import numpy as np

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.op import OpPoly
from symop.core.monomial import Monomial
from symop.core.terms import OpTerm
from symop.core.types.operator_kind import OperatorKind
from symop.polynomial.channels.primitives.linear_mode_unitary import (
    LinearModeMap,
    _apply_ketpoly_to_vacuum,
    apply_to_densitypoly,
    apply_to_ketpoly,
    apply_to_oppoly,
    make_substitution,
)

from tests.polynomial.state._builders import make_test_mode


class TestLinearModeMap(unittest.TestCase):
    def test_rejects_wrong_matrix_shape(self):
        mode = make_test_mode(name="a", path="p0")
        U = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            LinearModeMap(modes=(mode,), U=U)

    def test_rejects_non_unitary_matrix_when_check_enabled(self):
        mode = make_test_mode(name="a", path="p0")
        U = np.array([[2.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            LinearModeMap(modes=(mode,), U=U, check_unitary=True)

    def test_accepts_unitary_matrix_when_check_enabled(self):
        mode = make_test_mode(name="a", path="p0")
        U = np.array([[1.0]], dtype=np.complex128)

        lmap = LinearModeMap(modes=(mode,), U=U, check_unitary=True)

        self.assertEqual(lmap.modes, (mode,))
        self.assertTrue(np.array_equal(lmap.U, U))


class TestLinearModeMapSubstitution(unittest.TestCase):
    def test_unknown_mode_is_left_unchanged(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        U = np.array([[1.0]], dtype=np.complex128)
        lmap = LinearModeMap(modes=(mode_a,), U=U)
        subst = make_substitution(lmap)

        result = subst(mode_b.cre)

        self.assertEqual(result, [(1.0 + 0.0j, mode_b.cre)])

    def test_creation_operator_is_rewritten_by_column(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        U = np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
            ],
            dtype=np.complex128,
        )
        lmap = LinearModeMap(modes=(mode_a, mode_b), U=U)
        subst = make_substitution(lmap)

        result = subst(mode_a.cre)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 1.0 + 0.0j)
        self.assertEqual(result[1][0], 2.0 + 0.0j)
        self.assertEqual(result[0][1].mode, mode_a)
        self.assertEqual(result[1][1].mode, mode_b)
        self.assertEqual(result[0][1].kind, OperatorKind.CRE)
        self.assertEqual(result[1][1].kind, OperatorKind.CRE)

    def test_annihilation_operator_uses_conjugated_column(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        U = np.array(
            [
                [1.0 + 1.0j, 0.0],
                [2.0 - 3.0j, 0.0],
            ],
            dtype=np.complex128,
        )
        lmap = LinearModeMap(modes=(mode_a, mode_b), U=U)
        subst = make_substitution(lmap)

        result = subst(mode_a.ann)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], np.conjugate(U[0, 0]))
        self.assertEqual(result[1][0], np.conjugate(U[1, 0]))
        self.assertEqual(result[0][1].mode, mode_a)
        self.assertEqual(result[1][1].mode, mode_b)
        self.assertEqual(result[0][1].kind, OperatorKind.ANN)
        self.assertEqual(result[1][1].kind, OperatorKind.ANN)


class TestApplyKetpolyToVacuum(unittest.TestCase):
    def test_keeps_creator_only_and_identity_terms(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly(
            (
                KetPoly.identity().terms[0],
                KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=2.0).terms[0],
                KetPoly.from_ops(creators=(), annihilators=(mode.ann,), coeff=3.0).terms[0],
            )
        )

        result = _apply_ketpoly_to_vacuum(poly)

        self.assertEqual(len(result.terms), 2)
        self.assertTrue(all(t.monomial.is_creator_only or t.monomial.is_identity for t in result.terms))

    def test_drops_annihilator_only_terms(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(), annihilators=(mode.ann,), coeff=1.0)

        result = _apply_ketpoly_to_vacuum(poly)

        self.assertEqual(len(result.terms), 0)


class TestLinearModeMapApplyHelpers(unittest.TestCase):
    def test_apply_to_ketpoly_identity_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        lmap = LinearModeMap(
            modes=(mode,),
            U=np.array([[1.0]], dtype=np.complex128),
        )

        result = apply_to_ketpoly(poly, lmap=lmap)

        self.assertEqual(result, poly.combine_like_terms())

    def test_apply_to_ketpoly_swaps_modes(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        poly = KetPoly.from_ops(creators=(mode_a.cre,), annihilators=(), coeff=1.0)
        U = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
        lmap = LinearModeMap(modes=(mode_a, mode_b), U=U)

        result = apply_to_ketpoly(poly, lmap=lmap)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].monomial.creators[0].mode, mode_b)

    def test_apply_to_densitypoly_identity_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)
        lmap = LinearModeMap(
            modes=(mode,),
            U=np.array([[1.0]], dtype=np.complex128),
        )

        result = apply_to_densitypoly(rho, lmap=lmap)

        self.assertEqual(result, rho.combine_like_terms())

    def test_apply_to_oppoly_identity_is_noop(self):
        mode = make_test_mode(name="a", path="p0")
        poly = OpPoly((OpTerm(coeff=2.0, ops=(mode.cre, mode.ann)),)).combine_like_terms()
        lmap = LinearModeMap(
            modes=(mode,),
            U=np.array([[1.0]], dtype=np.complex128),
        )

        result = apply_to_oppoly(poly, lmap=lmap)

        self.assertEqual(result, poly)
