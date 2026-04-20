from __future__ import annotations

import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.op import OpPoly
from symop.core.terms import OpTerm
from symop.polynomial.rewrites.substitution import (
    _normalize_word_to_monomials,
    expand_word_substitution,
    rewrite_densitypoly,
    rewrite_ketpoly,
    rewrite_oppoly,
)

from tests.polynomial.state._builders import make_test_mode


class TestExpandWordSubstitution(unittest.TestCase):
    def test_empty_word_returns_identity_expansion(self):
        def subst_fn(op):
            return [(1.0, op)]

        result = expand_word_substitution((), subst_fn)

        self.assertEqual(result, [(1.0 + 0.0j, ())])

    def test_single_operator_single_replacement(self):
        mode = make_test_mode(name="a", path="p0")

        def subst_fn(op):
            return [(2.0, op)]

        result = expand_word_substitution((mode.cre,), subst_fn)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 2.0 + 0.0j)
        self.assertEqual(result[0][1], (mode.cre,))

    def test_single_operator_multiple_replacements(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")

        def subst_fn(op):
            return [(1.0, mode_a.cre), (2.0, mode_b.cre)]

        result = expand_word_substitution((mode_a.cre,), subst_fn)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (1.0 + 0.0j, (mode_a.cre,)))
        self.assertEqual(result[1], (2.0 + 0.0j, (mode_b.cre,)))

    def test_two_operator_cartesian_product(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")

        def subst_fn(op):
            if op is mode_a.cre:
                return [(1.0, mode_a.cre), (2.0, mode_b.cre)]
            return [(3.0, mode_b.cre), (4.0, mode_a.cre)]

        result = expand_word_substitution((mode_a.cre, mode_b.cre), subst_fn)

        self.assertEqual(len(result), 4)
        coeffs = {coeff for coeff, _ in result}
        self.assertEqual(coeffs, {3.0 + 0.0j, 4.0 + 0.0j, 6.0 + 0.0j, 8.0 + 0.0j})

    def test_eps_prunes_small_branches(self):
        mode = make_test_mode(name="a", path="p0")

        def subst_fn(op):
            return [(1e-15, op), (1.0, op)]

        result = expand_word_substitution((mode.cre,), subst_fn, eps=1e-12)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1.0 + 0.0j)

    def test_zero_length_word_not_pruned(self):
        def subst_fn(op):
            return [(1.0, op)]

        result = expand_word_substitution((), subst_fn, eps=1e-12)

        self.assertEqual(result, [(1.0 + 0.0j, ())])


class TestNormalizeWordToMonomials(unittest.TestCase):
    def test_empty_word_produces_identity_monomial(self):
        result = _normalize_word_to_monomials((), eps=1e-12)

        self.assertEqual(len(result), 1)
        coeff, monomial = result[0]
        self.assertEqual(coeff, 1.0 + 0.0j)
        self.assertTrue(monomial.is_identity)

    def test_creator_word_produces_nonempty_result(self):
        mode = make_test_mode(name="a", path="p0")

        result = _normalize_word_to_monomials((mode.cre,), eps=1e-12)

        self.assertGreaterEqual(len(result), 1)
        coeffs = [coeff for coeff, _ in result]
        self.assertIn(1.0 + 0.0j, coeffs)

    def test_annihilator_word_produces_nonempty_result(self):
        mode = make_test_mode(name="a", path="p0")

        result = _normalize_word_to_monomials((mode.ann,), eps=1e-12)

        self.assertGreaterEqual(len(result), 1)

    def test_mixed_word_produces_nonempty_result(self):
        mode = make_test_mode(name="a", path="p0")

        result = _normalize_word_to_monomials((mode.ann, mode.cre), eps=1e-12)

        self.assertGreaterEqual(len(result), 1)


class TestRewriteKetPoly(unittest.TestCase):
    def test_identity_substitution_leaves_ketpoly_unchanged(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=2.0)

        def subst_fn(op):
            return [(1.0, op)]

        result = rewrite_ketpoly(poly, subst_fn)

        self.assertEqual(result, poly.combine_like_terms())

    def test_sum_substitution_expands_single_creator_term(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        poly = KetPoly.from_ops(creators=(mode_a.cre,), annihilators=(), coeff=1.0)

        def subst_fn(op):
            return [(1.0, mode_a.cre), (2.0, mode_b.cre)]

        result = rewrite_ketpoly(poly, subst_fn)

        self.assertEqual(len(result.terms), 2)
        coeffs = {term.coeff for term in result.terms}
        self.assertEqual(coeffs, {1.0 + 0.0j, 2.0 + 0.0j})

    def test_eps_prunes_small_substitution_terms(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        poly = KetPoly.from_ops(creators=(mode_a.cre,), annihilators=(), coeff=1.0)

        def subst_fn(op):
            return [(1e-15, mode_a.cre), (1.0, mode_b.cre)]

        result = rewrite_ketpoly(poly, subst_fn, eps=1e-12)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].monomial.creators[0].mode, mode_b)

    def test_like_terms_are_combined(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)

        def subst_fn(op):
            return [(1.0, mode.cre), (2.0, mode.cre)]

        result = rewrite_ketpoly(poly, subst_fn)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].coeff, 3.0 + 0.0j)


class TestRewriteDensityPoly(unittest.TestCase):
    def test_identity_substitution_leaves_densitypoly_unchanged(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        def subst_fn(op):
            return [(1.0, op)]

        result = rewrite_densitypoly(rho, subst_fn)

        self.assertEqual(result, rho.combine_like_terms())

    def test_left_and_right_substitutions_are_applied(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(creators=(mode_a.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        def left_subst(op):
            return [(1.0, mode_b.cre)]

        def right_subst(op):
            return [(1.0, mode_a.cre)]

        result = rewrite_densitypoly(rho, left_subst, right_subst)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].left.creators[0].mode, mode_b)
        self.assertEqual(result.terms[0].right.creators[0].mode, mode_a)

    def test_none_right_substitution_reuses_left_function(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(creators=(mode_a.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        def subst_fn(op):
            return [(1.0, mode_b.cre)]

        result = rewrite_densitypoly(rho, subst_fn)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].left.creators[0].mode, mode_b)
        self.assertEqual(result.terms[0].right.creators[0].mode, mode_b)

    def test_sum_substitution_expands_densitypoly(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(creators=(mode_a.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        def subst_fn(op):
            return [(1.0, mode_a.cre), (2.0, mode_b.cre)]

        result = rewrite_densitypoly(rho, subst_fn)

        self.assertEqual(len(result.terms), 4)
        coeffs = [term.coeff for term in result.terms]
        self.assertIn(1.0 + 0.0j, coeffs)
        self.assertIn(2.0 + 0.0j, coeffs)
        self.assertIn(4.0 + 0.0j, coeffs)

    def test_like_terms_are_combined_in_density_rewrite(self):
        mode = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        def subst_fn(op):
            return [(1.0, mode.cre), (2.0, mode.cre)]

        result = rewrite_densitypoly(rho, subst_fn)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].coeff, 9.0 + 0.0j)


class TestRewriteOpPoly(unittest.TestCase):
    def test_identity_substitution_leaves_oppoly_unchanged(self):
        mode = make_test_mode(name="a", path="p0")
        poly = OpPoly((OpTerm(coeff=2.0, ops=(mode.cre, mode.ann)),)).combine_like_terms()

        def subst_fn(op):
            return [(1.0, op)]

        result = rewrite_oppoly(poly, subst_fn)

        self.assertEqual(result, poly)

    def test_sum_substitution_expands_oppoly(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        poly = OpPoly((OpTerm(coeff=1.0, ops=(mode_a.cre,)),))

        def subst_fn(op):
            return [(1.0, mode_a.cre), (2.0, mode_b.cre)]

        result = rewrite_oppoly(poly, subst_fn)

        self.assertEqual(len(result.terms), 2)
        coeffs = {term.coeff for term in result.terms}
        self.assertEqual(coeffs, {1.0 + 0.0j, 2.0 + 0.0j})

    def test_eps_prunes_small_operator_terms(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        poly = OpPoly((OpTerm(coeff=1.0, ops=(mode_a.cre,)),))

        def subst_fn(op):
            return [(1e-15, mode_a.cre), (1.0, mode_b.cre)]

        result = rewrite_oppoly(poly, subst_fn, eps=1e-12)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].ops[0].mode, mode_b)

    def test_like_terms_are_combined_in_operator_rewrite(self):
        mode = make_test_mode(name="a", path="p0")
        poly = OpPoly((OpTerm(coeff=1.0, ops=(mode.cre,)),))

        def subst_fn(op):
            return [(1.0, mode.cre), (2.0, mode.cre)]

        result = rewrite_oppoly(poly, subst_fn)

        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].coeff, 3.0 + 0.0j)
