from __future__ import annotations

import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm, KetTerm
from symop.polynomial.rewrites.relabel_modes import (
    _map_ops,
    _maybe_replace_mode,
    density_relabel_modes,
    ket_relabel_modes,
)

from tests.polynomial.state._builders import make_test_mode


class FakeNonDataclassLikeOp:
    def __init__(self, mode):
        self.mode = mode


class TestMaybeReplaceMode(unittest.TestCase):
    def test_returns_same_operator_when_signature_not_mapped(self):
        mode = make_test_mode(name="a", path="p0")
        op = mode.cre

        result = _maybe_replace_mode(op, {})

        self.assertIs(result, op)

    def test_replaces_operator_when_signature_is_mapped(self):
        old_mode = make_test_mode(name="a", path="p0")
        new_mode = make_test_mode(name="a", path="p1")
        op = old_mode.cre

        result = _maybe_replace_mode(op, {old_mode.signature: new_mode})

        self.assertIsNot(result, op)
        self.assertEqual(result.mode, new_mode)
        self.assertEqual(result.kind, op.kind)

    def test_raises_when_replacement_is_needed_for_non_dataclass_operator(self):
        old_mode = make_test_mode(name="a", path="p0")
        new_mode = make_test_mode(name="a", path="p1")
        op = FakeNonDataclassLikeOp(old_mode)

        with self.assertRaises(TypeError):
            _maybe_replace_mode(op, {old_mode.signature: new_mode})  # type: ignore[arg-type]


class TestMapOps(unittest.TestCase):
    def test_returns_same_tuple_when_nothing_changes(self):
        mode = make_test_mode(name="a", path="p0")
        ops = (mode.cre, mode.ann)

        result = _map_ops(ops, {})

        self.assertIs(result, ops)

    def test_returns_new_tuple_when_operator_changes(self):
        old_mode = make_test_mode(name="a", path="p0")
        new_mode = make_test_mode(name="a", path="p1")
        ops = (old_mode.cre, old_mode.ann)

        result = _map_ops(ops, {old_mode.signature: new_mode})

        self.assertIsNot(result, ops)
        self.assertEqual(result[0].mode, new_mode)
        self.assertEqual(result[1].mode, new_mode)

    def test_only_matching_operator_is_replaced(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        new_mode_a = make_test_mode(name="a", path="p2")
        ops = (mode_a.cre, mode_b.cre)

        result = _map_ops(ops, {mode_a.signature: new_mode_a})

        self.assertEqual(result[0].mode, new_mode_a)
        self.assertIs(result[1], mode_b.cre)


class TestKetRelabelModes(unittest.TestCase):
    def test_empty_mode_map_returns_same_terms_tuple(self):
        mode = make_test_mode(name="a", path="p0")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)

        result = ket_relabel_modes(poly.terms, mode_map={})

        self.assertIs(result, poly.terms)

    def test_unmatched_mode_map_keeps_same_term_objects(self):
        mode = make_test_mode(name="a", path="p0")
        other = make_test_mode(name="b", path="p1")
        poly = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)

        result = ket_relabel_modes(poly.terms, mode_map={other.signature: other})

        self.assertIs(result[0], poly.terms[0])

    def test_relabels_creator_mode_in_ket_term(self):
        old_mode = make_test_mode(name="a", path="p0")
        new_mode = make_test_mode(name="a", path="p1")
        poly = KetPoly.from_ops(creators=(old_mode.cre,), annihilators=(), coeff=1.0)

        result = ket_relabel_modes(poly.terms, mode_map={old_mode.signature: new_mode})

        self.assertEqual(result[0].monomial.creators[0].mode, new_mode)
        self.assertEqual(result[0].coeff, poly.terms[0].coeff)

    def test_relabels_annihilator_mode_in_ket_term(self):
        old_mode = make_test_mode(name="a", path="p0")
        new_mode = make_test_mode(name="a", path="p1")
        term = KetTerm(
            1.0,
            Monomial(creators=(), annihilators=(old_mode.ann,)),
        )

        result = ket_relabel_modes((term,), mode_map={old_mode.signature: new_mode})

        self.assertEqual(result[0].monomial.annihilators[0].mode, new_mode)

    def test_relabels_both_creator_and_annihilator(self):
        old_mode = make_test_mode(name="a", path="p0")
        new_mode = make_test_mode(name="a", path="p1")
        term = KetTerm(
            1.0,
            Monomial(
                creators=(old_mode.cre,),
                annihilators=(old_mode.ann,),
            ),
        )

        result = ket_relabel_modes((term,), mode_map={old_mode.signature: new_mode})

        self.assertEqual(result[0].monomial.creators[0].mode, new_mode)
        self.assertEqual(result[0].monomial.annihilators[0].mode, new_mode)

    def test_only_changed_terms_are_rebuilt(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        new_mode_a = make_test_mode(name="a", path="p2")

        term_a = KetTerm(
            1.0,
            Monomial(creators=(mode_a.cre,), annihilators=()),
        )
        term_b = KetTerm(
            2.0,
            Monomial(creators=(mode_b.cre,), annihilators=()),
        )

        result = ket_relabel_modes(
            (term_a, term_b),
            mode_map={mode_a.signature: new_mode_a},
        )

        self.assertIsNot(result[0], term_a)
        self.assertIs(result[1], term_b)
        self.assertEqual(result[0].monomial.creators[0].mode, new_mode_a)


class TestDensityRelabelModes(unittest.TestCase):
    def test_empty_mode_map_returns_same_terms_tuple(self):
        ket = KetPoly.identity()
        rho = DensityPoly.pure(ket)

        result = density_relabel_modes(rho.terms, mode_map={})

        self.assertIs(result, rho.terms)

    def test_unmatched_mode_map_keeps_same_term_objects(self):
        mode = make_test_mode(name="a", path="p0")
        other = make_test_mode(name="b", path="p1")
        ket = KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = density_relabel_modes(rho.terms, mode_map={other.signature: other})

        self.assertIs(result[0], rho.terms[0])

    def test_relabels_left_and_right_modes_in_density_term(self):
        old_mode = make_test_mode(name="a", path="p0")
        new_mode = make_test_mode(name="a", path="p1")
        ket = KetPoly.from_ops(creators=(old_mode.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = density_relabel_modes(
            rho.terms,
            mode_map={old_mode.signature: new_mode},
        )

        self.assertEqual(result[0].left.creators[0].mode, new_mode)
        self.assertEqual(result[0].right.creators[0].mode, new_mode)

    def test_relabels_selected_side_without_touching_other_term(self):
        mode_a = make_test_mode(name="a", path="p0")
        mode_b = make_test_mode(name="b", path="p1")
        new_mode_a = make_test_mode(name="a", path="p2")

        term_a = DensityTerm(
            1.0,
            Monomial(creators=(mode_a.cre,), annihilators=()),
            Monomial(creators=(mode_a.cre,), annihilators=()),
        )
        term_b = DensityTerm(
            2.0,
            Monomial(creators=(mode_b.cre,), annihilators=()),
            Monomial(creators=(mode_b.cre,), annihilators=()),
        )

        result = density_relabel_modes(
            (term_a, term_b),
            mode_map={mode_a.signature: new_mode_a},
        )

        self.assertIsNot(result[0], term_a)
        self.assertIs(result[1], term_b)
        self.assertEqual(result[0].left.creators[0].mode, new_mode_a)
        self.assertEqual(result[0].right.creators[0].mode, new_mode_a)
