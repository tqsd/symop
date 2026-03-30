import unittest

from symop.ccr.algebra.ket.poly import KetPoly
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import (
    make_fake_op_poly,
    make_mode,
    make_op_term,
    set_hermitian_overlap,
)


class TestKetPolyConstructors(unittest.TestCase):
    def test_identity(self) -> None:
        poly = KetPoly.identity()

        self.assertTrue(poly.is_identity)
        self.assertTrue(poly.is_creator_only)
        self.assertTrue(poly.is_annihilator_only)
        self.assertEqual(poly.creation_count, 0)
        self.assertEqual(poly.annihilation_count, 0)
        self.assertEqual(poly.total_degree, 0)
        self.assertEqual(poly.mode_count, 0)

    def test_from_ops(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(
            creators=(mode.cre,),
            annihilators=(mode.ann,),
            coeff=2.0,
        )

        expected = KetPoly(
            (
                KetTerm(
                    coeff=2.0,
                    monomial=Monomial(
                        creators=(mode.cre,),
                        annihilators=(mode.ann,),
                    ),
                ),
            )
        )
        self.assertEqual(poly, expected)

    def test_from_word(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_word(ops=(mode.ann, mode.cre))

        expected = KetPoly(
            tuple(
                sorted(
                    (
                        KetTerm(coeff=1.0, monomial=Monomial.identity()),
                        KetTerm(
                            coeff=1.0,
                            monomial=Monomial(
                                creators=(mode.cre,),
                                annihilators=(mode.ann,),
                            ),
                        ),
                    ),
                    key=lambda t: t.monomial.signature,
                )
            )
        )
        self.assertEqual(poly, expected)


class TestKetPolyCoreMethods(unittest.TestCase):
    def test_combine_like_terms(self) -> None:
        mode = make_mode()

        poly = KetPoly(
            (
                KetTerm(coeff=2.0, monomial=Monomial(creators=(mode.cre,))),
                KetTerm(coeff=3.0, monomial=Monomial(creators=(mode.cre,))),
            )
        )

        result = poly.combine_like_terms()

        expected = KetPoly(
            (
                KetTerm(coeff=5.0, monomial=Monomial(creators=(mode.cre,))),
            )
        )
        self.assertEqual(result, expected)

    def test_scaled(self) -> None:
        mode = make_mode()

        poly = KetPoly(
            (
                KetTerm(coeff=2.0 + 1.0j, monomial=Monomial(creators=(mode.cre,))),
            )
        )

        result = poly.scaled(3.0 - 2.0j)

        expected = KetPoly(
            (
                KetTerm(
                    coeff=(3.0 - 2.0j) * (2.0 + 1.0j),
                    monomial=Monomial(creators=(mode.cre,)),
                ),
            )
        )
        self.assertEqual(result, expected)

    def test_multiply(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = KetPoly.from_ops(creators=(mode_a.cre,), coeff=2.0)
        right = KetPoly.from_ops(creators=(mode_b.cre,), coeff=3.0)

        result = left.multiply(right)

        expected = KetPoly.from_ops(
            creators=(mode_a.cre, mode_b.cre),
            coeff=6.0,
        )
        self.assertEqual(result, expected)

    def test_apply_word(self) -> None:
        mode = make_mode()
        poly = KetPoly.from_ops(coeff=1.0)

        result = poly.apply_word((mode.cre,))

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=1.0)
        self.assertEqual(result, expected)

    def test_apply_words(self) -> None:
        mode = make_mode()
        poly = KetPoly.from_ops(coeff=1.0)

        result = poly.apply_words(
            (
                (2.0, (mode.cre,)),
                (3.0, (mode.cre,)),
            )
        )

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=5.0)
        self.assertEqual(result, expected)

    def test_inner(self) -> None:
        mode = make_mode()

        a = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        b = KetPoly.from_ops(creators=(mode.cre,), coeff=3.0)

        result = a.inner(b)

        self.assertEqual(result, 6.0 + 0.0j)

    def test_norm2(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)

        self.assertEqual(poly.norm2(), 4.0)

    def test_normalize(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        normalized = poly.normalize()

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=1.0)
        self.assertEqual(normalized, expected)

    def test_normalize_raises_for_near_zero_norm(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=0.0)

        with self.assertRaises(ValueError):
            poly.normalize()

    def test_is_normalized(self) -> None:
        mode = make_mode()

        self.assertTrue(KetPoly.from_ops(creators=(mode.cre,), coeff=1.0).is_normalized())
        self.assertFalse(KetPoly.from_ops(creators=(mode.cre,), coeff=2.0).is_normalized())


class TestKetPolyProperties(unittest.TestCase):
    def test_is_creator_only(self) -> None:
        mode = make_mode()

        self.assertTrue(KetPoly.from_ops(creators=(mode.cre,)).is_creator_only)
        self.assertFalse(KetPoly.from_ops(annihilators=(mode.ann,)).is_creator_only)

    def test_is_annihilator_only(self) -> None:
        mode = make_mode()

        self.assertTrue(KetPoly.from_ops(annihilators=(mode.ann,)).is_annihilator_only)
        self.assertFalse(KetPoly.from_ops(creators=(mode.cre,)).is_annihilator_only)

    def test_is_identity(self) -> None:
        mode = make_mode()

        self.assertTrue(KetPoly.identity().is_identity)
        self.assertFalse(KetPoly.from_ops(creators=(mode.cre,)).is_identity)

    def test_creation_annihilation_and_total_degree(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        poly = KetPoly(
            (
                KetTerm(
                    coeff=1.0,
                    monomial=Monomial(
                        creators=(mode_a.cre, mode_b.cre),
                        annihilators=(mode_a.ann,),
                    ),
                ),
                KetTerm(
                    coeff=1.0,
                    monomial=Monomial(
                        creators=(),
                        annihilators=(mode_b.ann,),
                    ),
                ),
            )
        )

        self.assertEqual(poly.creation_count, 2)
        self.assertEqual(poly.annihilation_count, 2)
        self.assertEqual(poly.total_degree, 4)

    def test_unique_modes_and_mode_count(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        poly = KetPoly(
            (
                KetTerm(
                    coeff=1.0,
                    monomial=Monomial(
                        creators=(mode_a.cre, mode_b.cre, mode_a.cre),
                        annihilators=(mode_b.ann,),
                    ),
                ),
            )
        )

        self.assertEqual(poly.unique_modes, (mode_a, mode_b))
        self.assertEqual(poly.mode_count, 2)

    def test_require_creator_only_passes_for_creator_only(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,))
        poly.require_creator_only()

    def test_require_creator_only_raises_for_non_creator_only(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(annihilators=(mode.ann,))

        with self.assertRaises(ValueError):
            poly.require_creator_only()


class TestKetPolyOperators(unittest.TestCase):
    def test_add(self) -> None:
        mode = make_mode()

        left = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        right = KetPoly.from_ops(creators=(mode.cre,), coeff=3.0)

        result = left + right

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=5.0)
        self.assertEqual(result, expected)

    def test_sub(self) -> None:
        mode = make_mode()

        left = KetPoly.from_ops(creators=(mode.cre,), coeff=5.0)
        right = KetPoly.from_ops(creators=(mode.cre,), coeff=3.0)

        result = left - right

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        self.assertEqual(result, expected)

    def test_neg(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        result = -poly

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=-2.0)
        self.assertEqual(result, expected)

    def test_truediv(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=6.0)
        result = poly / 3.0

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        self.assertEqual(result, expected)

    def test_truediv_raises_for_non_scalar(self) -> None:
        mode = make_mode()
        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=1.0)

        with self.assertRaises(TypeError):
            _ = poly / "x"

    def test_truediv_raises_for_zero(self) -> None:
        mode = make_mode()
        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=1.0)

        with self.assertRaises(ZeroDivisionError):
            _ = poly / 0.0

    def test_eq(self) -> None:
        mode = make_mode()

        left = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        right = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)

        self.assertEqual(left, right)

    def test_eq_dunder_with_other_type_returns_not_implemented(self) -> None:
        poly = KetPoly.identity()

        result = poly.__eq__(object())

        self.assertIs(result, NotImplemented)

    def test_mul_by_scalar(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        result = poly * 3.0

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=6.0)
        self.assertEqual(result, expected)

    def test_mul_by_poly(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = KetPoly.from_ops(creators=(mode_a.cre,), coeff=2.0)
        right = KetPoly.from_ops(creators=(mode_b.cre,), coeff=3.0)

        result = left * right

        expected = KetPoly.from_ops(creators=(mode_a.cre, mode_b.cre), coeff=6.0)
        self.assertEqual(result, expected)

    def test_rmul(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)
        result = 3.0 * poly

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=6.0)
        self.assertEqual(result, expected)

    def test_bool(self) -> None:
        self.assertFalse(KetPoly())
        self.assertTrue(KetPoly.identity())

    def test_repr_contains_class_name(self) -> None:
        text = repr(KetPoly.identity())
        self.assertIn("KetPoly", text)

    def test_rmatmul_with_fake_op_poly(self) -> None:
        mode = make_mode()

        poly = KetPoly.from_ops(coeff=1.0)
        op_poly = make_fake_op_poly(
            make_op_term(ops=(mode.cre,), coeff=2.0),
            make_op_term(ops=(mode.cre,), coeff=3.0),
        )

        result = op_poly @ poly

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=5.0)
        self.assertEqual(result, expected)

    def test_rmatmul_returns_not_implemented_for_other_type(self) -> None:
        poly = KetPoly.identity()

        result = poly.__rmatmul__(object())

        self.assertIs(result, NotImplemented)


class TestKetPolyHermitianBehavior(unittest.TestCase):
    def test_inner_is_hermitian_for_overlapping_modes(self) -> None:
        envelope_table = {}

        mode_a = make_mode(
            path="p",
            polarization="h",
            envelope="env_a",
            envelope_table=envelope_table,
        )
        mode_b = make_mode(
            path="p",
            polarization="h",
            envelope="env_b",
            envelope_table=envelope_table,
        )

        set_hermitian_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            0.2 + 0.7j,
        )

        a = KetPoly.from_ops(creators=(mode_a.cre,), coeff=2.0 - 1.0j)
        b = KetPoly.from_ops(creators=(mode_b.cre,), coeff=3.0 + 4.0j)

        ab = a.inner(b)
        ba = b.inner(a)

        self.assertEqual(ab, ba.conjugate())


if __name__ == "__main__":
    unittest.main()
