import unittest

from symop.core.terms.op_term import OpTerm

from tests.core.support.fakes import make_mode


class TestOpTerm(unittest.TestCase):
    def test_identity_uses_empty_word_and_given_coefficient(self) -> None:
        term = OpTerm.identity(2.0 + 3.0j)

        self.assertEqual(term.ops, ())
        self.assertEqual(term.coeff, 2.0 + 3.0j)
        self.assertEqual(len(term), 0)

    def test_scaled_multiplies_coefficient_only(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre, mode.ann), coeff=2.0 + 1.0j)

        scaled = term.scaled(3.0 - 2.0j)

        self.assertEqual(scaled.ops, term.ops)
        self.assertEqual(scaled.coeff, (2.0 + 1.0j) * (3.0 - 2.0j))
        self.assertEqual(term.coeff, 2.0 + 1.0j)

    def test_adjoint_reverses_order_takes_daggers_and_conjugates_coeff(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = OpTerm(
            ops=(mode_a.cre, mode_b.ann, mode_a.ann),
            coeff=2.0 + 3.0j,
        )

        adj = term.adjoint()

        self.assertEqual(
            adj.ops,
            (mode_a.cre, mode_b.cre, mode_a.ann),
        )
        self.assertEqual(adj.coeff, 2.0 - 3.0j)

    def test_double_adjoint_returns_original_term(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = OpTerm(
            ops=(mode_a.cre, mode_b.ann, mode_a.ann),
            coeff=1.0 + 2.0j,
        )

        self.assertEqual(term.adjoint().adjoint(), term)

    def test_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = OpTerm(
            ops=(mode_a.cre, mode_b.ann),
            coeff=7.0,
        )

        self.assertEqual(
            term.signature,
            ("op_term", (mode_a.cre.signature, mode_b.ann.signature)),
        )

    def test_approx_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = OpTerm(
            ops=(mode_a.cre, mode_b.ann),
            coeff=7.0,
        )

        self.assertEqual(
            term.approx_signature(decimals=7, ignore_global_phase=True),
            (
                "op_term_approx",
                (
                    mode_a.cre.approx_signature(
                        decimals=7,
                        ignore_global_phase=True,
                    ),
                    mode_b.ann.approx_signature(
                        decimals=7,
                        ignore_global_phase=True,
                    ),
                ),
            ),
        )

    def test_len_returns_number_of_operators(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        self.assertEqual(
            len(OpTerm(ops=(mode_a.cre, mode_b.ann, mode_a.ann))),
            3,
        )

    def test_iter_returns_operators_in_word_order(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = OpTerm(ops=(mode_a.cre, mode_b.ann, mode_a.ann))

        self.assertEqual(list(term), [mode_a.cre, mode_b.ann, mode_a.ann])

    def test_neg_returns_additive_inverse(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=2.0 + 3.0j)

        neg = -term

        self.assertEqual(neg.ops, term.ops)
        self.assertEqual(neg.coeff, -(2.0 + 3.0j))

    def test_mul_by_scalar(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=2.0 + 1.0j)

        result = term * (3.0 - 2.0j)

        self.assertEqual(result.ops, term.ops)
        self.assertEqual(result.coeff, (2.0 + 1.0j) * (3.0 - 2.0j))

    def test_rmul_by_scalar(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=2.0 + 1.0j)

        result = (3.0 - 2.0j) * term

        self.assertEqual(result.ops, term.ops)
        self.assertEqual(result.coeff, (3.0 - 2.0j) * (2.0 + 1.0j))

    def test_mul_with_non_scalar_returns_not_implemented(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=1.0)

        result = term.__mul__("x")

        self.assertIs(result, NotImplemented)

    def test_rmul_with_non_scalar_returns_not_implemented(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=1.0)

        result = term.__rmul__("x")

        self.assertIs(result, NotImplemented)

    def test_truediv_by_scalar(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=6.0 + 3.0j)

        result = term / 3.0

        self.assertEqual(result.ops, term.ops)
        self.assertEqual(result.coeff, (6.0 + 3.0j) / 3.0)

    def test_truediv_by_non_scalar_raises_type_error(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=1.0)

        with self.assertRaises(TypeError):
            _ = term / "x"

    def test_truediv_by_zero_raises_zero_division_error(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=1.0)

        with self.assertRaises(ZeroDivisionError):
            _ = term / 0.0

    def test_eq_true_for_same_ops_and_coeff(self) -> None:
        mode = make_mode()

        left = OpTerm(ops=(mode.cre, mode.ann), coeff=2.0)
        right = OpTerm(ops=(mode.cre, mode.ann), coeff=2.0)

        self.assertEqual(left, right)

    def test_eq_false_if_ops_differ(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = OpTerm(ops=(mode_a.cre,), coeff=2.0)
        right = OpTerm(ops=(mode_b.cre,), coeff=2.0)

        self.assertNotEqual(left, right)

    def test_eq_false_if_coeff_differs(self) -> None:
        mode = make_mode()

        left = OpTerm(ops=(mode.cre,), coeff=2.0)
        right = OpTerm(ops=(mode.cre,), coeff=3.0)

        self.assertNotEqual(left, right)

    def test_eq_with_other_type_returns_not_implemented_from_dunder(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=1.0)

        result = term.__eq__(object())

        self.assertIs(result, NotImplemented)

    def test_bool_false_only_for_exact_zero_coefficient(self) -> None:
        mode = make_mode()

        self.assertFalse(OpTerm(ops=(mode.cre,), coeff=0.0))
        self.assertTrue(OpTerm(ops=(mode.cre,), coeff=1.0))
        self.assertTrue(OpTerm(ops=(mode.cre,), coeff=1.0j))

    def test_repr_contains_class_name_ops_and_coeff(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=2.0 + 1.0j)

        text = repr(term)

        self.assertIn("OpTerm", text)
        self.assertIn("ops=", text)
        self.assertIn("coeff=", text)


if __name__ == "__main__":
    unittest.main()
