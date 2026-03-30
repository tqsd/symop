import unittest

from symop.ccr.algebra.op.from_words import from_words
from symop.core.terms.op_term import OpTerm

from tests.ccr.support.fakes import make_mode


class TestFromWords(unittest.TestCase):
    def test_empty_words_return_empty_tuple(self) -> None:
        result = from_words((), term_factory=OpTerm)
        self.assertEqual(result, ())

    def test_words_without_coeffs_get_unit_coefficients(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = from_words(
            (
                (mode_a.cre,),
                (mode_b.ann, mode_a.cre),
            ),
            term_factory=OpTerm,
        )

        expected = (
            OpTerm(ops=(mode_a.cre,), coeff=1.0),
            OpTerm(ops=(mode_b.ann, mode_a.cre), coeff=1.0),
        )
        self.assertEqual(result, expected)

    def test_words_with_coeffs_use_given_coefficients(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = from_words(
            (
                (mode_a.cre,),
                (mode_b.ann,),
            ),
            coeffs=(2.0, 3.0 + 1.0j),
            term_factory=OpTerm,
        )

        expected = (
            OpTerm(ops=(mode_a.cre,), coeff=2.0),
            OpTerm(ops=(mode_b.ann,), coeff=3.0 + 1.0j),
        )
        self.assertEqual(result, expected)

    def test_general_iterables_are_materialized(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        words = (
            (op for op in (mode_a.cre, mode_b.ann)),
        )

        result = from_words(words, term_factory=OpTerm)

        expected = (
            OpTerm(ops=(mode_a.cre, mode_b.ann), coeff=1.0),
        )
        self.assertEqual(result, expected)

    def test_mismatched_coeff_length_raises_value_error(self) -> None:
        mode = make_mode()

        with self.assertRaises(ValueError):
            from_words(
                (
                    (mode.cre,),
                    (mode.ann,),
                ),
                coeffs=(1.0,),
                term_factory=OpTerm,
            )


if __name__ == "__main__":
    unittest.main()
