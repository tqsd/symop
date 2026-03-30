import unittest

from symop.ccr.algebra.op.combine import combine_like_terms
from symop.core.terms.op_term import OpTerm

from tests.ccr.support.fakes import make_mode


class TestCombineLikeTerms(unittest.TestCase):
    def test_empty_input_returns_empty_tuple(self) -> None:
        result = combine_like_terms((), term_factory=OpTerm)
        self.assertEqual(result, ())

    def test_single_term_is_preserved(self) -> None:
        mode = make_mode()
        term = OpTerm(ops=(mode.cre,), coeff=2.0)

        result = combine_like_terms((term,), term_factory=OpTerm)

        self.assertEqual(result, (term,))

    def test_like_terms_are_combined(self) -> None:
        mode = make_mode()
        terms = (
            OpTerm(ops=(mode.cre,), coeff=2.0),
            OpTerm(ops=(mode.cre,), coeff=3.0 + 1.0j),
        )

        result = combine_like_terms(terms, term_factory=OpTerm)

        expected = (
            OpTerm(ops=(mode.cre,), coeff=5.0 + 1.0j),
        )
        self.assertEqual(result, expected)

    def test_unlike_terms_are_not_combined(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            OpTerm(ops=(mode_a.cre,), coeff=2.0),
            OpTerm(ops=(mode_b.cre,), coeff=3.0),
        )

        result = combine_like_terms(terms, term_factory=OpTerm)

        self.assertEqual(result, terms)

    def test_zero_sum_bucket_is_removed(self) -> None:
        mode = make_mode()

        terms = (
            OpTerm(ops=(mode.cre,), coeff=2.0),
            OpTerm(ops=(mode.cre,), coeff=-2.0),
        )

        result = combine_like_terms(terms, term_factory=OpTerm)

        self.assertEqual(result, ())

    def test_exact_mode_differences_prevent_combining_when_approx_false(self) -> None:
        mode_a = make_mode(path="p", polarization="h", envelope="env_a")
        mode_b = make_mode(path="p", polarization="h", envelope="env_b")

        terms = (
            OpTerm(ops=(mode_a.cre,), coeff=2.0),
            OpTerm(ops=(mode_b.cre,), coeff=3.0),
        )

        result = combine_like_terms(
            terms,
            approx=False,
            term_factory=OpTerm,
        )

        self.assertEqual(result, terms)

    def test_approximate_combining_path_can_merge_terms(self) -> None:
        mode_a = make_mode(path="p", polarization="h", envelope="env")
        mode_b = make_mode(path="p", polarization="h", envelope="env")

        terms = (
            OpTerm(ops=(mode_a.cre,), coeff=2.0),
            OpTerm(ops=(mode_b.cre,), coeff=3.0),
        )

        result = combine_like_terms(
            terms,
            approx=True,
            term_factory=OpTerm,
            decimals=12,
            ignore_global_phase=True,
        )

        expected = (
            OpTerm(ops=(mode_a.cre,), coeff=5.0),
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
