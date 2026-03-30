import unittest

from symop.ccr.algebra.density.combine import combine_like_terms_density
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode


class TestCombineLikeTermsDensity(unittest.TestCase):
    def test_empty_input_returns_empty_tuple(self) -> None:
        result = combine_like_terms_density(())
        self.assertEqual(result, ())

    def test_single_term_is_returned_unchanged(self) -> None:
        mode = make_mode()
        term = DensityTerm(
            coeff=2.0 + 1.0j,
            left=Monomial(creators=(mode.cre,), annihilators=()),
            right=Monomial.identity(),
        )

        result = combine_like_terms_density((term,))

        self.assertEqual(result, (term,))

    def test_like_terms_are_combined(self) -> None:
        mode = make_mode()
        left = Monomial(creators=(mode.cre,), annihilators=())
        right = Monomial(annihilators=(mode.ann,))

        terms = (
            DensityTerm(coeff=2.0, left=left, right=right),
            DensityTerm(coeff=3.0 + 1.0j, left=left, right=right),
        )

        result = combine_like_terms_density(terms)

        expected = (
            DensityTerm(coeff=5.0 + 1.0j, left=left, right=right),
        )
        self.assertEqual(result, expected)

    def test_distinct_left_monomials_are_not_combined(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        right = Monomial.identity()
        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=right,
            ),
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=right,
            ),
        )

        result = combine_like_terms_density(terms)

        expected = tuple(
            sorted(terms, key=lambda t: (t.left.signature, t.right.signature))
        )
        self.assertEqual(result, expected)

    def test_distinct_right_monomials_are_not_combined(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = Monomial.identity()
        terms = (
            DensityTerm(
                coeff=2.0,
                left=left,
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
            DensityTerm(
                coeff=3.0,
                left=left,
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        result = combine_like_terms_density(terms)

        expected = tuple(
            sorted(terms, key=lambda t: (t.left.signature, t.right.signature))
        )
        self.assertEqual(result, expected)

    def test_near_zero_combined_term_is_dropped(self) -> None:
        mode = make_mode()
        left = Monomial(creators=(mode.cre,), annihilators=())
        right = Monomial.identity()

        terms = (
            DensityTerm(coeff=1.0, left=left, right=right),
            DensityTerm(coeff=-1.0 + 1e-14j, left=left, right=right),
        )

        result = combine_like_terms_density(terms, eps=1e-12)

        self.assertEqual(result, ())

    def test_term_with_abs_equal_to_eps_is_kept(self) -> None:
        mode = make_mode()
        left = Monomial(creators=(mode.cre,), annihilators=())
        right = Monomial.identity()

        result = combine_like_terms_density(
            (DensityTerm(coeff=1e-12, left=left, right=right),),
            eps=1e-12,
        )

        expected = (
            DensityTerm(coeff=1e-12, left=left, right=right),
        )
        self.assertEqual(result, expected)

    def test_output_is_sorted_by_left_then_right_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = combine_like_terms_density(terms)

        expected = tuple(
            sorted(terms, key=lambda t: (t.left.signature, t.right.signature))
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
