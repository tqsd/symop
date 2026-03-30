import unittest

from symop.ccr.algebra.density.scale import density_scale
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode


class TestDensityScale(unittest.TestCase):
    def test_empty_density_scales_to_empty(self) -> None:
        self.assertEqual(density_scale((), 2.0), ())

    def test_single_term_is_scaled(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0 + 1.0j,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = density_scale(terms, 3.0 - 2.0j)

        expected = (
            DensityTerm(
                coeff=(3.0 - 2.0j) * (2.0 + 1.0j),
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_multiple_terms_preserve_left_and_right_monomials(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
            DensityTerm(
                coeff=3.0j,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        result = density_scale(terms, -2.0)

        expected = (
            DensityTerm(
                coeff=-4.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
            DensityTerm(
                coeff=-6.0j,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)

    def test_scaling_by_zero_zeroes_coefficients(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = density_scale(terms, 0.0)

        expected = (
            DensityTerm(
                coeff=0.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
