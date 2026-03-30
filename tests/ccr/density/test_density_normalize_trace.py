import unittest

from symop.ccr.algebra.density.normalize_trace import density_normalize_trace
from symop.ccr.algebra.density.trace import density_trace
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode


class TestDensityNormalizeTrace(unittest.TestCase):
    def test_normalize_trace_of_identity_density_is_unchanged(self) -> None:
        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        result = density_normalize_trace(terms)

        self.assertEqual(result, terms)
        self.assertEqual(density_trace(result), 1.0 + 0.0j)

    def test_normalize_trace_rescales_coefficients(self) -> None:
        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        result = density_normalize_trace(terms)

        expected = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)
        self.assertEqual(density_trace(result), 1.0 + 0.0j)

    def test_normalize_trace_with_complex_trace(self) -> None:
        terms = (
            DensityTerm(
                coeff=2.0 + 2.0j,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        result = density_normalize_trace(terms)

        self.assertEqual(density_trace(result), 1.0 + 0.0j)

    def test_normalize_trace_raises_for_zero_trace(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        with self.assertRaises(ValueError):
            density_normalize_trace(terms)

    def test_normalize_trace_raises_for_near_zero_trace(self) -> None:
        terms = (
            DensityTerm(
                coeff=1e-16,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        with self.assertRaises(ValueError):
            density_normalize_trace(terms, eps=1e-14)


if __name__ == "__main__":
    unittest.main()
