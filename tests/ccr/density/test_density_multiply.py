import unittest

from symop.ccr.algebra.density.multiply import density_multiply
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestDensityMultiply(unittest.TestCase):
    def test_empty_times_empty_is_empty(self) -> None:
        self.assertEqual(density_multiply((), ()), ())

    def test_empty_left_is_empty(self) -> None:
        mode = make_mode()
        right = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(density_multiply((), right), ())

    def test_empty_right_is_empty(self) -> None:
        mode = make_mode()
        left = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        self.assertEqual(density_multiply(left, ()), ())

    def test_identity_times_identity_is_identity(self) -> None:
        ident = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        self.assertEqual(density_multiply(ident, ident), ident)

    def test_product_uses_inner_contraction_of_middle_monomials(self) -> None:
        mode = make_mode()

        left_terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        right_terms = (
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = density_multiply(left_terms, right_terms)

        expected = (
            DensityTerm(
                coeff=6.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_orthogonal_middle_overlap_gives_zero(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left_terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
        )
        right_terms = (
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        self.assertEqual(density_multiply(left_terms, right_terms), ())

    def test_overlapping_middle_overlap_is_used(self) -> None:
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

        overlap = 0.25 + 0.5j
        set_hermitian_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            overlap,
        )

        left_terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
        )
        right_terms = (
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = density_multiply(left_terms, right_terms)

        expected = (
            DensityTerm(
                coeff=6.0 * overlap,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_multiple_products_are_combined(self) -> None:
        mode = make_mode()

        left_terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial(identity := (), annihilators=())  # replaced below
            ),
        )
        left_terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
            DensityTerm(
                coeff=3.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        right_terms = (
            DensityTerm(
                coeff=5.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = density_multiply(left_terms, right_terms)

        expected = (
            DensityTerm(
                coeff=25.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_small_overlap_is_dropped(self) -> None:
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
            1e-15 + 0.0j,
        )

        left_terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
        )
        right_terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        self.assertEqual(density_multiply(left_terms, right_terms, eps=1e-12), ())


if __name__ == "__main__":
    unittest.main()
