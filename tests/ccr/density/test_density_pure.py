import unittest

from symop.ccr.algebra.density.pure import density_pure
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode


class TestDensityPure(unittest.TestCase):
    def test_empty_ket_gives_empty_density(self) -> None:
        self.assertEqual(density_pure(()), ())

    def test_single_identity_ket_gives_rank_one_identity_density(self) -> None:
        ket_terms = (
            KetTerm(coeff=2.0, monomial=Monomial.identity()),
        )

        result = density_pure(ket_terms)

        expected = (
            DensityTerm(
                coeff=4.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_single_nontrivial_ket_term(self) -> None:
        mode = make_mode()

        ket_terms = (
            KetTerm(
                coeff=2.0 - 1.0j,
                monomial=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = density_pure(ket_terms)

        expected = (
            DensityTerm(
                coeff=(2.0 - 1.0j) * (2.0 - 1.0j).conjugate(),
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)

    def test_two_term_ket_expands_to_outer_product_sum(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        ket_terms = (
            KetTerm(
                coeff=2.0,
                monomial=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
            KetTerm(
                coeff=3.0j,
                monomial=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        result = density_pure(ket_terms)

        expected = tuple(
            sorted(
                (
                    DensityTerm(
                        coeff=4.0,
                        left=Monomial(creators=(mode_a.cre,), annihilators=()),
                        right=Monomial(creators=(mode_a.cre,), annihilators=()),
                    ),
                    DensityTerm(
                        coeff=-6.0j,
                        left=Monomial(creators=(mode_a.cre,), annihilators=()),
                        right=Monomial(creators=(mode_b.cre,), annihilators=()),
                    ),
                    DensityTerm(
                        coeff=6.0j,
                        left=Monomial(creators=(mode_b.cre,), annihilators=()),
                        right=Monomial(creators=(mode_a.cre,), annihilators=()),
                    ),
                    DensityTerm(
                        coeff=9.0,
                        left=Monomial(creators=(mode_b.cre,), annihilators=()),
                        right=Monomial(creators=(mode_b.cre,), annihilators=()),
                    ),
                ),
                key=lambda t: t.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_like_density_terms_are_combined(self) -> None:
        mode = make_mode()

        ket_terms = (
            KetTerm(
                coeff=2.0,
                monomial=Monomial(creators=(mode.cre,), annihilators=()),
            ),
            KetTerm(
                coeff=3.0,
                monomial=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = density_pure(ket_terms)

        expected = (
            DensityTerm(
                coeff=25.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
