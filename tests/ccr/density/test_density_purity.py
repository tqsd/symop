import unittest

from symop.ccr.algebra.density.pure import density_pure
from symop.ccr.algebra.density.purity import density_purity
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode


class TestDensityPurity(unittest.TestCase):
    def test_empty_density_has_zero_purity(self) -> None:
        self.assertEqual(density_purity(()), 0.0)

    def test_identity_rank_one_density_has_purity_one(self) -> None:
        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        self.assertEqual(density_purity(terms), 1.0)

    def test_pure_single_particle_state_has_purity_one(self) -> None:
        mode = make_mode()

        ket_terms = (
            KetTerm(
                coeff=1.0,
                monomial=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        rho = density_pure(ket_terms)

        self.assertEqual(density_purity(rho), 1.0)

    def test_scaled_pure_state_has_scaled_purity(self) -> None:
        mode = make_mode()

        ket_terms = (
            KetTerm(
                coeff=2.0,
                monomial=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )
        rho = density_pure(ket_terms)

        self.assertEqual(density_purity(rho), 16.0)

    def test_simple_mixed_like_density_has_expected_purity(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        rho = (
            DensityTerm(
                coeff=0.5,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
            DensityTerm(
                coeff=0.5,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        self.assertEqual(density_purity(rho), 0.5)


if __name__ == "__main__":
    unittest.main()
