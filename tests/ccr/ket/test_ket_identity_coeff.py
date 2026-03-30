import unittest

from symop.ccr.algebra.ket.identity_coeff import identity_coeff
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode


class TestIdentityCoeff(unittest.TestCase):
    def test_empty_terms_return_zero(self) -> None:
        result = identity_coeff(())
        self.assertEqual(result, 0.0 + 0.0j)

    def test_returns_zero_when_no_identity_term_is_present(self) -> None:
        mode = make_mode()

        terms = (
            KetTerm(coeff=2.0, monomial=Monomial(creators=(mode.cre,))),
            KetTerm(coeff=3.0, monomial=Monomial(annihilators=(mode.ann,))),
        )

        result = identity_coeff(terms)

        self.assertEqual(result, 0.0 + 0.0j)

    def test_returns_identity_coefficient_when_present(self) -> None:
        mode = make_mode()

        terms = (
            KetTerm(coeff=2.0, monomial=Monomial(creators=(mode.cre,))),
            KetTerm(coeff=3.0 + 1.0j, monomial=Monomial.identity()),
            KetTerm(coeff=4.0, monomial=Monomial(annihilators=(mode.ann,))),
        )

        result = identity_coeff(terms)

        self.assertEqual(result, 3.0 + 1.0j)

    def test_returns_first_identity_term_if_multiple_are_present(self) -> None:
        terms = (
            KetTerm(coeff=2.0, monomial=Monomial.identity()),
            KetTerm(coeff=5.0, monomial=Monomial.identity()),
        )

        result = identity_coeff(terms)

        self.assertEqual(result, 2.0)


if __name__ == "__main__":
    unittest.main()
