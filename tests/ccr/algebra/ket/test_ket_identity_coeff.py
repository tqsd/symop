import unittest

from symop.ccr.algebra.ket.identity_coeff import identity_coeff
from symop.core.monomial import Monomial
from symop.core.terms import KetTerm


class TestIdentityCoeff(unittest.TestCase):
    def test_returns_zero_if_no_identity(self):
        m = Monomial(creators=(), annihilators=(object(),))
        terms = (KetTerm(1.0 + 0.0j, m),)
        self.assertEqual(identity_coeff(terms), 0.0 + 0.0j)

    def test_extracts_identity_coefficient(self):
        m_id = Monomial.identity()
        m_other = Monomial(creators=(object(),), annihilators=())

        terms = (
            KetTerm(2.0 + 1.0j, m_other),
            KetTerm(3.0 - 2.0j, m_id),
        )

        self.assertEqual(identity_coeff(terms), 3.0 - 2.0j)

    def test_multiple_identity_terms_returns_first(self):
        m_id = Monomial.identity()

        terms = (
            KetTerm(1.0 + 0.0j, m_id),
            KetTerm(5.0 + 0.0j, m_id),
        )

        # combine_like_terms should normally remove duplicates,
        # but identity_coeff itself just returns first.
        self.assertEqual(identity_coeff(terms), 1.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
