import unittest

from symop.ccr.algebra.ket.scale import ket_scale
from symop.core.monomial import Monomial
from symop.core.terms import KetTerm


class TestKetScale(unittest.TestCase):
    def test_scaling_identity(self):
        m = Monomial.identity()
        terms = (KetTerm(2.0 + 1.0j, m),)

        out = ket_scale(terms, 3.0)

        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].monomial.is_identity)
        self.assertEqual(out[0].coeff, (3.0 * (2.0 + 1.0j)))

    def test_scaling_multiple_terms(self):
        m1 = Monomial.identity()
        m2 = Monomial(creators=(object(),), annihilators=())

        terms = (
            KetTerm(1.0 + 0.0j, m1),
            KetTerm(-2.0 + 3.0j, m2),
        )

        out = ket_scale(terms, -1.0)

        self.assertEqual(out[0].coeff, -1.0 + 0.0j)
        self.assertEqual(out[1].coeff, 2.0 - 3.0j)

    def test_scaling_by_zero(self):
        m = Monomial.identity()
        terms = (KetTerm(5.0 + 2.0j, m),)

        out = ket_scale(terms, 0.0)

        self.assertEqual(out[0].coeff, 0.0 + 0.0j)

    def test_empty_input(self):
        out = ket_scale((), 2.0)
        self.assertEqual(out, ())


if __name__ == "__main__":
    unittest.main()
