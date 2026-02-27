import unittest

from symop.ccr.algebra.ket.combine import combine_like_terms_ket
from symop.core.monomial import Monomial
from symop.core.terms import KetTerm


class TestCombineLikeTermsKet(unittest.TestCase):
    def test_combines_equal_monomials_and_sums_coeffs(self):
        m = Monomial.identity()
        terms = (
            KetTerm(1.0 + 0.0j, m),
            KetTerm(2.0 - 1.0j, m),
        )
        out = combine_like_terms_ket(terms)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].coeff, 3.0 - 1.0j)
        self.assertEqual(out[0].monomial.signature, m.signature)

    def test_drops_terms_below_eps(self):
        m = Monomial.identity()
        terms = (KetTerm(1.0e-13 + 0.0j, m),)
        out = combine_like_terms_ket(terms, eps=1.0e-12)
        self.assertEqual(out, ())

    def test_keeps_distinct_monomials(self):
        # Two different monomials must not be merged.
        m1 = Monomial.identity()

        # Create a distinct monomial by using dummy objects in creators.
        # We can use object() because Monomial only uses op.signature in signature.
        class _Op:
            def __init__(self, sig):
                self._sig = sig
                self.mode = self  # not used here

            @property
            def signature(self):
                return self._sig

            def approx_signature(self, *, decimals=12, ignore_global_phase=False):
                return ("approx", self._sig, decimals, ignore_global_phase)

        op = _Op(("lop", "adag", "x"))
        m2 = Monomial(creators=(op,), annihilators=())

        terms = (KetTerm(1.0 + 0.0j, m1), KetTerm(1.0 + 0.0j, m2))
        out = combine_like_terms_ket(terms)
        self.assertEqual(len(out), 2)
        self.assertNotEqual(out[0].monomial.signature, out[1].monomial.signature)


if __name__ == "__main__":
    unittest.main()
