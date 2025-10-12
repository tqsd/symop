from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm
from symop_proto.algebra.density.scale import density_scale


class TestDensityScale(ExtendedTestCase):
    def test_single_term_scaling(self):
        m = Monomial()
        rho = (DensityTerm(2.0, left=m, right=m),)
        out = density_scale(rho, 3.0)
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 6.0 + 0.0j)

    def test_multiple_terms_scaling(self):
        m = Monomial()
        rho = (
            DensityTerm(1.0, left=m, right=m),
            DensityTerm(2.0, left=m, right=m),
        )
        out = density_scale(rho, -1j)
        coeffs = [t.coeff for t in out]
        self.assertIn(-1j, coeffs)
        self.assertIn(-2j, coeffs)

    def test_empty_input_returns_empty(self):
        out = density_scale((), 5.0)
        self.assertEqual(out, ())

    def test_does_not_modify_original(self):
        m = Monomial()
        rho = (DensityTerm(1.0, left=m, right=m),)
        out = density_scale(rho, 10.0)
        self.assertComplexAlmostEqual(rho[0].coeff, 1.0)
        self.assertComplexAlmostEqual(out[0].coeff, 10.0)


if __name__ == "__main__":
    unittest.main()
