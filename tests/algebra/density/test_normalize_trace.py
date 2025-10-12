from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm
from symop_proto.algebra.density.trace import density_trace
from symop_proto.algebra.density.normalize_trace import density_normalize_trace


class TestDensityNormalizeTrace(ExtendedTestCase):
    def test_already_normalized_identity(self):
        # rho = 1 * |I><I|  -> trace = 1
        rho = (DensityTerm(1.0, left=Monomial(), right=Monomial()),)
        out = density_normalize_trace(rho)
        # unchanged
        self.assertEqual(out, rho)
        # trace == 1
        tr = density_trace(out)
        self.assertComplexAlmostEqual(tr, 1.0 + 0.0j)

    def test_scales_real_trace(self):
        # rho = 2 * |I><I|  -> trace = 2  -> scale by 1/2
        rho = (DensityTerm(2.0, left=Monomial(), right=Monomial()),)
        out = density_normalize_trace(rho)
        tr = density_trace(out)
        self.assertComplexAlmostEqual(tr, 1.0 + 0.0j)
        # coefficient scaled by 1/2
        self.assertComplexAlmostEqual(out[0].coeff, 1.0 + 0.0j)

    def test_scales_complex_trace(self):
        # rho = (1+2j) * |I><I|  -> trace = 1+2j  -> scale by 1/(1+2j)
        c = 1.0 + 2.0j
        rho = (DensityTerm(c, left=Monomial(), right=Monomial()),)
        out = density_normalize_trace(rho)
        tr = density_trace(out)
        self.assertComplexAlmostEqual(tr, 1.0 + 0.0j)
        # coefficient equals c*(1/c) = 1
        self.assertComplexAlmostEqual(out[0].coeff, 1.0 + 0.0j)

    def test_linearity_multiple_terms(self):
        # rho = 3*|I><I| + 1*|I><I| -> trace = 4 -> scale by 1/4
        rho = (
            DensityTerm(3.0, left=Monomial(), right=Monomial()),
            DensityTerm(1.0, left=Monomial(), right=Monomial()),
        )
        out = density_normalize_trace(rho)
        tr = density_trace(out)
        self.assertComplexAlmostEqual(tr, 1.0 + 0.0j)
        # coefficients scaled by 1/4
        self.assertComplexAlmostEqual(out[0].coeff + out[1].coeff, 1.0 + 0.0j)
        # individual factors are correct
        self.assertComplexAlmostEqual(out[0].coeff, 0.75 + 0.0j)
        self.assertComplexAlmostEqual(out[1].coeff, 0.25 + 0.0j)

    def test_raises_on_too_small_trace(self):
        # Near-zero trace -> raises
        tiny = 1e-15
        rho = (DensityTerm(tiny, left=Monomial(), right=Monomial()),)
        with self.assertRaises(ValueError):
            density_normalize_trace(rho, eps=1e-14)


if __name__ == "__main__":
    unittest.main()
