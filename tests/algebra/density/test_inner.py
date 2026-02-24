from __future__ import annotations
import unittest
from typing import Tuple

from tests.utils.case import ExtendedTestCase

from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm
from symop_proto.algebra.density.inner import density_inner


class TestDensityInner(ExtendedTestCase):
    def test_empty_against_anything_is_zero(self):
        rho_empty: Tuple[DensityTerm, ...] = ()
        rho_one = (DensityTerm(1.0, left=Monomial(), right=Monomial()),)
        self.assertComplexAlmostEqual(density_inner(rho_empty, rho_empty), 0.0 + 0.0j)
        self.assertComplexAlmostEqual(density_inner(rho_empty, rho_one), 0.0 + 0.0j)
        self.assertComplexAlmostEqual(density_inner(rho_one, rho_empty), 0.0 + 0.0j)

    def test_identity_with_identity_is_one(self):
        rho = (DensityTerm(1.0, left=Monomial(), right=Monomial()),)
        val = density_inner(rho, rho)
        self.assertComplexAlmostEqual(val, 1.0 + 0.0j)

    def test_scalars_and_conjugation(self):
        a = 2.0 + 3.0j
        b = 4.0 - 5.0j
        A = (DensityTerm(a, left=Monomial(), right=Monomial()),)
        B = (DensityTerm(b, left=Monomial(), right=Monomial()),)
        val = density_inner(A, B)
        self.assertComplexAlmostEqual(val, a.conjugate() * b)

    def test_linearity_over_sum(self):
        a1 = 1.0 + 0.5j
        a2 = -0.25 + 2.0j
        b = 0.75 - 1.25j
        A = (
            DensityTerm(a1, left=Monomial(), right=Monomial()),
            DensityTerm(a2, left=Monomial(), right=Monomial()),
        )
        B = (DensityTerm(b, left=Monomial(), right=Monomial()),)
        expected = a1.conjugate() * b + a2.conjugate() * b
        val = density_inner(A, B)
        self.assertComplexAlmostEqual(val, expected)

    def test_conjugate_symmetry(self):
        A = (
            DensityTerm(1.0 + 2.0j, left=Monomial(), right=Monomial()),
            DensityTerm(-0.5j, left=Monomial(), right=Monomial()),
        )
        B = (DensityTerm(0.5 - 0.25j, left=Monomial(), right=Monomial()),)
        ab = density_inner(A, B)
        ba = density_inner(B, A)
        self.assertComplexAlmostEqual(ab, ba.conjugate())


if __name__ == "__main__":
    unittest.main()
