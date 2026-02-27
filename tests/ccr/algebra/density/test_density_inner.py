from __future__ import annotations

import unittest

from symop.ccr.algebra.density.inner import density_inner
from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm


class TestDensityInner(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_density_inner_identity_reduces_to_coeff_inner_product(
        self,
    ) -> None:
        # With identity monomials on both sides, overlaps are 1:
        # <A,B> = sum_i sum_j conj(ci) * cj = conj(sum_i ci) * (sum_j cj).
        I = self._identity_monomial()

        a = (
            DensityTerm(coeff=1.0 + 2.0j, left=I, right=I),
            DensityTerm(coeff=-3.0 + 0.5j, left=I, right=I),
        )
        b = (
            DensityTerm(coeff=2.0 - 1.0j, left=I, right=I),
            DensityTerm(coeff=0.25 + 0.0j, left=I, right=I),
            DensityTerm(coeff=-1.0 + 4.0j, left=I, right=I),
        )

        sum_a = (1.0 + 2.0j) + (-3.0 + 0.5j)
        sum_b = (2.0 - 1.0j) + (0.25 + 0.0j) + (-1.0 + 4.0j)
        expected = sum_a.conjugate() * sum_b

        out = density_inner(a, b)
        self.assertAlmostEqual(out.real, expected.real)
        self.assertAlmostEqual(out.imag, expected.imag)

    def test_density_inner_is_hermitian(self) -> None:
        I = self._identity_monomial()

        a = (
            DensityTerm(coeff=1.25 - 0.5j, left=I, right=I),
            DensityTerm(coeff=-2.0 + 3.0j, left=I, right=I),
        )
        b = (DensityTerm(coeff=0.75 + 1.5j, left=I, right=I),)

        ab = density_inner(a, b)
        ba = density_inner(b, a)

        self.assertAlmostEqual(ab.real, ba.conjugate().real)
        self.assertAlmostEqual(ab.imag, ba.conjugate().imag)

    def test_density_inner_is_linear_in_second_argument(self) -> None:
        I = self._identity_monomial()

        a = (
            DensityTerm(coeff=1.0 + 0.0j, left=I, right=I),
            DensityTerm(coeff=0.5 - 1.0j, left=I, right=I),
        )
        b1 = (DensityTerm(coeff=2.0 + 3.0j, left=I, right=I),)
        b2 = (
            DensityTerm(coeff=-1.0 + 0.25j, left=I, right=I),
            DensityTerm(coeff=0.0 + 2.0j, left=I, right=I),
        )

        out = density_inner(a, (*b1, *b2))
        expected = density_inner(a, b1) + density_inner(a, b2)

        self.assertAlmostEqual(out.real, expected.real)
        self.assertAlmostEqual(out.imag, expected.imag)
