from __future__ import annotations

import unittest

from symop.ccr.algebra.density.scale import density_scale
from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm


class TestDensityScale(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_density_scale_multiplies_coefficients(self) -> None:
        I = self._identity_monomial()

        terms = (
            DensityTerm(coeff=1.0 + 2.0j, left=I, right=I),
            DensityTerm(coeff=-3.0 + 0.5j, left=I, right=I),
        )

        c = 2.0 - 1.0j
        out = density_scale(terms, c)

        self.assertEqual(len(out), 2)
        for t_in, t_out in zip(terms, out, strict=False):
            expected = c * t_in.coeff
            self.assertAlmostEqual(t_out.coeff.real, expected.real)
            self.assertAlmostEqual(t_out.coeff.imag, expected.imag)
            self.assertEqual(t_out.left.signature, t_in.left.signature)
            self.assertEqual(t_out.right.signature, t_in.right.signature)

    def test_density_scale_by_zero(self) -> None:
        I = self._identity_monomial()

        terms = (DensityTerm(coeff=5.0 + 0.0j, left=I, right=I),)

        out = density_scale(terms, 0.0 + 0.0j)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].coeff, 0.0 + 0.0j)

    def test_density_scale_empty_input(self) -> None:
        out = density_scale((), 3.0 + 0.0j)
        self.assertEqual(out, ())
