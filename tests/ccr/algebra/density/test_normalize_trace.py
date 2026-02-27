from __future__ import annotations

import unittest

from symop.ccr.algebra.density.normalize_trace import density_normalize_trace
from symop.ccr.algebra.density.trace import density_trace
from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm


class TestDensityNormalizeTrace(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_normalize_trace_makes_unit_trace(self) -> None:
        # For identity monomials, Tr(c |I><I|) = c * <I|I> and <I|I> should be 1.
        I = self._identity_monomial()

        rho = (DensityTerm(coeff=2.0 + 0.0j, left=I, right=I),)
        rho_n = density_normalize_trace(rho)

        tr = density_trace(rho_n)
        self.assertAlmostEqual(tr.real, 1.0)
        self.assertAlmostEqual(tr.imag, 0.0)

    def test_normalize_trace_scales_by_inverse_trace(self) -> None:
        I = self._identity_monomial()

        # Trace should be 4+0j for this input if <I|I>=1.
        rho = (DensityTerm(coeff=4.0 + 0.0j, left=I, right=I),)
        rho_n = density_normalize_trace(rho)

        self.assertEqual(len(rho_n), 1)
        self.assertAlmostEqual(rho_n[0].coeff.real, 1.0)
        self.assertAlmostEqual(rho_n[0].coeff.imag, 0.0)

    def test_normalize_trace_raises_on_near_zero_trace(self) -> None:
        I = self._identity_monomial()

        rho = (DensityTerm(coeff=0.0 + 0.0j, left=I, right=I),)

        with self.assertRaises(ValueError):
            density_normalize_trace(rho, eps=1e-14)
