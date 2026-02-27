from __future__ import annotations

import unittest

from symop.ccr.algebra.density.trace import density_trace
from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm


class TestDensityTrace(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_trace_identity_term_equals_coefficient(self) -> None:
        I = self._identity_monomial()

        rho = (DensityTerm(coeff=3.0 + 2.0j, left=I, right=I),)

        tr = density_trace(rho)

        # Since <I|I> = 1
        self.assertAlmostEqual(tr.real, 3.0)
        self.assertAlmostEqual(tr.imag, 2.0)

    def test_trace_multiple_identity_terms(self) -> None:
        I = self._identity_monomial()

        rho = (
            DensityTerm(coeff=1.0 + 0.0j, left=I, right=I),
            DensityTerm(coeff=2.5 + 1.0j, left=I, right=I),
        )

        tr = density_trace(rho)

        expected = (1.0 + 0.0j) + (2.5 + 1.0j)
        self.assertAlmostEqual(tr.real, expected.real)
        self.assertAlmostEqual(tr.imag, expected.imag)

    def test_trace_empty_density_is_zero(self) -> None:
        tr = density_trace(())
        self.assertEqual(tr, 0.0 + 0.0j)
