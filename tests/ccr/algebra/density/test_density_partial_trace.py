from __future__ import annotations

import unittest

from symop.ccr.algebra.density.partial_trace import density_partial_trace
from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm


class TestDensityPartialTrace(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_partial_trace_empty_trace_set_is_identity(self) -> None:
        I = self._identity_monomial()
        rho = (DensityTerm(coeff=2.0 + 0.5j, left=I, right=I),)

        out = density_partial_trace(rho, ())
        self.assertEqual(out, rho)

    def test_partial_trace_over_unrelated_signature_keeps_identity_term(
        self,
    ) -> None:
        I = self._identity_monomial()
        rho = (DensityTerm(coeff=-3.0 + 1.0j, left=I, right=I),)

        out = density_partial_trace(rho, (("some_mode_signature", 123),))
        self.assertEqual(out, rho)

    def test_partial_trace_combines_identical_kept_terms(self) -> None:
        I = self._identity_monomial()

        # Two identical terms should remain identical after tracing and then combine.
        rho = (
            DensityTerm(coeff=1.0 + 0.0j, left=I, right=I),
            DensityTerm(coeff=2.0 + 0.0j, left=I, right=I),
        )

        out = density_partial_trace(rho, (("any",),))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].coeff.real, 3.0)
        self.assertAlmostEqual(out[0].coeff.imag, 0.0)
        self.assertEqual(out[0].left.signature, I.signature)
        self.assertEqual(out[0].right.signature, I.signature)
