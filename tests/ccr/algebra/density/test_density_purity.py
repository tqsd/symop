from __future__ import annotations

import unittest

from symop.ccr.algebra.density.normalize_trace import density_normalize_trace
from symop.ccr.algebra.density.pure import density_pure
from symop.ccr.algebra.density.purity import density_purity
from symop.core.monomial import Monomial
from symop.core.terms import KetTerm


class TestDensityPurity(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_purity_of_single_pure_state_is_one_after_normalization(
        self,
    ) -> None:
        I = self._identity_monomial()

        ket = (KetTerm(coeff=2.0 + 0.0j, monomial=I),)
        rho = density_pure(ket)
        rho = density_normalize_trace(rho)

        purity = density_purity(rho)
        self.assertAlmostEqual(purity, 1.0)

    def test_purity_of_convex_combo_of_same_projector_is_one(self) -> None:
        I = self._identity_monomial()

        ket1 = (KetTerm(coeff=1.0 + 0.0j, monomial=I),)
        ket2 = (KetTerm(coeff=2.0 + 0.0j, monomial=I),)

        rho1 = density_pure(ket1)
        rho2 = density_pure(ket2)

        # Convex combination of proportional projectors -> same projector after normalization.
        rho_mix = (
            *[type(t)(coeff=0.5 * t.coeff, left=t.left, right=t.right) for t in rho1],
            *[type(t)(coeff=0.5 * t.coeff, left=t.left, right=t.right) for t in rho2],
        )

        rho_mix = density_normalize_trace(rho_mix)
        purity = density_purity(rho_mix)

        self.assertAlmostEqual(purity, 1.0, places=12)

    def test_purity_empty_density_is_zero(self) -> None:
        purity = density_purity(())
        self.assertEqual(purity, 0.0)
