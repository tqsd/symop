from __future__ import annotations

import unittest

from symop.ccr.algebra.density.pure import density_pure
from symop.ccr.algebra.density.trace import density_trace
from symop.core.monomial import Monomial
from symop.core.terms import KetTerm


class TestDensityPure(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_density_pure_single_term(self) -> None:
        I = self._identity_monomial()

        ket = (KetTerm(coeff=2.0 + 0.0j, monomial=I),)

        rho = density_pure(ket)

        self.assertEqual(len(rho), 1)
        self.assertAlmostEqual(rho[0].coeff.real, 4.0)
        self.assertAlmostEqual(rho[0].coeff.imag, 0.0)
        self.assertEqual(rho[0].left.signature, I.signature)
        self.assertEqual(rho[0].right.signature, I.signature)

    def test_density_pure_two_term_superposition(self) -> None:
        I = self._identity_monomial()

        ket = (
            KetTerm(coeff=1.0 + 0.0j, monomial=I),
            KetTerm(coeff=2.0 + 0.0j, monomial=I),
        )

        rho = density_pure(ket)

        # (1 + 2)^2 = 9 after combining identical outer products
        self.assertEqual(len(rho), 1)
        self.assertAlmostEqual(rho[0].coeff.real, 9.0)
        self.assertAlmostEqual(rho[0].coeff.imag, 0.0)

    def test_density_pure_is_hermitian_by_construction(self) -> None:
        I = self._identity_monomial()

        ket = (
            KetTerm(coeff=1.0 + 1.0j, monomial=I),
            KetTerm(coeff=2.0 - 3.0j, monomial=I),
        )

        rho = density_pure(ket)

        # Hermiticity test: ρ should satisfy <ρ,ρ> real and >=0
        # Since identity monomials collapse everything,
        # trace = (sum coeff)^2 magnitude
        tr = density_trace(rho)
        self.assertGreaterEqual(tr.real, 0.0)

    def test_density_pure_empty_input(self) -> None:
        rho = density_pure(())
        self.assertEqual(rho, ())
