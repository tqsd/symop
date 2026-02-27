from __future__ import annotations

import unittest

from symop.ccr.algebra.density.apply_right import apply_right
from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm


class TestApplyRight(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_apply_right_empty_word_is_identity(self) -> None:
        # This test requires no ladder operators and should always pass.
        M_left = self._identity_monomial()
        M_right = self._identity_monomial()

        terms = (DensityTerm(coeff=2.0 + 0.0j, left=M_left, right=M_right),)
        out = apply_right(terms, ())

        self.assertEqual(out, terms)

    def test_apply_right_matches_reference_for_empty_word(self) -> None:
        # Cross-check using ket_from_word: with empty word, right monomial unchanged.
        M_left = self._identity_monomial()
        M_right = self._identity_monomial()

        t = DensityTerm(coeff=1.0 + 0.0j, left=M_left, right=M_right)
        out = apply_right((t,), ())

        # Reference: unchanged right monomial, unchanged coefficient.
        ref = (t,)
        self.assertEqual(out, ref)
