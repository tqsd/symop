from __future__ import annotations

import unittest

from symop.ccr.algebra.density.expand_monomial_times_word import (
    expand_monomial_times_word,
)
from symop.ccr.algebra.density.expand_word_times_monomial import (
    expand_word_times_monomial,
)
from symop.ccr.algebra.ket.from_word import ket_from_word
from symop.core.monomial import Monomial


class TestExpandHelpers(unittest.TestCase):
    def _identity_monomial(self) -> Monomial:
        # Prefer an explicit constructor if you have one; fall back to empty creators/annihilators.
        ident = getattr(Monomial, "identity", None)
        if callable(ident):
            return ident()
        return Monomial(creators=(), annihilators=())

    def test_expand_monomial_times_word_identity_word_identity_monomial(
        self,
    ) -> None:
        M = self._identity_monomial()
        out = expand_monomial_times_word(M, ())

        ref = list(ket_from_word(ops=()))
        self.assertEqual(out, ref)

    def test_expand_monomial_times_word_empty_word_matches_ket_from_word(
        self,
    ) -> None:
        M = self._identity_monomial()

        out = expand_monomial_times_word(M, ())
        ref = list(ket_from_word(ops=(*M.creators, *M.annihilators)))

        self.assertEqual(out, ref)

    def test_expand_word_times_monomial_empty_word_is_identity_action(
        self,
    ) -> None:
        M = self._identity_monomial()

        out = expand_word_times_monomial((), M)

        # For empty word: W * M = M. Compare against the canonical path via ket_from_word.
        ref = list(ket_from_word(ops=(*M.creators, *M.annihilators)))

        self.assertEqual(out, ref)

    def test_expand_word_times_monomial_returns_single_term_for_identity(
        self,
    ) -> None:
        M = self._identity_monomial()

        out = expand_word_times_monomial((), M)

        self.assertEqual(len(out), 1)
        t = out[0]

        # Structural checks that should hold for the identity monomial term.
        self.assertTrue(hasattr(t, "coeff"))
        self.assertTrue(hasattr(t, "monomial"))
        self.assertAlmostEqual(t.coeff, 1.0 + 0.0j)
        self.assertEqual(t.monomial.signature, M.signature)
