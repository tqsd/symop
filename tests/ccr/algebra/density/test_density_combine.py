from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch


@dataclass(frozen=True)
class DummyMonomial:
    signature: tuple


@dataclass(frozen=True)
class DummyDensityTerm:
    coeff: complex
    left: DummyMonomial
    right: DummyMonomial


class TestCombineLikeTermsDensity(unittest.TestCase):
    def test_combines_coefficients_drops_near_zero_and_sorts(self) -> None:
        # Two different (left,right) pairs; first pair cancels to ~0 and is dropped.
        a = DummyMonomial(signature=("A",))
        b = DummyMonomial(signature=("B",))
        c = DummyMonomial(signature=("C",))
        d = DummyMonomial(signature=("D",))

        t1 = DummyDensityTerm(coeff=1.0 + 0.0j, left=b, right=d)
        t2 = DummyDensityTerm(coeff=-1.0 + 0.0j, left=b, right=d)  # cancels t1
        t3 = DummyDensityTerm(coeff=2.0 + 0.0j, left=a, right=c)

        # Define signature keys explicitly.
        def fake_sig_density(term, *, approx, decimals, ignore_global_phase):
            # Verify forwarded arguments exist and are wired.
            self.assertIsInstance(approx, bool)
            self.assertIsInstance(decimals, int)
            self.assertIsInstance(ignore_global_phase, bool)
            return (term.left.signature, term.right.signature)

        with patch(
            "symop.ccr.algebra.density.combine.sig_density",
            side_effect=fake_sig_density,
        ):
            from symop.ccr.algebra.density.combine import (
                combine_like_terms_density,
            )

            out = combine_like_terms_density((t1, t2, t3), eps=1e-12)

        # Only the (a,c) term should remain, because (b,d) cancels.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].coeff, 2.0 + 0.0j)
        self.assertEqual(out[0].left.signature, ("A",))
        self.assertEqual(out[0].right.signature, ("C",))

    def test_sorts_by_left_then_right_signature(self) -> None:
        # Create terms that survive combining, then check sort order.
        a = DummyMonomial(signature=("A",))
        b = DummyMonomial(signature=("B",))
        c = DummyMonomial(signature=("C",))
        d = DummyMonomial(signature=("D",))

        t1 = DummyDensityTerm(coeff=1.0 + 0.0j, left=b, right=c)
        t2 = DummyDensityTerm(coeff=1.0 + 0.0j, left=a, right=d)

        def fake_sig_density(term, *, approx, decimals, ignore_global_phase):
            return (term.left.signature, term.right.signature)

        with patch(
            "symop.ccr.algebra.density.combine.sig_density",
            side_effect=fake_sig_density,
        ):
            from symop.ccr.algebra.density.combine import (
                combine_like_terms_density,
            )

            out = combine_like_terms_density((t1, t2))

        # Sorted: (A,D) then (B,C)
        self.assertEqual([t.left.signature for t in out], [("A",), ("B",)])
        self.assertEqual([t.right.signature for t in out], [("D",), ("C",)])
