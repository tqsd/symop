from __future__ import annotations

import math
import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.op.poly import OpPoly
from symop.core.monomial import Monomial
from symop.core.operators import ModeOp
from symop.core.terms import DensityTerm, KetTerm
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


def _mk_mode(
    tag: str, *, omega0: float = 2.0, sigma: float = 1.0, tau: float = 0.0
) -> ModeOp:
    env = GaussianEnvelope(omega0=omega0, sigma=sigma, tau=tau, phi0=0.0)
    lab = ModeLabel(path=PathLabel(tag), pol=PolarizationLabel.H(), envelope=env)
    return ModeOp(label=lab)


def _mono_from_word(word) -> Monomial:
    """
    Build a *normally ordered* Monomial from an ordered ladder-op word.

    Your Monomial stores two tuples: creators then annihilators.
    For tests we keep it strict: we only accept already-normal-ordered words.
    """
    creators = []
    annihilators = []
    seen_ann = False
    for op in word:
        if op.is_creation:
            if seen_ann:
                raise ValueError(
                    "Test helper expects normal order (all creators before annihilators)."
                )
            creators.append(op)
        else:
            seen_ann = True
            annihilators.append(op)
    return Monomial(creators=tuple(creators), annihilators=tuple(annihilators))


class TestDensityPoly(unittest.TestCase):
    def setUp(self) -> None:
        self.m1 = _mk_mode("m1")
        self.m2 = _mk_mode("m2", tau=0.25)

    def test_zero_identity_bool_len_iter(self) -> None:
        z = DensityPoly.zero()
        self.assertFalse(bool(z))
        self.assertEqual(len(z), 0)
        self.assertEqual(tuple(iter(z)), ())

        I = DensityPoly.identity()
        self.assertTrue(bool(I))
        self.assertEqual(len(I), 1)
        dt = I.terms[0]
        self.assertEqual(dt.coeff, 1.0 + 0.0j)
        self.assertTrue(dt.left.is_identity)
        self.assertTrue(dt.right.is_identity)
        self.assertTrue(I.is_identity_left)
        self.assertTrue(I.is_identity_right)
        self.assertTrue(I.is_diagonal_in_monomials)
        self.assertEqual(I.mode_count, 0)
        self.assertEqual(I.unique_modes, ())

    def test_structural_equality_is_exact(self) -> None:
        I1 = DensityPoly.identity()
        I2 = DensityPoly.identity()
        self.assertEqual(I1, I2)

        # Structural equality: term tuples must match exactly
        rho_a = DensityPoly((I1.terms[0],))
        rho_b = DensityPoly((I1.terms[0], I1.terms[0]))
        self.assertNotEqual(rho_a, rho_b)

    def test_scaled_mul_rmul_div_neg_sub(self) -> None:
        I = DensityPoly.identity()

        a = I.scaled(2.0 + 0.0j)
        b = I * (2.0 + 0.0j)
        c = (2.0 + 0.0j) * I
        self.assertEqual(a, b)
        self.assertEqual(b, c)
        self.assertEqual(a.terms[0].coeff, 2.0 + 0.0j)

        d = I / (2.0 + 0.0j)
        self.assertEqual(d.terms[0].coeff, 0.5 + 0.0j)

        with self.assertRaises(TypeError):
            _ = I / "nope"  # type: ignore[operator]

        with self.assertRaises(ZeroDivisionError):
            _ = I / 0.0

        self.assertEqual((-I).terms[0].coeff, -1.0 + 0.0j)
        self.assertFalse(bool(I - I))
        self.assertEqual((I - I).normalize().terms, ())

    def test_add_concatenates_then_merges(self) -> None:
        I = DensityPoly.identity()
        out = I + I
        self.assertEqual(len(out), 1)
        self.assertEqual(out.terms[0].coeff, 2.0 + 0.0j)

    def test_combine_like_terms_eps_removal(self) -> None:
        Id = Monomial.identity()
        t1 = DensityTerm(coeff=1.0 + 0.0j, left=Id, right=Id)
        t2 = DensityTerm(coeff=-1.0 + 0.0j, left=Id, right=Id)

        rho = DensityPoly((t1, t2))
        rho2 = rho.combine_like_terms(eps=1e-12)
        self.assertEqual(rho2.terms, ())
        self.assertFalse(bool(rho2))

    def test_trace_identity_and_trace_normalization(self) -> None:
        I = DensityPoly.identity()
        self.assertEqual(I.trace(), 1.0 + 0.0j)
        self.assertTrue(I.is_trace_normalized())

        rho = I.scaled(3.0 + 0.0j)
        self.assertEqual(rho.trace(), 3.0 + 0.0j)

        rho_n = rho.normalize_trace()
        self.assertTrue(rho_n.is_trace_normalized())
        self.assertAlmostEqual(abs(rho_n.trace() - (1.0 + 0.0j)), 0.0, places=14)

        with self.assertRaises(ValueError):
            _ = DensityPoly.zero().normalize_trace(eps=1e-14)

    def test_require_trace_normalized(self) -> None:
        DensityPoly.identity().require_trace_normalized()

        with self.assertRaises(ValueError):
            DensityPoly.identity().scaled(2.0).require_trace_normalized()

    def test_inner_and_hs_norm_for_scaled_identity(self) -> None:
        I = DensityPoly.identity()
        c = 2.5 - 1.0j
        rho = I.scaled(c)

        inn = rho.inner(rho)
        self.assertAlmostEqual(inn.imag, 0.0, places=14)
        self.assertAlmostEqual(inn.real, abs(c) ** 2, places=14)

        self.assertAlmostEqual(rho.hs_norm2(), abs(c) ** 2, places=14)
        self.assertAlmostEqual(rho.hs_norm(), abs(c), places=14)

    def test_purity_for_scaled_identity(self) -> None:
        I = DensityPoly.identity()
        self.assertAlmostEqual(I.purity(), 1.0, places=14)
        self.assertTrue(I.is_pure())

        rho = I.scaled(2.0)
        self.assertAlmostEqual(rho.purity(), 4.0, places=14)
        self.assertFalse(rho.is_pure())

        rho_n = rho.normalize_trace()
        self.assertAlmostEqual(rho_n.purity(), 1.0, places=14)
        self.assertTrue(rho_n.is_pure())

    def test_flags_diagonal_identity(self) -> None:
        adag1 = self.m1.create
        a1 = self.m1.ann

        n_mono = Monomial(creators=(adag1,), annihilators=(a1,))
        rho_diag = DensityPoly(
            (DensityTerm(coeff=1.0 + 0.0j, left=n_mono, right=n_mono),)
        )
        self.assertTrue(rho_diag.is_diagonal_in_monomials)
        self.assertFalse(rho_diag.is_identity_left)
        self.assertFalse(rho_diag.is_identity_right)

        rho_off = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0 + 0.0j,
                    left=n_mono,
                    right=Monomial(creators=(adag1,), annihilators=()),
                ),
            )
        )
        self.assertFalse(rho_off.is_diagonal_in_monomials)

    def test_unique_modes_and_mode_count(self) -> None:
        a1 = self.m1.ann
        a2 = self.m2.ann

        m1mono = Monomial(creators=(), annihilators=(a1,))
        m2mono = Monomial(creators=(), annihilators=(a2,))

        rho = DensityPoly(
            (
                DensityTerm(coeff=1.0 + 0.0j, left=m1mono, right=m1mono),
                DensityTerm(coeff=1.0 + 0.0j, left=m2mono, right=m2mono),
            )
        )

        ms = rho.unique_modes
        self.assertEqual(len(ms), 2)
        self.assertEqual(rho.mode_count, 2)
        self.assertEqual(ms[0].signature, self.m1.signature)
        self.assertEqual(ms[1].signature, self.m2.signature)

    def test_is_block_diagonal_by_modes(self) -> None:
        a1 = self.m1.ann
        a2 = self.m2.ann

        # Left [m1], right [m1] -> ok
        t_ok = DensityTerm(
            coeff=1.0 + 0.0j,
            left=Monomial(creators=(), annihilators=(a1,)),
            right=Monomial(creators=(), annihilators=(a1,)),
        )
        self.assertTrue(DensityPoly((t_ok,)).is_block_diagonal_by_modes())

        # Left [m1, m2], right [m2, m1] -> mismatch
        t_bad = DensityTerm(
            coeff=1.0 + 0.0j,
            left=Monomial(creators=(), annihilators=(a1, a2)),
            right=Monomial(creators=(), annihilators=(a2, a1)),
        )
        self.assertFalse(DensityPoly((t_bad,)).is_block_diagonal_by_modes())

    def test_apply_left_right_empty_word_no_change(self) -> None:
        I = DensityPoly.identity()
        self.assertEqual(I.apply_left(()), I)
        self.assertEqual(I.apply_right(()), I)

    def test_matmul_with_oppoly_identity_and_linearity(self) -> None:
        adag1 = self.m1.create
        a1 = self.m1.ann

        L = Monomial(creators=(adag1,), annihilators=())
        R = Monomial(creators=(), annihilators=(a1,))
        rho = DensityPoly((DensityTerm(coeff=2.0 + 0.0j, left=L, right=R),))

        # rho @ I = rho
        OI = OpPoly.identity()
        self.assertEqual(rho @ OI, rho)

        # Linearity: rho@(O1+O2) == rho@O1 + rho@O2 (after merging)
        O1 = OpPoly.from_words([[a1]])
        O2 = OpPoly.from_words([[adag1]]) * (0.5 + 0.0j)

        left = (rho @ (O1 + O2)).combine_like_terms()
        right = ((rho @ O1) + (rho @ O2)).combine_like_terms()
        self.assertEqual(left, right)

    def test_partial_trace_smoke(self) -> None:
        # This is intentionally a "does it run + returns DensityPoly" test.
        # Exact content depends on your density_partial_trace semantics.
        a1 = self.m1.ann
        a2 = self.m2.ann

        mono = Monomial(creators=(), annihilators=(a1, a2))
        rho = DensityPoly((DensityTerm(coeff=1.0 + 0.0j, left=mono, right=mono),))

        red = rho.partial_trace([self.m2])
        self.assertIsInstance(red, DensityPoly)
        self.assertIsInstance(red.terms, tuple)

    def test_pure_constructor_smoke(self) -> None:
        adag1 = self.m1.create
        Id = Monomial.identity()
        mono1 = Monomial(creators=(adag1,), annihilators=())

        ket_terms = (
            KetTerm(coeff=1.0 + 0.0j, monomial=Id),
            KetTerm(coeff=0.25 + 0.0j, monomial=mono1),
        )

        rho = DensityPoly.pure(ket_terms)
        self.assertIsInstance(rho, DensityPoly)
        self.assertTrue(bool(rho))

        tr = rho.trace()
        self.assertTrue(math.isfinite(tr.real))
        self.assertTrue(math.isfinite(tr.imag))
        self.assertAlmostEqual(tr.imag, 0.0, places=12)
        self.assertGreaterEqual(tr.real, 0.0)


if __name__ == "__main__":
    unittest.main()
