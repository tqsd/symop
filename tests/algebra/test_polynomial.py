from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.polynomial import (
    KetPoly,
    DensityPoly,
)
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm
from symop_proto.core.operators import ModeOp
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel


def make_mode(
    path: str = "A",
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    tau: float = 0.0,
    phi: float = 0.0,
) -> ModeOp:
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), PolarizationLabel.H())
    return ModeOp(env=env, label=label)


class TestKetPoly(ExtendedTestCase):
    def test_from_ops_creators_only(self):
        a = make_mode("A")
        kp = KetPoly.from_ops(creators=(a.create,), coeff=2.0)
        self.assertEqual(len(kp.terms), 1)
        t = kp.terms[0]
        self.assertComplexAlmostEqual(t.coeff, 2.0 + 0.0j)
        self.assertEqual(t.monomial.signature, Monomial(creators=(a.create,)).signature)

    def test_from_word_adag_then_a_same_mode(self):
        m = make_mode("A")
        kp = KetPoly.from_word(ops=(m.create, m.ann))
        # Expect identity + a^dag a (both +1)
        sigs = {t.monomial.signature for t in kp.terms}
        adag_a_sig = Monomial(creators=(m.create,), annihilators=(m.ann,)).signature
        self.assertEqual(sigs, {adag_a_sig})

    def test_scaled_and_inner_and_norm(self):
        a = make_mode("A")
        kp = KetPoly.from_ops(creators=(a.create,), coeff=2.0)
        self.assertAlmostEqual(kp.norm2(), 4.0)  # <2 a^dag | 2 a^dag> = 4
        kq = kp.scaled(0.5)
        self.assertAlmostEqual(kq.norm2(), 1.0)
        self.assertAlmostEqual(kp.inner(kp), 4.0 + 0.0j)

    def test_add_and_mul(self):
        a = make_mode("A")
        b = make_mode("B")
        kp1 = KetPoly.from_ops(creators=(a.create,), coeff=1.0)
        kp2 = KetPoly.from_ops(creators=(b.create,), coeff=2.0)
        s = kp1 + kp2
        self.assertEqual(len(s.terms), 2)
        p = kp1 * 3.0
        self.assertComplexAlmostEqual(p.terms[0].coeff, 3.0 + 0.0j)

    def test_apply_word(self):
        m = make_mode("A")
        kp = KetPoly.from_ops(creators=(m.create,), coeff=1.0)
        out = kp.apply_word((m.ann,))
        # Expect identity + a^dag a
        sigs = {t.monomial.signature for t in out.terms}
        self.assertIn(Monomial().signature, sigs)
        self.assertIn(
            Monomial(creators=(m.create,), annihilators=(m.ann,)).signature,
            sigs,
        )

    def test_unique_modes_and_mode_count(self):
        a = make_mode("A")
        b = make_mode("B")
        kp = KetPoly.from_ops(creators=(a.create, b.create))
        self.assertEqual(kp.mode_count, 2)
        sigs = [m.signature for m in kp.unique_modes]
        self.assertIn(a.signature, sigs)
        self.assertIn(b.signature, sigs)

    def test_require_creator_only(self):
        a = make_mode("A")
        kp = KetPoly.from_ops(creators=(a.create,))
        kp.require_creator_only()  # should not raise
        # build a ket with an annihilator via from_word
        kw = KetPoly.from_word(ops=(a.ann,))
        with self.assertRaises(ValueError):
            kw.require_creator_only()


class TestDensityPoly(ExtendedTestCase):
    def test_pure_and_trace_and_purity(self):
        a = make_mode("A")
        psi = KetPoly.from_ops(creators=(a.create,), coeff=1.0)
        rho = DensityPoly.pure(psi).normalize_trace()
        self.assertTrue(rho.is_trace_normalized())
        # pure state's purity should be ~1
        self.assertAlmostEqual(rho.purity(), 1.0, places=12)

    def test_scaled_and_combine_like_terms(self):
        m = Monomial()
        rho = DensityPoly(terms=(DensityTerm(1.0, m, m), DensityTerm(2.0, m, m)))
        comb = rho.combine_like_terms()
        self.assertEqual(len(comb.terms), 1)
        self.assertComplexAlmostEqual(comb.terms[0].coeff, 3.0 + 0.0j)
        sc = comb.scaled(-1j)
        self.assertComplexAlmostEqual(sc.terms[0].coeff, -3j)

    def test_apply_left_right(self):
        m = make_mode("A")
        rho0 = DensityPoly(terms=(DensityTerm(1.0, Monomial(), Monomial()),))
        # Left apply a^dag then a
        rhoL = rho0.apply_left((m.create, m.ann))
        sigsL = {t.left.signature for t in rhoL.terms}
        self.assertEqual(
            sigsL,
            {Monomial(creators=(m.create,), annihilators=(m.ann,)).signature},
        )
        # Right apply a (acts as (a^dag) on the right via reversed-dagger expansion)
        rhoR = rho0.apply_right((m.ann,))
        sigsR = {t.right.signature for t in rhoR.terms}
        self.assertEqual(sigsR, {Monomial(creators=(m.create,)).signature})

    def test_trace_and_normalize_trace(self):
        m = Monomial()
        rho = DensityPoly(terms=(DensityTerm(2.0, m, m),))
        self.assertComplexAlmostEqual(rho.trace(), 2.0 + 0.0j)
        rhoN = rho.normalize_trace()
        self.assertTrue(rhoN.is_trace_normalized())
        self.assertComplexAlmostEqual(rhoN.trace(), 1.0 + 0.0j)

    def test_partial_trace_merges_terms(self):
        a = make_mode("A")
        # Two identical terms on mode A -> after tracing A they merge on identity
        rho = DensityPoly(
            terms=(
                DensityTerm(
                    1.0,
                    Monomial(creators=(a.create,)),
                    Monomial(creators=(a.create,)),
                ),
                DensityTerm(
                    2.0,
                    Monomial(creators=(a.create,)),
                    Monomial(creators=(a.create,)),
                ),
            )
        )
        out = rho.partial_trace((a,))
        self.assertEqual(len(out.terms), 1)
        self.assertEqual(out.terms[0].left.signature, Monomial().signature)
        self.assertEqual(out.terms[0].right.signature, Monomial().signature)
        self.assertComplexAlmostEqual(out.terms[0].coeff, 3.0 + 0.0j)

    def test_unique_modes_and_block_diagonal(self):
        a = make_mode("A")
        b = make_mode("B")
        rho = DensityPoly(
            terms=(
                DensityTerm(
                    1.0,
                    Monomial(creators=(a.create,)),
                    Monomial(creators=(a.create,)),
                ),
                DensityTerm(
                    1.0,
                    Monomial(creators=(b.create,)),
                    Monomial(creators=(b.create,)),
                ),
            )
        )
        self.assertEqual(rho.mode_count, 2)
        self.assertTrue(rho.is_block_diagonal_by_modes())


if __name__ == "__main__":
    unittest.main()
