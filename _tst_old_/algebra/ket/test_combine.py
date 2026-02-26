from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.ket.combine import combine_like_terms_ket
from symop_proto.core.terms import KetTerm
from symop_proto.core.monomial import Monomial
from symop_proto.core.operators import ModeOp
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel


def make_mode(
    path: str = "A",
    *,
    omega=1.0,
    sigma=0.3,
    tau=0.0,
    phi=0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


class TestCombineLikeTermsKet(ExtendedTestCase):
    def test_empty(self):
        self.assertEqual(combine_like_terms_ket(()), ())

    def test_exact_combine_and_sum_coeffs(self):
        m = make_mode("A")
        mono = Monomial(creators=(m.create,))
        t1 = KetTerm(1.0 + 2.0j, mono)
        t2 = KetTerm(-1.0 + 2.0j, mono)
        out = combine_like_terms_ket((t1, t2))
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 0.0 + 4.0j)
        self.assertEqual(out[0].monomial.signature, mono.signature)

    def test_drops_small_coeffs_with_eps(self):
        m = make_mode("A")
        mono_keep = Monomial(creators=(m.create,))
        mono_drop = Monomial(
            creators=(m.create,)
        )  # same key; use different coeffs in separate calls

        # One term just below eps -> dropped
        t_small = KetTerm(0.5e-12, mono_drop)
        out1 = combine_like_terms_ket((t_small,), eps=1e-12)
        self.assertEqual(out1, ())

        # Mixed: one small (drop), one larger (keep)
        t_keep = KetTerm(2.0e-12, mono_keep)
        out2 = combine_like_terms_ket((t_small, t_keep), eps=1e-12)
        self.assertEqual(len(out2), 1)
        self.assertComplexAlmostEqual(out2[0].coeff, 2.0e-12)
        self.assertEqual(out2[0].monomial.signature, mono_keep.signature)

    def test_approx_false_does_not_combine_nearby_envs(self):
        m1 = make_mode("A", tau=0.0)
        m2 = make_mode("A", tau=1e-11)  # tiny env change -> different exact signature
        t1 = KetTerm(1.0, Monomial(creators=(m1.create,)))
        t2 = KetTerm(2.0, Monomial(creators=(m2.create,)))
        out = combine_like_terms_ket((t1, t2), approx=False)
        self.assertEqual(len(out), 2)
        # ensure original terms are present (order may be sorted)
        sigs = [t.monomial.signature for t in out]
        self.assertIn(t1.monomial.signature, sigs)
        self.assertIn(t2.monomial.signature, sigs)

    def test_approx_true_does_combine_with_rounding(self):
        m1 = make_mode("A", tau=0.0)
        m2 = make_mode("A", tau=1e-11)  # will collapse under decimals=8
        t1 = KetTerm(1.0 + 0j, Monomial(creators=(m1.create,)))
        t2 = KetTerm(2.5 + 0j, Monomial(creators=(m2.create,)))
        out = combine_like_terms_ket((t1, t2), approx=True, decimals=8)
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 3.5 + 0j)

    def test_sorted_by_monomial_signature(self):
        a = make_mode("A")
        b = make_mode("B")
        # Create out-of-order terms
        tB = KetTerm(1.0, Monomial(creators=(b.create,)))
        tA = KetTerm(1.0, Monomial(creators=(a.create,)))
        out = combine_like_terms_ket((tB, tA))
        sigs = [t.monomial.signature for t in out]
        self.assertLessEqual(sigs[0], sigs[1])  # sorted non-decreasing
        # And contains the expected keys
        self.assertEqual(set(sigs), {tA.monomial.signature, tB.monomial.signature})


if __name__ == "__main__":
    unittest.main()
