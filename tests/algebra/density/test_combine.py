from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel

from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm

from symop_proto.algebra.density.combine import combine_like_terms_density


def make_mode(
    path: str = "A",
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    tau: float = 0.0,
    phi: float = 0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


class TestCombineLikeTermsDensity(ExtendedTestCase):
    def test_empty(self):
        self.assertEqual(combine_like_terms_density(()), ())

    def test_exact_combine_and_sum_coeffs(self):
        mA = make_mode("A")
        mB = make_mode("B")
        L = Monomial(creators=(mA.create,))
        R = Monomial(annihilators=(mB.ann,))
        d1 = DensityTerm(1.0 + 2.0j, left=L, right=R)
        d2 = DensityTerm(-1.0 + 3.0j, left=L, right=R)
        out = combine_like_terms_density((d1, d2))
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 0.0 + 5.0j)
        self.assertEqual(out[0].left.signature, L.signature)
        self.assertEqual(out[0].right.signature, R.signature)

    def test_drops_small_coeffs_with_eps(self):
        mA = make_mode("A")
        L = Monomial(creators=(mA.create,))
        R = Monomial()
        small = DensityTerm(5e-13 + 4e-13j, left=L, right=R)
        out1 = combine_like_terms_density((small,), eps=1e-12)
        self.assertEqual(out1, ())

        keep = DensityTerm(2e-12, left=L, right=R)
        out2 = combine_like_terms_density((small, keep), eps=1e-12)
        self.assertEqual(len(out2), 1)
        self.assertComplexAlmostEqual(out2[0].coeff, 2e-12 + 0j)
        self.assertEqual(out2[0].left.signature, L.signature)
        self.assertEqual(out2[0].right.signature, R.signature)

    def test_approx_true_merges_nearby_envs(self):
        """Near-identical envelopes (tiny tau difference) should merge with approx=True."""
        m1L = make_mode("A", tau=0.0)
        m1R = make_mode("B", tau=0.0)
        m2L = make_mode("A", tau=1e-11)  # tiny shift
        m2R = make_mode("B", tau=1e-11)

        d1 = DensityTerm(
            1.0,
            left=Monomial(creators=(m1L.create,)),
            right=Monomial(creators=(m1R.create,)),
        )
        d2 = DensityTerm(
            2.5,
            left=Monomial(creators=(m2L.create,)),
            right=Monomial(creators=(m2R.create,)),
        )

        # Exact: distinct
        out_exact = combine_like_terms_density((d1, d2), approx=False)
        self.assertEqual(len(out_exact), 2)

        # Approx (e.g., decimals=8): merged
        out_approx = combine_like_terms_density((d1, d2), approx=True, decimals=8)
        self.assertEqual(len(out_approx), 1)
        self.assertComplexAlmostEqual(out_approx[0].coeff, 3.5 + 0j)

    def test_sorted_by_left_right_signature(self):
        """Output is sorted lexicographically by (left.signature, right.signature)."""
        a = make_mode("A")
        b = make_mode("B")
        # Create deliberately out-of-order inputs
        d2 = DensityTerm(1.0, left=Monomial(creators=(b.create,)), right=Monomial())
        d1 = DensityTerm(1.0, left=Monomial(creators=(a.create,)), right=Monomial())

        out = combine_like_terms_density((d2, d1))
        pairs = [(t.left.signature, t.right.signature) for t in out]
        self.assertLessEqual(pairs[0], pairs[1])
        self.assertEqual(
            set(pairs),
            {
                (
                    Monomial(creators=(a.create,)).signature,
                    Monomial().signature,
                ),
                (
                    Monomial(creators=(b.create,)).signature,
                    Monomial().signature,
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
