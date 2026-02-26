from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel

from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm, DensityTerm


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


class TestKetTerm(ExtendedTestCase):
    def test_adjoint_conjugates_coeff_and_daggers_monomial(self):
        ma = make_mode("A")
        mb = make_mode("B")
        m = Monomial(creators=(ma.create, mb.create), annihilators=(ma.ann,))
        kt = KetTerm(coeff=1.0 + 2.0j, monomial=m)

        adj = kt.adjoint()
        # coeff conjugated
        self.assertComplexAlmostEqual(adj.coeff, (1.0 - 2.0j))
        # double adjoint returns original
        self.assertEqual(kt.signature, adj.adjoint().signature)

        # monomial structure flipped & daggered: check via signatures
        m_adj = m.adjoint()
        self.assertEqual(adj.monomial.signature, m_adj.signature)

    def test_signature_and_approx_signature(self):
        # exact signature equals monomial signature
        ma1 = make_mode("A", tau=0.0)
        m1 = Monomial(creators=(ma1.create,))
        kt1 = KetTerm(1.0, m1)

        ma2 = make_mode("A", tau=1e-11)  # tiny env change
        kt2 = KetTerm(1.0, Monomial(creators=(ma2.create,)))

        # exact signatures usually differ
        self.assertNotEqual(kt1.signature, kt2.signature)
        # approx signatures can match with coarse rounding
        self.assertEqual(
            kt1.approx_signature(decimals=8),
            kt2.approx_signature(decimals=8),
        )


class TestDensityTerm(ExtendedTestCase):
    def test_adjoint_swaps_left_right_and_conjugates(self):
        ma = make_mode("A")
        mb = make_mode("B")
        left = Monomial(creators=(ma.create,))
        right = Monomial(creators=(mb.create,), annihilators=(ma.ann,))

        rho = DensityTerm(coeff=-0.5 + 0.25j, left=left, right=right)
        adj = rho.adjoint()

        # coeff conjugated
        self.assertComplexAlmostEqual(adj.coeff, (-0.5 - 0.25j))
        # left/right swapped
        self.assertEqual(adj.left.signature, right.signature)
        self.assertEqual(adj.right.signature, left.signature)
        # double adjoint returns original
        self.assertEqual(rho.signature, adj.adjoint().signature)

    def test_signature_and_approx_signature(self):
        mL1 = Monomial(creators=(make_mode("A", tau=0.0).create,))
        mR1 = Monomial(creators=(make_mode("B", tau=0.0).create,))

        mL2 = Monomial(creators=(make_mode("A", tau=1e-11).create,))
        mR2 = Monomial(creators=(make_mode("B", tau=1e-11).create,))

        d1 = DensityTerm(1.0, left=mL1, right=mR1)
        d2 = DensityTerm(1.0, left=mL2, right=mR2)

        # exact signatures likely differ
        self.assertNotEqual(d1.signature, d2.signature)
        # approx signatures can match with rounding
        self.assertEqual(
            d1.approx_signature(decimals=8),
            d2.approx_signature(decimals=8),
        )

    def test_signature_shape(self):
        mL = Monomial(creators=(make_mode("A").create,))
        mR = Monomial(creators=(make_mode("B").create,))
        d = DensityTerm(1.0, left=mL, right=mR)
        sig = d.signature
        self.assertEqual(sig[0], "DT")
        self.assertEqual(sig[1], "L")
        self.assertEqual(sig[3], "R")
        # ensure monomial signatures are embedded
        self.assertIsInstance(sig[2], tuple)
        self.assertIsInstance(sig[4], tuple)


if __name__ == "__main__":
    unittest.main()
