from __future__ import annotations

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm, DensityTerm

from symop_proto.algebra.common.signatures import (
    sig_mono,
    sig_ket,
    sig_density,
)


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


class TestSignatureHelpers(ExtendedTestCase):
    def test_sig_mono_exact_vs_approx(self):
        m1 = Monomial(creators=(make_mode("A", tau=0.0).create,))
        m2 = Monomial(creators=(make_mode("A", tau=1e-11).create,))

        # exact signatures should differ (tiny env change)
        self.assertNotEqual(sig_mono(m1), sig_mono(m2))

        # approx signatures can match with coarse rounding
        self.assertEqual(
            sig_mono(m1, approx=True, decimals=8),
            sig_mono(m2, approx=True, decimals=8),
        )

        # but at stricter rounding they may differ
        self.assertNotEqual(
            sig_mono(m1, approx=True, decimals=14),
            sig_mono(m2, approx=True, decimals=14),
        )

        # returns a tuple
        self.assertIsInstance(sig_mono(m1), tuple)
        self.assertIsInstance(sig_mono(m1, approx=True, decimals=8), tuple)

    def test_sig_ket_delegates_and_rounds(self):
        m1 = Monomial(creators=(make_mode("A", tau=0.0).create,))
        m2 = Monomial(creators=(make_mode("A", tau=1e-11).create,))
        kt1 = KetTerm(1.0, m1)
        kt2 = KetTerm(1.0, m2)

        # exact differs
        self.assertNotEqual(sig_ket(kt1), sig_ket(kt2))
        # approx can match
        self.assertEqual(
            sig_ket(kt1, approx=True, decimals=8),
            sig_ket(kt2, approx=True, decimals=8),
        )
        # type is tuple
        self.assertIsInstance(sig_ket(kt1), tuple)
        self.assertIsInstance(sig_ket(kt1, approx=True, decimals=8), tuple)

    def test_sig_density_delegates_and_rounds(self):
        L1 = Monomial(creators=(make_mode("A", tau=0.0).create,))
        R1 = Monomial(creators=(make_mode("B", tau=0.0).create,))
        L2 = Monomial(creators=(make_mode("A", tau=1e-11).create,))
        R2 = Monomial(creators=(make_mode("B", tau=1e-11).create,))

        d1 = DensityTerm(1.0, left=L1, right=R1)
        d2 = DensityTerm(1.0, left=L2, right=R2)

        # exact differs
        self.assertNotEqual(sig_density(d1), sig_density(d2))
        # approx can match
        self.assertEqual(
            sig_density(d1, approx=True, decimals=8),
            sig_density(d2, approx=True, decimals=8),
        )
        # type is tuple
        self.assertIsInstance(sig_density(d1), tuple)
        self.assertIsInstance(sig_density(d1, approx=True, decimals=8), tuple)
