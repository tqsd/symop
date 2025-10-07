from __future__ import annotations
from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp, OperatorKind


def make_mode(
    path: str = "A",
    pol: PolarizationLabel | None = None,
    *,
    omega=1.0,
    sigma=0.3,
    tau=0.0,
    phi=0.0,
) -> ModeOp:
    if pol is None:
        pol = PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


class TestOperators(ExtendedTestCase):
    def test_flags_and_dagger(self):
        m = make_mode()
        self.assertTrue(m.ann.is_annihilation)
        self.assertFalse(m.ann.is_creation)
        self.assertTrue(m.create.is_creation)
        self.assertFalse(m.create.is_annihilation)
        self.assertIs(m.ann.dagger(), m.create)
        self.assertIs(m.create.dagger(), m.ann)

    def test_mode_signature_shapes(self):
        m = make_mode()
        sig = m.signature
        self.assertEqual(sig[0], "mode")
        self.assertIsInstance(sig[1], tuple)  # env sig
        self.assertIsInstance(sig[2], tuple)  # label sig
        lsig = m.create.signature
        self.assertEqual(lsig[0], "lop")
        self.assertEqual(lsig[1], OperatorKind.CREATE.value)

    def test_commutator_same_mode_is_one(self):
        m = make_mode()
        z = m.ann.commutator(m.create)
        self.assertComplexAlmostEqual(z, 1.0 + 0j, rtol=1e-12, atol=1e-12)
        z2 = m.create.commutator(m.ann)
        self.assertComplexAlmostEqual(z2, -1.0 + 0j, rtol=1e-12, atol=1e-12)
        self.assertComplexAlmostEqual(m.ann.commutator(m.ann), 0.0 + 0j)
        self.assertComplexAlmostEqual(m.create.commutator(m.create), 0.0 + 0j)

    def test_commutator_zero_for_orthogonal_paths(self):
        a = make_mode("A")
        b = make_mode("B")  # path-orthogonal
        self.assertComplexAlmostEqual(a.ann.commutator(b.create), 0.0 + 0j)

    def test_commutator_zero_for_orthogonal_polarizations(self):
        aH = make_mode("A", PolarizationLabel.H())
        aV = make_mode("A", PolarizationLabel.V())
        self.assertComplexAlmostEqual(aH.ann.commutator(aV.create), 0.0 + 0j)

    def test_commutator_nonorthogonal_polarization(self):
        aH = make_mode("A", PolarizationLabel.H())
        aD = make_mode("A", PolarizationLabel.D())  # <H|D> = 1/sqrt(2)
        s = 2**-0.5
        z = aH.ann.commutator(aD.create)
        self.assertComplexAlmostEqual(z, s + 0j, rtol=1e-12, atol=1e-12)

    def test_commutator_complex_phase_from_polarization(self):
        # <V|R> = -i / sqrt(2)
        aV = make_mode("A", PolarizationLabel.V())
        aR = make_mode("A", PolarizationLabel.R())
        expected = (-1j) * (2**-0.5)
        z = aV.ann.commutator(aR.create)
        self.assertComplexAlmostEqual(z, expected, rtol=1e-12, atol=1e-12)

    def test_commutator_with_different_envelopes(self):
        # envelope overlap enters multiplicatively
        m1 = make_mode(
            "A", PolarizationLabel.H(), omega=1.0, sigma=0.35, tau=-0.2
        )
        m2 = make_mode(
            "A", PolarizationLabel.H(), omega=1.3, sigma=0.5, tau=+0.1
        )
        env_overlap = m1.env.overlap(m2.env)  # analytic Gaussian overlap
        z = m1.ann.commutator(m2.create)
        self.assertComplexAlmostEqual(z, env_overlap, rtol=1e-12, atol=1e-12)

    def test_approx_signature_rounding(self):
        m1 = make_mode(tau=0.0)
        m2 = make_mode(tau=1e-11)  # tiny difference
        self.assertNotEqual(m1.signature, m2.signature)
        self.assertEqual(
            m1.approx_signature(decimals=8),
            m2.approx_signature(decimals=8),
        )
