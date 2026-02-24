from __future__ import annotations

import unittest

import numpy as np

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial

from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.evaluator import GaussianEvaluator


class TestGaussianEvaluator(unittest.TestCase):
    def assert_allclose(self, a, b, atol=1e-12, rtol=1e-12, msg=""):
        ok = np.allclose(a, b, atol=atol, rtol=rtol)
        if not ok:
            aa = np.asarray(a)
            bb = np.asarray(b)
            diff = np.max(np.abs(aa - bb))
            self.fail(msg or f"not close; max abs diff = {diff}")

    def setUp(self):
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)

        self.m1 = ModeOp(
            env=env,
            label=ModeLabel(PathLabel("A"), PolarizationLabel.H()),
        )
        self.m2 = ModeOp(
            env=env,
            label=ModeLabel(PathLabel("B"), PolarizationLabel.H()),
        )

        self.B = ModeBasis.build([self.m1, self.m2])
        self.vac = GaussianCore.vacuum(self.B)
        self.ev_vac = GaussianEvaluator(self.vac)

    def test_identity_monomial(self):
        mon_id = Monomial()
        self.assert_allclose(self.ev_vac.expect_monomial(mon_id), 1.0 + 0.0j)

    def test_vacuum_basic_expectations(self):
        mon_a1 = Monomial(creators=(), annihilators=(self.m1.ann,))
        self.assert_allclose(self.ev_vac.expect_monomial(mon_a1), 0.0 + 0.0j)

        mon_adag1 = Monomial(creators=(self.m1.create,), annihilators=())
        self.assert_allclose(
            self.ev_vac.expect_monomial(mon_adag1), 0.0 + 0.0j
        )

        mon_n1 = Monomial(
            creators=(self.m1.create,), annihilators=(self.m1.ann,)
        )
        self.assert_allclose(self.ev_vac.expect_monomial(mon_n1), 0.0 + 0.0j)

        mon_a1a2 = Monomial(
            creators=(), annihilators=(self.m1.ann, self.m2.ann)
        )
        self.assert_allclose(self.ev_vac.expect_monomial(mon_a1a2), 0.0 + 0.0j)

    def test_commutator_identity_diagonal(self):
        # For any state: <a_i a_i^dag> = G_ii + <a_i^dag a_i>.
        mon_n1 = Monomial(
            creators=(self.m1.create,), annihilators=(self.m1.ann,)
        )
        rhs = self.B.gram[0, 0] + self.ev_vac.expect_monomial(mon_n1)
        self.assert_allclose(rhs, self.B.gram[0, 0])

    def test_coherent_first_and_second_moments(self):
        alpha = np.array([0.3 + 0.1j, -0.2 + 0.5j], dtype=complex)
        core = GaussianCore.coherent(self.B, alpha)
        ev = GaussianEvaluator(core)

        mon_a1 = Monomial(creators=(), annihilators=(self.m1.ann,))
        mon_a2 = Monomial(creators=(), annihilators=(self.m2.ann,))
        self.assert_allclose(ev.expect_monomial(mon_a1), alpha[0])
        self.assert_allclose(ev.expect_monomial(mon_a2), alpha[1])

        mon_n1 = Monomial(
            creators=(self.m1.create,), annihilators=(self.m1.ann,)
        )
        mon_n2 = Monomial(
            creators=(self.m2.create,), annihilators=(self.m2.ann,)
        )
        self.assert_allclose(ev.expect_monomial(mon_n1), abs(alpha[0]) ** 2)
        self.assert_allclose(ev.expect_monomial(mon_n2), abs(alpha[1]) ** 2)

        mon_adag1_a2 = Monomial(
            creators=(self.m1.create,), annihilators=(self.m2.ann,)
        )
        self.assert_allclose(
            ev.expect_monomial(mon_adag1_a2),
            np.conjugate(alpha[0]) * alpha[1],
        )

    def test_two_mode_squeezed_matches_M_and_N(self):
        r = 0.7
        s2 = np.sinh(r) ** 2
        sc = np.sinh(r) * np.cosh(r)

        alpha0 = np.zeros((2,), dtype=complex)
        N = np.array([[s2, 0.0], [0.0, s2]], dtype=complex)
        M = np.array([[0.0, sc], [sc, 0.0]], dtype=complex)

        core = GaussianCore.from_moments(self.B, alpha=alpha0, N=N, M=M)
        ev = GaussianEvaluator(core)

        mon_a1a2 = Monomial(
            creators=(), annihilators=(self.m1.ann, self.m2.ann)
        )
        self.assert_allclose(ev.expect_monomial(mon_a1a2), M[0, 1])

        mon_n1 = Monomial(
            creators=(self.m1.create,), annihilators=(self.m1.ann,)
        )
        mon_n2 = Monomial(
            creators=(self.m2.create,), annihilators=(self.m2.ann,)
        )
        self.assert_allclose(ev.expect_monomial(mon_n1), N[0, 0])
        self.assert_allclose(ev.expect_monomial(mon_n2), N[1, 1])

        mon_a1a1 = Monomial(
            creators=(), annihilators=(self.m1.ann, self.m1.ann)
        )
        self.assert_allclose(ev.expect_monomial(mon_a1a1), M[0, 0])

    def test_fourth_order_thermal_like_single_mode(self):
        # This is the regression trap for "forgot to sum over pairings".
        #
        # Single mode with alpha=0, M=0, N=nbar:
        # <a^dag a> = nbar
        # <a^dag a^dag a a> = 2 * nbar^2 (Gaussian / Wick)
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m = ModeOp(
            env=env,
            label=ModeLabel(PathLabel("T"), PolarizationLabel.H()),
        )
        B1 = ModeBasis.build([m])

        nbar = 0.37
        alpha0 = np.zeros((1,), dtype=complex)
        N = np.array([[nbar]], dtype=complex)
        M = np.array([[0.0]], dtype=complex)

        core = GaussianCore.from_moments(B1, alpha=alpha0, N=N, M=M)
        ev = GaussianEvaluator(core)

        mon_n = Monomial(creators=(m.create,), annihilators=(m.ann,))
        self.assert_allclose(ev.expect_monomial(mon_n), nbar)

        mon_adag2_a2 = Monomial(
            creators=(m.create, m.create),
            annihilators=(m.ann, m.ann),
        )
        self.assert_allclose(
            ev.expect_monomial(mon_adag2_a2), 2.0 * nbar * nbar
        )

    def test_nonorthogonal_offdiag_commutator_identity(self):
        # Build two modes that overlap partially in time so Gram has off-diagonal.
        env1 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        env2 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.2, phi0=0.0)

        m1 = ModeOp(
            env=env1,
            label=ModeLabel(PathLabel("A"), PolarizationLabel.H()),
        )
        m2 = ModeOp(
            env=env2,
            label=ModeLabel(PathLabel("A"), PolarizationLabel.H()),
        )

        B = ModeBasis.build([m1, m2])
        core = GaussianCore.vacuum(B)
        ev = GaussianEvaluator(core)

        # In vacuum: <a2^dag a1> = 0 (normal ordered)
        mon_adag2_a1 = Monomial(creators=(m2.create,), annihilators=(m1.ann,))
        self.assert_allclose(ev.expect_monomial(mon_adag2_a1), 0.0 + 0.0j)

        # Identity: <a1 a2^dag> = G_12 + <a2^dag a1> = G_12
        g12 = B.gram[0, 1]
        rhs = g12 + ev.expect_monomial(mon_adag2_a1)
        self.assert_allclose(rhs, g12)

        # Sanity: ensure we actually got a nontrivial overlap
        self.assertTrue(abs(g12) > 1e-6)

    def test_raises_for_mode_not_in_basis(self):
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m3 = ModeOp(
            env=env,
            label=ModeLabel(PathLabel("C"), PolarizationLabel.H()),
        )
        mon = Monomial(creators=(), annihilators=(m3.ann,))

        with self.assertRaises(KeyError):
            self.ev_vac.expect_monomial(mon)


if __name__ == "__main__":
    unittest.main()
