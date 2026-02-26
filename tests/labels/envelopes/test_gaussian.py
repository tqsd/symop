import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope


class TestGaussianEnvelope(unittest.TestCase):
    def setUp(self):
        self.env = GaussianEnvelope(
            omega0=5.0,
            sigma=0.7,
            tau=1.2,
            phi0=0.3,
        )

    def test_omega_sigma(self):
        self.assertAlmostEqual(self.env.omega_sigma, 1.0 / 0.7, places=14)

    def test_time_eval_basic_properties(self):
        t = np.linspace(-5.0, 5.0, 201)
        z = self.env.time_eval(t)

        self.assertTrue(np.iscomplexobj(z))
        self.assertEqual(z.shape, t.shape)

        # peak magnitude near tau
        idx_peak = np.argmax(np.abs(z))
        self.assertAlmostEqual(t[idx_peak], self.env.tau, places=1)

    def test_time_eval_raises_on_bad_sigma(self):
        bad = GaussianEnvelope(omega0=0.0, sigma=0.0, tau=0.0, phi0=0.0)
        with self.assertRaises(ValueError):
            bad.time_eval([0.0, 1.0])

        bad2 = GaussianEnvelope(omega0=0.0, sigma=float("nan"), tau=0.0, phi0=0.0)
        with self.assertRaises(ValueError):
            bad2.time_eval([0.0, 1.0])

    def test_freq_eval_basic_properties(self):
        w = np.linspace(-10.0, 10.0, 401)
        Z = self.env.freq_eval(w)

        self.assertTrue(np.iscomplexobj(Z))
        self.assertEqual(Z.shape, w.shape)

        # peak magnitude near omega0
        idx_peak = np.argmax(np.abs(Z))
        self.assertAlmostEqual(w[idx_peak], self.env.omega0, places=1)

    def test_freq_eval_raises_on_bad_sigma(self):
        bad = GaussianEnvelope(omega0=0.0, sigma=0.0, tau=0.0, phi0=0.0)
        with self.assertRaises(ValueError):
            bad.freq_eval([0.0, 1.0])

        bad2 = GaussianEnvelope(omega0=0.0, sigma=float("inf"), tau=0.0, phi0=0.0)
        with self.assertRaises(ValueError):
            bad2.freq_eval([0.0, 1.0])

    def test_center_and_scale(self):
        center, scale = self.env.center_and_scale()
        self.assertEqual(center, self.env.tau)
        self.assertEqual(scale, self.env.sigma)

    def test_delayed(self):
        env2 = self.env.delayed(0.5)
        self.assertNotEqual(env2.tau, self.env.tau)
        self.assertAlmostEqual(env2.tau, self.env.tau + 0.5, places=14)
        self.assertEqual(env2.omega0, self.env.omega0)

    def test_phased(self):
        env2 = self.env.phased(0.7)
        self.assertAlmostEqual(env2.phi0, self.env.phi0 + 0.7, places=14)
        self.assertEqual(env2.tau, self.env.tau)

    def test_signature(self):
        sig = self.env.signature
        self.assertEqual(sig[0], "gauss")
        self.assertEqual(len(sig), 5)

    def test_approx_signature(self):
        approx = self.env.approx_signature(decimals=3, ignore_global_phase=False)
        self.assertEqual(approx[0], "gauss_approx")
        self.assertEqual(len(approx), 5)

        approx2 = self.env.approx_signature(decimals=3, ignore_global_phase=True)
        self.assertEqual(approx2[-1], 0.0)

    def test_latex_property(self):
        latex = self.env.latex
        self.assertIsInstance(latex, str)
        self.assertIn(r"\zeta(t)", latex)

    def test_frozen_dataclass(self):
        with self.assertRaises(FrozenInstanceError):
            self.env.tau = 0.0  # type: ignore[misc]

    def test_overlap_self_and_symmetry(self):
        env1 = GaussianEnvelope(
            omega0=3.0,
            sigma=0.8,
            tau=0.5,
            phi0=0.1,
        )

        env2 = GaussianEnvelope(
            omega0=3.0,
            sigma=0.8,
            tau=0.5,
            phi0=0.1,
        )

        # Self-overlap should be ~1 (normalized mode)
        ov_self = env1.overlap(env1)
        self.assertIsInstance(ov_self, complex)
        self.assertAlmostEqual(abs(ov_self), 1.0, places=6)

        # Overlap should be symmetric conjugate
        ov12 = env1.overlap(env2)
        ov21 = env2.overlap(env1)
        self.assertAlmostEqual(ov12, np.conjugate(ov21), places=6)


if __name__ == "__main__":
    unittest.main()
