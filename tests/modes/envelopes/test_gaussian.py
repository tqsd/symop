import math
import unittest

import numpy as np

from symop.modes.envelopes.base import _overlap_numeric
from symop.modes.envelopes.gaussian import GaussianEnvelope


class TestGaussianEnvelope(unittest.TestCase):
    def assertComplexAlmostEqual(
        self,
        left: complex,
        right: complex,
        places: int = 10,
    ) -> None:
        self.assertAlmostEqual(left.real, right.real, places=places)
        self.assertAlmostEqual(left.imag, right.imag, places=places)

    def test_omega_sigma_is_inverse_sigma(self) -> None:
        env = GaussianEnvelope(omega0=2.0, sigma=4.0, tau=1.0)
        self.assertAlmostEqual(env.omega_sigma, 0.25)

    def test_center_and_scale_returns_tau_and_sigma(self) -> None:
        env = GaussianEnvelope(omega0=2.0, sigma=3.0, tau=5.0)
        self.assertEqual(env.center_and_scale(), (5.0, 3.0))

    def test_delayed_returns_shifted_copy(self) -> None:
        env = GaussianEnvelope(omega0=2.0, sigma=1.5, tau=0.5, phi0=0.2)
        delayed = env.delayed(1.25)

        self.assertEqual(delayed.omega0, env.omega0)
        self.assertEqual(delayed.sigma, env.sigma)
        self.assertEqual(delayed.tau, 1.75)
        self.assertEqual(delayed.phi0, env.phi0)

    def test_phased_returns_phase_shifted_copy(self) -> None:
        env = GaussianEnvelope(omega0=2.0, sigma=1.5, tau=0.5, phi0=0.2)
        phased = env.phased(0.7)

        self.assertEqual(phased.omega0, env.omega0)
        self.assertEqual(phased.sigma, env.sigma)
        self.assertEqual(phased.tau, env.tau)
        self.assertAlmostEqual(phased.phi0, 0.9)

    def test_time_eval_rejects_non_positive_sigma(self) -> None:
        env = GaussianEnvelope(omega0=0.0, sigma=0.0, tau=0.0)
        t = np.linspace(-1.0, 1.0, 8)
        with self.assertRaises(ValueError):
            env.time_eval(t)

    def test_freq_eval_rejects_non_positive_sigma(self) -> None:
        env = GaussianEnvelope(omega0=0.0, sigma=-1.0, tau=0.0)
        w = np.linspace(-1.0, 1.0, 8)
        with self.assertRaises(ValueError):
            env.freq_eval(w)

    def test_self_overlap_is_one(self) -> None:
        env = GaussianEnvelope(omega0=1.2, sigma=0.8, tau=-0.4, phi0=0.3)
        self.assertComplexAlmostEqual(env.overlap(env), 1.0 + 0.0j, places=10)
        self.assertAlmostEqual(env.norm2(), 1.0, places=10)

    def test_overlap_is_conjugate_symmetric(self) -> None:
        left = GaussianEnvelope(omega0=1.0, sigma=0.7, tau=-0.2, phi0=0.1)
        right = GaussianEnvelope(omega0=1.4, sigma=1.1, tau=0.5, phi0=-0.6)

        ov_lr = left.overlap(right)
        ov_rl = right.overlap(left)

        self.assertComplexAlmostEqual(ov_lr, ov_rl.conjugate(), places=10)

    def test_closed_form_overlap_matches_numeric_quadrature(self) -> None:
        left = GaussianEnvelope(omega0=1.1, sigma=0.9, tau=-0.4, phi0=0.2)
        right = GaussianEnvelope(omega0=1.7, sigma=1.3, tau=0.8, phi0=-0.1)

        closed = left.overlap_gaussian_closed(right)

        c1, s1 = left.center_and_scale()
        c2, s2 = right.center_and_scale()
        center = 0.5 * (c1 + c2)
        window = 12.0 * max(s1, s2)

        numeric = _overlap_numeric(
            left.time_eval,
            right.time_eval,
            tmin=center - window,
            tmax=center + window,
            n=2**14,
        )

        self.assertAlmostEqual(closed.real, numeric.real, places=6)
        self.assertAlmostEqual(closed.imag, numeric.imag, places=6)

    def test_delay_reduces_self_overlap_magnitude(self) -> None:
        env = GaussianEnvelope(omega0=0.8, sigma=1.0, tau=0.0)
        delayed = env.delayed(3.0)

        overlap = env.overlap(delayed)

        self.assertLess(abs(overlap), 1.0)
        self.assertGreaterEqual(abs(overlap), 0.0)

    def test_global_phase_changes_overlap_by_phase_only(self) -> None:
        env = GaussianEnvelope(omega0=0.8, sigma=1.0, tau=0.0, phi0=0.0)
        phased = env.phased(math.pi / 4.0)

        overlap = env.overlap(phased)

        self.assertAlmostEqual(abs(overlap), 1.0, places=10)
        self.assertAlmostEqual(
            overlap.real,
            math.cos(math.pi / 4.0),
            places=10,
        )
        self.assertAlmostEqual(
            overlap.imag,
            math.sin(math.pi / 4.0),
            places=10,
        )

    def test_signature_is_stable(self) -> None:
        env = GaussianEnvelope(omega0=1.0, sigma=2.0, tau=3.0, phi0=4.0)
        self.assertEqual(
            env.signature,
            ("gauss", 1.0, 2.0, 3.0, 4.0),
        )

    def test_approx_signature_can_ignore_global_phase(self) -> None:
        env1 = GaussianEnvelope(omega0=1.0, sigma=2.0, tau=3.0, phi0=0.2)
        env2 = GaussianEnvelope(omega0=1.0, sigma=2.0, tau=3.0, phi0=1.7)

        self.assertNotEqual(env1.approx_signature(), env2.approx_signature())
        self.assertEqual(
            env1.approx_signature(ignore_global_phase=True),
            env2.approx_signature(ignore_global_phase=True),
        )
