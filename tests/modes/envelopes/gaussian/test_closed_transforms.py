import unittest

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope
from symop.modes.transfer.gaussian.constant_phase import ConstantPhase
from symop.modes.transfer.gaussian.time_delay import TimeDelay
from symop.modes.transfer.gaussian.lowpass import GaussianLowpass


class TestGaussianClosedTransfers(unittest.TestCase):
    def test_constant_phase_preserves_transmission(self) -> None:
        env = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=2.0, phi0=0.1)
        transfer = ConstantPhase(phi0=0.3)

        out_env, eta = transfer.apply_to_gaussian(env)

        self.assertIsInstance(out_env, GaussianEnvelope)
        self.assertAlmostEqual(eta, 1.0)
        self.assertAlmostEqual(out_env.tau, env.tau)
        self.assertAlmostEqual(out_env.phi0, env.phi0 + 0.3)

    def test_time_delay_shifts_tau(self) -> None:
        env = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=2.0, phi0=0.1)
        transfer = TimeDelay(tau=1.25)

        out_env, eta = transfer.apply_to_gaussian(env)

        self.assertIsInstance(out_env, GaussianEnvelope)
        self.assertAlmostEqual(eta, 1.0)
        self.assertAlmostEqual(out_env.tau, 3.25)
        self.assertAlmostEqual(out_env.phi0, env.phi0)

    def test_lowpass_returns_gaussian_closed_env_and_eta_in_range(self) -> None:
        env = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=0.0, phi0=0.0)
        transfer = GaussianLowpass(w0=1.0, sigma_w=1.0)

        out_env, eta = transfer.apply_to_gaussian(env)

        self.assertIsInstance(out_env, (GaussianEnvelope, GaussianMixtureEnvelope))
        self.assertGreaterEqual(eta, 0.0)
        self.assertLessEqual(eta, 1.0)
