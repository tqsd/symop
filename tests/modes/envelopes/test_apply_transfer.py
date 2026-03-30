import unittest

from symop.modes.envelopes.filtered import FilteredEnvelope
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.transfer.apply import apply_transfer
from symop.modes.transfer.gaussian.lowpass import GaussianLowpass
from symop.modes.transfer.rect_bandpass import RectBandpass


class TestApplyTransfer(unittest.TestCase):
    def test_analytic_path_for_gaussian_closed_transfer(self) -> None:
        env = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=0.0, phi0=0.0)
        transfer = GaussianLowpass(w0=1.0, sigma_w=1.0)

        out_env, eta = apply_transfer(transfer, env)

        self.assertNotIsInstance(out_env, FilteredEnvelope)
        self.assertGreaterEqual(eta, 0.0)
        self.assertLessEqual(eta, 1.0)

    def test_numerical_path_for_generic_transfer(self) -> None:
        env = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=0.0, phi0=0.0)
        transfer = RectBandpass(w0=1.0, width=1.0)

        out_env, eta = apply_transfer(transfer, env)

        self.assertIsInstance(out_env, FilteredEnvelope)
        self.assertGreaterEqual(eta, 0.0)
        self.assertLessEqual(eta, 1.0)
