import unittest

import numpy as np

from symop.modes.transfer.gaussian.lowpass import GaussianLowpass


class TestGaussianLowpass(unittest.TestCase):
    def test_call_matches_formula(self) -> None:
        transfer = GaussianLowpass(w0=2.0, sigma_w=0.5)
        w = np.array([2.0, 2.5, 3.0], dtype=float)

        out = transfer(w)
        expected = np.exp(-0.5 * ((w - 2.0) / 0.5) ** 2).astype(complex)

        np.testing.assert_allclose(out, expected)

    def test_center_value_is_one(self) -> None:
        transfer = GaussianLowpass(w0=1.5, sigma_w=0.25)

        out = transfer(np.array([1.5], dtype=float))

        np.testing.assert_allclose(out, np.array([1.0 + 0.0j]))

    def test_invalid_sigma_raises(self) -> None:
        with self.assertRaises(ValueError):
            GaussianLowpass(w0=0.0, sigma_w=0.0)
