import unittest

import numpy as np

from symop.modes.transfer.supergaussian_bandpass import SuperGaussianBandpass


class TestSuperGaussianBandpass(unittest.TestCase):
    def test_order_one_reduces_to_gaussian_shape(self) -> None:
        transfer = SuperGaussianBandpass(w0=0.0, sigma_w=2.0, order=1)
        w = np.array([0.0, 1.0, 2.0], dtype=float)

        out = transfer(w)
        expected = np.exp(-0.5 * ((w - 0.0) / 2.0) ** 2).astype(complex)

        np.testing.assert_allclose(out, expected)

    def test_invalid_order_raises(self) -> None:
        with self.assertRaises(ValueError):
            SuperGaussianBandpass(w0=0.0, sigma_w=1.0, order=0)
