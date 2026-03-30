import unittest

import numpy as np

from symop.modes.transfer.gaussian.highpass import GaussianHighpass


class TestGaussianHighpass(unittest.TestCase):
    def test_call_matches_formula(self) -> None:
        transfer = GaussianHighpass(w0=1.0, sigma_w=2.0)
        w = np.array([1.0, 2.0, 5.0], dtype=float)

        out = transfer(w)
        expected = 1.0 - np.exp(-0.5 * ((w - 1.0) / 2.0) ** 2)

        np.testing.assert_allclose(out, expected.astype(complex))

    def test_center_value_is_zero(self) -> None:
        transfer = GaussianHighpass(w0=3.0, sigma_w=1.0)

        out = transfer(np.array([3.0], dtype=float))

        np.testing.assert_allclose(out, np.array([0.0 + 0.0j]))
