import unittest

import numpy as np

from symop.modes.transfer.rect_bandpass import RectBandpass


class TestRectBandpass(unittest.TestCase):
    def test_call_matches_rectangular_window(self) -> None:
        transfer = RectBandpass(w0=10.0, width=4.0)
        w = np.array([7.9, 8.0, 10.0, 12.0, 12.1], dtype=float)

        out = transfer(w)
        expected = np.array([0, 1, 1, 1, 0], dtype=complex)

        np.testing.assert_allclose(out, expected)

    def test_invalid_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            RectBandpass(w0=0.0, width=-1.0)
