import unittest

import numpy as np

from symop.modes.transfer.rect_bandpass import RectBandpass


class TestRectBandpass(unittest.TestCase):
    def test_signature(self):
        tf = RectBandpass(w0=1.5, width=2.0)
        self.assertEqual(tf.signature, ("rect_bandpass", 1.5, 2.0))

    def test_approx_signature_rounds(self):
        tf = RectBandpass(w0=1.234567, width=9.876543)
        got = tf.approx_signature(decimals=3, ignore_global_phase=True)
        self.assertEqual(
            got,
            (
                "rect_bandpass_approx",
                round(1.234567, 3),
                round(9.876543, 3),
            ),
        )

    def test_call_inside_outside_and_edges(self):
        w0 = 10.0
        width = 4.0
        half = 0.5 * width
        tf = RectBandpass(w0=w0, width=width)

        w = np.array(
            [
                w0 - half - 1e-9,  # outside
                w0 - half,  # edge (inside)
                w0,  # center
                w0 + half,  # edge (inside)
                w0 + half + 1e-9,  # outside
            ],
            dtype=float,
        )
        out = tf(w)

        expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=complex)
        np.testing.assert_allclose(out, expected)
        self.assertTrue(np.iscomplexobj(out))
        self.assertEqual(out.shape, w.shape)

    def test_call_accepts_python_sequence(self):
        tf = RectBandpass(w0=0.0, width=2.0)  # half-width = 1
        out = tf([-1.0, -1.1, 0.0, 0.9, 1.0, 1.0001])

        expected = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (6,))

    def test_call_empty_input(self):
        tf = RectBandpass(w0=0.0, width=1.0)
        w = np.array([], dtype=float)
        out = tf(w)

        self.assertEqual(out.shape, (0,))
        self.assertTrue(np.iscomplexobj(out))

    def test_call_is_only_zeros_or_ones(self):
        tf = RectBandpass(w0=0.3, width=1.2)
        w = np.linspace(-10.0, 10.0, 2001)
        out = tf(w)

        vals = np.unique(out.real)
        for v in vals:
            self.assertTrue(v == 0.0 or v == 1.0)
        self.assertTrue(np.all(out.imag == 0.0))

    def test_call_raises_on_nonpositive_width(self):
        tf0 = RectBandpass(w0=0.0, width=0.0)
        with self.assertRaises(ValueError):
            _ = tf0([0.0, 1.0])

        tfneg = RectBandpass(w0=0.0, width=-1.0)
        with self.assertRaises(ValueError):
            _ = tfneg([0.0, 1.0])

    def test_call_raises_on_nonfinite_width(self):
        tfnan = RectBandpass(w0=0.0, width=float("nan"))
        with self.assertRaises(ValueError):
            _ = tfnan([0.0, 1.0])

        tfinf = RectBandpass(w0=0.0, width=float("inf"))
        with self.assertRaises(ValueError):
            _ = tfinf([0.0, 1.0])

    def test_call_does_not_modify_input(self):
        tf = RectBandpass(w0=1.0, width=2.0)
        w = np.array([0.0, 1.0, 2.0], dtype=float)
        w_copy = w.copy()

        _ = tf(w)
        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
