import unittest

import numpy as np

from symop.modes.transfer.gaussian_bandpass import GaussianBandpass


class TestGaussianBandpass(unittest.TestCase):
    def test_signature(self):
        tf = GaussianBandpass(w0=1.5, sigma_w=2.0)
        self.assertEqual(tf.signature, ("gauss_bandpass", 1.5, 2.0))

    def test_approx_signature_rounds(self):
        tf = GaussianBandpass(w0=1.234567, sigma_w=9.876543)
        got = tf.approx_signature(decimals=3)
        self.assertEqual(
            got,
            (
                "gauss_bandpass_approx",
                round(1.234567, 3),
                round(9.876543, 3),
            ),
        )

    def test_call_basic_values(self):
        w0 = 10.0
        sigma = 2.0
        tf = GaussianBandpass(w0=w0, sigma_w=sigma)

        w = np.array([w0, w0 + sigma, w0 + 2.0 * sigma], dtype=float)
        out = tf(w)

        expected = np.exp(-0.5 * ((w - w0) / sigma) ** 2).astype(complex)
        np.testing.assert_allclose(out, expected)

        self.assertTrue(np.iscomplexobj(out))
        self.assertEqual(out.shape, w.shape)

    def test_call_accepts_python_sequence(self):
        tf = GaussianBandpass(w0=0.0, sigma_w=1.0)
        out = tf([0.0, 1.0, 2.0])

        expected = np.exp(-0.5 * np.array([0.0, 1.0, 2.0], dtype=float) ** 2).astype(
            complex
        )
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3,))

    def test_call_empty_input(self):
        tf = GaussianBandpass(w0=0.0, sigma_w=1.0)
        w = np.array([], dtype=float)
        out = tf(w)

        self.assertEqual(out.shape, (0,))
        self.assertTrue(np.iscomplexobj(out))

    def test_call_is_real_and_positive(self):
        tf = GaussianBandpass(w0=0.0, sigma_w=1.0)
        w = np.linspace(-5.0, 5.0, 51)
        out = tf(w)

        self.assertTrue(np.all(np.isreal(out)))
        self.assertTrue(np.all(out.real >= 0.0))
        self.assertTrue(np.all(out.real <= 1.0 + 1e-15))

    def test_call_symmetric_about_w0(self):
        w0 = 2.3
        sigma = 0.7
        tf = GaussianBandpass(w0=w0, sigma_w=sigma)

        d = np.array([0.0, 0.1, 0.5, 1.2], dtype=float)
        w_left = w0 - d
        w_right = w0 + d

        out_left = tf(w_left)
        out_right = tf(w_right)

        np.testing.assert_allclose(out_left, out_right)

    def test_call_raises_on_nonpositive_sigma(self):
        tf0 = GaussianBandpass(w0=0.0, sigma_w=0.0)
        with self.assertRaises(ValueError):
            _ = tf0([0.0, 1.0])

        tfneg = GaussianBandpass(w0=0.0, sigma_w=-1.0)
        with self.assertRaises(ValueError):
            _ = tfneg([0.0, 1.0])

    def test_call_raises_on_nonfinite_sigma(self):
        tfnan = GaussianBandpass(w0=0.0, sigma_w=float("nan"))
        with self.assertRaises(ValueError):
            _ = tfnan([0.0, 1.0])

        tfinf = GaussianBandpass(w0=0.0, sigma_w=float("inf"))
        with self.assertRaises(ValueError):
            _ = tfinf([0.0, 1.0])

    def test_call_does_not_modify_input(self):
        tf = GaussianBandpass(w0=1.0, sigma_w=2.0)
        w = np.array([1.0, 2.0, 3.0], dtype=float)
        w_copy = w.copy()

        _ = tf(w)
        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
