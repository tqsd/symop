import unittest

import numpy as np

from symop.modes.transfer.supergaussian_bandpass import SuperGaussianBandpass


class TestSuperGaussianBandpass(unittest.TestCase):
    def test_signature(self):
        tf = SuperGaussianBandpass(w0=1.5, sigma_w=2.0, order=3)
        self.assertEqual(tf.signature, ("supergauss_bandpass", 1.5, 2.0, 3))

    def test_approx_signature_rounds(self):
        tf = SuperGaussianBandpass(w0=1.234567, sigma_w=9.876543, order=2)
        got = tf.approx_signature(decimals=3, ignore_global_phase=True)
        self.assertEqual(
            got,
            (
                "supergauss_bandpass_approx",
                round(1.234567, 3),
                round(9.876543, 3),
                2,
            ),
        )

    def test_call_matches_definition(self):
        w0 = 10.0
        sigma = 2.0
        m = 3
        tf = SuperGaussianBandpass(w0=w0, sigma_w=sigma, order=m)

        w = np.array([w0, w0 + sigma, w0 + 2.0 * sigma], dtype=float)
        out = tf(w)

        x = (w - w0) / sigma
        expected = np.exp(-0.5 * np.power(x * x, m)).astype(complex)
        np.testing.assert_allclose(out, expected)

        self.assertTrue(np.iscomplexobj(out))
        self.assertEqual(out.shape, w.shape)

    def test_call_order_one_matches_gaussian_shape(self):
        w0 = 0.7
        sigma = 1.3
        tf = SuperGaussianBandpass(w0=w0, sigma_w=sigma, order=1)

        w = np.array([w0 - sigma, w0, w0 + sigma], dtype=float)
        out = tf(w)

        expected = np.exp(-0.5 * ((w - w0) / sigma) ** 2).astype(complex)
        np.testing.assert_allclose(out, expected)

    def test_call_is_real_and_in_0_1(self):
        tf = SuperGaussianBandpass(w0=0.0, sigma_w=1.0, order=4)
        w = np.linspace(-5.0, 5.0, 201)
        out = tf(w)

        self.assertTrue(np.all(np.isreal(out)))
        self.assertTrue(np.all(out.real >= 0.0))
        self.assertTrue(np.all(out.real <= 1.0 + 1e-15))

        center = tf(np.array([0.0], dtype=float)).real[0]
        self.assertAlmostEqual(center, 1.0, places=14)

    def test_call_symmetric_about_w0(self):
        w0 = 2.3
        sigma = 0.7
        tf = SuperGaussianBandpass(w0=w0, sigma_w=sigma, order=2)

        d = np.array([0.0, 0.1, 0.5, 1.2], dtype=float)
        out_left = tf(w0 - d)
        out_right = tf(w0 + d)

        np.testing.assert_allclose(out_left, out_right)

    def test_call_accepts_python_sequence(self):
        tf = SuperGaussianBandpass(w0=0.0, sigma_w=1.0, order=2)
        out = tf([0.0, 1.0, 2.0])

        w = np.array([0.0, 1.0, 2.0], dtype=float)
        expected = np.exp(-0.5 * np.power(w * w, 2)).astype(complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3,))

    def test_call_empty_input(self):
        tf = SuperGaussianBandpass(w0=0.0, sigma_w=1.0, order=2)
        w = np.array([], dtype=float)
        out = tf(w)

        self.assertEqual(out.shape, (0,))
        self.assertTrue(np.iscomplexobj(out))

    def test_call_raises_on_order_less_than_one(self):
        tf = SuperGaussianBandpass(w0=0.0, sigma_w=1.0, order=0)
        with self.assertRaises(ValueError):
            _ = tf([0.0, 1.0])

        tf2 = SuperGaussianBandpass(w0=0.0, sigma_w=1.0, order=-3)
        with self.assertRaises(ValueError):
            _ = tf2([0.0, 1.0])

    def test_call_raises_on_nonpositive_sigma(self):
        tf0 = SuperGaussianBandpass(w0=0.0, sigma_w=0.0, order=2)
        with self.assertRaises(ValueError):
            _ = tf0([0.0, 1.0])

        tfneg = SuperGaussianBandpass(w0=0.0, sigma_w=-1.0, order=2)
        with self.assertRaises(ValueError):
            _ = tfneg([0.0, 1.0])

    def test_call_raises_on_nonfinite_sigma(self):
        tfnan = SuperGaussianBandpass(w0=0.0, sigma_w=float("nan"), order=2)
        with self.assertRaises(ValueError):
            _ = tfnan([0.0, 1.0])

        tfinf = SuperGaussianBandpass(w0=0.0, sigma_w=float("inf"), order=2)
        with self.assertRaises(ValueError):
            _ = tfinf([0.0, 1.0])

    def test_call_does_not_modify_input(self):
        tf = SuperGaussianBandpass(w0=1.0, sigma_w=2.0, order=2)
        w = np.array([1.0, 2.0, 3.0], dtype=float)
        w_copy = w.copy()

        _ = tf(w)
        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
