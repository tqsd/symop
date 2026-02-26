import unittest

import numpy as np

from symop.modes.transfer.time_delay import TimeDelay


class TestTimeDelay(unittest.TestCase):
    def test_signature(self):
        tf = TimeDelay(tau=1.25)
        self.assertEqual(tf.signature, ("time_delay", 1.25))

    def test_approx_signature_rounds(self):
        tf = TimeDelay(tau=1.23456789)
        got = tf.approx_signature(decimals=4, ignore_global_phase=True)
        self.assertEqual(got, ("time_delay_approx", round(1.23456789, 4)))

    def test_call_matches_definition(self):
        tau = 0.7
        tf = TimeDelay(tau=tau)

        w = np.array([0.0, 1.0, -2.5, 10.0], dtype=float)
        out = tf(w)

        expected = np.exp(-1.0j * w * tau).astype(complex)
        np.testing.assert_allclose(out, expected)

        self.assertTrue(np.iscomplexobj(out))
        self.assertEqual(out.shape, w.shape)

    def test_call_unit_magnitude(self):
        tf = TimeDelay(tau=2.1)
        w = np.linspace(-100.0, 100.0, 1001, dtype=float)
        out = tf(w)

        np.testing.assert_allclose(np.abs(out), np.ones_like(w), rtol=0.0, atol=1e-13)

    def test_call_accepts_python_sequence(self):
        tf = TimeDelay(tau=0.5)
        out = tf([0.0, 1.0, 2.0])

        w = np.array([0.0, 1.0, 2.0], dtype=float)
        expected = np.exp(-1.0j * w * 0.5).astype(complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3,))

    def test_call_empty_input(self):
        tf = TimeDelay(tau=0.1)
        w = np.array([], dtype=float)
        out = tf(w)

        self.assertEqual(out.shape, (0,))
        self.assertTrue(np.iscomplexobj(out))

    def test_call_raises_on_nonfinite_tau(self):
        tf_nan = TimeDelay(tau=float("nan"))
        with self.assertRaises(ValueError):
            _ = tf_nan([0.0, 1.0])

        tf_inf = TimeDelay(tau=float("inf"))
        with self.assertRaises(ValueError):
            _ = tf_inf([0.0, 1.0])

    def test_call_does_not_modify_input(self):
        tf = TimeDelay(tau=0.3)
        w = np.array([1.0, 2.0, 3.0], dtype=float)
        w_copy = w.copy()

        _ = tf(w)
        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
