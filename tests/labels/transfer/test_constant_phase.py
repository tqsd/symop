import unittest

import numpy as np

from symop.modes.transfer.constant_phase import ConstantPhase


class TestConstantPhase(unittest.TestCase):
    def test_signature(self):
        tf = ConstantPhase(phi0=0.125)
        self.assertEqual(tf.signature, ("const_phase", 0.125))

    def test_approx_signature_rounds(self):
        tf = ConstantPhase(phi0=1.23456789)
        got = tf.approx_signature(decimals=3, ignore_global_phase=False)
        self.assertEqual(got, ("const_phase_approx", round(1.23456789, 3)))

    def test_approx_signature_ignore_global_phase(self):
        tf = ConstantPhase(phi0=1.234)
        got = tf.approx_signature(decimals=12, ignore_global_phase=True)
        self.assertEqual(got, ("const_phase_approx", 0.0))

    def test_call_returns_constant_complex_array(self):
        phi0 = 0.7
        tf = ConstantPhase(phi0=phi0)

        w = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        out = tf(w)

        expected = np.exp(1j * phi0) * np.ones_like(w, dtype=complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, w.shape)
        self.assertTrue(np.iscomplexobj(out))

    def test_call_accepts_python_sequence(self):
        phi0 = -0.2
        tf = ConstantPhase(phi0=phi0)

        out = tf([10.0, 20.0, 30.0])
        expected = np.exp(1j * phi0) * np.ones(3, dtype=complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3,))

    def test_call_empty_input(self):
        tf = ConstantPhase(phi0=0.1)
        w = np.array([], dtype=float)
        out = tf(w)
        self.assertEqual(out.shape, (0,))
        self.assertTrue(np.iscomplexobj(out))

    def test_call_raises_on_nonfinite_phi0(self):
        tf = ConstantPhase(phi0=float("nan"))
        with self.assertRaises(ValueError):
            _ = tf([0.0, 1.0])

        tf2 = ConstantPhase(phi0=float("inf"))
        with self.assertRaises(ValueError):
            _ = tf2([0.0, 1.0])

    def test_call_does_not_modify_input(self):
        tf = ConstantPhase(phi0=0.3)
        w = np.array([1.0, 2.0, 3.0], dtype=float)
        w_copy = w.copy()

        _ = tf(w)
        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
