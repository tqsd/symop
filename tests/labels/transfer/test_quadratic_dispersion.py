import unittest

import numpy as np

from symop.modes.transfer.quadratic_dispersion import QuadraticDispersion


class TestQuadraticDispersion(unittest.TestCase):
    def test_signature(self):
        tf = QuadraticDispersion(beta2=1.25, w_ref=-0.5)
        self.assertEqual(tf.signature, ("quad_dispersion", 1.25, -0.5))

    def test_approx_signature_rounds(self):
        tf = QuadraticDispersion(beta2=1.23456789, w_ref=9.87654321)
        got = tf.approx_signature(decimals=4, ignore_global_phase=True)
        self.assertEqual(
            got,
            (
                "quad_dispersion_approx",
                round(1.23456789, 4),
                round(9.87654321, 4),
            ),
        )

    def test_call_matches_definition(self):
        beta2 = 0.8
        w_ref = 2.0
        tf = QuadraticDispersion(beta2=beta2, w_ref=w_ref)

        w = np.array([w_ref, w_ref + 0.5, w_ref - 1.25], dtype=float)
        out = tf(w)

        dw = w - w_ref
        expected = np.exp(-1.0j * 0.5 * beta2 * dw * dw).astype(complex)
        np.testing.assert_allclose(out, expected)

        self.assertTrue(np.iscomplexobj(out))
        self.assertEqual(out.shape, w.shape)

    def test_call_unit_magnitude(self):
        tf = QuadraticDispersion(beta2=1.7, w_ref=-0.3)
        w = np.linspace(-5.0, 5.0, 201, dtype=float)
        out = tf(w)

        np.testing.assert_allclose(np.abs(out), np.ones_like(w), rtol=0.0, atol=1e-13)

    def test_call_even_symmetry_about_w_ref(self):
        beta2 = 0.9
        w_ref = 1.1
        tf = QuadraticDispersion(beta2=beta2, w_ref=w_ref)

        d = np.array([0.0, 0.2, 1.0, 2.5], dtype=float)
        w_left = w_ref - d
        w_right = w_ref + d

        out_left = tf(w_left)
        out_right = tf(w_right)

        np.testing.assert_allclose(out_left, out_right)

    def test_call_accepts_python_sequence(self):
        tf = QuadraticDispersion(beta2=0.5, w_ref=0.0)
        out = tf([0.0, 1.0, 2.0])

        w = np.array([0.0, 1.0, 2.0], dtype=float)
        expected = np.exp(-1.0j * 0.5 * 0.5 * w * w).astype(complex)
        np.testing.assert_allclose(out, expected)
        self.assertEqual(out.shape, (3,))

    def test_call_empty_input(self):
        tf = QuadraticDispersion(beta2=0.1, w_ref=0.0)
        w = np.array([], dtype=float)
        out = tf(w)

        self.assertEqual(out.shape, (0,))
        self.assertTrue(np.iscomplexobj(out))

    def test_call_raises_on_nonfinite_beta2(self):
        tf_nan = QuadraticDispersion(beta2=float("nan"), w_ref=0.0)
        with self.assertRaises(ValueError):
            _ = tf_nan([0.0, 1.0])

        tf_inf = QuadraticDispersion(beta2=float("inf"), w_ref=0.0)
        with self.assertRaises(ValueError):
            _ = tf_inf([0.0, 1.0])

    def test_call_raises_on_nonfinite_w_ref(self):
        tf_nan = QuadraticDispersion(beta2=0.1, w_ref=float("nan"))
        with self.assertRaises(ValueError):
            _ = tf_nan([0.0, 1.0])

        tf_inf = QuadraticDispersion(beta2=0.1, w_ref=float("inf"))
        with self.assertRaises(ValueError):
            _ = tf_inf([0.0, 1.0])

    def test_call_does_not_modify_input(self):
        tf = QuadraticDispersion(beta2=0.3, w_ref=1.0)
        w = np.array([1.0, 2.0, 3.0], dtype=float)
        w_copy = w.copy()

        _ = tf(w)
        np.testing.assert_array_equal(w, w_copy)


if __name__ == "__main__":
    unittest.main()
