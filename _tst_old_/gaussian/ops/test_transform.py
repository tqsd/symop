from __future__ import annotations

import unittest
import numpy as np

from symop_proto.gaussian.ops.transform import (
    ladder_change_of_basis,
    quadrature_to_ladder_affine,
    ladder_to_quadrature_affine,
)
import symop_proto.gaussian.ops.transform as transform


class TestLadderChangeOfBasis(unittest.TestCase):
    def test_shapes_and_inverse(self) -> None:
        k = 3
        T, Tinv = ladder_change_of_basis(k)

        self.assertEqual(T.shape, (2 * k, 2 * k))
        self.assertEqual(Tinv.shape, (2 * k, 2 * k))

        I = np.eye(2 * k, dtype=complex)
        self.assertTrue(np.allclose(T @ Tinv, I, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(Tinv @ T, I, atol=1e-12, rtol=0.0))

    def test_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            ladder_change_of_basis(0)
        with self.assertRaises(ValueError):
            ladder_change_of_basis(-1)


class TestQuadratureToLadderAffine(unittest.TestCase):
    def test_round_trip_identity(self) -> None:

        # right before the failing assertion
        k = 2
        k2 = 2 * k

        Xq = np.eye(k2, dtype=float)
        Yq = np.zeros((k2, k2), dtype=float)
        dq = np.array([0.1, -0.2, 0.3, 0.4], dtype=float)

        Xl, Yl, dl = quadrature_to_ladder_affine(Xq, Yq, dq)
        Xq2, Yq2, dq2 = ladder_to_quadrature_affine(Xl, Yl, dl)

        self.assertTrue(np.allclose(Xq2, Xq, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(Yq2, Yq, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(dq2, dq.astype(complex), atol=1e-12, rtol=0.0))

    def test_dq_none_returns_zero_complex(self) -> None:
        k = 1
        k2 = 2 * k
        Xq = np.eye(k2, dtype=float)
        Yq = np.zeros((k2, k2), dtype=float)

        Xl, Yl, dl = quadrature_to_ladder_affine(Xq, Yq, None)

        self.assertEqual(Xl.shape, (k2, k2))
        self.assertEqual(Yl.shape, (k2, k2))
        self.assertEqual(dl.shape, (k2,))
        self.assertEqual(dl.dtype, np.dtype(complex))
        self.assertTrue(
            np.allclose(dl, np.zeros((k2,), dtype=complex), atol=0.0, rtol=0.0)
        )

    def test_single_mode_rotation_becomes_phase_diagonal(self) -> None:
        k = 1
        theta = 0.37

        c = float(np.cos(theta))
        s = float(np.sin(theta))

        Xq = np.array([[c, -s], [s, c]], dtype=float)
        Yq = np.zeros((2, 2), dtype=float)
        dq = np.zeros((2,), dtype=float)

        Xl, Yl, dl = quadrature_to_ladder_affine(Xq, Yq, dq)

        expected = np.array(
            [
                [np.exp(-1j * theta), 0.0 + 0.0j],
                [0.0 + 0.0j, np.exp(1j * theta)],
            ],
            dtype=complex,
        )

        self.assertTrue(np.allclose(Yl, 0.0, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(dl, 0.0, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(Xl, expected, atol=1e-12, rtol=0.0))

    def test_shape_validation(self) -> None:
        # Xq not square
        Xq = np.zeros((2, 3), dtype=float)
        Yq = np.zeros((2, 2), dtype=float)
        with self.assertRaises(ValueError):
            quadrature_to_ladder_affine(Xq, Yq)

        # Yq wrong shape
        Xq = np.eye(2, dtype=float)
        Yq = np.zeros((4, 4), dtype=float)
        with self.assertRaises(ValueError):
            quadrature_to_ladder_affine(Xq, Yq)

        # odd dimension
        Xq = np.eye(3, dtype=float)
        Yq = np.eye(3, dtype=float)
        with self.assertRaises(ValueError):
            quadrature_to_ladder_affine(Xq, Yq)

        # dq wrong length
        Xq = np.eye(2, dtype=float)
        Yq = np.zeros((2, 2), dtype=float)
        dq = np.zeros((4,), dtype=float)
        with self.assertRaises(ValueError):
            quadrature_to_ladder_affine(Xq, Yq, dq)

    def test_finite_check_rejects_nan(self) -> None:
        Xq = np.eye(2, dtype=float)
        Yq = np.zeros((2, 2), dtype=float)
        Xq[0, 0] = np.nan  # type: ignore[assignment]

        with self.assertRaises(ValueError):
            quadrature_to_ladder_affine(Xq, Yq, None, check_finite=True)

        # but allowed if check_finite=False
        Xl, Yl, dl = quadrature_to_ladder_affine(Xq, Yq, None, check_finite=False)
        self.assertEqual(Xl.shape, (2, 2))
        self.assertEqual(Yl.shape, (2, 2))
        self.assertEqual(dl.shape, (2,))


class TestLadderToQuadratureAffine(unittest.TestCase):
    def test_round_trip_random(self) -> None:
        rng = np.random.default_rng(1234)
        k = 2
        k2 = 2 * k

        Xl = rng.normal(size=(k2, k2)) + 1j * rng.normal(size=(k2, k2))
        Yl = rng.normal(size=(k2, k2)) + 1j * rng.normal(size=(k2, k2))
        dl = rng.normal(size=(k2,)) + 1j * rng.normal(size=(k2,))

        Xq, Yq, dq = ladder_to_quadrature_affine(Xl, Yl, dl)
        Xl2, Yl2, dl2 = quadrature_to_ladder_affine(Xq, Yq, dq)

        self.assertTrue(np.allclose(Xl2, Xl, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(Yl2, Yl, atol=1e-12, rtol=0.0))
        self.assertTrue(np.allclose(dl2, dl, atol=1e-12, rtol=0.0))

    def test_shape_validation(self) -> None:
        Xl = np.zeros((2, 3), dtype=complex)
        Yl = np.zeros((2, 2), dtype=complex)
        with self.assertRaises(ValueError):
            ladder_to_quadrature_affine(Xl, Yl)

        Xl = np.eye(2, dtype=complex)
        Yl = np.zeros((4, 4), dtype=complex)
        with self.assertRaises(ValueError):
            ladder_to_quadrature_affine(Xl, Yl)

        Xl = np.eye(3, dtype=complex)
        Yl = np.eye(3, dtype=complex)
        with self.assertRaises(ValueError):
            ladder_to_quadrature_affine(Xl, Yl)

        Xl = np.eye(2, dtype=complex)
        Yl = np.zeros((2, 2), dtype=complex)
        dl = np.zeros((4,), dtype=complex)
        with self.assertRaises(ValueError):
            ladder_to_quadrature_affine(Xl, Yl, dl)

    def test_finite_check_rejects_inf(self) -> None:
        Xl = np.eye(2, dtype=complex)
        Yl = np.zeros((2, 2), dtype=complex)
        Yl[0, 1] = np.inf + 0j  # type: ignore[assignment]

        with self.assertRaises(ValueError):
            ladder_to_quadrature_affine(Xl, Yl, None, check_finite=True)

        Xq, Yq, dq = ladder_to_quadrature_affine(Xl, Yl, None, check_finite=False)
        self.assertEqual(Xq.shape, (2, 2))
        self.assertEqual(Yq.shape, (2, 2))
        self.assertEqual(dq.shape, (2,))


if __name__ == "__main__":
    unittest.main()
