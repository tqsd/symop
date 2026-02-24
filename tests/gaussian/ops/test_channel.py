from __future__ import annotations

import unittest

import numpy as np

from symop_proto.gaussian.ops.channel import embed_subset_affine


class TestEmbedSubsetAffine(unittest.TestCase):
    def test_embed_identity_local_returns_identity_global(self) -> None:
        n = 2
        idx = [1]
        X = np.eye(2, dtype=complex)
        Y = np.zeros((2, 2), dtype=complex)

        X_full, Y_full, d0_full = embed_subset_affine(n, idx, X, Y)

        self.assertEqual(X_full.shape, (4, 4))
        self.assertEqual(Y_full.shape, (4, 4))
        self.assertEqual(d0_full.shape, (4,))

        self.assertTrue(np.allclose(X_full, np.eye(4, dtype=complex)))
        self.assertTrue(np.allclose(Y_full, np.zeros((4, 4), dtype=complex)))
        self.assertTrue(np.allclose(d0_full, np.zeros((4,), dtype=complex)))

    def test_embed_nontrivial_local_block_only_affects_selected_mode(
        self,
    ) -> None:
        n = 2
        idx = [1]

        # Ladder ordering r = (a0, a1, a0_dag, a1_dag)
        # Selected mode is 1, so affected indices are [1, 3]
        X = np.array(
            [[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=complex
        )
        Y = np.array(
            [[0.1 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.2 + 0.0j]], dtype=complex
        )
        d0 = np.array([5.0 + 0.0j, 7.0 + 0.0j], dtype=complex)

        X_full, Y_full, d0_full = embed_subset_affine(n, idx, X, Y, d0)

        expected_X = np.eye(4, dtype=complex)
        expected_Y = np.zeros((4, 4), dtype=complex)
        expected_d0 = np.zeros((4,), dtype=complex)

        expected_X[np.ix_([1, 3], [1, 3])] = X
        expected_Y[np.ix_([1, 3], [1, 3])] = Y
        expected_d0[[1, 3]] = d0

        self.assertTrue(np.allclose(X_full, expected_X))
        self.assertTrue(np.allclose(Y_full, expected_Y))
        self.assertTrue(np.allclose(d0_full, expected_d0))

        # Ensure untouched indices remain identity / zero
        untouched = [0, 2]
        self.assertTrue(
            np.allclose(
                X_full[np.ix_(untouched, untouched)], np.eye(2, dtype=complex)
            )
        )
        self.assertTrue(
            np.allclose(
                Y_full[np.ix_(untouched, untouched)],
                np.zeros((2, 2), dtype=complex),
            )
        )
        self.assertTrue(np.allclose(d0_full[untouched], 0.0))

    def test_multi_index_subset_order_is_respected(self) -> None:
        n = 3
        idx = [2, 0]  # order matters for the local (2k)x(2k) block

        lad_idx = [2, 0, 5, 3]  # (a2, a0, a2_dag, a0_dag)

        X = np.arange(16, dtype=float).reshape(4, 4).astype(complex)
        Y = (np.arange(16, dtype=float).reshape(4, 4) * 0.01).astype(complex)

        X_full, Y_full, d0_full = embed_subset_affine(n, idx, X, Y)

        expected_X = np.eye(6, dtype=complex)
        expected_Y = np.zeros((6, 6), dtype=complex)
        expected_d0 = np.zeros((6,), dtype=complex)

        expected_X[np.ix_(lad_idx, lad_idx)] = X
        expected_Y[np.ix_(lad_idx, lad_idx)] = Y

        self.assertTrue(np.allclose(X_full, expected_X))
        self.assertTrue(np.allclose(Y_full, expected_Y))
        self.assertTrue(np.allclose(d0_full, expected_d0))

    def test_idx_empty_raises(self) -> None:
        n = 2
        X = np.eye(0, dtype=complex)
        Y = np.zeros((0, 0), dtype=complex)
        with self.assertRaises(ValueError):
            embed_subset_affine(n, [], X, Y)

    def test_idx_out_of_range_raises(self) -> None:
        n = 2
        X = np.eye(2, dtype=complex)
        Y = np.zeros((2, 2), dtype=complex)
        with self.assertRaises(IndexError):
            embed_subset_affine(n, [2], X, Y)

    def test_idx_duplicates_raises(self) -> None:
        n = 2
        X = np.eye(4, dtype=complex)
        Y = np.zeros((4, 4), dtype=complex)
        with self.assertRaises(ValueError):
            embed_subset_affine(n, [1, 1], X, Y)

    def test_shape_mismatch_raises(self) -> None:
        n = 2
        idx = [1]  # k=1, so k2=2
        X_bad = np.eye(4, dtype=complex)  # should be (2,2)
        Y = np.zeros((2, 2), dtype=complex)
        with self.assertRaises(ValueError):
            embed_subset_affine(n, idx, X_bad, Y)

        X = np.eye(2, dtype=complex)
        Y_bad = np.zeros((4, 4), dtype=complex)
        with self.assertRaises(ValueError):
            embed_subset_affine(n, idx, X, Y_bad)

        d0_bad = np.zeros((4,), dtype=complex)
        with self.assertRaises(ValueError):
            embed_subset_affine(n, idx, X, Y, d0_bad)

    def test_check_finite_rejects_nan_inf(self) -> None:
        n = 2
        idx = [0]  # k2=2

        X = np.eye(2, dtype=complex)
        Y = np.zeros((2, 2), dtype=complex)

        X_nan = X.copy()
        X_nan[0, 0] = np.nan + 0.0j
        with self.assertRaises(ValueError):
            embed_subset_affine(n, idx, X_nan, Y, check_finite=True)

        Y_inf = Y.copy()
        Y_inf[1, 1] = np.inf + 0.0j
        with self.assertRaises(ValueError):
            embed_subset_affine(n, idx, X, Y_inf, check_finite=True)

        d0 = np.zeros((2,), dtype=complex)
        d0[0] = np.nan + 0.0j
        with self.assertRaises(ValueError):
            embed_subset_affine(n, idx, X, Y, d0, check_finite=True)

    def test_check_finite_false_allows_nan_inf(self) -> None:
        n = 2
        idx = [0]
        X = np.eye(2, dtype=complex)
        Y = np.zeros((2, 2), dtype=complex)

        X[0, 0] = np.nan + 0.0j
        Y[1, 1] = np.inf + 0.0j
        d0 = np.array([np.nan + 0.0j, 0.0 + 0.0j], dtype=complex)

        X_full, Y_full, d0_full = embed_subset_affine(
            n, idx, X, Y, d0, check_finite=False
        )
        self.assertEqual(X_full.shape, (4, 4))
        self.assertEqual(Y_full.shape, (4, 4))
        self.assertEqual(d0_full.shape, (4,))


if __name__ == "__main__":
    unittest.main()
