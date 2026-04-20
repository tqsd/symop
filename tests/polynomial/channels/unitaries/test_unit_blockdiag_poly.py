from __future__ import annotations

import unittest

import numpy as np

from symop.polynomial.channels.unitaries.blockdiag import (
    block_diag,
    embed_1,
    embed_2,
    embed_u,
)


class TestBlockDiag(unittest.TestCase):
    def test_block_diag_with_no_blocks_returns_empty_matrix(self):
        U = block_diag()

        self.assertEqual(U.shape, (0, 0))

    def test_block_diag_with_one_block_returns_same_block(self):
        A = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)

        U = block_diag(A)

        self.assertTrue(np.allclose(U, A))

    def test_block_diag_with_two_blocks(self):
        A = np.asarray([[1.0]], dtype=np.complex128)
        B = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.complex128)

        U = block_diag(A, B)
        expected = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 3.0],
                [0.0, 4.0, 5.0],
            ],
            dtype=np.complex128,
        )

        self.assertTrue(np.allclose(U, expected))

    def test_block_diag_raises_for_non_square_block(self):
        A = np.asarray([[1.0, 2.0, 3.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            block_diag(A)


class TestEmbed1(unittest.TestCase):
    def test_embed_1_replaces_single_diagonal_entry(self):
        U = embed_1(n=3, i=1, u1=1j)
        expected = np.eye(3, dtype=np.complex128)
        expected[1, 1] = 1j

        self.assertTrue(np.allclose(U, expected))

    def test_embed_1_raises_for_invalid_n(self):
        with self.assertRaises(ValueError):
            embed_1(n=0, i=0, u1=1.0)

    def test_embed_1_raises_for_index_out_of_range(self):
        with self.assertRaises(ValueError):
            embed_1(n=3, i=3, u1=1.0)


class TestEmbed2(unittest.TestCase):
    def test_embed_2_embeds_block_into_identity(self):
        U2 = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        U = embed_2(n=4, i=1, j=3, U2=U2)

        expected = np.eye(4, dtype=np.complex128)
        expected[np.ix_([1, 3], [1, 3])] = U2

        self.assertTrue(np.allclose(U, expected))

    def test_embed_2_raises_for_invalid_n(self):
        U2 = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            embed_2(n=1, i=0, j=1, U2=U2)

    def test_embed_2_raises_for_equal_indices(self):
        U2 = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            embed_2(n=3, i=1, j=1, U2=U2)

    def test_embed_2_raises_for_indices_out_of_range(self):
        U2 = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            embed_2(n=3, i=0, j=3, U2=U2)

    def test_embed_2_raises_for_wrong_shape(self):
        U2 = np.eye(3, dtype=np.complex128)

        with self.assertRaises(ValueError):
            embed_2(n=4, i=0, j=1, U2=U2)


class TestEmbedU(unittest.TestCase):
    def test_embed_u_with_empty_indices_returns_identity(self):
        U = embed_u(n=4, indices=(), Uk=np.eye(0, dtype=np.complex128))

        self.assertTrue(np.allclose(U, np.eye(4, dtype=np.complex128)))

    def test_embed_u_embeds_square_block(self):
        Uk = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        U = embed_u(n=4, indices=(0, 2), Uk=Uk)

        expected = np.eye(4, dtype=np.complex128)
        expected[np.ix_([0, 2], [0, 2])] = Uk

        self.assertTrue(np.allclose(U, expected))

    def test_embed_u_raises_for_invalid_n(self):
        with self.assertRaises(ValueError):
            embed_u(n=0, indices=(0,), Uk=np.eye(1, dtype=np.complex128))

    def test_embed_u_raises_for_duplicate_indices(self):
        with self.assertRaises(ValueError):
            embed_u(n=3, indices=(1, 1), Uk=np.eye(2, dtype=np.complex128))

    def test_embed_u_raises_for_indices_out_of_range(self):
        with self.assertRaises(ValueError):
            embed_u(n=3, indices=(0, 3), Uk=np.eye(2, dtype=np.complex128))

    def test_embed_u_raises_for_non_square_matrix(self):
        Uk = np.asarray([[1.0, 2.0, 3.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            embed_u(n=3, indices=(0,), Uk=Uk)

    def test_embed_u_raises_for_size_mismatch(self):
        Uk = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            embed_u(n=4, indices=(0, 1, 2), Uk=Uk)

    def test_embed_u_optional_unitary_check_passes_for_unitary(self):
        Uk = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        U = embed_u(n=3, indices=(0, 2), Uk=Uk, check_unitary=True)

        expected = np.eye(3, dtype=np.complex128)
        expected[np.ix_([0, 2], [0, 2])] = Uk
        self.assertTrue(np.allclose(U, expected))

    def test_embed_u_optional_unitary_check_raises_for_non_unitary(self):
        Uk = np.asarray([[2.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            embed_u(n=2, indices=(1,), Uk=Uk, check_unitary=True)
