from __future__ import annotations

import unittest

import numpy as np

from symop.polynomial.channels.unitaries.swap import swap_u


class TestSwapUnitary(unittest.TestCase):
    def test_swap_u_matches_expected_matrix(self):
        U = swap_u()

        expected = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        self.assertTrue(np.allclose(U, expected))

    def test_swap_u_is_unitary(self):
        U = swap_u()

        lhs = U.conjugate().T @ U
        self.assertTrue(np.allclose(lhs, np.eye(2, dtype=np.complex128)))

    def test_swap_u_optional_unitary_check_passes(self):
        U = swap_u(check_unitary=True)

        expected = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        self.assertTrue(np.allclose(U, expected))
