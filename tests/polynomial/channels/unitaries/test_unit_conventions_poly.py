from __future__ import annotations

import unittest

import numpy as np

from symop.polynomial.channels.unitaries.conventions import (
    require_dim,
    require_square,
    require_unitary_optional,
)


class TestRequireSquare(unittest.TestCase):
    def test_accepts_square_matrix(self):
        U = np.eye(2, dtype=np.complex128)

        require_square(U)

    def test_raises_for_non_2d_array(self):
        U = np.asarray([1.0, 2.0], dtype=np.complex128)

        with self.assertRaises(ValueError):
            require_square(U)

    def test_raises_for_non_square_matrix(self):
        U = np.asarray([[1.0, 2.0, 3.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            require_square(U)


class TestRequireDim(unittest.TestCase):
    def test_accepts_matching_shape(self):
        U = np.eye(3, dtype=np.complex128)

        require_dim(U, n=3)

    def test_raises_for_shape_mismatch(self):
        U = np.eye(2, dtype=np.complex128)

        with self.assertRaises(ValueError):
            require_dim(U, n=3)

    def test_raises_for_non_square_before_dim_check(self):
        U = np.asarray([[1.0, 2.0, 3.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            require_dim(U, n=1)


class TestRequireUnitaryOptional(unittest.TestCase):
    def test_noop_when_check_disabled(self):
        U = np.asarray([[2.0]], dtype=np.complex128)

        require_unitary_optional(U, check_unitary=False)

    def test_accepts_unitary_when_check_enabled(self):
        U = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

        require_unitary_optional(U, check_unitary=True)

    def test_raises_for_non_unitary_when_check_enabled(self):
        U = np.asarray([[2.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            require_unitary_optional(U, check_unitary=True)

    def test_raises_for_non_square_input(self):
        U = np.asarray([[1.0, 2.0, 3.0]], dtype=np.complex128)

        with self.assertRaises(ValueError):
            require_unitary_optional(U, check_unitary=True)
