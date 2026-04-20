from __future__ import annotations

import math
import unittest

import numpy as np

from symop.polynomial.channels.unitaries.beamsplitter import (
    beamsplitter_u,
    loss_dilation_u,
)


class TestBeamsplitterUnitary(unittest.TestCase):
    def test_beamsplitter_u_has_correct_shape(self):
        U = beamsplitter_u(t=1.0, r=0.0)

        self.assertEqual(U.shape, (2, 2))

    def test_beamsplitter_u_identity_case(self):
        U = beamsplitter_u(t=1.0, r=0.0, phi_t=0.0, phi_r=0.0)

        self.assertTrue(np.allclose(U, np.eye(2, dtype=np.complex128)))

    def test_beamsplitter_u_is_unitary_for_physical_parameters(self):
        t = math.sqrt(0.7)
        r = math.sqrt(0.3)
        U = beamsplitter_u(t=t, r=r, phi_t=0.2, phi_r=-0.4)

        lhs = U.conjugate().T @ U
        self.assertTrue(np.allclose(lhs, np.eye(2, dtype=np.complex128)))

    def test_beamsplitter_u_matches_expected_formula(self):
        t = 0.6
        r = 0.8
        phi_t = 0.3
        phi_r = -0.2

        U = beamsplitter_u(t=t, r=r, phi_t=phi_t, phi_r=phi_r)

        et = complex(math.cos(phi_t), math.sin(phi_t))
        er = complex(math.cos(phi_r), math.sin(phi_r))
        expected = np.asarray(
            [
                [t * et, r * er],
                [-r * np.conjugate(er), t * np.conjugate(et)],
            ],
            dtype=np.complex128,
        )

        self.assertTrue(np.allclose(U, expected))


class TestLossDilationUnitary(unittest.TestCase):
    def test_loss_dilation_eta_one_is_identity(self):
        U = loss_dilation_u(eta=1.0)

        self.assertTrue(np.allclose(U, np.eye(2, dtype=np.complex128)))

    def test_loss_dilation_eta_zero_matches_full_loss_case(self):
        U = loss_dilation_u(eta=0.0)
        expected = beamsplitter_u(t=0.0, r=1.0, phi_t=0.0, phi_r=0.0)

        self.assertTrue(np.allclose(U, expected))

    def test_loss_dilation_is_unitary_for_valid_eta(self):
        U = loss_dilation_u(eta=0.25)

        lhs = U.conjugate().T @ U
        self.assertTrue(np.allclose(lhs, np.eye(2, dtype=np.complex128)))

    def test_loss_dilation_entries_match_sqrt_eta_convention(self):
        eta = 0.36
        U = loss_dilation_u(eta=eta)

        self.assertAlmostEqual(U[0, 0].real, math.sqrt(eta), places=12)
        self.assertAlmostEqual(U[0, 1].real, math.sqrt(1.0 - eta), places=12)
        self.assertAlmostEqual(U[1, 0].real, -math.sqrt(1.0 - eta), places=12)
        self.assertAlmostEqual(U[1, 1].real, math.sqrt(eta), places=12)
