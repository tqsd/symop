from __future__ import annotations

import math
import unittest

import numpy as np

from symop.polynomial.channels.unitaries.phase import phase_u


class TestPhaseUnitary(unittest.TestCase):
    def test_phase_u_has_correct_shape(self):
        U = phase_u(phi=0.0)

        self.assertEqual(U.shape, (1, 1))

    def test_phase_u_zero_is_identity(self):
        U = phase_u(phi=0.0)

        self.assertTrue(np.allclose(U, np.asarray([[1.0 + 0.0j]], dtype=np.complex128)))

    def test_phase_u_pi_is_minus_one(self):
        U = phase_u(phi=math.pi)

        self.assertAlmostEqual(U[0, 0].real, -1.0, places=12)
        self.assertAlmostEqual(U[0, 0].imag, 0.0, places=12)

    def test_phase_u_matches_exp_i_phi(self):
        phi = 0.7
        U = phase_u(phi=phi)
        expected = complex(math.cos(phi), math.sin(phi))

        self.assertAlmostEqual(U[0, 0].real, expected.real, places=12)
        self.assertAlmostEqual(U[0, 0].imag, expected.imag, places=12)

    def test_phase_u_is_unit_modulus(self):
        phi = -1.2
        U = phase_u(phi=phi)

        self.assertAlmostEqual(abs(U[0, 0]), 1.0, places=12)
