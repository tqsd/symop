# tests/test_unitaries_unittest.py
from __future__ import annotations
import math
import numpy as np

from symop_proto.rewrites.functions.unitaries import (
    identity_unitary,
    phase_unitary,
    beamsplitter_unitary,
    pol_rotation_unitary,
)

from tests.utils.case import ExtendedTestCase


class TestUnitaries(ExtendedTestCase):
    def test_identity_unitary(self):
        for n in (0, 1, 3):
            U = identity_unitary(n)
            self.assertArrayAllClose(U, np.eye(n, dtype=np.complex128))
            self.assertUnitary(U)
        with self.assertRaises(ValueError):
            identity_unitary(-1)

    def test_phase_unitary_is_unitary_and_value(self):
        phi = 0.37
        U = phase_unitary(phi)
        # shape check via array equality
        self.assertArrayAllClose(U.shape, (1, 1))
        self.assertUnitary(U)
        self.assertComplexAlmostEqual(U[0, 0], np.exp(1j * phi))

    def test_beamsplitter_unitary_is_unitary_and_5050_values(self):
        th = math.pi / 4  # 50:50
        U = beamsplitter_unitary(th, phi=0.0)
        self.assertArrayAllClose(U.shape, (2, 2))
        self.assertUnitary(U)
        s2 = 1 / math.sqrt(2)
        expected = np.array([[s2, s2], [-s2, s2]], dtype=np.complex128)
        self.assertArrayAllClose(U, expected, atol=1e-12, rtol=1e-12)

    def test_pol_rotation_unitary_is_unitary_hwp_swap_at_45deg(self):
        # Half-wave plate (chi = pi) at 45° should swap H<->V up to phases.
        th = math.pi / 4
        chi = math.pi
        U = pol_rotation_unitary(th, chi)
        self.assertArrayAllClose(U.shape, (2, 2))
        self.assertUnitary(U)
        # Magnitudes should match a swap matrix [[0,1],[1,0]] (phases ignored here).
        mags = np.abs(U)
        self.assertArrayAllClose(mags, np.array([[0.0, 1.0], [1.0, 0.0]]))
