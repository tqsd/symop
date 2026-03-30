import math
import unittest

import numpy as np

from symop.modes.transfer.gaussian.constant_phase import ConstantPhase


class TestConstantPhase(unittest.TestCase):
    def test_call_returns_constant_phase(self) -> None:
        transfer = ConstantPhase(phi0=math.pi / 3.0)
        w = np.array([-2.0, 0.0, 5.0], dtype=float)

        out = transfer(w)
        expected = np.exp(1j * math.pi / 3.0) * np.ones_like(w, dtype=complex)

        np.testing.assert_allclose(out, expected)

    def test_invalid_phi0_raises(self) -> None:
        with self.assertRaises(ValueError):
            ConstantPhase(phi0=float("nan"))

    def test_ignore_global_phase_in_approx_signature(self) -> None:
        transfer = ConstantPhase(phi0=1.23456789)

        sig = transfer.approx_signature(decimals=4, ignore_global_phase=True)

        self.assertEqual(sig, ("const_phase_approx", 0.0))
