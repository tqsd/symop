import unittest

import numpy as np

from symop.modes.transfer.quadratic_dispersion import QuadraticDispersion


class TestQuadraticDispersion(unittest.TestCase):
    def test_call_has_unit_modulus(self) -> None:
        transfer = QuadraticDispersion(beta2=1.5, w_ref=2.0)
        w = np.array([-1.0, 0.0, 2.0, 5.0], dtype=float)

        out = transfer(w)

        np.testing.assert_allclose(np.abs(out), np.ones_like(w))

    def test_reference_frequency_has_unit_phase(self) -> None:
        transfer = QuadraticDispersion(beta2=3.0, w_ref=4.0)

        out = transfer(np.array([4.0], dtype=float))

        np.testing.assert_allclose(out, np.array([1.0 + 0.0j]))
