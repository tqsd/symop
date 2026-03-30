import unittest

import numpy as np

from symop.modes.transfer.cascade import Cascade
from symop.modes.transfer.gaussian.constant_phase import ConstantPhase
from symop.modes.transfer.gaussian.lowpass import GaussianLowpass


class TestCascade(unittest.TestCase):
    def test_call_multiplies_parts(self) -> None:
        a = ConstantPhase(phi0=0.3)
        b = GaussianLowpass(w0=0.0, sigma_w=1.0)
        cascade = Cascade(parts=(a, b))

        w = np.array([-1.0, 0.0, 1.0], dtype=float)

        out = cascade(w)
        expected = a(w) * b(w)

        np.testing.assert_allclose(out, expected)

    def test_signature_depends_on_order(self) -> None:
        a = ConstantPhase(phi0=0.3)
        b = GaussianLowpass(w0=0.0, sigma_w=1.0)

        c1 = Cascade(parts=(a, b))
        c2 = Cascade(parts=(b, a))

        self.assertNotEqual(c1.signature, c2.signature)
