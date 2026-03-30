import unittest

import numpy as np

from symop.modes.transfer.gaussian.time_delay import TimeDelay


class TestTimeDelay(unittest.TestCase):
    def test_call_matches_formula(self) -> None:
        tau = 2.5
        transfer = TimeDelay(tau=tau)
        w = np.array([-1.0, 0.0, 3.0], dtype=float)

        out = transfer(w)
        expected = np.exp(-1j * w * tau)

        np.testing.assert_allclose(out, expected)

    def test_invalid_tau_raises(self) -> None:
        with self.assertRaises(ValueError):
            TimeDelay(tau=float("inf"))
