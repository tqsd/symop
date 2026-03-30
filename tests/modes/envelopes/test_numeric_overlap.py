import math
import unittest

import numpy as np

from symop.modes.envelopes.base import _overlap_numeric


class TestNumericOverlap(unittest.TestCase):
    def assertComplexAlmostEqual(
        self,
        left: complex,
        right: complex,
        places: int = 10,
    ) -> None:
        self.assertAlmostEqual(left.real, right.real, places=places)
        self.assertAlmostEqual(left.imag, right.imag, places=places)

    def test_overlap_of_identical_gaussians_is_close_to_expected(self) -> None:
        def f(t):
            return np.exp(-(t**2))

        value = _overlap_numeric(f, f, tmin=-10.0, tmax=10.0, n=2**14)

        expected = math.sqrt(math.pi / 2.0)
        self.assertAlmostEqual(value.imag, 0.0, places=10)
        self.assertAlmostEqual(value.real, expected, places=6)

    def test_overlap_is_conjugate_symmetric(self) -> None:
        def f1(t):
            return np.exp(-(t**2)) * np.exp(1j * 0.3 * t)

        def f2(t):
            return np.exp(-((t - 0.5) ** 2))

        left = _overlap_numeric(f1, f2, tmin=-10.0, tmax=10.0, n=2**14)
        right = _overlap_numeric(f2, f1, tmin=-10.0, tmax=10.0, n=2**14)

        self.assertComplexAlmostEqual(left, right.conjugate(), places=8)

    def test_overlap_with_zero_function_is_zero(self) -> None:
        def f1(t):
            return np.exp(-(t**2))

        def f0(t):
            return np.zeros_like(t, dtype=complex)

        value = _overlap_numeric(f1, f0, tmin=-10.0, tmax=10.0, n=2**12)
        self.assertComplexAlmostEqual(value, 0.0 + 0.0j, places=10)

    def test_non_finite_values_from_first_function_raise(self) -> None:
        def bad(t):
            y = np.ones_like(t, dtype=complex)
            y[3] = np.nan
            return y

        def good(t):
            return np.ones_like(t, dtype=complex)

        with self.assertRaises(ValueError):
            _overlap_numeric(bad, good, tmin=-1.0, tmax=1.0, n=16)

    def test_non_finite_values_from_second_function_raise(self) -> None:
        def good(t):
            return np.ones_like(t, dtype=complex)

        def bad(t):
            y = np.ones_like(t, dtype=complex)
            y[5] = np.inf
            return y

        with self.assertRaises(ValueError):
            _overlap_numeric(good, bad, tmin=-1.0, tmax=1.0, n=16)
