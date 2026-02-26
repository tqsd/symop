import unittest

import numpy as np

import symop.modes.envelopes.base as base_mod
from symop.modes.envelopes.base import BaseEnvelope, _overlap_numeric
from symop.modes.protocols import SupportsOverlapWithGeneric


class _ConstEnvelope(BaseEnvelope):
    def __init__(self, value, *, center=0.0, scale=1.0):
        self._value = complex(value)
        self._center = float(center)
        self._scale = float(scale)

    def time_eval(self, t):
        t = np.asarray(t, dtype=float)
        return (np.ones_like(t, dtype=complex) * self._value).astype(complex)

    def freq_eval(self, w):
        w = np.asarray(w, dtype=float)
        return (np.ones_like(w, dtype=complex) * self._value).astype(complex)

    def delayed(self, dt):
        return _ConstEnvelope(
            self._value, center=self._center + float(dt), scale=self._scale
        )

    def phased(self, dphi):
        return _ConstEnvelope(
            self._value * np.exp(1j * float(dphi)),
            center=self._center,
            scale=self._scale,
        )

    @property
    def signature(self):
        return (
            "const_env",
            float(self._value.real),
            float(self._value.imag),
            self._center,
            self._scale,
        )

    def approx_signature(self, *, decimals=12, ignore_global_phase=False):
        r = round
        return (
            "const_env_approx",
            r(float(self._value.real), decimals),
            r(float(self._value.imag), decimals),
            r(float(self._center), decimals),
            r(float(self._scale), decimals),
        )

    def center_and_scale(self):
        return self._center, self._scale


class _GenericOverlapEnv(_ConstEnvelope, SupportsOverlapWithGeneric):
    def __init__(self, value, *, return_value, center=0.0, scale=1.0):
        super().__init__(value, center=center, scale=scale)
        self._return_value = complex(return_value)
        self.calls = []

    def overlap_with_generic(self, other):
        self.calls.append(("overlap_with_generic", other))
        return self._return_value


class TestOverlapNumeric(unittest.TestCase):
    def test_overlap_numeric_constant_functions(self):
        a = 2.0 + 1.0j
        b = -0.5 + 0.25j

        def f1(t):
            t = np.asarray(t, dtype=float)
            return (np.ones_like(t, dtype=complex) * a).astype(complex)

        def f2(t):
            t = np.asarray(t, dtype=float)
            return (np.ones_like(t, dtype=complex) * b).astype(complex)

        tmin = -3.0
        tmax = 5.0
        expected = np.conjugate(a) * b * (tmax - tmin)

        got = _overlap_numeric(f1, f2, tmin=tmin, tmax=tmax, n=4096)
        self.assertIsInstance(got, complex)
        self.assertAlmostEqual(got.real, expected.real, places=6)
        self.assertAlmostEqual(got.imag, expected.imag, places=6)

    def test_overlap_numeric_raises_on_nonfinite(self):
        def f_bad(_t):
            return np.array([1.0 + 0.0j, np.nan + 0.0j], dtype=complex)

        def f_ok(_t):
            return np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=complex)

        with self.assertRaises(ValueError):
            _ = _overlap_numeric(f_bad, f_ok, tmin=0.0, tmax=1.0, n=2)

        with self.assertRaises(ValueError):
            _ = _overlap_numeric(f_ok, f_bad, tmin=0.0, tmax=1.0, n=2)


class TestBaseEnvelopeOverlapDispatch(unittest.TestCase):
    def test_overlap_prefers_other_supports_generic(self):
        self_env = _ConstEnvelope(1.0 + 0.0j)
        other_env = _GenericOverlapEnv(1.0 + 0.0j, return_value=0.123 + 0.0j)

        ov = self_env.overlap(other_env)
        self.assertEqual(ov, 0.123 + 0.0j)
        self.assertEqual(len(other_env.calls), 1)
        self.assertIs(other_env.calls[0][1], self_env)

    def test_overlap_uses_self_supports_generic_if_other_does_not(self):
        self_env = _GenericOverlapEnv(1.0 + 0.0j, return_value=0.5 + 0.25j)
        other_env = _ConstEnvelope(2.0 + 0.0j)

        ov = self_env.overlap(other_env)
        self.assertEqual(ov, 0.5 + 0.25j)
        self.assertEqual(len(self_env.calls), 1)
        self.assertIs(self_env.calls[0][1], other_env)


class TestBaseEnvelopePlotPaths(unittest.TestCase):
    def test_plot_raises_without_matplotlib(self):
        env = _ConstEnvelope(1.0 + 0.0j)

        old_plt = base_mod.plt
        try:
            base_mod.plt = None
            with self.assertRaises(RuntimeError):
                _ = env.plot()
        finally:
            base_mod.plt = old_plt

    def test_plot_many_raises_on_empty(self):
        with self.assertRaises(ValueError):
            _ = BaseEnvelope.plot_many([])

    def test_plot_many_calls_plot_and_raises_without_matplotlib(self):
        envs = [_ConstEnvelope(1.0 + 0.0j), _ConstEnvelope(2.0 + 0.0j)]

        old_plt = base_mod.plt
        try:
            base_mod.plt = None
            with self.assertRaises(RuntimeError):
                _ = BaseEnvelope.plot_many(envs)
        finally:
            base_mod.plt = old_plt


if __name__ == "__main__":
    unittest.main()
