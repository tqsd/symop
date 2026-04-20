from __future__ import annotations

import unittest

import numpy as np

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.polynomial.channels.models.amplifier import (
    AmplifierSpec,
    _require_gain,
    amplifier_bogoliubov_xy,
    amplifier_densitypoly,
    amplifier_densitypoly_by_mode,
    amplifier_densitypoly_many,
)

from tests.polynomial.state._builders import make_test_mode


class TestAmplifierHelpers(unittest.TestCase):
    def test_require_gain_accepts_one_and_above(self):
        self.assertEqual(_require_gain(1.0), 1.0)
        self.assertEqual(_require_gain(2.5), 2.5)

    def test_require_gain_rejects_nonfinite_or_less_than_one(self):
        with self.assertRaises(ValueError):
            _require_gain(float("inf"))
        with self.assertRaises(ValueError):
            _require_gain(float("nan"))
        with self.assertRaises(ValueError):
            _require_gain(0.999)

    def test_amplifier_bogoliubov_xy_for_gain_one(self):
        X, Y = amplifier_bogoliubov_xy(gain=1.0)

        self.assertTrue(np.array_equal(X, np.eye(2, dtype=np.complex128)))
        self.assertTrue(np.array_equal(Y, np.zeros((2, 2), dtype=np.complex128)))

    def test_amplifier_bogoliubov_xy_for_gain_two(self):
        X, Y = amplifier_bogoliubov_xy(gain=2.0)

        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(Y.shape, (2, 2))
        self.assertAlmostEqual(X[0, 0].real, np.sqrt(2.0), places=12)
        self.assertAlmostEqual(X[1, 1].real, np.sqrt(2.0), places=12)
        self.assertAlmostEqual(Y[0, 1].real, 1.0, places=12)
        self.assertAlmostEqual(Y[1, 0].real, 1.0, places=12)


class TestAmplifierChannels(unittest.TestCase):
    def test_amplifier_densitypoly_gain_one_is_noop(self):
        signal = make_test_mode(name="a", path="p0")
        env = make_test_mode(name="env", path="env0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = amplifier_densitypoly(
            rho,
            signal_mode=signal,
            env_mode=env,
            gain=1.0,
        )

        self.assertEqual(result, rho.combine_like_terms())

    def test_amplifier_densitypoly_normalizes_trace(self):
        signal = make_test_mode(name="a", path="p0")
        env = make_test_mode(name="env", path="env0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = amplifier_densitypoly(
            rho,
            signal_mode=signal,
            env_mode=env,
            gain=1.3,
            normalize_trace=True,
        )

        self.assertEqual(result.trace(), 1.0 + 0.0j)

    def test_amplifier_densitypoly_many_applies_specs(self):
        signal_a = make_test_mode(name="a", path="p0")
        signal_b = make_test_mode(name="b", path="p1")
        env_a = make_test_mode(name="env_a", path="env0")
        env_b = make_test_mode(name="env_b", path="env1")
        ket = KetPoly.from_ops(creators=(signal_a.cre, signal_b.cre), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = amplifier_densitypoly_many(
            rho,
            specs=(
                AmplifierSpec(signal_mode=signal_a, env_mode=env_a, gain=1.0),
                AmplifierSpec(signal_mode=signal_b, env_mode=env_b, gain=1.0),
            ),
        )

        self.assertEqual(result, rho.combine_like_terms())

    def test_amplifier_densitypoly_by_mode_uses_mappings(self):
        signal = make_test_mode(name="a", path="p0")
        env = make_test_mode(name="env", path="env0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = amplifier_densitypoly_by_mode(
            rho,
            gain_by_mode={signal: 1.0},
            env_by_signal_mode={signal: env},
        )

        self.assertEqual(result, rho.combine_like_terms())

    def test_amplifier_densitypoly_by_mode_raises_for_missing_env(self):
        signal = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        with self.assertRaises(KeyError):
            amplifier_densitypoly_by_mode(
                rho,
                gain_by_mode={signal: 1.0},
                env_by_signal_mode={},
            )
