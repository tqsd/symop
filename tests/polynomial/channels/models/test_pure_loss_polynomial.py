from __future__ import annotations

import unittest

import numpy as np

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.polynomial.channels.models.pure_loss import (
    PureLossSpec,
    _require_eta,
    pure_loss_densitypoly,
    pure_loss_densitypoly_by_mode,
    pure_loss_densitypoly_many,
)

from tests.polynomial.state._builders import make_test_mode


class TestPureLossHelpers(unittest.TestCase):
    def test_require_eta_accepts_closed_interval(self):
        self.assertEqual(_require_eta(0.0), 0.0)
        self.assertEqual(_require_eta(1.0), 1.0)
        self.assertEqual(_require_eta(0.3), 0.3)

    def test_require_eta_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            _require_eta(float("nan"))
        with self.assertRaises(ValueError):
            _require_eta(float("inf"))
        with self.assertRaises(ValueError):
            _require_eta(-0.1)
        with self.assertRaises(ValueError):
            _require_eta(1.1)


class TestPureLossChannels(unittest.TestCase):
    def test_pure_loss_eta_one_is_noop(self):
        signal = make_test_mode(name="a", path="p0")
        env = make_test_mode(name="env", path="env0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = pure_loss_densitypoly(
            rho,
            signal_mode=signal,
            env_mode=env,
            eta=1.0,
        )

        self.assertEqual(result, rho.combine_like_terms())

    def test_pure_loss_eta_zero_returns_trace_normalized_output(self):
        signal = make_test_mode(name="a", path="p0")
        env = make_test_mode(name="env", path="env0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = pure_loss_densitypoly(
            rho,
            signal_mode=signal,
            env_mode=env,
            eta=0.0,
            normalize_trace=True,
        )

        self.assertEqual(result.trace(), 1.0 + 0.0j)

    def test_pure_loss_many_with_identity_specs_is_noop(self):
        signal_a = make_test_mode(name="a", path="p0")
        signal_b = make_test_mode(name="b", path="p1")
        env_a = make_test_mode(name="env_a", path="env0")
        env_b = make_test_mode(name="env_b", path="env1")
        ket = KetPoly.from_ops(creators=(signal_a.cre, signal_b.cre), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = pure_loss_densitypoly_many(
            rho,
            specs=(
                PureLossSpec(signal_mode=signal_a, env_mode=env_a, eta=1.0),
                PureLossSpec(signal_mode=signal_b, env_mode=env_b, eta=1.0),
            ),
        )

        self.assertEqual(result, rho.combine_like_terms())

    def test_pure_loss_by_mode_uses_mappings(self):
        signal = make_test_mode(name="a", path="p0")
        env = make_test_mode(name="env", path="env0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        result = pure_loss_densitypoly_by_mode(
            rho,
            eta_by_mode={signal: 1.0},
            env_by_signal_mode={signal: env},
        )

        self.assertEqual(result, rho.combine_like_terms())

    def test_pure_loss_by_mode_raises_for_missing_env(self):
        signal = make_test_mode(name="a", path="p0")
        ket = KetPoly.from_ops(creators=(signal.cre,), annihilators=(), coeff=1.0)
        rho = DensityPoly.pure(ket)

        with self.assertRaises(KeyError):
            pure_loss_densitypoly_by_mode(
                rho,
                eta_by_mode={signal: 1.0},
                env_by_signal_mode={},
            )
