# tests/modes/envelopes/test_filtered.py

import unittest
from dataclasses import dataclass

import numpy as np

from symop.modes.envelopes.filtered import FilteredEnvelope
from symop.modes.envelopes.gaussian import GaussianEnvelope


@dataclass(frozen=True)
class ConstantTransfer:
    value: complex

    def __call__(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        return np.full_like(w, fill_value=self.value, dtype=complex)

    @property
    def signature(self):
        return ("constant_transfer", float(self.value.real), float(self.value.imag))


@dataclass(frozen=True)
class LinearPhaseTransfer:
    delay: float

    def __call__(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        return np.exp(-1j * w * self.delay)

    @property
    def signature(self):
        return ("linear_phase_transfer", float(self.delay))


class TestFilteredEnvelope(unittest.TestCase):
    def assertComplexAlmostEqual(
        self,
        left: complex,
        right: complex,
        places: int = 8,
    ) -> None:
        self.assertAlmostEqual(left.real, right.real, places=places)
        self.assertAlmostEqual(left.imag, right.imag, places=places)

    def test_signature_contains_base_and_transfer(self) -> None:
        base = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0, phi0=0.2)
        transfer = ConstantTransfer(1.0 + 0.0j)
        env = FilteredEnvelope(base=base, transfer=transfer)

        self.assertEqual(
            env.signature,
            ("filtered", base.signature, transfer.signature),
        )

    def test_approx_signature_forwards_to_base(self) -> None:
        base = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0, phi0=0.2)
        transfer = ConstantTransfer(1.0 + 0.0j)
        env = FilteredEnvelope(base=base, transfer=transfer)

        self.assertEqual(
            env.approx_signature(ignore_global_phase=True),
            (
                "filtered_approx",
                base.approx_signature(ignore_global_phase=True),
                transfer.signature,
                int(env.n_fft),
                float(env.w_span_sigma),
            ),
        )

    def test_identity_transfer_preserves_norm(self) -> None:
        base = GaussianEnvelope(omega0=1.0, sigma=0.8, tau=-0.4, phi0=0.2)
        env = FilteredEnvelope(
            base=base,
            transfer=ConstantTransfer(1.0 + 0.0j),
            n_fft=2**12,
            w_span_sigma=10.0,
        )

        self.assertAlmostEqual(env.norm2(), 1.0, places=4)
        self.assertAlmostEqual(env.eta, 1.0, places=4)

    def test_constant_gain_is_renormalized(self) -> None:
        base = GaussianEnvelope(omega0=0.5, sigma=0.9, tau=0.3)
        env = FilteredEnvelope(
            base=base,
            transfer=ConstantTransfer(3.0 + 0.0j),
            n_fft=2**12,
            w_span_sigma=10.0,
        )

        self.assertAlmostEqual(env.norm2(), 1.0, places=4)
        self.assertAlmostEqual(env.eta, 9.0, places=3)

    def test_zero_transfer_raises_when_normalization_is_requested(self) -> None:
        base = GaussianEnvelope(omega0=0.5, sigma=0.9, tau=0.3)
        env = FilteredEnvelope(
            base=base,
            transfer=ConstantTransfer(0.0 + 0.0j),
            n_fft=2**10,
            w_span_sigma=8.0,
        )

        with self.assertRaises(ValueError):
            _ = env.freq_eval(np.linspace(-2.0, 2.0, 16))

    def test_freq_eval_with_identity_transfer_matches_base_spectrum(self) -> None:
        base = GaussianEnvelope(omega0=1.2, sigma=0.7, tau=-0.2, phi0=0.4)
        env = FilteredEnvelope(
            base=base,
            transfer=ConstantTransfer(1.0 + 0.0j),
            n_fft=2**12,
            w_span_sigma=10.0,
        )

        w = np.linspace(base.omega0 - 4.0, base.omega0 + 4.0, 128)
        actual = env.freq_eval(w)
        expected = base.freq_eval(w)

        np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-4)

    def test_delayed_returns_filtered_envelope_with_delayed_base(self) -> None:
        base = GaussianEnvelope(omega0=1.0, sigma=0.8, tau=0.5)
        transfer = ConstantTransfer(1.0 + 0.0j)
        env = FilteredEnvelope(
            base=base,
            transfer=transfer,
            n_fft=2**10,
            w_span_sigma=8.0,
        )

        delayed = env.delayed(1.25)

        self.assertAlmostEqual(delayed.base.tau, 1.75)
        self.assertEqual(delayed.transfer, transfer)
        self.assertEqual(delayed.n_fft, env.n_fft)
        self.assertEqual(delayed.w_span_sigma, env.w_span_sigma)

    def test_phased_returns_filtered_envelope_with_phased_base(self) -> None:
        base = GaussianEnvelope(omega0=1.0, sigma=0.8, tau=0.5, phi0=0.2)
        transfer = ConstantTransfer(1.0 + 0.0j)
        env = FilteredEnvelope(
            base=base,
            transfer=transfer,
            n_fft=2**10,
            w_span_sigma=8.0,
        )

        phased = env.phased(0.7)

        self.assertAlmostEqual(phased.base.phi0, 0.9)
        self.assertEqual(phased.transfer, transfer)

    def test_overlap_with_generic_self_is_one_for_identity_transfer(self) -> None:
        base = GaussianEnvelope(omega0=1.0, sigma=0.8, tau=-0.4, phi0=0.2)
        env = FilteredEnvelope(
            base=base,
            transfer=ConstantTransfer(1.0 + 0.0j),
            n_fft=2**12,
            w_span_sigma=10.0,
        )

        overlap = env.overlap_with_generic(env)
        self.assertAlmostEqual(overlap.real, 1.0, places=4)
        self.assertAlmostEqual(overlap.imag, 0.0, places=4)

    def test_linear_phase_transfer_preserves_norm(self) -> None:
        base = GaussianEnvelope(omega0=0.7, sigma=1.0, tau=0.1)
        env = FilteredEnvelope(
            base=base,
            transfer=LinearPhaseTransfer(0.8),
            n_fft=2**12,
            w_span_sigma=10.0,
        )

        self.assertAlmostEqual(env.norm2(), 1.0, places=4)

    def test_center_and_scale_is_finite(self) -> None:
        base = GaussianEnvelope(omega0=0.7, sigma=1.0, tau=0.1)
        env = FilteredEnvelope(
            base=base,
            transfer=ConstantTransfer(1.0 + 0.0j),
            n_fft=2**12,
            w_span_sigma=10.0,
        )

        center, scale = env.center_and_scale()

        self.assertTrue(np.isfinite(center))
        self.assertTrue(np.isfinite(scale))
        self.assertGreater(scale, 0.0)
