import unittest

import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope


class TestGaussianMixtureEnvelope(unittest.TestCase):
    def assertComplexAlmostEqual(
        self,
        left: complex,
        right: complex,
        places: int = 10,
    ) -> None:
        self.assertAlmostEqual(left.real, right.real, places=places)
        self.assertAlmostEqual(left.imag, right.imag, places=places)

    def test_rejects_empty_components(self) -> None:
        with self.assertRaises(ValueError):
            GaussianMixtureEnvelope(
                components=(),
                weights=np.array([], dtype=complex),
            )

    def test_rejects_non_1d_weights(self) -> None:
        g = GaussianEnvelope(omega0=0.0, sigma=1.0, tau=0.0)
        with self.assertRaises(ValueError):
            GaussianMixtureEnvelope(
                components=(g,),
                weights=np.array([[1.0 + 0.0j]], dtype=complex),
            )

    def test_rejects_length_mismatch(self) -> None:
        g1 = GaussianEnvelope(omega0=0.0, sigma=1.0, tau=0.0)
        g2 = GaussianEnvelope(omega0=0.0, sigma=1.0, tau=1.0)

        with self.assertRaises(ValueError):
            GaussianMixtureEnvelope(
                components=(g1, g2),
                weights=np.array([1.0 + 0.0j], dtype=complex),
            )

    def test_single_component_is_normalized(self) -> None:
        g = GaussianEnvelope(omega0=0.0, sigma=1.0, tau=0.0)
        mix = GaussianMixtureEnvelope(
            components=(g,),
            weights=np.array([3.0 + 0.0j], dtype=complex),
        )

        self.assertAlmostEqual(abs(mix.weights[0]), 1.0, places=10)
        self.assertComplexAlmostEqual(mix.overlap(mix), 1.0 + 0.0j, places=10)

    def test_time_eval_is_linear_superposition(self) -> None:
        g1 = GaussianEnvelope(omega0=0.2, sigma=0.8, tau=-0.5, phi0=0.1)
        g2 = GaussianEnvelope(omega0=0.2, sigma=1.1, tau=0.7, phi0=-0.3)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, 0.25 - 0.5j], dtype=complex),
        )

        t = np.linspace(-3.0, 3.0, 128)
        actual = mix.time_eval(t)
        expected = (
            mix.weights[0] * g1.time_eval(t)
            + mix.weights[1] * g2.time_eval(t)
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_freq_eval_is_linear_superposition(self) -> None:
        g1 = GaussianEnvelope(omega0=0.2, sigma=0.8, tau=-0.5, phi0=0.1)
        g2 = GaussianEnvelope(omega0=0.2, sigma=1.1, tau=0.7, phi0=-0.3)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, 0.25 - 0.5j], dtype=complex),
        )

        w = np.linspace(-4.0, 4.0, 128)
        actual = mix.freq_eval(w)
        expected = (
            mix.weights[0] * g1.freq_eval(w)
            + mix.weights[1] * g2.freq_eval(w)
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_self_overlap_is_one(self) -> None:
        g1 = GaussianEnvelope(omega0=0.0, sigma=0.8, tau=-1.0)
        g2 = GaussianEnvelope(omega0=0.0, sigma=1.2, tau=1.0)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, 0.5 + 0.2j], dtype=complex),
        )

        self.assertComplexAlmostEqual(mix.overlap(mix), 1.0 + 0.0j, places=10)
        self.assertAlmostEqual(mix.norm2(), 1.0, places=10)

    def test_overlap_with_single_gaussian_matches_manual_sum(self) -> None:
        g1 = GaussianEnvelope(omega0=0.1, sigma=0.8, tau=-0.5, phi0=0.2)
        g2 = GaussianEnvelope(omega0=0.1, sigma=1.0, tau=0.9, phi0=-0.4)
        target = GaussianEnvelope(omega0=0.3, sigma=0.9, tau=0.2, phi0=0.7)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, -0.3 + 0.5j], dtype=complex),
        )

        actual = mix.overlap_gaussian_closed(target)
        expected = (
            np.conjugate(mix.weights[0]) * g1.overlap_gaussian_closed(target)
            + np.conjugate(mix.weights[1]) * g2.overlap_gaussian_closed(target)
        )

        self.assertComplexAlmostEqual(actual, expected, places=10)

    def test_overlap_with_other_mixture_matches_manual_sum(self) -> None:
        g1 = GaussianEnvelope(omega0=0.0, sigma=0.8, tau=-0.5)
        g2 = GaussianEnvelope(omega0=0.0, sigma=1.0, tau=0.9)
        h1 = GaussianEnvelope(omega0=0.2, sigma=0.7, tau=-0.2)
        h2 = GaussianEnvelope(omega0=0.2, sigma=1.3, tau=0.6)

        left = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, 0.2 - 0.1j], dtype=complex),
        )
        right = GaussianMixtureEnvelope(
            components=(h1, h2),
            weights=np.array([0.6 + 0.0j, -0.4 + 0.3j], dtype=complex),
        )

        actual = left.overlap_gaussian_closed(right)

        expected = 0.0 + 0.0j
        for ci, gi in zip(left.weights, left.components, strict=True):
            for dj, hj in zip(right.weights, right.components, strict=True):
                expected += np.conjugate(ci) * dj * gi.overlap_gaussian_closed(hj)

        self.assertComplexAlmostEqual(actual, expected, places=10)

    def test_delayed_shifts_component_centers(self) -> None:
        g1 = GaussianEnvelope(omega0=0.0, sigma=0.8, tau=-1.0)
        g2 = GaussianEnvelope(omega0=0.0, sigma=1.2, tau=1.0)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, 0.5 + 0.2j], dtype=complex),
        )
        delayed = mix.delayed(0.75)

        self.assertAlmostEqual(delayed.components[0].tau, -0.25)
        self.assertAlmostEqual(delayed.components[1].tau, 1.75)
        np.testing.assert_allclose(delayed.weights, mix.weights)

    def test_phased_preserves_norm(self) -> None:
        g1 = GaussianEnvelope(omega0=0.0, sigma=0.8, tau=-1.0)
        g2 = GaussianEnvelope(omega0=0.0, sigma=1.2, tau=1.0)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, 0.5 + 0.2j], dtype=complex),
        )
        phased = mix.phased(0.9)

        self.assertComplexAlmostEqual(phased.overlap(phased), 1.0 + 0.0j, places=10)

    def test_center_and_scale_returns_weighted_center_and_max_sigma(self) -> None:
        g1 = GaussianEnvelope(omega0=0.0, sigma=0.8, tau=-2.0)
        g2 = GaussianEnvelope(omega0=0.0, sigma=1.5, tau=4.0)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([2.0 + 0.0j, 1.0 + 0.0j], dtype=complex),
        )

        center, scale = mix.center_and_scale()

        self.assertGreater(center, -2.0)
        self.assertLess(center, 4.0)
        self.assertAlmostEqual(scale, 1.5)

    def test_approx_signature_can_ignore_global_phase(self) -> None:
        g1 = GaussianEnvelope(omega0=0.0, sigma=0.8, tau=-1.0, phi0=0.2)
        g2 = GaussianEnvelope(omega0=0.0, sigma=1.2, tau=1.0, phi0=-0.4)

        mix = GaussianMixtureEnvelope(
            components=(g1, g2),
            weights=np.array([1.0 + 0.0j, 0.5 + 0.2j], dtype=complex),
        )
        phased = mix.phased(0.7)

        self.assertEqual(
            mix.approx_signature(ignore_global_phase=True),
            phased.approx_signature(ignore_global_phase=True),
        )
