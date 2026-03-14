import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope


class TestGaussianMixtureEnvelope(unittest.TestCase):
    def setUp(self):
        self.g1 = GaussianEnvelope(omega0=3.0, sigma=0.7, tau=0.0, phi0=0.0)
        self.g2 = GaussianEnvelope(omega0=3.0, sigma=0.9, tau=1.0, phi0=0.2)

    def test_frozen_dataclass(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.5 + 0.1j], dtype=complex),
        )
        with self.assertRaises(FrozenInstanceError):
            mix.report = None  # type: ignore[misc]

    def test_rejects_empty_components(self):
        with self.assertRaises(ValueError):
            GaussianMixtureEnvelope(
                components=(),
                weights=np.array([], dtype=complex),
            )

    def test_rejects_weight_shape(self):
        with self.assertRaises(ValueError):
            GaussianMixtureEnvelope(
                components=(self.g1, self.g2),
                weights=np.zeros((2, 1), dtype=complex),
            )

    def test_rejects_weight_length_mismatch(self):
        with self.assertRaises(ValueError):
            GaussianMixtureEnvelope(
                components=(self.g1, self.g2),
                weights=np.array([1.0 + 0.0j], dtype=complex),
            )

    def test_normalizes_on_construction(self):
        w = np.array([2.0 + 0.0j, 0.3 + 0.4j], dtype=complex)
        mix = GaussianMixtureEnvelope(components=(self.g1, self.g2), weights=w)

        ov = mix.overlap(mix)
        self.assertAlmostEqual(float(np.real(ov)), 1.0, places=10)
        self.assertAlmostEqual(float(np.imag(ov)), 0.0, places=10)

    def test_time_eval_shape_and_complex(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.2 + 0.0j], dtype=complex),
        )
        t = np.linspace(-2.0, 3.0, 501)
        z = mix.time_eval(t)
        self.assertTrue(np.iscomplexobj(z))
        self.assertEqual(z.shape, t.shape)

    def test_freq_eval_shape_and_complex(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.2 + 0.0j], dtype=complex),
        )
        w = np.linspace(-10.0, 10.0, 401)
        Z = mix.freq_eval(w)
        self.assertTrue(np.iscomplexobj(Z))
        self.assertEqual(Z.shape, w.shape)

    def test_delayed_shifts_component_taus(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.2 + 0.0j], dtype=complex),
        )
        dt = 0.75
        mix2 = mix.delayed(dt)

        self.assertAlmostEqual(
            mix2.components[0].tau, self.g1.tau + dt, places=14
        )
        self.assertAlmostEqual(
            mix2.components[1].tau, self.g2.tau + dt, places=14
        )

        # Overlap between mix and delayed version should generally be < 1
        ov = mix.overlap(mix2)
        self.assertLessEqual(abs(ov), 1.0 + 1e-10)

    def test_phased_applies_global_phase(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.2 + 0.1j], dtype=complex),
        )
        dphi = 0.9
        mix2 = mix.phased(dphi)

        # Since both are normalized, overlap should be exp(i dphi) up to numerical error
        ov = mix.overlap(mix2)
        self.assertAlmostEqual(abs(ov), 1.0, places=10)

        expected = complex(np.cos(dphi), np.sin(dphi))
        self.assertAlmostEqual(ov, expected, places=10)

    def test_center_and_scale_returns_finite(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.2 + 0.1j], dtype=complex),
        )
        c, s = mix.center_and_scale()
        self.assertTrue(np.isfinite(c))
        self.assertTrue(np.isfinite(s))
        self.assertGreater(s, 0.0)

    def test_overlap_closed_form_with_gaussian_matches_sum(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([0.6 + 0.0j, 0.2 - 0.3j], dtype=complex),
        )
        h = GaussianEnvelope(omega0=3.0, sigma=0.8, tau=0.3, phi0=-0.1)

        # closed-form formula: sum_i conj(c_i) <g_i, h>
        ref = np.conjugate(mix.weights[0]) * self.g1.overlap_gaussian_closed(
            h
        ) + np.conjugate(mix.weights[1]) * self.g2.overlap_gaussian_closed(h)
        got = mix.overlap(h)
        self.assertAlmostEqual(got, ref, places=10)

    def test_overlap_closed_form_mix_mix_matches_double_sum(self):
        mix1 = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([0.7 + 0.0j, -0.1 + 0.2j], dtype=complex),
        )
        h1 = GaussianEnvelope(omega0=3.0, sigma=0.75, tau=-0.2, phi0=0.05)
        h2 = GaussianEnvelope(omega0=3.0, sigma=0.95, tau=1.4, phi0=-0.15)
        mix2 = GaussianMixtureEnvelope(
            components=(h1, h2),
            weights=np.array([0.3 + 0.4j, 0.5 + 0.0j], dtype=complex),
        )

        # double sum: sum_{i,j} conj(c_i) d_j <g_i, h_j>
        ref = 0.0 + 0.0j
        for ci, gi in zip(mix1.weights, mix1.components, strict=True):
            for dj, hj in zip(mix2.weights, mix2.components, strict=True):
                ref += np.conjugate(ci) * dj * gi.overlap_gaussian_closed(hj)

        got = mix1.overlap(mix2)
        self.assertAlmostEqual(got, ref, places=10)

        # conjugate symmetry check
        got21 = mix2.overlap(mix1)
        self.assertAlmostEqual(got, np.conjugate(got21), places=10)

    def test_signature_structure(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.2 + 0.0j], dtype=complex),
        )
        sig = mix.signature
        self.assertEqual(sig[0], "gauss_mix")

    def test_latex_property(self):
        mix = GaussianMixtureEnvelope(
            components=(self.g1, self.g2),
            weights=np.array([1.0 + 0.0j, 0.2 + 0.0j], dtype=complex),
        )
        latex = mix.latex
        self.assertIsInstance(latex, str)
        self.assertIn(r"\zeta(t)", latex)


if __name__ == "__main__":
    unittest.main()
