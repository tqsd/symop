import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from symop.modes.labels.polarization import PolarizationLabel


class TestPolarizationLabel(unittest.TestCase):
    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            PolarizationLabel((0.0 + 0.0j, 0.0 + 0.0j))

    def test_normalization(self):
        p = PolarizationLabel((2.0 + 0.0j, 0.0 + 0.0j))
        a, b = p.jones
        self.assertAlmostEqual(abs(a) ** 2 + abs(b) ** 2, 1.0, places=14)
        self.assertEqual(p.jones, (1.0 + 0.0j, 0.0 + 0.0j))

    def test_global_phase_canonicalization(self):
        s = 2**-0.5
        p1 = PolarizationLabel((s + 0.0j, s + 0.0j))
        p2 = PolarizationLabel((1j * s, 1j * s))

        self.assertEqual(p1.jones, p2.jones)
        self.assertEqual(p1.signature, p2.signature)

    def test_overlap_basic(self):
        h = PolarizationLabel.H()
        v = PolarizationLabel.V()

        self.assertAlmostEqual(abs(h.overlap(h)), 1.0, places=14)
        self.assertAlmostEqual(abs(v.overlap(v)), 1.0, places=14)
        self.assertAlmostEqual(abs(h.overlap(v)), 0.0, places=14)

    def test_overlap_conjugation_rule(self):
        pol_r = PolarizationLabel.R()
        pol_l = PolarizationLabel.L()

        self.assertAlmostEqual(abs(pol_r.overlap(pol_r)), 1.0, places=14)
        self.assertAlmostEqual(abs(pol_l.overlap(pol_l)), 1.0, places=14)
        self.assertAlmostEqual(abs(pol_r.overlap(pol_l)), 0.0, places=14)

    def test_linear_factory(self):
        theta = np.pi / 6.0
        p = PolarizationLabel.linear(theta)
        a, b = p.jones

        self.assertAlmostEqual(a.real, np.cos(theta), places=14)
        self.assertAlmostEqual(b.real, np.sin(theta), places=14)
        self.assertAlmostEqual(abs(a) ** 2 + abs(b) ** 2, 1.0, places=14)

    def test_rotated(self):
        h = PolarizationLabel.H()
        v = PolarizationLabel.V()

        h_rot = h.rotated(np.pi / 2.0)
        self.assertAlmostEqual(abs(h_rot.overlap(v)), 1.0, places=14)

    def test_signature_consistency(self):
        p = PolarizationLabel.D()
        sig = p.signature

        self.assertEqual(sig[0], "pol")
        self.assertEqual(len(sig), 5)

    def test_approx_signature_rounding(self):
        p = PolarizationLabel((1.0 + 1e-14j, 0.0 + 0.0j))
        approx = p.approx_signature(decimals=6)

        self.assertEqual(approx[0], "pol_approx")
        self.assertEqual(len(approx), 5)

    def test_frozen_dataclass(self):
        p = PolarizationLabel.H()
        with self.assertRaises(FrozenInstanceError):
            p.jones = (0.0 + 0.0j, 1.0 + 0.0j)  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
