import unittest
from dataclasses import FrozenInstanceError

from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


class DummyEnvelope:
    def __init__(self, name):
        self.name = name

    @property
    def signature(self):
        return ("env", self.name)

    def approx_signature(self, *, decimals=12, ignore_global_phase=False):
        return ("env_approx", self.name, decimals, ignore_global_phase)


class TestModeLabel(unittest.TestCase):
    def setUp(self):
        self.path_a = PathLabel("a")
        self.path_b = PathLabel("b")

        self.pol_h = PolarizationLabel.H()
        self.pol_v = PolarizationLabel.V()

        self.env1 = DummyEnvelope("e1")
        self.env2 = DummyEnvelope("e2")

    def test_overlap_factorizes_path_and_pol(self):
        m1 = ModeLabel(self.path_a, self.pol_h, self.env1)
        m2 = ModeLabel(self.path_a, self.pol_h, self.env2)
        m3 = ModeLabel(self.path_b, self.pol_h, self.env1)
        m4 = ModeLabel(self.path_a, self.pol_v, self.env1)

        # Same path and pol → overlap 1
        self.assertAlmostEqual(abs(m1.overlap(m2)), 1.0, places=14)

        # Different path → 0
        self.assertAlmostEqual(abs(m1.overlap(m3)), 0.0, places=14)

        # Different polarization → 0
        self.assertAlmostEqual(abs(m1.overlap(m4)), 0.0, places=14)

    def test_signature_structure(self):
        m = ModeLabel(self.path_a, self.pol_h, self.env1)
        sig = m.signature

        self.assertEqual(sig[0], "mode_label")
        self.assertEqual(sig[1], self.path_a.signature)
        self.assertEqual(sig[2], self.pol_h.signature)
        self.assertEqual(sig[3], self.env1.signature)

    def test_approx_signature_forwards(self):
        m = ModeLabel(self.path_a, self.pol_h, self.env1)
        approx = m.approx_signature(decimals=3, ignore_global_phase=True)

        self.assertEqual(approx[0], "mode_label_approx")
        self.assertEqual(
            approx[1],
            self.path_a.approx_signature(decimals=3, ignore_global_phase=True),
        )
        self.assertEqual(
            approx[2],
            self.pol_h.approx_signature(decimals=3, ignore_global_phase=True),
        )
        self.assertEqual(
            approx[3],
            self.env1.approx_signature(decimals=3, ignore_global_phase=True),
        )

    def test_with_path(self):
        m1 = ModeLabel(self.path_a, self.pol_h, self.env1)
        m2 = m1.with_path(self.path_b)

        self.assertNotEqual(m1, m2)
        self.assertEqual(m2.path, self.path_b)
        self.assertEqual(m2.pol, self.pol_h)
        self.assertEqual(m2.envelope, self.env1)

    def test_with_pol(self):
        m1 = ModeLabel(self.path_a, self.pol_h, self.env1)
        m2 = m1.with_pol(self.pol_v)

        self.assertNotEqual(m1, m2)
        self.assertEqual(m2.path, self.path_a)
        self.assertEqual(m2.pol, self.pol_v)
        self.assertEqual(m2.envelope, self.env1)

    def test_with_envelope(self):
        m1 = ModeLabel(self.path_a, self.pol_h, self.env1)
        m2 = m1.with_envelope(self.env2)

        self.assertNotEqual(m1, m2)
        self.assertEqual(m2.path, self.path_a)
        self.assertEqual(m2.pol, self.pol_h)
        self.assertEqual(m2.envelope, self.env2)

    def test_frozen_dataclass(self):
        m = ModeLabel(self.path_a, self.pol_h, self.env1)
        with self.assertRaises(FrozenInstanceError):
            m.path = self.path_b  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
