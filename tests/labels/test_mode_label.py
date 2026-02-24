from __future__ import annotations

from tests.utils.case import ExtendedTestCase
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel


class TestModeLabel(ExtendedTestCase):
    def test_init_and_signature(self):
        ml = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        sig = ml.signature
        self.assertIsInstance(sig, tuple)
        self.assertEqual(sig[0], "mode_label")

    def test_overlap_same_path_same_pol(self):
        ml = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        self.assertComplexAlmostEqual(ml.overlap(ml), 1.0 + 0j)

    def test_overlap_different_paths_is_zero(self):
        aH = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        bH = ModeLabel(PathLabel("B"), PolarizationLabel.H())
        self.assertComplexAlmostEqual(aH.overlap(bH), 0.0 + 0j)
        self.assertComplexAlmostEqual(bH.overlap(aH), 0.0 + 0j)

    def test_overlap_same_path_orthogonal_pol_is_zero(self):
        aH = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        aV = ModeLabel(PathLabel("A"), PolarizationLabel.V())
        self.assertComplexAlmostEqual(aH.overlap(aV), 0.0 + 0j)
        self.assertComplexAlmostEqual(aV.overlap(aH), 0.0 + 0j)

    def test_overlap_same_path_nonorth_pol(self):
        s = 2**-0.5
        aH = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        aD = ModeLabel(PathLabel("A"), PolarizationLabel.D())
        self.assertComplexAlmostEqual(aH.overlap(aD), s + 0j)
        self.assertComplexAlmostEqual(aH.overlap(aD), aD.overlap(aH).conjugate())

    def test_overlap_with_phase_in_pol(self):
        s = 2**-0.5
        aH = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        aD = ModeLabel(PathLabel("A"), PolarizationLabel.D())
        aD_phased = ModeLabel(PathLabel("A"), PolarizationLabel((1j * s, 1j * s)))
        self.assertComplexAlmostEqual(aH.overlap(aD), aH.overlap(aD_phased))

    def test_approx_signature(self):
        s = 2**-0.5
        a = ModeLabel(PathLabel("A"), PolarizationLabel((s, s)))
        b = ModeLabel(PathLabel("A"), PolarizationLabel((s + 1e-11, s - 1e-11)))
        self.assertEqual(
            a.approx_signature(decimals=8),
            b.approx_signature(decimals=8),
        )

    def test_signature_phase_invariance_via_pol(self):
        s = 2**-0.5
        a = ModeLabel(PathLabel("A"), PolarizationLabel((s, s)))
        b = ModeLabel(PathLabel("A"), PolarizationLabel((1j * s, 1j * s)))
        self.assertEqual(a.signature, b.signature)
