import numpy as np
from symop_proto.labels.polarization_label import PolarizationLabel
from tests.utils.case import ExtendedTestCase


class TestPolarizationLabel(ExtendedTestCase):

    def test_initiation(self):
        s = 2**-0.5
        pl_H = PolarizationLabel.H()
        self.assertComplexAlmostEqual(pl_H.jones[0], 1)
        self.assertComplexAlmostEqual(pl_H.jones[1], 0)
        pl_V = PolarizationLabel.V()
        self.assertComplexAlmostEqual(pl_V.jones[0], 0)
        self.assertComplexAlmostEqual(pl_V.jones[1], 1)
        pl_D = PolarizationLabel.D()
        self.assertComplexAlmostEqual(pl_D.jones[0], s)
        self.assertComplexAlmostEqual(pl_D.jones[1], s)
        pl_A = PolarizationLabel.A()
        self.assertComplexAlmostEqual(pl_A.jones[0], s)
        self.assertComplexAlmostEqual(pl_A.jones[1], -s)
        pl_R = PolarizationLabel.R()
        self.assertComplexAlmostEqual(pl_R.jones[0], s)
        self.assertComplexAlmostEqual(pl_R.jones[1], -1j * s)
        pl_L = PolarizationLabel.L()
        self.assertComplexAlmostEqual(pl_L.jones[0], s)
        self.assertComplexAlmostEqual(pl_L.jones[1], 1j * s)

    def test_linear_preset(self):
        s = 2**-0.5
        self.assertComplexAlmostEqual(
            PolarizationLabel.linear(0.0).jones[0], 1.0 + 0j
        )
        self.assertComplexAlmostEqual(
            PolarizationLabel.linear(0.0).jones[1], 0.0 + 0j
        )

        self.assertComplexAlmostEqual(
            PolarizationLabel.linear(np.pi / 2).jones[0], 0.0 + 0j
        )
        self.assertComplexAlmostEqual(
            PolarizationLabel.linear(np.pi / 2).jones[1], 1.0 + 0j
        )
        self.assertComplexAlmostEqual(
            PolarizationLabel.linear(np.pi / 4).jones[0], s + 0j
        )
        self.assertComplexAlmostEqual(
            PolarizationLabel.linear(np.pi / 4).jones[1], s + 0j
        )

    def test_rotated_from_H_matches_expected(self):
        H = PolarizationLabel.H()
        V = PolarizationLabel.V()
        A = PolarizationLabel.A()

        self.assertComplexAlmostEqual(
            H.rotated(np.pi / 2).jones[0], V.jones[0]
        )
        self.assertComplexAlmostEqual(
            H.rotated(np.pi / 2).jones[1], V.jones[1]
        )

        r = H.rotated(np.pi / 4)
        self.assertComplexAlmostEqual(r.jones[0], A.jones[0])
        self.assertComplexAlmostEqual(r.jones[1], A.jones[1])

    def test_global_phase_removed_and_normalized(self):
        s = 2**-0.5
        # D with a global phase i should canonicalize back to D
        D = PolarizationLabel.D()
        phased = PolarizationLabel((1j * s, 1j * s))
        self.assertComplexAlmostEqual(phased.jones[0], D.jones[0])
        self.assertComplexAlmostEqual(phased.jones[1], D.jones[1])
        # (-1, 0) should also canonicalize to H (global phase -1 removed)
        m = PolarizationLabel((-1 + 0j, 0 + 0j))
        self.assertComplexAlmostEqual(m.jones[0], 1.0 + 0j)
        self.assertComplexAlmostEqual(m.jones[1], 0.0 + 0j)
        # unnormalized input gets normalized
        n = PolarizationLabel((10 + 0j, 0 + 0j))
        self.assertComplexAlmostEqual(n.jones[0], 1.0 + 0j)
        self.assertComplexAlmostEqual(n.jones[1], 0.0 + 0j)

    def test_overlap(self):
        s = complex(2**-0.5)
        pl_H = PolarizationLabel.H()
        pl_V = PolarizationLabel.V()
        self.assertEqual(pl_H.overlap(pl_V), 0)
        self.assertEqual(pl_V.overlap(pl_H), 0)
        pl_D = PolarizationLabel.D()
        self.assertComplexAlmostEqual(pl_H.overlap(pl_D), s)
        self.assertComplexAlmostEqual(pl_D.overlap(pl_H), s)
        self.assertComplexAlmostEqual(pl_V.overlap(pl_D), s)
        self.assertComplexAlmostEqual(pl_D.overlap(pl_V), s)

    def test_signatures_and_approx(self):
        # same physical state with different global phase -> same signature after canonicalization
        s = 2**-0.5
        a = PolarizationLabel((s, s))
        b = PolarizationLabel((1j * s, 1j * s))  # same up to global phase
        self.assertEqual(a.signature, b.signature)

        # nearly equal vectors -> same approx_signature at lower precision, but not necessarily same exact signature
        a2 = PolarizationLabel((s, s))
        b2 = PolarizationLabel((s + 1e-11, s - 1e-11))
        # they normalize and phase-fix; signatures might still differ slightly,
        # but approx_signature with decimals=8 should match
        self.assertEqual(
            a2.approx_signature(decimals=8), b2.approx_signature(decimals=8)
        )

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            _ = PolarizationLabel((0.0 + 0j, 0.0 + 0j))
