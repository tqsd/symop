import unittest
import numpy as np

from symop_proto.gaussian.basis import ModeBasis
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp


class TestModeBasis(unittest.TestCase):
    def setUp(self):
        # Two temporal modes on same path (partially overlapping)
        env1 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        env2 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.2, phi0=0.0)

        self.m1 = ModeOp(
            env=env1, label=ModeLabel(PathLabel("A"), PolarizationLabel.H())
        )
        self.m2 = ModeOp(
            env=env2, label=ModeLabel(PathLabel("A"), PolarizationLabel.H())
        )

        # Orthogonal polarization
        self.m3 = ModeOp(
            env=env1, label=ModeLabel(PathLabel("A"), PolarizationLabel.V())
        )

    # ------------------------------------------------------------
    # Basic construction
    # ------------------------------------------------------------

    def test_build_basic(self):
        basis = ModeBasis.build([self.m1, self.m2])

        self.assertEqual(basis.n, 2)
        self.assertEqual(basis.gram.shape, (2, 2))

    def test_duplicate_modes_are_merged(self):
        basis = ModeBasis.build([self.m1, self.m1, self.m1])

        self.assertEqual(basis.n, 1)
        self.assertEqual(basis.gram.shape, (1, 1))

    # ------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------

    def test_index_mapping_consistency(self):
        basis = ModeBasis.build([self.m1, self.m2])

        for i, m in enumerate(basis.modes):
            self.assertEqual(basis.index_of(m), i)

    def test_require_index_of_raises(self):
        basis = ModeBasis.build([self.m1])

        with self.assertRaises(KeyError):
            basis.require_index_of(self.m2)

    # ------------------------------------------------------------
    # Gram properties
    # ------------------------------------------------------------

    def test_gram_is_hermitian(self):
        basis = ModeBasis.build([self.m1, self.m2])
        self.assertTrue(basis.is_hermitian())

    def test_positive_semidefinite(self):
        basis = ModeBasis.build([self.m1, self.m2])
        self.assertTrue(basis.is_positive_semidefinite())

    def test_canonical_for_orthogonal_modes(self):
        basis = ModeBasis.build([self.m1, self.m3])

        # m1 and m3 differ in polarization → orthogonal
        self.assertTrue(basis.is_canonical(eps=1e-10))

    # ------------------------------------------------------------
    # Union behavior
    # ------------------------------------------------------------

    def test_union_extends_basis(self):
        b1 = ModeBasis.build([self.m1])
        b2 = b1.union([self.m2])

        self.assertEqual(b2.n, 2)
        self.assertEqual(b2.index_of(self.m1), 0)
        self.assertEqual(b2.index_of(self.m2), 1)

    def test_union_does_not_duplicate(self):
        b1 = ModeBasis.build([self.m1])
        b2 = b1.union([self.m1])

        self.assertEqual(b2.n, 1)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def test_validate_passes(self):
        basis = ModeBasis.build([self.m1, self.m2])
        basis.validate(hermitian=True, psd=True)

    def test_invalid_gram_shape_raises(self):
        with self.assertRaises(ValueError):
            ModeBasis(
                modes=(self.m1,),
                gram=np.zeros((2, 2), dtype=complex),
                index_by_sig={self.m1.signature: 0},
            )


if __name__ == "__main__":
    unittest.main()
