from __future__ import annotations

import unittest

import numpy as np

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp
from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore


class TestGaussianCoreStructuralOps(unittest.TestCase):
    def _make_modes(self):
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(
            env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H())
        )
        m2 = ModeOp(
            env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H())
        )
        m3 = ModeOp(
            env=env, label=ModeLabel(PathLabel("C"), PolarizationLabel.H())
        )
        return env, m1, m2, m3

    def test_keep_slices_moments_and_gram(self):
        _, m1, m2, m3 = self._make_modes()
        B = ModeBasis.build([m1, m2, m3])

        alpha = np.array([1.0 + 0.1j, -0.2 + 0.0j, 0.3 - 0.4j], dtype=complex)
        core = GaussianCore.coherent(B, alpha)

        sub = core.keep((m3, m1))

        # Ordering should follow requested modes (m3 then m1)
        self.assertEqual(sub.basis.n, 2)
        self.assertEqual(sub.basis.modes[0].signature, m3.signature)
        self.assertEqual(sub.basis.modes[1].signature, m1.signature)

        # Alpha matches slice in that order
        self.assertTrue(
            np.allclose(
                sub.alpha, np.array([alpha[2], alpha[0]], dtype=complex)
            )
        )

        # Gram matches the corresponding sub-block of original gram
        idx = np.array([2, 0], dtype=int)
        G_expected = B.gram[np.ix_(idx, idx)]
        self.assertTrue(np.allclose(sub.basis.gram, G_expected))

        # N and M blocks match
        self.assertTrue(np.allclose(sub.N, core.N[np.ix_(idx, idx)]))
        self.assertTrue(np.allclose(sub.M, core.M[np.ix_(idx, idx)]))

    def test_trace_out_equals_keep_complement(self):
        _, m1, m2, m3 = self._make_modes()
        B = ModeBasis.build([m1, m2, m3])

        alpha = np.array([0.1 + 0.0j, 0.2 + 0.0j, 0.3 + 0.0j], dtype=complex)
        core = GaussianCore.coherent(B, alpha)

        out = core.trace_out((m2,))
        keep = core.keep((m1, m3))

        self.assertEqual(out.basis.n, keep.basis.n)
        self.assertTrue(np.allclose(out.alpha, keep.alpha))
        self.assertTrue(np.allclose(out.N, keep.N))
        self.assertTrue(np.allclose(out.M, keep.M))
        self.assertTrue(np.allclose(out.basis.gram, keep.basis.gram))

    def test_extend_with_vacuum_appends_zero_blocks(self):
        _, m1, m2, m3 = self._make_modes()
        B = ModeBasis.build([m1])
        alpha = np.array([1.0 + 0.0j], dtype=complex)
        core = GaussianCore.coherent(B, alpha)

        ext = core.extend_with_vacuum((m2, m3))

        self.assertEqual(ext.basis.n, 3)

        # Old block preserved
        self.assertTrue(np.allclose(ext.alpha[0], core.alpha[0]))
        self.assertTrue(np.allclose(ext.N[0:1, 0:1], core.N))
        self.assertTrue(np.allclose(ext.M[0:1, 0:1], core.M))

        # New modes vacuum-initialized (no correlations added)
        self.assertTrue(np.allclose(ext.alpha[1:], 0.0))
        self.assertTrue(np.allclose(ext.N[1:, :], 0.0))
        self.assertTrue(np.allclose(ext.N[:, 1:], 0.0))
        self.assertTrue(np.allclose(ext.M[1:, :], 0.0))
        self.assertTrue(np.allclose(ext.M[:, 1:], 0.0))

    def test_extend_with_existing_mode_is_noop(self):
        _, m1, _, _ = self._make_modes()
        B = ModeBasis.build([m1])
        core = GaussianCore.vacuum(B)

        ext = core.extend_with_vacuum((m1,))
        # You return self if basis unchanged
        self.assertIs(ext, core)


if __name__ == "__main__":
    unittest.main()
