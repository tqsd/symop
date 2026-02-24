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
from symop_proto.gaussian.ops.passive import apply_passive_unitary_subset


class TestGaussianOpsPassive(unittest.TestCase):
    def assert_allclose(self, a, b, atol=1e-12, rtol=1e-12, msg=""):
        ok = np.allclose(a, b, atol=atol, rtol=rtol)
        if not ok:
            diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            self.fail(msg or f"not close; max abs diff = {diff}")

    def setUp(self) -> None:
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        self.m1 = ModeOp(
            env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H())
        )
        self.m2 = ModeOp(
            env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H())
        )
        self.B = ModeBasis.build([self.m1, self.m2])

    def test_empty_idx_is_identity(self):
        core = GaussianCore.coherent(
            self.B, np.array([0.2 + 0.1j, -0.4 + 0.0j], dtype=complex)
        )
        U = np.array([[np.exp(1j * 0.3)]], dtype=complex)
        out = apply_passive_unitary_subset(core, idx=[], U=U)
        self.assert_allclose(out.alpha, core.alpha)
        self.assert_allclose(out.N, core.N)
        self.assert_allclose(out.M, core.M)

    def test_vacuum_invariant_under_unitary(self):
        core = GaussianCore.vacuum(self.B)
        theta = np.pi / 4.0
        U = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=complex,
        )
        out = apply_passive_unitary_subset(
            core, idx=[0, 1], U=U, check_unitary=True
        )

        self.assert_allclose(out.alpha, core.alpha)
        self.assert_allclose(out.N, core.N)
        self.assert_allclose(out.M, core.M)

    def test_coherent_alpha_transforms(self):
        alpha = np.array([1.0 + 0.2j, -0.3 + 0.0j], dtype=complex)
        core = GaussianCore.coherent(self.B, alpha)

        theta = np.pi / 4.0
        U = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=complex,
        )
        out = apply_passive_unitary_subset(
            core, idx=[0, 1], U=U, check_unitary=True
        )
        self.assert_allclose(out.alpha, U @ alpha)

        # For displaced vacuum, the result should still be displaced vacuum
        expected = GaussianCore.coherent(self.B, U @ alpha)
        self.assert_allclose(out.N, expected.N)
        self.assert_allclose(out.M, expected.M)

    def test_local_phase_updates_pairing_correlation(self):
        r = 0.7
        s2 = np.sinh(r) ** 2
        sc = np.sinh(r) * np.cosh(r)

        alpha0 = np.zeros((2,), dtype=complex)
        N = np.array([[s2, 0.0], [0.0, s2]], dtype=complex)
        M = np.array([[0.0, sc], [sc, 0.0]], dtype=complex)

        core = GaussianCore.from_moments(self.B, alpha=alpha0, N=N, M=M)

        phi = 0.31
        U = np.array([[np.exp(1j * phi), 0.0], [0.0, 1.0]], dtype=complex)
        out = apply_passive_unitary_subset(
            core, idx=[0, 1], U=U, check_unitary=True
        )

        self.assert_allclose(out.N, core.N)
        self.assert_allclose(out.M[0, 1], np.exp(1j * phi) * core.M[0, 1])
        self.assert_allclose(out.M[1, 0], np.exp(1j * phi) * core.M[1, 0])

    def test_raises_on_shape_mismatch(self):
        core = GaussianCore.vacuum(self.B)
        U = np.eye(3, dtype=complex)
        with self.assertRaises(ValueError):
            apply_passive_unitary_subset(core, idx=[0, 1], U=U)

    def test_raises_on_non_unitary_if_check_enabled(self):
        core = GaussianCore.vacuum(self.B)
        U = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
        with self.assertRaises(ValueError):
            apply_passive_unitary_subset(
                core, idx=[0, 1], U=U, check_unitary=True
            )


if __name__ == "__main__":
    unittest.main()
