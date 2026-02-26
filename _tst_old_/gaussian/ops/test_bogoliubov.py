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
from symop_proto.gaussian.ops.bogoliubov import apply_bogoliubov_subset


def _make_mode(path: str) -> ModeOp:
    env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
    return ModeOp(
        env=env,
        label=ModeLabel(PathLabel(path), PolarizationLabel.H()),
    )


class TestApplyBogoliubovSubset(unittest.TestCase):
    def test_single_mode_squeezing_on_vacuum(self) -> None:
        m = _make_mode("A")
        B = ModeBasis.build([m])
        core = GaussianCore.vacuum(B)

        r = 0.6
        phi = 0.1
        ch = np.cosh(r)
        sh = np.sinh(r)

        U = np.array([[ch]], dtype=complex)
        V = np.array([[np.exp(1j * phi) * sh]], dtype=complex)

        out = apply_bogoliubov_subset(core, idx=[0], U=U, V=V, check_ccr=True)

        self.assertTrue(
            np.allclose(out.alpha, np.zeros((1,), dtype=complex), atol=1e-12)
        )

        expected_N = sh * sh
        expected_M = np.exp(1j * phi) * sh * ch

        self.assertTrue(np.allclose(out.N[0, 0].real, expected_N, atol=1e-12))
        self.assertTrue(np.allclose(out.M[0, 0], expected_M, atol=1e-12))

        # Symmetry stabilization
        self.assertTrue(np.allclose(out.N, out.N.conj().T, atol=1e-12))
        self.assertTrue(np.allclose(out.M, out.M.T, atol=1e-12))

    def test_two_mode_squeezing_on_vacuum(self) -> None:
        m1 = _make_mode("A")
        m2 = _make_mode("B")
        B = ModeBasis.build([m1, m2])
        core = GaussianCore.vacuum(B)

        r = 0.4
        ch = np.cosh(r)
        sh = np.sinh(r)

        # a1' = ch a1 + sh a2^dag
        # a2' = ch a2 + sh a1^dag
        U = np.array([[ch, 0.0], [0.0, ch]], dtype=complex)
        V = np.array([[0.0, sh], [sh, 0.0]], dtype=complex)

        out = apply_bogoliubov_subset(core, idx=[0, 1], U=U, V=V, check_ccr=True)

        expected_N = sh * sh
        expected_M12 = sh * ch

        self.assertTrue(
            np.allclose(
                np.diag(out.N).real,
                np.array([expected_N, expected_N]),
                atol=1e-12,
            )
        )
        self.assertTrue(np.allclose(out.M[0, 1], expected_M12, atol=1e-12))
        self.assertTrue(np.allclose(out.M[1, 0], expected_M12, atol=1e-12))

    def test_subset_updates_cross_correlations(self) -> None:
        # Three modes: squeeze only first mode. Cross-correlations with others must update.
        m1 = _make_mode("A")
        m2 = _make_mode("B")
        m3 = _make_mode("C")
        B = ModeBasis.build([m1, m2, m3])

        alpha = np.array([1.0 + 0.2j, -0.5 + 0.1j, 0.7 - 0.3j], dtype=complex)
        core = GaussianCore.coherent(B, alpha)

        r = 0.3
        phi = 0.0
        ch = np.cosh(r)
        sh = np.sinh(r)

        U = np.array([[ch]], dtype=complex)
        V = np.array([[np.exp(1j * phi) * sh]], dtype=complex)

        out = apply_bogoliubov_subset(core, idx=[0], U=U, V=V, check_ccr=True)

        # Unacted modes keep their alpha exactly (U_full identity, V_full zero there).
        self.assertTrue(np.allclose(out.alpha[1:], core.alpha[1:], atol=1e-12))

        # Acted mode alpha transforms: alpha1' = ch*alpha1 + sh*alpha1_conj
        expected_alpha0 = ch * core.alpha[0] + sh * core.alpha[0].conj()
        self.assertTrue(np.allclose(out.alpha[0], expected_alpha0, atol=1e-12))

        # Cross term in N for coherent states is alpha_i^* alpha_j.
        # Since alpha0 changes, N_{0,1} and N_{1,0} must change accordingly.
        expected_N01 = out.alpha[0].conj() * out.alpha[1]
        expected_N10 = out.alpha[1].conj() * out.alpha[0]
        self.assertTrue(np.allclose(out.N[0, 1], expected_N01, atol=1e-12))
        self.assertTrue(np.allclose(out.N[1, 0], expected_N10, atol=1e-12))

    def test_ccr_violation_raises(self) -> None:
        m1 = _make_mode("A")
        m2 = _make_mode("B")
        B = ModeBasis.build([m1, m2])
        core = GaussianCore.vacuum(B)

        # Clearly invalid: U=0, V=0 does not preserve [a,a^dag]=I
        U = np.zeros((2, 2), dtype=complex)
        V = np.zeros((2, 2), dtype=complex)

        with self.assertRaises(ValueError):
            apply_bogoliubov_subset(core, idx=[0, 1], U=U, V=V, check_ccr=True)

    def test_bad_shapes_raise(self) -> None:
        m1 = _make_mode("A")
        m2 = _make_mode("B")
        B = ModeBasis.build([m1, m2])
        core = GaussianCore.vacuum(B)

        U = np.eye(2, dtype=complex)
        V = np.eye(1, dtype=complex)

        with self.assertRaises(ValueError):
            apply_bogoliubov_subset(core, idx=[0, 1], U=U, V=V, check_ccr=False)


if __name__ == "__main__":
    unittest.main()
