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

from symop_proto.gaussian.maps.passive import (
    PassiveUnitary,
    PhaseShift,
    BeamSplitter,
)


class TestGaussianPassiveMaps(unittest.TestCase):
    def assert_allclose(self, a, b, atol=1e-12, rtol=1e-12, msg: str = "") -> None:
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
        self.m3 = ModeOp(
            env=env, label=ModeLabel(PathLabel("C"), PolarizationLabel.H())
        )

        self.B2 = ModeBasis.build([self.m1, self.m2])
        self.B3 = ModeBasis.build([self.m1, self.m2, self.m3])

    def test_phaseshift_coherent_first_moment(self) -> None:
        alpha = np.array([0.4 + 0.1j, -0.2 + 0.5j], dtype=complex)
        core = GaussianCore.coherent(self.B2, alpha)

        phi = 0.37
        out = PhaseShift(mode=self.m1, phi=phi, check_unitary=True).apply(core)

        expected = alpha.copy()
        expected[0] *= np.exp(1j * phi)
        self.assert_allclose(out.alpha, expected)

    def test_beamsplitter_coherent_matches_U_alpha(self) -> None:
        alpha = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        core = GaussianCore.coherent(self.B2, alpha)

        bs = BeamSplitter(
            mode1=self.m1,
            mode2=self.m2,
            theta=np.pi / 4.0,
            phi=0.0,
            check_unitary=True,
        )
        out = bs.apply(core)

        self.assert_allclose(out.alpha, bs.U @ alpha)

    def test_beamsplitter_is_unitary(self) -> None:
        bs = BeamSplitter(mode1=self.m1, mode2=self.m2, theta=0.3, phi=0.9)
        U = bs.U
        I = np.eye(2, dtype=complex)
        self.assert_allclose(U.conj().T @ U, I)

    def test_passiveunitary_coherent_updates_second_moments_consistently(
        self,
    ) -> None:
        # For coherent states, N = outer(conj(alpha), alpha), M = outer(alpha, alpha).
        alpha = np.array([0.1 + 0.2j, -0.3 + 0.1j], dtype=complex)
        core = GaussianCore.coherent(self.B2, alpha)

        theta = 0.4
        U = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=complex,
        )

        out = PassiveUnitary(modes=(self.m1, self.m2), U=U, check_unitary=True).apply(
            core
        )

        alpha2 = U @ alpha
        expected = GaussianCore.coherent(self.B2, alpha2)
        self.assert_allclose(out.alpha, expected.alpha)
        self.assert_allclose(out.N, expected.N)
        self.assert_allclose(out.M, expected.M)

    def test_phaseshift_on_two_mode_squeezed_updates_pairing_phase(
        self,
    ) -> None:
        # Two-mode squeezed vacuum correlations: M12 != 0, alpha = 0.
        r = 0.7
        s2 = np.sinh(r) ** 2
        sc = np.sinh(r) * np.cosh(r)

        alpha0 = np.zeros((2,), dtype=complex)
        N = np.array([[s2, 0.0], [0.0, s2]], dtype=complex)
        M = np.array([[0.0, sc], [sc, 0.0]], dtype=complex)

        core = GaussianCore.from_moments(self.B2, alpha=alpha0, N=N, M=M)

        phi = 0.31
        out = PhaseShift(mode=self.m1, phi=phi, check_unitary=True).apply(core)

        self.assert_allclose(out.alpha, core.alpha)
        self.assert_allclose(out.N, core.N)
        self.assert_allclose(out.M[0, 1], np.exp(1j * phi) * core.M[0, 1])
        self.assert_allclose(out.M[1, 0], np.exp(1j * phi) * core.M[1, 0])

    def test_subset_action_updates_cross_blocks_in_3mode_state(self) -> None:
        # Build a 3-mode state with cross-correlations between (m1,m2) and (m3).
        alpha = np.array([0.1 + 0.0j, -0.2 + 0.1j, 0.05 - 0.02j], dtype=complex)

        # Start from vacuum-like base (alpha only), then inject some correlations.
        core = GaussianCore.from_moments(
            self.B3,
            alpha=alpha,
            N=np.zeros((3, 3), dtype=complex),
            M=np.zeros((3, 3), dtype=complex),
        )

        N = core.N.copy()
        M = core.M.copy()

        # Add some arbitrary Hermitian N cross correlations with mode3:
        N[0, 2] = 0.03 + 0.04j
        N[2, 0] = np.conjugate(N[0, 2])
        N[1, 2] = -0.02 + 0.01j
        N[2, 1] = np.conjugate(N[1, 2])

        # Add some symmetric M cross correlations with mode3:
        M[0, 2] = 0.02 - 0.01j
        M[2, 0] = M[0, 2]
        M[1, 2] = -0.01 + 0.03j
        M[2, 1] = M[1, 2]

        core = GaussianCore.from_moments(self.B3, alpha=alpha, N=N, M=M)

        # Apply a phase shift only on mode1 (subset size 1).
        phi = 0.6
        out = PhaseShift(mode=self.m1, phi=phi, check_unitary=True).apply(core)

        # Cross-block expectation: entries touching index 0 get phased appropriately.
        e = np.exp(1j * phi)

        # alpha
        self.assert_allclose(out.alpha[0], e * core.alpha[0])
        self.assert_allclose(out.alpha[1], core.alpha[1])
        self.assert_allclose(out.alpha[2], core.alpha[2])

        # N: N_0,2 = <a0^dag a2> should pick up e^{-i phi} on the creation index 0
        self.assert_allclose(out.N[0, 2], np.conjugate(e) * core.N[0, 2])
        # N_2,0 = <a2^dag a0> should pick up e^{i phi} on annihilation index 0
        self.assert_allclose(out.N[2, 0], e * core.N[2, 0])

        # M: M_0,2 = <a0 a2> picks up e^{i phi} on index 0
        self.assert_allclose(out.M[0, 2], e * core.M[0, 2])
        self.assert_allclose(out.M[2, 0], e * core.M[2, 0])

        # Invariants preserved
        self.assert_allclose(out.N, out.N.conj().T)
        self.assert_allclose(out.M, out.M.T)

    def test_unitary_check_rejects_non_unitary(self) -> None:
        core = GaussianCore.vacuum(self.B2)

        U_bad = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
        op = PassiveUnitary(modes=(self.m1, self.m2), U=U_bad, check_unitary=True)

        with self.assertRaises(ValueError):
            op.apply(core)

    def test_shape_mismatch_raises(self) -> None:
        core = GaussianCore.vacuum(self.B2)

        U_wrong = np.eye(3, dtype=complex)
        op = PassiveUnitary(modes=(self.m1, self.m2), U=U_wrong, check_unitary=False)

        with self.assertRaises(ValueError):
            op.apply(core)

    def test_mode_not_in_basis_raises(self) -> None:
        core = GaussianCore.vacuum(self.B2)

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        mX = ModeOp(env=env, label=ModeLabel(PathLabel("X"), PolarizationLabel.H()))

        op = PhaseShift(mode=mX, phi=0.1)
        with self.assertRaises(KeyError):
            op.apply(core)


if __name__ == "__main__":
    unittest.main()
