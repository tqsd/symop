import unittest
import numpy as np

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp
from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore


class TestGaussianCore(unittest.TestCase):

    def setUp(self):
        # Build an orthogonal two-mode basis using different paths.
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)

        m1 = ModeOp(
            env=env,
            label=ModeLabel(PathLabel("A"), PolarizationLabel.H()),
        )
        m2 = ModeOp(
            env=env,
            label=ModeLabel(PathLabel("B"), PolarizationLabel.H()),
        )

        self.basis = ModeBasis.build([m1, m2])
        self.n = self.basis.n

    def assert_allclose(self, a, b, atol=1e-12, rtol=1e-12, msg=None):
        self.assertTrue(np.allclose(a, b, atol=atol, rtol=rtol), msg=msg)

    # -------------------------
    # Construction and shapes
    # -------------------------

    def test_vacuum_shapes(self):
        core = GaussianCore.vacuum(self.basis)
        self.assertEqual(core.alpha.shape, (self.n,))
        self.assertEqual(core.N.shape, (self.n, self.n))
        self.assertEqual(core.M.shape, (self.n, self.n))

    def test_vacuum_zero_moments(self):
        core = GaussianCore.vacuum(self.basis)
        self.assert_allclose(core.alpha, np.zeros((self.n,), dtype=complex))
        self.assert_allclose(core.N, np.zeros((self.n, self.n), dtype=complex))
        self.assert_allclose(core.M, np.zeros((self.n, self.n), dtype=complex))

    def test_from_moments_defaults_to_zero(self):
        core = GaussianCore.from_moments(self.basis)
        self.assert_allclose(core.alpha, 0.0)
        self.assert_allclose(core.N, 0.0)
        self.assert_allclose(core.M, 0.0)

    def test_from_moments_copies_inputs(self):
        alpha = np.array([1.0 + 0.2j, -0.5 + 0.0j], dtype=complex)
        N = np.eye(self.n, dtype=complex)
        M = np.zeros((self.n, self.n), dtype=complex)

        core = GaussianCore.from_moments(self.basis, alpha=alpha, N=N, M=M)

        # Mutate the original arrays and ensure core didn't change
        alpha[0] = 999.0 + 0.0j
        N[0, 0] = 999.0 + 0.0j
        M[0, 0] = 999.0 + 0.0j

        self.assertNotEqual(core.alpha[0], alpha[0])
        self.assertNotEqual(core.N[0, 0], N[0, 0])
        self.assertNotEqual(core.M[0, 0], M[0, 0])

    # -------------------------
    # Algebraic constraints
    # -------------------------

    def test_N_must_be_hermitian(self):
        alpha = np.zeros((self.n,), dtype=complex)
        N = np.zeros((self.n, self.n), dtype=complex)
        N[0, 1] = 1.0 + 0.0j
        N[1, 0] = 0.0 + 0.0j  # not conjugate -> non-Hermitian
        M = np.zeros((self.n, self.n), dtype=complex)

        with self.assertRaises(ValueError):
            GaussianCore.from_moments(self.basis, alpha=alpha, N=N, M=M)

    def test_M_must_be_symmetric(self):
        alpha = np.zeros((self.n,), dtype=complex)
        N = np.zeros((self.n, self.n), dtype=complex)
        M = np.zeros((self.n, self.n), dtype=complex)
        M[0, 1] = 1.0 + 0.0j
        M[1, 0] = 2.0 + 0.0j  # not symmetric

        with self.assertRaises(ValueError):
            GaussianCore.from_moments(self.basis, alpha=alpha, N=N, M=M)

    # -------------------------
    # Centered moments
    # -------------------------

    def test_centered_moments_vacuum(self):
        core = GaussianCore.vacuum(self.basis)
        N0, M0 = core.centered_moments()
        self.assert_allclose(N0, 0.0)
        self.assert_allclose(M0, 0.0)

    def test_centered_moments_coherent(self):
        alpha = np.array([1.0 + 0.2j, -0.5 + 0.0j], dtype=complex)
        core = GaussianCore.from_moments(self.basis, alpha=alpha)

        N0, M0 = core.centered_moments()
        # For a "pure displacement on vacuum" (N=M=0):
        # N0 = -alpha^* alpha^T, M0 = -alpha alpha^T
        expected_N0 = -np.outer(np.conjugate(alpha), alpha)
        expected_M0 = -np.outer(alpha, alpha)

        self.assert_allclose(N0, expected_N0)
        self.assert_allclose(M0, expected_M0)

    # -------------------------
    # Quadrature mean
    # -------------------------

    def test_quadrature_mean_vacuum(self):
        core = GaussianCore.vacuum(self.basis)
        d = core.quadrature_mean()
        self.assertEqual(d.shape, (2 * self.n,))
        self.assert_allclose(d, 0.0)

    def test_quadrature_mean_coherent(self):
        alpha = np.array([1.0 + 0.2j, -0.5 + 0.0j], dtype=complex)
        core = GaussianCore.from_moments(self.basis, alpha=alpha)

        d = core.quadrature_mean()
        expected_x = np.sqrt(2.0) * np.real(alpha)
        expected_p = np.sqrt(2.0) * np.imag(alpha)
        expected = np.concatenate([expected_x, expected_p], axis=0)

        self.assert_allclose(d, expected)

    # -------------------------
    # Quadrature covariance
    # -------------------------

    def test_quadrature_covariance_shapes_and_symmetry(self):
        core = GaussianCore.vacuum(self.basis)
        V = core.quadrature_covariance()
        self.assertEqual(V.shape, (2 * self.n, 2 * self.n))
        self.assert_allclose(V, V.T, msg="Covariance must be symmetric")

    def test_quadrature_covariance_is_real(self):
        core = GaussianCore.vacuum(self.basis)
        V = core.quadrature_covariance()
        self.assertTrue(np.issubdtype(V.dtype, np.floating))

    def test_quadrature_covariance_vacuum_canonical_basis(self):
        # For an orthogonal basis with Gram ~ I, vacuum covariance should be (1/2) I.
        self.assertTrue(self.basis.is_canonical(eps=1e-10))

        core = GaussianCore.vacuum(self.basis)
        V = core.quadrature_covariance()

        expected = 0.5 * np.eye(2 * self.n, dtype=float)
        self.assert_allclose(V, expected, atol=1e-10, rtol=1e-10)

    def test_quadrature_covariance_displacement_does_not_change_cov(self):
        alpha = np.array([0.3 + 0.1j, -0.2 + 0.5j], dtype=complex)

        core0 = GaussianCore.vacuum(self.basis)

        core1 = GaussianCore.coherent(self.basis, alpha)

        V0 = core0.quadrature_covariance()
        V1 = core1.quadrature_covariance()

        self.assert_allclose(V0, V1, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
