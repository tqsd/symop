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
from symop_proto.gaussian.physicality import project_to_physical_gaussian


def _make_mode(path: str) -> ModeOp:
    env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
    return ModeOp(
        env=env, label=ModeLabel(PathLabel(path), PolarizationLabel.H())
    )


class TestGaussianPhysicality(unittest.TestCase):
    def test_vacuum_is_physical(self) -> None:
        m = _make_mode("A")
        B = ModeBasis.build([m])
        core = GaussianCore.vacuum(B)

        self.assertTrue(core.is_physical(atol=1e-12))
        core.validate_physical(atol=1e-12)

    def test_unphysical_negative_number_variance_detected(self) -> None:
        m = _make_mode("A")
        B = ModeBasis.build([m])

        alpha = np.zeros((1,), dtype=complex)

        # Deliberately unphysical: negative <a^dag a>
        N = np.array([[-0.1 + 0.0j]], dtype=complex)
        M = np.array([[0.0 + 0.0j]], dtype=complex)

        core = GaussianCore.from_moments(B, alpha=alpha, N=N, M=M)

        self.assertFalse(core.is_physical(atol=1e-12))
        with self.assertRaises(ValueError):
            core.validate_physical(atol=1e-12)

    def test_projection_repairs_unphysical_state(self) -> None:
        m = _make_mode("A")
        B = ModeBasis.build([m])

        alpha = np.zeros((1,), dtype=complex)
        N = np.array([[-0.1 + 0.0j]], dtype=complex)
        M = np.array([[0.0 + 0.0j]], dtype=complex)
        core = GaussianCore.from_moments(B, alpha=alpha, N=N, M=M)

        self.assertFalse(core.is_physical(atol=1e-12))

        fixed, rep = project_to_physical_gaussian(
            core, method="psd_uncertainty", atol=1e-12
        )

        self.assertTrue(fixed.is_physical(atol=1e-10))
        self.assertGreaterEqual(rep.added_noise, 0.0)

    def test_projection_noop_on_physical_state(self) -> None:
        m = _make_mode("A")
        B = ModeBasis.build([m])
        core = GaussianCore.vacuum(B)

        fixed, rep = project_to_physical_gaussian(
            core, method="psd_uncertainty", atol=1e-12
        )

        self.assertTrue(np.allclose(fixed.alpha, core.alpha, atol=1e-12))
        self.assertTrue(np.allclose(fixed.N, core.N, atol=1e-12))
        self.assertTrue(np.allclose(fixed.M, core.M, atol=1e-12))
        self.assertEqual(rep.added_noise, 0.0)

    def test_isotropic_noise_method_also_repairs(self) -> None:
        m = _make_mode("A")
        B = ModeBasis.build([m])

        alpha = np.zeros((1,), dtype=complex)
        N = np.array([[-0.2 + 0.0j]], dtype=complex)
        M = np.array([[0.0 + 0.0j]], dtype=complex)
        core = GaussianCore.from_moments(B, alpha=alpha, N=N, M=M)

        fixed, rep = project_to_physical_gaussian(
            core, method="add_isotropic_noise", atol=1e-12
        )

        self.assertTrue(fixed.is_physical(atol=1e-10))
        self.assertGreater(rep.added_noise, 0.0)


if __name__ == "__main__":
    unittest.main()
