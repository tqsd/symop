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
from symop_proto.gaussian.maps.bogoliubov import Bogoliubov
from symop_proto.gaussian.ops.bogoliubov import apply_bogoliubov_subset


def _make_mode(path: str) -> ModeOp:
    env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
    return ModeOp(
        env=env,
        label=ModeLabel(PathLabel(path), PolarizationLabel.H()),
    )


class TestBogoliubovMap(unittest.TestCase):
    def test_map_matches_kernel_single_mode(self) -> None:
        m = _make_mode("A")
        B = ModeBasis.build([m])
        core = GaussianCore.vacuum(B)

        r = 0.6
        phi = 0.1
        ch = np.cosh(r)
        sh = np.sinh(r)

        U = np.array([[ch]], dtype=complex)
        V = np.array([[np.exp(1j * phi) * sh]], dtype=complex)

        out_map = Bogoliubov(modes=(m,), U=U, V=V, check_ccr=True).apply(core)
        out_kernel = apply_bogoliubov_subset(
            core, idx=[0], U=U, V=V, check_ccr=True
        )

        self.assertTrue(
            np.allclose(out_map.alpha, out_kernel.alpha, atol=1e-12)
        )
        self.assertTrue(np.allclose(out_map.N, out_kernel.N, atol=1e-12))
        self.assertTrue(np.allclose(out_map.M, out_kernel.M, atol=1e-12))

    def test_map_matches_kernel_two_mode_subset_ordering(self) -> None:
        m1 = _make_mode("A")
        m2 = _make_mode("B")
        B = ModeBasis.build([m1, m2])
        core = GaussianCore.vacuum(B)

        r = 0.4
        ch = np.cosh(r)
        sh = np.sinh(r)

        U = np.array([[ch, 0.0], [0.0, ch]], dtype=complex)
        V = np.array([[0.0, sh], [sh, 0.0]], dtype=complex)

        out_map = Bogoliubov(modes=(m1, m2), U=U, V=V, check_ccr=True).apply(
            core
        )
        out_kernel = apply_bogoliubov_subset(
            core, idx=[0, 1], U=U, V=V, check_ccr=True
        )

        self.assertTrue(np.allclose(out_map.N, out_kernel.N, atol=1e-12))
        self.assertTrue(np.allclose(out_map.M, out_kernel.M, atol=1e-12))

    def test_mode_not_in_basis_raises(self) -> None:
        m1 = _make_mode("A")
        m2 = _make_mode("B")
        B = ModeBasis.build([m1])
        core = GaussianCore.vacuum(B)

        U = np.array([[1.0]], dtype=complex)
        V = np.array([[0.0]], dtype=complex)

        op = Bogoliubov(modes=(m2,), U=U, V=V, check_ccr=False)

        with self.assertRaises(KeyError):
            _ = op.apply(core)


if __name__ == "__main__":
    unittest.main()
