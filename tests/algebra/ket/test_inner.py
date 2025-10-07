from __future__ import annotations

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.ket.inner import ket_inner
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm
from symop_proto.core.operators import ModeOp
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel


def make_mode(
    path: str = "A",
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    tau: float = 0.0,
    phi: float = 0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


class TestKetInner(ExtendedTestCase):
    def test_vacuum_inner_is_one(self):
        # |0> corresponds to monomial with no creators/annihilators
        psi = (KetTerm(1.0, Monomial((), ())),)
        self.assertComplexAlmostEqual(ket_inner(psi, psi), 1.0 + 0j)

    def test_single_photon_same_mode(self):
        m = make_mode("A")
        psi = (KetTerm(1.0, Monomial((m.create,), ())),)
        self.assertComplexAlmostEqual(ket_inner(psi, psi), 1.0 + 0j)

    def test_orthogonal_paths_give_zero(self):
        a = make_mode("A", pol=PolarizationLabel.H())
        b = make_mode("B", pol=PolarizationLabel.H())  # path orthogonal
        psi_a = (KetTerm(1.0, Monomial((a.create,), ())),)
        psi_b = (KetTerm(1.0, Monomial((b.create,), ())),)
        self.assertComplexAlmostEqual(ket_inner(psi_a, psi_b), 0.0 + 0j)
        self.assertComplexAlmostEqual(ket_inner(psi_b, psi_a), 0.0 + 0j)

    def test_nonorthogonal_polarization_overlap(self):
        # <H|D> = 1/sqrt(2)
        aH = make_mode("A", pol=PolarizationLabel.H())
        aD = make_mode("A", pol=PolarizationLabel.D())
        psi_H = (KetTerm(1.0, Monomial((aH.create,), ())),)
        psi_D = (KetTerm(1.0, Monomial((aD.create,), ())),)
        expected = 2**-0.5
        self.assertComplexAlmostEqual(ket_inner(psi_H, psi_D), expected + 0j)
        self.assertComplexAlmostEqual(ket_inner(psi_D, psi_H), expected + 0j)

    def test_superposition_norm_and_orthogonality(self):
        a = make_mode("A")
        b = make_mode("B")  # orthogonal path to A
        psi_a = KetTerm(
            2.0, Monomial((a.create,), ())
        )  # amplitude 2 -> |2|^2 = 4
        psi_b = KetTerm(
            1.0j, Monomial((b.create,), ())
        )  # amplitude i -> |i|^2 = 1
        psi = (psi_a, psi_b)
        # <psi|psi> = |2|^2 + |i|^2 = 5
        self.assertComplexAlmostEqual(ket_inner(psi, psi), 5.0 + 0j)

    def test_conjugate_symmetry(self):
        aH = make_mode("A", pol=PolarizationLabel.H())
        aR = make_mode("A", pol=PolarizationLabel.R())  # <H|R> = 1/sqrt(2)
        psi1 = (KetTerm(1.0 + 2.0j, Monomial((aH.create,), ())),)
        psi2 = (KetTerm(-0.5 + 0.25j, Monomial((aR.create,), ())),)
        z12 = ket_inner(psi1, psi2)
        z21 = ket_inner(psi2, psi1)
        self.assertComplexAlmostEqual(
            z12, z21.conjugate(), rtol=1e-12, atol=1e-12
        )
