from __future__ import annotations
import unittest

from symop_proto.algebra.density.overlap_right_left import overlap_right_left
from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel

from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial


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


class TestOverlapRightLeft(ExtendedTestCase):
    def test_identity_identity_is_one(self):
        I = Monomial()
        self.assertComplexAlmostEqual(overlap_right_left(I, I), 1.0 + 0.0j)

    def test_identity_against_creator_is_zero(self):
        m = make_mode("A")
        I = Monomial()
        L = Monomial(creators=(m.create,))
        # <I | a^dag> = 0 and <a^dag | I> = 0
        self.assertComplexAlmostEqual(overlap_right_left(I, L), 0.0 + 0.0j)
        self.assertComplexAlmostEqual(overlap_right_left(L, I), 0.0 + 0.0j)

    def test_matching_single_creator_gives_one(self):
        m = make_mode("A")
        R = Monomial(creators=(m.create,))
        L = Monomial(creators=(m.create,))
        # <a^dag | a^dag> -> identity coefficient of (a)(a^dag) is +1
        self.assertComplexAlmostEqual(overlap_right_left(R, L), 1.0 + 0.0j)

    def test_mismatched_modes_give_zero(self):
        a = make_mode("A")
        b = make_mode("B")  # orthogonal path label -> zero commutator
        R = Monomial(creators=(a.create,))
        L = Monomial(creators=(b.create,))
        self.assertComplexAlmostEqual(overlap_right_left(R, L), 0.0 + 0.0j)

    def test_creator_annihilator_case(self):
        m = make_mode("A")
        R = Monomial(annihilators=(m.ann,))
        L = Monomial(annihilators=(m.ann,))
        # <a | a> -> identity coefficient of (a^dag)(a) also +1
        self.assertComplexAlmostEqual(overlap_right_left(R, L), 1.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
