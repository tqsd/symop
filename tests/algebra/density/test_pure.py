from __future__ import annotations
import unittest
from symop_proto.core.operators import ModeOp
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from tests.utils.case import ExtendedTestCase

from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm
from symop_proto.algebra.density.pure import density_pure


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


class TestDensityPure(ExtendedTestCase):
    def test_single_ket_gives_single_density(self):
        k = (KetTerm(1.0, Monomial()),)
        out = density_pure(k)
        self.assertEqual(len(out), 1)
        t = out[0]
        self.assertEqual(t.left.signature, Monomial().signature)
        self.assertEqual(t.right.signature, Monomial().signature)
        self.assertComplexAlmostEqual(t.coeff, 1.0 + 0.0j)

    def test_two_kets_produces_single_merged_term(self):
        k1 = KetTerm(1.0, Monomial())
        k2 = KetTerm(2.0, Monomial())
        out = density_pure((k1, k2))
        # all four pairs collapse into one |I><I|
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].left.signature, Monomial().signature)
        self.assertEqual(out[0].right.signature, Monomial().signature)
        # total coeff = |sum c_i|^2 = |1+2|^2 = 9
        self.assertComplexAlmostEqual(out[0].coeff, 9.0 + 0.0j)

    def test_conjugation_applied_correctly_total_coeff(self):
        k1 = KetTerm(1.0 + 1.0j, Monomial())
        k2 = KetTerm(1.0 - 1.0j, Monomial())
        out = density_pure((k1, k2))
        # all pairs collapse; coeff = |(1+i) + (1-i)|^2 = |2|^2 = 4
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 4.0 + 0.0j)

    def test_combination_merges_identical_pairs(self):
        k1 = KetTerm(1.0, Monomial())
        k2 = KetTerm(1.0, Monomial())
        out = density_pure((k1, k2))
        # Since all monomials are the same, should merge into a single term
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 4.0 + 0.0j)

    def test_per_pair_coefficients_when_monomials_differ(self):
        mA = make_mode("A")
        mB = make_mode("B")
        k1 = KetTerm(1.0 + 1.0j, Monomial(creators=(mA.create,)))
        k2 = KetTerm(1.0 - 1.0j, Monomial(creators=(mB.create,)))
        out = density_pure((k1, k2))
        # Now we should have 4 distinct (left,right) pairs
        self.assertEqual(len(out), 4)
        coeff_map = {(t.left.signature, t.right.signature): t.coeff for t in out}
        self.assertComplexAlmostEqual(
            coeff_map[(k1.monomial.signature, k2.monomial.signature)],
            (1.0 + 1.0j) * (1.0 - 1.0j).conjugate(),  # = (1+i)*(1+i) = 2i
        )


if __name__ == "__main__":
    unittest.main()
