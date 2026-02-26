from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel

from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm

from symop_proto.algebra.density.apply_left import density_apply_left


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


class TestDensityApplyLeft(ExtendedTestCase):
    def test_identity_word_no_change(self):
        """Empty word leaves density terms unchanged."""
        m = make_mode("A")
        rho = (DensityTerm(2.0, left=Monomial(), right=Monomial(creators=(m.create,))),)
        out = density_apply_left(rho, word=())  # empty iterable
        self.assertEqual(out, rho)

    def test_single_creator_on_left_identity(self):
        m = make_mode("A")
        rho = (DensityTerm(3.0, left=Monomial(), right=Monomial()),)
        out = density_apply_left(rho, word=(m.create,))
        self.assertEqual(len(out), 1)
        t = out[0]
        self.assertComplexAlmostEqual(t.coeff, 3.0 + 0j)
        self.assertEqual(
            t.left.signature,
            Monomial(creators=(m.create,), annihilators=()).signature,
        )
        self.assertEqual(t.right.signature, Monomial().signature)

    def test_combines_like_terms_after_application(self):
        """Linearity: two identical outcomes combine coefficients."""
        m = make_mode("A")
        rho = (
            DensityTerm(1.5, left=Monomial(), right=Monomial()),
            DensityTerm(2.5, left=Monomial(), right=Monomial()),
        )
        out = density_apply_left(rho, word=(m.create,))
        self.assertEqual(len(out), 1)
        t = out[0]
        self.assertComplexAlmostEqual(t.coeff, 4.0 + 0j)
        self.assertEqual(
            t.left.signature,
            Monomial(creators=(m.create,), annihilators=()).signature,
        )

    def test_creator_then_annihilator_gives_pass_only(self):
        m = make_mode("A")
        rho = (
            DensityTerm(1.0, left=Monomial(annihilators=(m.ann,)), right=Monomial()),
        )
        out = density_apply_left(rho, word=(m.create,))

        aa_sig = Monomial(creators=(m.create,), annihilators=(m.ann,)).signature
        left_sigs = {t.left.signature for t in out}
        self.assertEqual(left_sigs, {aa_sig})

        coeff_by_sig = {t.left.signature: t.coeff for t in out}
        self.assertComplexAlmostEqual(coeff_by_sig[aa_sig], 1.0 + 0j)

        for t in out:
            self.assertEqual(t.right.signature, Monomial().signature)

    def test_word_can_be_generator(self):
        """Word iterable may be a generator."""
        m = make_mode("A")
        rho = (DensityTerm(2.0, left=Monomial(), right=Monomial()),)
        word_gen = (op for op in (m.create,))  # generator
        out = density_apply_left(rho, word=word_gen)
        self.assertEqual(len(out), 1)
        self.assertEqual(
            out[0].left.signature,
            Monomial(creators=(m.create,), annihilators=()).signature,
        )
        self.assertComplexAlmostEqual(out[0].coeff, 2.0 + 0j)


if __name__ == "__main__":
    unittest.main()
