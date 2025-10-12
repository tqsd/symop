from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial


def make_mode(path: str, *, omega=1.0, sigma=0.3, tau=0.0, phi=0.0):
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), PolarizationLabel.H())
    return ModeOp(env=env, label=label)


class TestMonomial(ExtendedTestCase):

    def test_mode_ops_collects_unique_in_first_seen_order(self):
        ma = make_mode("A")
        mb = make_mode("B")
        # duplicate A in creators, plus A in annihilators
        m = Monomial(
            creators=(ma.create, ma.create, mb.create), annihilators=(ma.ann,)
        )
        modes = m.mode_ops
        self.assertEqual(
            tuple(md.signature for md in modes), (ma.signature, mb.signature)
        )

    def test_is_creator_only(self):
        ma = make_mode("A")
        self.assertTrue(Monomial(creators=(ma.create,)).is_creator_only)
        self.assertFalse(
            Monomial(
                creators=(ma.create,), annihilators=(ma.ann,)
            ).is_creator_only
        )

    def test_adjoint_swaps_and_daggers(self):
        ma = make_mode("A")
        mb = make_mode("B")
        m = Monomial(creators=(ma.create, mb.create), annihilators=(ma.ann,))
        md = m.adjoint()
        # After adjoint: creators come from original annihilators (daggered)
        # and annihilators come from original creators (daggered).
        creators_sig = tuple(op.signature for op in md.creators)
        annihilators_sig = tuple(op.signature for op in md.annihilators)

        self.assertEqual(creators_sig, (ma.create.signature,))
        self.assertCountEqual(  # order not guaranteed
            list(annihilators_sig), [ma.ann.signature, mb.ann.signature]
        )
        # Double-adjoint returns original
        self.assertEqual(m.signature, md.adjoint().signature)

    def test_signature_sorts_ops_within_sections(self):
        ma = make_mode("A")
        mb = make_mode("B")
        # Put B before A in creators; signature should sort them
        m1 = Monomial(creators=(mb.create, ma.create), annihilators=(ma.ann,))
        m2 = Monomial(creators=(ma.create, mb.create), annihilators=(ma.ann,))
        self.assertEqual(m1.signature, m2.signature)

    def test_approx_signature_rounds_envs(self):
        # Two nearly identical envs should share approx_signature at coarse decimals
        ma1 = make_mode("A", tau=0.0)
        # tiny difference in tau
        ma2 = make_mode("A", tau=1e-11)
        m1 = Monomial(creators=(ma1.create,))
        m2 = Monomial(creators=(ma2.create,))
        # exact signature differs
        self.assertNotEqual(m1.signature, m2.signature)
        # approx signature with decimals=8 should match
        self.assertEqual(
            m1.approx_signature(decimals=8), m2.approx_signature(decimals=8)
        )

    def test_signature_shape(self):
        ma = make_mode("A")
        m = Monomial(creators=(ma.create,), annihilators=())
        sig = m.signature
        # ("cre", <tuple>, "ann", <tuple>)
        self.assertEqual(sig[0], "cre")
        self.assertIsInstance(sig[1], tuple)
        self.assertEqual(sig[2], "ann")
        self.assertIsInstance(sig[3], tuple)


if __name__ == "__main__":
    unittest.main()
