from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.expand.word_times_monomial import (
    expand_word_times_monomial,
)
from symop_proto.core.monomial import Monomial
from symop_proto.core.operators import ModeOp
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel


def make_mode(path: str = "A") -> ModeOp:
    env = GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0, phi0=0.0)
    label = ModeLabel(PathLabel(path), PolarizationLabel.H())
    return ModeOp(env=env, label=label)


class TestExpandWordTimesMonomial(ExtendedTestCase):
    def test_empty_word_returns_same_monomial(self):
        m = make_mode("A")
        M = Monomial(creators=(m.create,))
        out = expand_word_times_monomial((), M)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].monomial.signature, M.signature)
        self.assertComplexAlmostEqual(out[0].coeff, 1.0 + 0.0j)

    def test_ann_then_creator_same_mode_gives_identity_and_adag_a(self):
        m = make_mode("A")
        M = Monomial(creators=(m.create,))
        out = expand_word_times_monomial((m.ann,), M)
        # Expect two terms: identity and a^dag a, both with coeff +1
        sigs = [t.monomial.signature for t in out]
        id_sig = Monomial().signature
        adag_a_sig = Monomial(creators=(m.create,), annihilators=(m.ann,)).signature
        self.assertEqual(set(sigs), {id_sig, adag_a_sig})
        coeff_by_sig = {t.monomial.signature: t.coeff for t in out}
        self.assertComplexAlmostEqual(coeff_by_sig[id_sig], 1.0 + 0.0j)
        self.assertComplexAlmostEqual(coeff_by_sig[adag_a_sig], 1.0 + 0.0j)

    def test_creation_on_different_mode_just_concatenates_creators(self):
        a = make_mode("A")
        b = make_mode("B")
        M = Monomial(creators=(a.create,))
        out = expand_word_times_monomial((b.create,), M)
        # Normal order: creators (b, a)
        self.assertEqual(len(out), 1)
        expected = Monomial(creators=(b.create, a.create))
        self.assertEqual(out[0].monomial.signature, expected.signature)
        self.assertComplexAlmostEqual(out[0].coeff, 1.0 + 0.0j)

    def test_word_can_be_generator(self):
        m = make_mode("A")
        M = Monomial(creators=(m.create,))
        word_gen = (op for op in (m.ann,))
        out = expand_word_times_monomial(word_gen, M)
        self.assertTrue(isinstance(out, list))
        self.assertGreaterEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
