from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.expand.monomial_times_word import (
    expand_monomial_times_word,
)
from symop_proto.core.monomial import Monomial
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.core.operators import ModeOp


def make_mode(path: str = "A") -> ModeOp:
    env = GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0)
    label = ModeLabel(PathLabel(path), PolarizationLabel.H())
    return ModeOp(env=env, label=label)


class TestExpandMonomialTimesWord(ExtendedTestCase):
    def test_empty_word_returns_same_monomial(self):
        m = make_mode("A")
        mono = Monomial(creators=(m.create,))
        out = expand_monomial_times_word(mono, ())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].monomial.signature, mono.signature)

    def test_appending_annihilator(self):
        m = make_mode("A")
        mono = Monomial(creators=(m.create,))
        out = expand_monomial_times_word(mono, (m.ann,))
        # Expect one terms: a_dag a
        sigs = [t.monomial.signature for t in out]
        adag_a_sig = Monomial(creators=(m.create,), annihilators=(m.ann,)).signature
        self.assertEqual(set(sigs), {adag_a_sig})

    def test_with_multiple_modes(self):
        a = make_mode("A")
        b = make_mode("B")
        mono = Monomial(creators=(a.create,))
        out = expand_monomial_times_word(mono, (b.create,))
        # Expect one combined normally ordered term (a_dag b_dag)
        self.assertEqual(len(out), 1)
        expected = Monomial(creators=(a.create, b.create))
        self.assertEqual(out[0].monomial.signature, expected.signature)

    def test_word_can_be_generator(self):
        m = make_mode("A")
        mono = Monomial(creators=(m.create,))
        word_gen = (op for op in (m.ann,))
        out = expand_monomial_times_word(mono, word_gen)
        self.assertIsInstance(out, list)
        self.assertGreaterEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
