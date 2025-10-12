from __future__ import annotations
import unittest
from typing import Tuple

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm

from symop_proto.algebra.ket.apply import (
    ket_apply_word,
    ket_apply_words_linear,
)
from symop_proto.algebra.ket.from_word import ket_from_word
from symop_proto.algebra.ket.combine import combine_like_terms_ket


def make_mode(
    path: str = "A",
    *,
    omega=1.0,
    sigma=0.3,
    tau=0.0,
    phi=0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


class TestKetApply(ExtendedTestCase):
    def test_apply_word_on_identity_gives_word(self):
        m = make_mode("A")
        ket_id: Tuple[KetTerm, ...] = (KetTerm(1.0, Monomial()),)
        out = ket_apply_word(ket_id, (m.create,))
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 1.0 + 0.0j)
        self.assertEqual(
            out[0].monomial.signature, Monomial(creators=(m.create,)).signature
        )

    def test_apply_words_linear_combines_like_terms(self):
        m = make_mode("A")
        ket_id: Tuple[KetTerm, ...] = (KetTerm(1.0, Monomial()),)
        out = ket_apply_words_linear(
            ket_id,
            terms=[
                (2.0, (m.create,)),
                (3.5, (m.create,)),
            ],
        )
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 5.5 + 0j)
        self.assertEqual(
            out[0].monomial.signature, Monomial(creators=(m.create,)).signature
        )

    def test_agrees_with_direct_expansion_for_nontrivial_order(self):
        m = make_mode("A")
        # |psi> = a^\dagger |vac>
        ket = (KetTerm(1.0, Monomial(creators=(m.create,))),)
        applied = ket_apply_word(ket, (m.ann,))
        # Directly expand the operator word a a^\dagger and then multiply by |vac>
        direct_word = ket_from_word(ops=(m.ann, m.create))
        direct = combine_like_terms_ket(
            direct_word
        )  # acting on |vac> is just the word itself
        # Both should be the same ket (up to canonical combination)
        self.assertEqual(
            [t.monomial.signature for t in applied],
            [t.monomial.signature for t in direct],
        )
        self.assertArrayAllClose(
            [t.coeff for t in applied],
            [t.coeff for t in direct],
        )


if __name__ == "__main__":
    unittest.main()
