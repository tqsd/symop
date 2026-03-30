import unittest

from symop.ccr.algebra.density.expand_monomial_times_word import (
    expand_monomial_times_word,
)
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestExpandMonomialTimesWord(unittest.TestCase):
    def test_empty_word_returns_original_monomial(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,), annihilators=())

        result = expand_monomial_times_word(monomial, ())

        expected = [
            KetTerm(coeff=1.0, monomial=monomial),
        ]
        self.assertEqual(result, expected)

    def test_identity_monomial_with_single_creator(self) -> None:
        mode = make_mode()

        result = expand_monomial_times_word(Monomial.identity(), (mode.cre,))

        expected = [
            KetTerm(
                coeff=1.0,
                monomial=Monomial(creators=(mode.cre,), annihilators=()),
            )
        ]
        self.assertEqual(result, expected)

    def test_identity_monomial_with_single_annihilator(self) -> None:
        mode = make_mode()

        result = expand_monomial_times_word(Monomial.identity(), (mode.ann,))

        expected = [
            KetTerm(
                coeff=1.0,
                monomial=Monomial(creators=(), annihilators=(mode.ann,)),
            )
        ]
        self.assertEqual(result, expected)

    def test_creator_monomial_times_creator_word_concatenates(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(creators=(mode_a.cre,), annihilators=())
        result = expand_monomial_times_word(monomial, (mode_b.cre,))

        expected = [
            KetTerm(
                coeff=1.0,
                monomial=Monomial(
                    creators=(mode_a.cre, mode_b.cre),
                    annihilators=(),
                ),
            )
        ]
        self.assertEqual(result, expected)

    def test_annihilator_then_creator_can_contract(self) -> None:
        mode = make_mode()

        monomial = Monomial(creators=(), annihilators=(mode.ann,))
        result = expand_monomial_times_word(monomial, (mode.cre,))

        expected = sorted(
            [
                KetTerm(coeff=1.0, monomial=Monomial.identity()),
                KetTerm(
                    coeff=1.0,
                    monomial=Monomial(
                        creators=(mode.cre,),
                        annihilators=(mode.ann,),
                    ),
                ),
            ],
            key=lambda t: t.monomial.signature,
        )
        self.assertEqual(result, expected)

    def test_annihilator_then_overlapping_creator_uses_overlap(self) -> None:
        envelope_table = {}

        mode_a = make_mode(
            path="p",
            polarization="h",
            envelope="env_a",
            envelope_table=envelope_table,
        )
        mode_b = make_mode(
            path="p",
            polarization="h",
            envelope="env_b",
            envelope_table=envelope_table,
        )

        overlap = 0.25 + 0.5j
        set_hermitian_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            overlap,
        )

        monomial = Monomial(creators=(), annihilators=(mode_a.ann,))
        result = expand_monomial_times_word(monomial, (mode_b.cre,))

        expected = sorted(
            [
                KetTerm(coeff=overlap, monomial=Monomial.identity()),
                KetTerm(
                    coeff=1.0,
                    monomial=Monomial(
                        creators=(mode_b.cre,),
                        annihilators=(mode_a.ann,),
                    ),
                ),
            ],
            key=lambda t: t.monomial.signature,
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
