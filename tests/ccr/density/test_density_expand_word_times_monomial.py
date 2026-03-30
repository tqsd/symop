import unittest

from symop.ccr.algebra.density.expand_word_times_monomial import (
    expand_word_times_monomial,
)
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestExpandWordTimesMonomial(unittest.TestCase):
    def test_empty_word_returns_original_monomial(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,), annihilators=())

        result = expand_word_times_monomial((), monomial)

        expected = [
            KetTerm(coeff=1.0, monomial=monomial),
        ]
        self.assertEqual(result, expected)

    def test_single_creator_applied_to_identity_monomial(self) -> None:
        mode = make_mode()

        result = expand_word_times_monomial((mode.cre,), Monomial.identity())

        expected = [
            KetTerm(
                coeff=1.0,
                monomial=Monomial(creators=(mode.cre,), annihilators=()),
            )
        ]
        self.assertEqual(result, expected)

    def test_single_annihilator_applied_to_identity_monomial(self) -> None:
        mode = make_mode()

        result = expand_word_times_monomial((mode.ann,), Monomial.identity())

        expected = [
            KetTerm(
                coeff=1.0,
                monomial=Monomial(creators=(), annihilators=(mode.ann,)),
            )
        ]
        self.assertEqual(result, expected)

    def test_creator_applied_to_creator_monomial_concatenates(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(creators=(mode_b.cre,), annihilators=())
        result = expand_word_times_monomial((mode_a.cre,), monomial)

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

    def test_annihilator_applied_to_matching_creator_monomial_can_contract(self) -> None:
        mode = make_mode()

        monomial = Monomial(creators=(mode.cre,), annihilators=())
        result = expand_word_times_monomial((mode.ann,), monomial)

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

    def test_annihilator_applied_to_overlapping_creator_monomial_uses_overlap(self) -> None:
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

        monomial = Monomial(creators=(mode_b.cre,), annihilators=())
        result = expand_word_times_monomial((mode_a.ann,), monomial)

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

    def test_multiple_operators_are_applied_in_word_order(self) -> None:
        mode = make_mode()

        result = expand_word_times_monomial(
            (mode.cre, mode.cre),
            Monomial.identity(),
        )

        expected = [
            KetTerm(
                coeff=1.0,
                monomial=Monomial(
                    creators=(mode.cre, mode.cre),
                    annihilators=(),
                ),
            )
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
