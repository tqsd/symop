import unittest

from symop.ccr.algebra.ket.from_word import ket_from_word
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import (
    make_mode,
    set_symmetric_overlap,
)


class TestKetFromWord(unittest.TestCase):
    def test_empty_word_returns_identity(self) -> None:
        result = ket_from_word(ops=())

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_single_annihilator_returns_annihilator_only_monomial(self) -> None:
        mode = make_mode()

        result = ket_from_word(ops=(mode.ann,))

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(),
                    annihilators=(mode.ann,),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_single_creator_returns_creator_only_monomial(self) -> None:
        mode = make_mode()

        result = ket_from_word(ops=(mode.cre,))

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode.cre,),
                    annihilators=(),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_already_normal_ordered_word_is_preserved(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = ket_from_word(
            ops=(mode_a.cre, mode_b.cre, mode_b.ann),
        )

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode_a.cre, mode_b.cre),
                    annihilators=(mode_b.ann,),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_annihilator_then_matching_creator_gives_pass_and_contraction(self) -> None:
        mode = make_mode()

        result = ket_from_word(ops=(mode.ann, mode.cre))

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial.identity(),
            ),
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode.cre,),
                    annihilators=(mode.ann,),
                ),
            ),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)

    def test_annihilator_then_non_overlapping_creator_gives_only_pass_term(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = ket_from_word(ops=(mode_a.ann, mode_b.cre))

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode_b.cre,),
                    annihilators=(mode_a.ann,),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_annihilator_then_overlapping_creator_uses_overlap_weight(self) -> None:
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

        set_symmetric_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            0.25 + 0.5j,
        )

        result = ket_from_word(ops=(mode_a.ann, mode_b.cre))

        expected = (
            KetTerm(
                coeff=0.25 + 0.5j,
                monomial=Monomial.identity(),
            ),
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode_b.cre,),
                    annihilators=(mode_a.ann,),
                ),
            ),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)

    def test_two_annihilators_then_creator_contracts_once_with_each(self) -> None:
        mode = make_mode()

        result = ket_from_word(ops=(mode.ann, mode.ann, mode.cre))

        expected = (
            KetTerm(
                coeff=2.0 + 0.0j,
                monomial=Monomial(
                    creators=(),
                    annihilators=(mode.ann,),
                ),
            ),
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode.cre,),
                    annihilators=(mode.ann, mode.ann),
                ),
            ),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)

    def test_two_distinct_annihilators_then_creator_contracts_matching_one_only(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = ket_from_word(ops=(mode_a.ann, mode_b.ann, mode_b.cre))

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(),
                    annihilators=(mode_a.ann,),
                ),
            ),
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode_b.cre,),
                    annihilators=(mode_a.ann, mode_b.ann),
                ),
            ),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)

    def test_eps_drops_small_terms(self) -> None:
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

        set_symmetric_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            1e-15 + 0.0j,
        )

        result = ket_from_word(
            ops=(mode_a.ann, mode_b.cre),
            eps=1e-12,
        )

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode_b.cre,),
                    annihilators=(mode_a.ann,),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_output_is_sorted_by_exact_monomial_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = ket_from_word(ops=(mode_b.ann, mode_a.cre))

        expected = (
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode_a.cre,),
                    annihilators=(mode_b.ann,),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_longer_word_builds_symbolic_sum(self) -> None:
        mode = make_mode()

        result = ket_from_word(ops=(mode.ann, mode.cre, mode.cre))

        expected = (
            KetTerm(
                coeff=2.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode.cre,),
                    annihilators=(),
                ),
            ),
            KetTerm(
                coeff=1.0 + 0.0j,
                monomial=Monomial(
                    creators=(mode.cre, mode.cre),
                    annihilators=(mode.ann,),
                ),
            ),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
