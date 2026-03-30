import unittest

from symop.ccr.algebra.ket.apply import (
    ket_apply_word,
    ket_apply_words_linear,
)
from symop.ccr.algebra.ket.from_ops import ket_from_ops
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode


class TestKetApplyWord(unittest.TestCase):
    def test_apply_empty_word_returns_same_ket(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(mode.cre,), coeff=2.0)

        result = ket_apply_word(ket, ())

        self.assertEqual(result, ket)

    def test_apply_single_creator_to_identity_ket(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_word(ket, (mode.cre,))

        expected = ket_from_ops(creators=(mode.cre,), coeff=1.0)
        self.assertEqual(result, expected)

    def test_apply_single_annihilator_to_identity_ket_keeps_symbolic_annihilator(
        self,
    ) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_word(ket, (mode.ann,))

        expected = ket_from_ops(annihilators=(mode.ann,), coeff=1.0)
        self.assertEqual(result, expected)

    def test_apply_creator_then_creator_builds_longer_creator_string(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        ket = ket_from_ops(creators=(mode_b.cre,), coeff=1.0)
        result = ket_apply_word(ket, (mode_a.cre,))

        expected = ket_from_ops(
            creators=(mode_a.cre, mode_b.cre),
            coeff=1.0,
        )
        self.assertEqual(result, expected)

    def test_apply_annihilator_to_matching_single_particle_ket_returns_symbolic_sum(
        self,
    ) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(mode.cre,), coeff=1.0)

        result = ket_apply_word(ket, (mode.ann,))

        expected = (
            *ket_from_ops(creators=(), annihilators=(), coeff=1.0),
            *ket_from_ops(
                creators=(mode.cre,),
                annihilators=(mode.ann,),
                coeff=1.0,
            ),
        )
        self.assertEqual(result, expected)

    def test_apply_word_respects_word_order(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_word(ket, (mode.cre, mode.cre))

        expected = ket_from_ops(creators=(mode.cre, mode.cre), coeff=1.0)
        self.assertEqual(result, expected)


class TestKetApplyWordsLinear(unittest.TestCase):
    def test_apply_words_linear_empty_terms_returns_zero(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(mode.cre,), coeff=1.0)

        result = ket_apply_words_linear(ket, ())

        self.assertEqual(result, ())

    def test_apply_words_linear_single_term_matches_apply_word(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_words_linear(
            ket,
            ((1.0, (mode.cre,)),),
        )

        expected = ket_apply_word(ket, (mode.cre,))
        self.assertEqual(result, expected)

    def test_apply_words_linear_skips_zero_coefficients(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_words_linear(
            ket,
            (
                (0.0, (mode_a.cre,)),
                (1.0, (mode_b.cre,)),
            ),
        )

        expected = ket_apply_word(ket, (mode_b.cre,))
        self.assertEqual(result, expected)

    def test_apply_words_linear_scales_non_unit_coefficients(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_words_linear(
            ket,
            ((2.0 + 3.0j, (mode.cre,)),),
        )

        expected = ket_from_ops(creators=(mode.cre,), coeff=2.0 + 3.0j)
        self.assertEqual(result, expected)

    def test_apply_words_linear_combines_like_terms(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_words_linear(
            ket,
            (
                (2.0, (mode.cre,)),
                (3.0, (mode.cre,)),
            ),
        )

        expected = ket_from_ops(creators=(mode.cre,), coeff=5.0)
        self.assertEqual(result, expected)

    def test_apply_words_linear_keeps_distinct_terms_separate(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_words_linear(
            ket,
            (
                (1.0, (mode_a.cre,)),
                (1.0, (mode_b.cre,)),
            ),
        )

        expected_a = KetTerm(
            coeff=1.0,
            monomial=ket_from_ops(creators=(mode_a.cre,), coeff=1.0)[0].monomial,
        )
        expected_b = KetTerm(
            coeff=1.0,
            monomial=ket_from_ops(creators=(mode_b.cre,), coeff=1.0)[0].monomial,
        )

        self.assertEqual(result, (expected_a, expected_b))

    def test_apply_words_linear_can_cancel_to_zero(self) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_words_linear(
            ket,
            (
                (1.0, (mode.cre,)),
                (-1.0, (mode.cre,)),
            ),
        )

        self.assertEqual(result, ())

    def test_apply_words_linear_with_annihilator_on_identity_ket_keeps_symbolic_annihilator(
        self,
    ) -> None:
        mode = make_mode()
        ket = ket_from_ops(creators=(), annihilators=(), coeff=1.0)

        result = ket_apply_words_linear(
            ket,
            ((1.0, (mode.ann,)),),
        )

        expected = ket_from_ops(annihilators=(mode.ann,), coeff=1.0)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
