import unittest

from symop.ccr.algebra.ket.from_ops import ket_from_ops
from symop.ccr.algebra.ket.multiply import ket_multiply
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode, set_symmetric_overlap


class TestKetMultiply(unittest.TestCase):
    def test_empty_times_empty_is_empty(self) -> None:
        result = ket_multiply((), ())
        self.assertEqual(result, ())

    def test_empty_left_factor_gives_empty(self) -> None:
        mode = make_mode()
        b = ket_from_ops(creators=(mode.cre,), coeff=1.0)

        result = ket_multiply((), b)

        self.assertEqual(result, ())

    def test_empty_right_factor_gives_empty(self) -> None:
        mode = make_mode()
        a = ket_from_ops(creators=(mode.cre,), coeff=1.0)

        result = ket_multiply(a, ())

        self.assertEqual(result, ())

    def test_identity_multiplies_as_neutral_element_on_left(self) -> None:
        mode = make_mode()

        a = ket_from_ops(coeff=1.0)
        b = ket_from_ops(creators=(mode.cre,), coeff=2.0)

        result = ket_multiply(a, b)

        self.assertEqual(result, b)

    def test_identity_multiplies_as_neutral_element_on_right(self) -> None:
        mode = make_mode()

        a = ket_from_ops(creators=(mode.cre,), coeff=2.0)
        b = ket_from_ops(coeff=1.0)

        result = ket_multiply(a, b)

        self.assertEqual(result, a)

    def test_creator_times_creator_concatenates(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = ket_from_ops(creators=(mode_a.cre,), coeff=2.0)
        b = ket_from_ops(creators=(mode_b.cre,), coeff=3.0)

        result = ket_multiply(a, b)

        expected = ket_from_ops(
            creators=(mode_a.cre, mode_b.cre),
            coeff=6.0,
        )
        self.assertEqual(result, expected)

    def test_annihilator_times_creator_same_mode_produces_contraction_and_pass_term(
        self,
    ) -> None:
        mode = make_mode()

        a = ket_from_ops(annihilators=(mode.ann,), coeff=2.0)
        b = ket_from_ops(creators=(mode.cre,), coeff=3.0)

        result = ket_multiply(a, b)

        expected = (
            KetTerm(
                coeff=6.0,
                monomial=Monomial.identity(),
            ),
            KetTerm(
                coeff=6.0,
                monomial=Monomial(
                    creators=(mode.cre,),
                    annihilators=(mode.ann,),
                ),
            ),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)

    def test_annihilator_times_creator_orthogonal_modes_gives_only_pass_term(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = ket_from_ops(annihilators=(mode_a.ann,), coeff=2.0)
        b = ket_from_ops(creators=(mode_b.cre,), coeff=3.0)

        result = ket_multiply(a, b)

        expected = (
            KetTerm(
                coeff=6.0,
                monomial=Monomial(
                    creators=(mode_b.cre,),
                    annihilators=(mode_a.ann,),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_annihilator_times_creator_overlapping_modes_uses_overlap_weight(self) -> None:
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
        set_symmetric_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            overlap,
        )

        a = ket_from_ops(annihilators=(mode_a.ann,), coeff=2.0)
        b = ket_from_ops(creators=(mode_b.cre,), coeff=3.0)

        result = ket_multiply(a, b)

        expected = (
            KetTerm(
                coeff=6.0 * overlap,
                monomial=Monomial.identity(),
            ),
            KetTerm(
                coeff=6.0,
                monomial=Monomial(
                    creators=(mode_b.cre,),
                    annihilators=(mode_a.ann,),
                ),
            ),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)

    def test_product_distributes_over_multiple_terms(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = (
            *ket_from_ops(creators=(mode_a.cre,), coeff=2.0),
            *ket_from_ops(creators=(mode_b.cre,), coeff=3.0),
        )
        b = ket_from_ops(coeff=5.0)

        result = ket_multiply(a, b)

        expected = tuple(
            sorted(
                (
                    KetTerm(
                        coeff=10.0,
                        monomial=Monomial(creators=(mode_a.cre,), annihilators=()),
                    ),
                    KetTerm(
                        coeff=15.0,
                        monomial=Monomial(creators=(mode_b.cre,), annihilators=()),
                    ),
                ),
                key=lambda t: t.monomial.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_small_input_terms_are_skipped(self) -> None:
        mode = make_mode()

        a = ket_from_ops(creators=(mode.cre,), coeff=1e-15)
        b = ket_from_ops(creators=(mode.cre,), coeff=2.0)

        result = ket_multiply(a, b, eps=1e-12)

        self.assertEqual(result, ())

    def test_zero_result_is_dropped_after_combination(self) -> None:
        mode = make_mode()

        a = (
            *ket_from_ops(creators=(mode.cre,), coeff=1.0),
            *ket_from_ops(creators=(mode.cre,), coeff=-1.0),
        )
        b = ket_from_ops(coeff=1.0)

        result = ket_multiply(a, b)

        self.assertEqual(result, ())

    def test_approx_plumbing_keeps_valid_result(self) -> None:
        mode = make_mode()

        a = ket_from_ops(creators=(mode.cre,), coeff=2.0)
        b = ket_from_ops(coeff=3.0)

        result = ket_multiply(
            a,
            b,
            approx=True,
            decimals=7,
            ignore_global_phase=True,
        )

        expected = ket_from_ops(creators=(mode.cre,), coeff=6.0)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
