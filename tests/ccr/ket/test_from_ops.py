import unittest

from symop.ccr.algebra.ket.from_ops import ket_from_ops
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode


class TestKetFromOps(unittest.TestCase):
    def test_empty_inputs_return_identity_term(self) -> None:
        result = ket_from_ops()

        expected = (
            KetTerm(
                coeff=1.0,
                monomial=Monomial(creators=(), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)

    def test_creators_only_constructs_creator_only_monomial(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = ket_from_ops(
            creators=(mode_a.cre, mode_b.cre),
            coeff=2.0,
        )

        expected = (
            KetTerm(
                coeff=2.0,
                monomial=Monomial(
                    creators=(mode_a.cre, mode_b.cre),
                    annihilators=(),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_annihilators_only_constructs_annihilator_only_monomial(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = ket_from_ops(
            annihilators=(mode_a.ann, mode_b.ann),
            coeff=3.0,
        )

        expected = (
            KetTerm(
                coeff=3.0,
                monomial=Monomial(
                    creators=(),
                    annihilators=(mode_a.ann, mode_b.ann),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_creators_and_annihilators_preserve_order(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")
        mode_c = make_mode(path="c")

        result = ket_from_ops(
            creators=(mode_b.cre, mode_a.cre),
            annihilators=(mode_c.ann, mode_b.ann),
            coeff=1.0 + 2.0j,
        )

        expected = (
            KetTerm(
                coeff=1.0 + 2.0j,
                monomial=Monomial(
                    creators=(mode_b.cre, mode_a.cre),
                    annihilators=(mode_c.ann, mode_b.ann),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_general_iterables_are_materialized(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        creators = (op for op in (mode_a.cre, mode_b.cre))
        annihilators = (op for op in (mode_b.ann,))

        result = ket_from_ops(
            creators=creators,
            annihilators=annihilators,
            coeff=4.0,
        )

        expected = (
            KetTerm(
                coeff=4.0,
                monomial=Monomial(
                    creators=(mode_a.cre, mode_b.cre),
                    annihilators=(mode_b.ann,),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_invalid_creator_operator_raises_value_error(self) -> None:
        mode = make_mode()

        with self.assertRaises(ValueError):
            ket_from_ops(creators=(mode.ann,))

    def test_invalid_annihilator_operator_raises_value_error(self) -> None:
        mode = make_mode()

        with self.assertRaises(ValueError):
            ket_from_ops(annihilators=(mode.cre,))

    def test_mixed_invalid_inputs_raise_value_error_on_creators_first(self) -> None:
        mode = make_mode()

        with self.assertRaises(ValueError):
            ket_from_ops(
                creators=(mode.cre, mode.ann),
                annihilators=(mode.ann,),
            )

    def test_coefficient_zero_is_dropped_by_canonicalization(self) -> None:
        mode = make_mode()

        result = ket_from_ops(
            creators=(mode.cre,),
            coeff=0.0,
        )

        self.assertEqual(result, ())

    def test_approx_flag_plumbing_keeps_single_term_result(self) -> None:
        mode = make_mode()

        result = ket_from_ops(
            creators=(mode.cre,),
            coeff=2.5,
            approx=True,
            decimals=7,
            ignore_global_phase=True,
        )

        expected = (
            KetTerm(
                coeff=2.5,
                monomial=Monomial(
                    creators=(mode.cre,),
                    annihilators=(),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_identity_term_has_expected_structure(self) -> None:
        result = ket_from_ops(coeff=3.0)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].coeff, 3.0)
        self.assertTrue(result[0].is_identity)
        self.assertEqual(result[0].creation_count, 0)
        self.assertEqual(result[0].annihilation_count, 0)

    def test_complex_coefficient_is_preserved(self) -> None:
        mode = make_mode()
        coeff = 2.0 - 3.0j

        result = ket_from_ops(
            creators=(mode.cre,),
            annihilators=(mode.ann,),
            coeff=coeff,
        )

        self.assertEqual(result[0].coeff, coeff)
        self.assertEqual(
            result[0].monomial,
            Monomial(
                creators=(mode.cre,),
                annihilators=(mode.ann,),
            ),
        )


if __name__ == "__main__":
    unittest.main()
