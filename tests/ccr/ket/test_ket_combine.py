import unittest

from symop.ccr.algebra.ket.combine import combine_like_terms_ket
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode


class TestCombineLikeTermsKet(unittest.TestCase):
    def test_empty_input_returns_empty_tuple(self) -> None:
        result = combine_like_terms_ket(())
        self.assertEqual(result, ())

    def test_single_term_is_returned_unchanged(self) -> None:
        mode = make_mode()
        term = KetTerm(
            coeff=2.0 + 1.0j,
            monomial=Monomial(creators=(mode.cre,)),
        )

        result = combine_like_terms_ket((term,))

        self.assertEqual(result, (term,))

    def test_exactly_like_terms_are_combined(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,))

        terms = (
            KetTerm(coeff=2.0, monomial=monomial),
            KetTerm(coeff=3.0 + 1.0j, monomial=monomial),
        )

        result = combine_like_terms_ket(terms)

        expected = (
            KetTerm(coeff=5.0 + 1.0j, monomial=monomial),
        )
        self.assertEqual(result, expected)

    def test_distinct_monomials_are_not_combined(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial_a = Monomial(creators=(mode_a.cre,))
        monomial_b = Monomial(creators=(mode_b.cre,))

        terms = (
            KetTerm(coeff=2.0, monomial=monomial_a),
            KetTerm(coeff=3.0, monomial=monomial_b),
        )

        result = combine_like_terms_ket(terms)

        expected = (
            KetTerm(coeff=2.0, monomial=monomial_a),
            KetTerm(coeff=3.0, monomial=monomial_b),
        )
        self.assertEqual(result, expected)

    def test_terms_with_same_modes_but_different_structure_are_not_combined(self) -> None:
        mode = make_mode()

        monomial_a = Monomial(creators=(mode.cre,), annihilators=())
        monomial_b = Monomial(creators=(), annihilators=(mode.ann,))

        terms = (
            KetTerm(coeff=1.0, monomial=monomial_a),
            KetTerm(coeff=1.0, monomial=monomial_b),
        )

        result = combine_like_terms_ket(terms)

        expected = (
            KetTerm(coeff=1.0, monomial=monomial_b),
            KetTerm(coeff=1.0, monomial=monomial_a),
        )
        expected = tuple(sorted(expected, key=lambda t: t.monomial.signature))
        self.assertEqual(result, expected)

    def test_near_zero_combined_coefficient_is_dropped(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,))

        terms = (
            KetTerm(coeff=1.0, monomial=monomial),
            KetTerm(coeff=-1.0 + 1e-14j, monomial=monomial),
        )

        result = combine_like_terms_ket(terms, eps=1e-12)

        self.assertEqual(result, ())

    def test_coefficient_with_magnitude_above_eps_is_kept(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,))

        terms = (
            KetTerm(coeff=1.0, monomial=monomial),
            KetTerm(coeff=-1.0 + 1e-6j, monomial=monomial),
        )

        result = combine_like_terms_ket(terms, eps=1e-12)

        expected = (
            KetTerm(coeff=1e-6j, monomial=monomial),
        )
        self.assertEqual(result, expected)

    def test_output_is_sorted_by_exact_monomial_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial_b = Monomial(creators=(mode_b.cre,))
        monomial_a = Monomial(creators=(mode_a.cre,))

        terms = (
            KetTerm(coeff=1.0, monomial=monomial_b),
            KetTerm(coeff=2.0, monomial=monomial_a),
        )

        result = combine_like_terms_ket(terms)

        expected = tuple(
            sorted(
                (
                    KetTerm(coeff=1.0, monomial=monomial_b),
                    KetTerm(coeff=2.0, monomial=monomial_a),
                ),
                key=lambda t: t.monomial.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_exact_mode_differences_prevent_combining_when_approx_false(self) -> None:
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

        monomial_a = Monomial(creators=(mode_a.cre,))
        monomial_b = Monomial(creators=(mode_b.cre,))

        terms = (
            KetTerm(coeff=2.0, monomial=monomial_a),
            KetTerm(coeff=3.0, monomial=monomial_b),
        )

        result = combine_like_terms_ket(terms, approx=False)

        expected = tuple(
            sorted(
                (
                    KetTerm(coeff=2.0, monomial=monomial_a),
                    KetTerm(coeff=3.0, monomial=monomial_b),
                ),
                key=lambda t: t.monomial.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_approximately_equal_terms_can_be_combined(self) -> None:
        mode_a = make_mode(path="p", polarization="h", envelope="env")
        mode_b = make_mode(path="p", polarization="h", envelope="env")

        monomial_a = Monomial(creators=(mode_a.cre,))
        monomial_b = Monomial(creators=(mode_b.cre,))

        terms = (
            KetTerm(coeff=2.0, monomial=monomial_a),
            KetTerm(coeff=3.0, monomial=monomial_b),
        )

        result = combine_like_terms_ket(
            terms,
            approx=True,
            decimals=12,
            ignore_global_phase=False,
        )

        expected = (
            KetTerm(coeff=5.0, monomial=monomial_a),
        )
        self.assertEqual(result, expected)

    def test_approximate_combining_can_ignore_global_phase(self) -> None:
        mode_a = make_mode(path="p", polarization="h", envelope="env")
        mode_b = make_mode(path="p", polarization="h", envelope="env")

        monomial_a = Monomial(creators=(mode_a.cre,))
        monomial_b = Monomial(creators=(mode_b.cre,))

        terms = (
            KetTerm(coeff=1.5, monomial=monomial_a),
            KetTerm(coeff=2.5, monomial=monomial_b),
        )

        result = combine_like_terms_ket(
            terms,
            approx=True,
            decimals=12,
            ignore_global_phase=True,
        )

        expected = (
            KetTerm(coeff=4.0, monomial=monomial_a),
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
