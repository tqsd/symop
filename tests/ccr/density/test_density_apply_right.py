import unittest

from symop.ccr.algebra.density.apply_right import apply_right
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestApplyRight(unittest.TestCase):
    def test_empty_terms_return_empty_tuple(self) -> None:
        result = apply_right((), ())
        self.assertEqual(result, ())

    def test_empty_word_leaves_density_terms_unchanged(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=2.0 + 1.0j,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(), annihilators=(mode_b.ann,)),
            ),
        )

        result = apply_right(terms, ())

        self.assertEqual(result, terms)

    def test_single_creator_applied_on_right_updates_only_right(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = apply_right(terms, (mode_b.cre,))

        expected = (
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(), annihilators=(mode_b.ann,)),
            ),
        )
        self.assertEqual(result, expected)

    def test_single_annihilator_applied_on_right_updates_only_right(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = apply_right(terms, (mode_b.ann,))

        expected = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)

    def test_matching_creator_annihilator_on_right_produces_two_terms(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = apply_right(terms, (mode.cre,))

        expected = tuple(
            sorted(
                (
                    DensityTerm(
                        coeff=2.0,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                    DensityTerm(
                        coeff=2.0,
                        left=Monomial.identity(),
                        right=Monomial(
                            creators=(mode.cre,),
                            annihilators=(mode.ann,),
                        ),
                    ),
                ),
                key=lambda t: t.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_overlap_weight_is_applied_to_contracted_right_term(self) -> None:
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

        terms = (
            DensityTerm(
                coeff=3.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
        )

        result = apply_right(terms, (mode_b.cre,))

        expected = tuple(
            sorted(
                (
                    DensityTerm(
                        coeff=3.0 * overlap.conjugate(),
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                    DensityTerm(
                        coeff=3.0,
                        left=Monomial.identity(),
                        right=Monomial(
                            creators=(mode_a.cre,),
                            annihilators=(mode_b.ann,),
                        ),
                    ),
                ),
                key=lambda t: t.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_left_monomial_is_preserved_for_all_generated_terms(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = Monomial(
            creators=(mode_a.cre,),
            annihilators=(mode_a.ann,),
        )
        terms = (
            DensityTerm(
                coeff=1.0,
                left=left,
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        result = apply_right(terms, (mode_b.ann,))

        self.assertTrue(all(term.left == left for term in result))

    def test_coefficients_from_multiple_input_terms_are_combined(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
            DensityTerm(
                coeff=3.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        result = apply_right(terms, (mode.ann,))

        expected = (
            DensityTerm(
                coeff=5.0,
                left=Monomial.identity(),
                right=Monomial(
                    creators=(mode.cre,),
                    annihilators=(),
                ),
            ),
        )
        self.assertEqual(result, expected)

    def test_distinct_left_monomials_remain_separate(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = apply_right(terms, (mode_a.ann,))

        expected = tuple(
            sorted(
                (
                    DensityTerm(
                        coeff=1.0,
                        left=Monomial(creators=(mode_a.cre,), annihilators=()),
                        right=Monomial(creators=(mode_a.cre,), annihilators=()),
                    ),
                    DensityTerm(
                        coeff=1.0,
                        left=Monomial(creators=(mode_b.cre,), annihilators=()),
                        right=Monomial(creators=(mode_a.cre,), annihilators=()),
                    ),
                ),
                key=lambda t: t.signature,
            )
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
