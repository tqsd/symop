import unittest

from symop.ccr.algebra.density.apply_left import apply_left
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestApplyLeft(unittest.TestCase):
    def test_empty_terms_return_empty_tuple(self) -> None:
        result = apply_left((), ())
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

        result = apply_left(terms, ())

        self.assertEqual(result, terms)

    def test_single_creator_applied_to_identity_left_updates_only_left(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=3.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        result = apply_left(terms, (mode_a.cre,))

        expected = (
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)

    def test_single_annihilator_applied_to_identity_left_keeps_symbolic_annihilator(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial(annihilators=(mode_b.ann,)),
            ),
        )

        result = apply_left(terms, (mode_a.ann,))

        expected = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(), annihilators=(mode_a.ann,)),
                right=Monomial(annihilators=(mode_b.ann,)),
            ),
        )
        self.assertEqual(result, expected)

    def test_matching_annihilator_creator_on_left_produces_two_terms(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = apply_left(terms, (mode.ann,))

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
                        left=Monomial(
                            creators=(mode.cre,),
                            annihilators=(mode.ann,),
                        ),
                        right=Monomial.identity(),
                    ),
                ),
                key=lambda t: t.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_overlap_weight_is_applied_to_contracted_left_term(self) -> None:
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
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        result = apply_left(terms, (mode_a.ann,))

        expected = tuple(
            sorted(
                (
                    DensityTerm(
                        coeff=3.0 * overlap,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                    DensityTerm(
                        coeff=3.0,
                        left=Monomial(
                            creators=(mode_b.cre,),
                            annihilators=(mode_a.ann,),
                        ),
                        right=Monomial.identity(),
                    ),
                ),
                key=lambda t: t.signature,
            )
        )
        self.assertEqual(result, expected)

    def test_right_monomial_is_preserved_for_all_generated_terms(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        right = Monomial(
            creators=(mode_b.cre,),
            annihilators=(mode_b.ann,),
        )
        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=right,
            ),
        )

        result = apply_left(terms, (mode_a.ann,))

        self.assertTrue(all(term.right == right for term in result))

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

        result = apply_left(terms, (mode.cre,))

        expected = (
            DensityTerm(
                coeff=5.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_distinct_right_monomials_remain_separate(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        result = apply_left(terms, (mode_a.cre,))

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
                        left=Monomial(creators=(mode_a.cre,), annihilators=()),
                        right=Monomial(creators=(mode_b.cre,), annihilators=()),
                    ),
                ),
                key=lambda t: t.signature,
            )
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
