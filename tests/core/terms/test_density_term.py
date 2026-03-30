import unittest

from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.core.support.fakes import make_mode


class TestDensityTerm(unittest.TestCase):
    def test_identity_returns_unit_identity_term(self) -> None:
        term = DensityTerm.identity()

        self.assertEqual(term.coeff, 1.0)
        self.assertTrue(term.is_identity_left)
        self.assertTrue(term.is_identity_right)
        self.assertTrue(term.is_creator_only_left)
        self.assertTrue(term.is_creator_only_right)
        self.assertTrue(term.is_creator_only)
        self.assertTrue(term.is_annihilator_only_left)
        self.assertTrue(term.is_annihilator_only_right)
        self.assertTrue(term.is_annihilator_only)
        self.assertTrue(term.is_diagonal_in_monomials)
        self.assertEqual(term.creation_count_left, 0)
        self.assertEqual(term.creation_count_right, 0)
        self.assertEqual(term.annihilation_count_left, 0)
        self.assertEqual(term.annihilation_count_right, 0)
        self.assertEqual(term.mode_ops_left, ())
        self.assertEqual(term.mode_ops_right, ())

    def test_adjoint_conjugates_coeff_and_swaps_left_and_right(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = Monomial(
            creators=(mode_a.cre,),
            annihilators=(mode_b.ann,),
        )
        right = Monomial(
            creators=(mode_b.cre,),
            annihilators=(),
        )
        term = DensityTerm(coeff=2.0 + 3.0j, left=left, right=right)

        adj = term.adjoint()

        self.assertEqual(adj.coeff, 2.0 - 3.0j)
        self.assertEqual(adj.left, right)
        self.assertEqual(adj.right, left)

    def test_scaled_multiplies_coefficient_only(self) -> None:
        mode = make_mode()

        left = Monomial(creators=(mode.cre,))
        right = Monomial(annihilators=(mode.ann,))
        term = DensityTerm(coeff=2.0 + 1.0j, left=left, right=right)

        scaled = term.scaled(3.0 - 2.0j)

        self.assertEqual(scaled.coeff, (2.0 + 1.0j) * (3.0 - 2.0j))
        self.assertEqual(scaled.left, left)
        self.assertEqual(scaled.right, right)
        self.assertEqual(term.coeff, 2.0 + 1.0j)

    def test_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = Monomial(creators=(mode_a.cre,))
        right = Monomial(annihilators=(mode_b.ann,))
        term = DensityTerm(coeff=5.0, left=left, right=right)

        self.assertEqual(
            term.signature,
            ("DT", "L", left.signature, "R", right.signature),
        )

    def test_approx_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = Monomial(creators=(mode_a.cre,))
        right = Monomial(annihilators=(mode_b.ann,))
        term = DensityTerm(coeff=5.0, left=left, right=right)

        self.assertEqual(
            term.approx_signature(decimals=7, ignore_global_phase=True),
            (
                "DT_approx",
                "L",
                left.approx_signature(
                    decimals=7,
                    ignore_global_phase=True,
                ),
                "R",
                right.approx_signature(
                    decimals=7,
                    ignore_global_phase=True,
                ),
            ),
        )

    def test_is_creator_only_left_true_for_creator_only_monomial(self) -> None:
        mode = make_mode()

        term = DensityTerm(
            coeff=1.0,
            left=Monomial(creators=(mode.cre,)),
            right=Monomial(annihilators=(mode.ann,)),
        )

        self.assertTrue(term.is_creator_only_left)
        self.assertFalse(term.is_creator_only_right)
        self.assertFalse(term.is_creator_only)

    def test_is_creator_only_treats_identity_as_creator_only(self) -> None:
        mode = make_mode()

        term = DensityTerm(
            coeff=1.0,
            left=Monomial(),
            right=Monomial(creators=(mode.cre,)),
        )

        self.assertTrue(term.is_creator_only_left)
        self.assertTrue(term.is_creator_only_right)
        self.assertTrue(term.is_creator_only)

    def test_is_annihilator_only_left_true_for_annihilator_only_monomial(self) -> None:
        mode = make_mode()

        term = DensityTerm(
            coeff=1.0,
            left=Monomial(annihilators=(mode.ann,)),
            right=Monomial(creators=(mode.cre,)),
        )

        self.assertTrue(term.is_annihilator_only_left)
        self.assertFalse(term.is_annihilator_only_right)
        self.assertFalse(term.is_annihilator_only)

    def test_is_annihilator_only_treats_identity_as_annihilator_only(self) -> None:
        mode = make_mode()

        term = DensityTerm(
            coeff=1.0,
            left=Monomial(),
            right=Monomial(annihilators=(mode.ann,)),
        )

        self.assertTrue(term.is_annihilator_only_left)
        self.assertTrue(term.is_annihilator_only_right)
        self.assertTrue(term.is_annihilator_only)

    def test_is_identity_left_and_right(self) -> None:
        mode = make_mode()

        term = DensityTerm(
            coeff=1.0,
            left=Monomial(),
            right=Monomial(creators=(mode.cre,)),
        )

        self.assertTrue(term.is_identity_left)
        self.assertFalse(term.is_identity_right)

    def test_is_diagonal_in_monomials_true_when_signatures_match(self) -> None:
        mode = make_mode()

        left = Monomial(creators=(mode.cre,), annihilators=(mode.ann,))
        right = Monomial(creators=(mode.cre,), annihilators=(mode.ann,))
        term = DensityTerm(coeff=1.0, left=left, right=right)

        self.assertTrue(term.is_diagonal_in_monomials)

    def test_is_diagonal_in_monomials_false_when_signatures_differ(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = Monomial(creators=(mode_a.cre,))
        right = Monomial(creators=(mode_b.cre,))
        term = DensityTerm(coeff=1.0, left=left, right=right)

        self.assertFalse(term.is_diagonal_in_monomials)

    def test_creation_count_left_and_right(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = DensityTerm(
            coeff=1.0,
            left=Monomial(creators=(mode_a.cre, mode_b.cre)),
            right=Monomial(creators=(mode_b.cre,)),
        )

        self.assertEqual(term.creation_count_left, 2)
        self.assertEqual(term.creation_count_right, 1)

    def test_annihilation_count_left_and_right(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = DensityTerm(
            coeff=1.0,
            left=Monomial(annihilators=(mode_a.ann,)),
            right=Monomial(annihilators=(mode_a.ann, mode_b.ann)),
        )

        self.assertEqual(term.annihilation_count_left, 1)
        self.assertEqual(term.annihilation_count_right, 2)

    def test_mode_ops_left_and_right_delegate_to_monomials(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = Monomial(
            creators=(mode_a.cre, mode_b.cre, mode_a.cre),
            annihilators=(),
        )
        right = Monomial(
            creators=(),
            annihilators=(mode_b.ann, mode_a.ann, mode_b.ann),
        )
        term = DensityTerm(coeff=1.0, left=left, right=right)

        self.assertEqual(term.mode_ops_left, (mode_a, mode_b))
        self.assertEqual(term.mode_ops_right, (mode_b, mode_a))

    def test_double_adjoint_returns_original_term(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = DensityTerm(
            coeff=1.0 + 2.0j,
            left=Monomial(
                creators=(mode_a.cre,),
                annihilators=(mode_b.ann,),
            ),
            right=Monomial(
                creators=(mode_b.cre,),
                annihilators=(),
            ),
        )

        self.assertEqual(term.adjoint().adjoint(), term)


if __name__ == "__main__":
    unittest.main()
