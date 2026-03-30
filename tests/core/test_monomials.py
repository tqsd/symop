import unittest

from symop.core.monomial import Monomial

from tests.core.support.fakes import make_mode


class TestMonomial(unittest.TestCase):
    def test_post_init_normalizes_lists_to_tuples(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,), annihilators=(mode.ann,))

        self.assertIsInstance(monomial.creators, tuple)
        self.assertIsInstance(monomial.annihilators, tuple)
        self.assertEqual(monomial.creators, (mode.cre,))
        self.assertEqual(monomial.annihilators, (mode.ann,))

    def test_identity_staticmethod_returns_empty_monomial(self) -> None:
        monomial = Monomial.identity()

        self.assertEqual(monomial.creators, ())
        self.assertEqual(monomial.annihilators, ())
        self.assertTrue(monomial.is_identity)
        self.assertFalse(monomial.has_creators)
        self.assertFalse(monomial.has_annihilators)

    def test_mode_ops_returns_unique_modes_in_first_occurrence_order(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre, mode_b.cre, mode_a.cre),
            annihilators=(mode_b.ann, mode_a.ann),
        )

        self.assertEqual(monomial.mode_ops, (mode_a, mode_b))

    def test_mode_ops_empty_for_identity(self) -> None:
        monomial = Monomial.identity()
        self.assertEqual(monomial.mode_ops, ())

    def test_adjoint_swaps_sides_and_applies_dagger(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre, mode_b.cre),
            annihilators=(mode_b.ann, mode_a.ann),
        )

        adj = monomial.adjoint()

        self.assertEqual(
            adj.creators,
            (mode_b.cre, mode_a.cre),
        )
        self.assertEqual(
            adj.annihilators,
            (mode_a.ann, mode_b.ann),
        )

    def test_adjoint_of_creator_only_becomes_annihilator_only(self) -> None:
        mode = make_mode()

        monomial = Monomial(creators=(mode.cre,))
        adj = monomial.adjoint()

        self.assertEqual(adj.creators, ())
        self.assertEqual(adj.annihilators, (mode.ann,))
        self.assertTrue(adj.is_annihilator_only)

    def test_adjoint_of_annihilator_only_becomes_creator_only(self) -> None:
        mode = make_mode()

        monomial = Monomial(annihilators=(mode.ann,))
        adj = monomial.adjoint()

        self.assertEqual(adj.creators, (mode.cre,))
        self.assertEqual(adj.annihilators, ())
        self.assertTrue(adj.is_creator_only)

    def test_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre,),
            annihilators=(mode_b.ann,),
        )

        self.assertEqual(
            monomial.signature,
            (
                "monomial",
                "cre",
                (mode_a.cre.signature,),
                "ann",
                (mode_b.ann.signature,),
            ),
        )

    def test_approx_signature(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre,),
            annihilators=(mode_b.ann,),
        )

        self.assertEqual(
            monomial.approx_signature(decimals=7, ignore_global_phase=True),
            (
                "monomial_approx",
                "cre",
                (
                    mode_a.cre.approx_signature(
                        decimals=7,
                        ignore_global_phase=True,
                    ),
                ),
                "ann",
                (
                    mode_b.ann.approx_signature(
                        decimals=7,
                        ignore_global_phase=True,
                    ),
                ),
            ),
        )

    def test_is_creator_only_true_only_when_creators_present_and_no_annihilators(self) -> None:
        mode = make_mode()

        self.assertTrue(Monomial(creators=(mode.cre,)).is_creator_only)
        self.assertFalse(Monomial().is_creator_only)
        self.assertFalse(
            Monomial(creators=(mode.cre,), annihilators=(mode.ann,)).is_creator_only
        )
        self.assertFalse(Monomial(annihilators=(mode.ann,)).is_creator_only)

    def test_is_annihilator_only_true_only_when_annihilators_present_and_no_creators(self) -> None:
        mode = make_mode()

        self.assertTrue(Monomial(annihilators=(mode.ann,)).is_annihilator_only)
        self.assertFalse(Monomial().is_annihilator_only)
        self.assertFalse(
            Monomial(creators=(mode.cre,), annihilators=(mode.ann,)).is_annihilator_only
        )
        self.assertFalse(Monomial(creators=(mode.cre,)).is_annihilator_only)

    def test_is_identity_true_only_for_empty_monomial(self) -> None:
        mode = make_mode()

        self.assertTrue(Monomial().is_identity)
        self.assertFalse(Monomial(creators=(mode.cre,)).is_identity)
        self.assertFalse(Monomial(annihilators=(mode.ann,)).is_identity)
        self.assertFalse(
            Monomial(creators=(mode.cre,), annihilators=(mode.ann,)).is_identity
        )

    def test_has_creators(self) -> None:
        mode = make_mode()

        self.assertFalse(Monomial().has_creators)
        self.assertTrue(Monomial(creators=(mode.cre,)).has_creators)
        self.assertTrue(
            Monomial(creators=(mode.cre,), annihilators=(mode.ann,)).has_creators
        )

    def test_has_annihilators(self) -> None:
        mode = make_mode()

        self.assertFalse(Monomial().has_annihilators)
        self.assertTrue(Monomial(annihilators=(mode.ann,)).has_annihilators)
        self.assertTrue(
            Monomial(creators=(mode.cre,), annihilators=(mode.ann,)).has_annihilators
        )

    def test_double_adjoint_returns_original_monomial(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre, mode_b.cre),
            annihilators=(mode_b.ann,),
        )

        self.assertEqual(monomial.adjoint().adjoint(), monomial)


if __name__ == "__main__":
    unittest.main()
