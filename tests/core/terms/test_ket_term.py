import unittest

from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.core.support.fakes import make_mode


class TestKetTerm(unittest.TestCase):
    def test_identity_returns_unit_identity_term(self) -> None:
        term = KetTerm.identity()

        self.assertEqual(term.coeff, 1.0)
        self.assertTrue(term.is_identity)
        self.assertFalse(term.is_creator_only)
        self.assertFalse(term.is_annihilator_only)
        self.assertEqual(term.creation_count, 0)
        self.assertEqual(term.annihilation_count, 0)
        self.assertEqual(term.total_degree, 0)
        self.assertEqual(term.mode_ops, ())

    def test_adjoint_conjugates_coeff_and_adjoins_monomial(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre,),
            annihilators=(mode_b.ann,),
        )
        term = KetTerm(coeff=2.0 + 3.0j, monomial=monomial)

        adj = term.adjoint()

        self.assertEqual(adj.coeff, 2.0 - 3.0j)
        self.assertEqual(adj.monomial, monomial.adjoint())

    def test_scaled_multiplies_coefficient_only(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,))
        term = KetTerm(coeff=2.0 + 1.0j, monomial=monomial)

        scaled = term.scaled(3.0 - 2.0j)

        self.assertEqual(scaled.coeff, (2.0 + 1.0j) * (3.0 - 2.0j))
        self.assertEqual(scaled.monomial, monomial)
        self.assertEqual(term.coeff, 2.0 + 1.0j)

    def test_signature(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,))
        term = KetTerm(coeff=5.0, monomial=monomial)

        self.assertEqual(
            term.signature,
            ("KT", monomial.signature),
        )

    def test_approx_signature(self) -> None:
        mode = make_mode()
        monomial = Monomial(creators=(mode.cre,))
        term = KetTerm(coeff=5.0, monomial=monomial)

        self.assertEqual(
            term.approx_signature(decimals=7, ignore_global_phase=True),
            (
                "KT_approx",
                monomial.approx_signature(
                    decimals=7,
                    ignore_global_phase=True,
                ),
            ),
        )

    def test_is_creator_only_delegates_to_monomial(self) -> None:
        mode = make_mode()

        creator_only = KetTerm(
            coeff=1.0,
            monomial=Monomial(creators=(mode.cre,)),
        )
        mixed = KetTerm(
            coeff=1.0,
            monomial=Monomial(
                creators=(mode.cre,),
                annihilators=(mode.ann,),
            ),
        )
        identity = KetTerm.identity()

        self.assertTrue(creator_only.is_creator_only)
        self.assertFalse(mixed.is_creator_only)
        self.assertFalse(identity.is_creator_only)

    def test_is_annihilator_only_delegates_to_monomial(self) -> None:
        mode = make_mode()

        annihilator_only = KetTerm(
            coeff=1.0,
            monomial=Monomial(annihilators=(mode.ann,)),
        )
        mixed = KetTerm(
            coeff=1.0,
            monomial=Monomial(
                creators=(mode.cre,),
                annihilators=(mode.ann,),
            ),
        )
        identity = KetTerm.identity()

        self.assertTrue(annihilator_only.is_annihilator_only)
        self.assertFalse(mixed.is_annihilator_only)
        self.assertFalse(identity.is_annihilator_only)

    def test_is_identity_delegates_to_monomial(self) -> None:
        mode = make_mode()

        self.assertTrue(KetTerm.identity().is_identity)
        self.assertFalse(
            KetTerm(coeff=1.0, monomial=Monomial(creators=(mode.cre,))).is_identity
        )
        self.assertFalse(
            KetTerm(
                coeff=1.0,
                monomial=Monomial(annihilators=(mode.ann,)),
            ).is_identity
        )

    def test_creation_count(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = KetTerm(
            coeff=1.0,
            monomial=Monomial(
                creators=(mode_a.cre, mode_b.cre),
                annihilators=(mode_a.ann,),
            ),
        )

        self.assertEqual(term.creation_count, 2)

    def test_annihilation_count(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = KetTerm(
            coeff=1.0,
            monomial=Monomial(
                creators=(mode_a.cre,),
                annihilators=(mode_a.ann, mode_b.ann),
            ),
        )

        self.assertEqual(term.annihilation_count, 2)

    def test_total_degree_is_sum_of_creation_and_annihilation_counts(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = KetTerm(
            coeff=1.0,
            monomial=Monomial(
                creators=(mode_a.cre, mode_b.cre),
                annihilators=(mode_b.ann,),
            ),
        )

        self.assertEqual(term.creation_count, 2)
        self.assertEqual(term.annihilation_count, 1)
        self.assertEqual(term.total_degree, 3)

    def test_mode_ops_delegates_to_monomial(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre, mode_b.cre, mode_a.cre),
            annihilators=(mode_b.ann,),
        )
        term = KetTerm(coeff=1.0, monomial=monomial)

        self.assertEqual(term.mode_ops, (mode_a, mode_b))

    def test_double_adjoint_returns_original_term(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        term = KetTerm(
            coeff=1.0 + 2.0j,
            monomial=Monomial(
                creators=(mode_a.cre,),
                annihilators=(mode_b.ann,),
            ),
        )

        self.assertEqual(term.adjoint().adjoint(), term)


if __name__ == "__main__":
    unittest.main()
