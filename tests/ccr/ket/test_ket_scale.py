import unittest

from symop.ccr.algebra.ket.scale import ket_scale
from symop.core.monomial import Monomial
from symop.core.terms.ket_term import KetTerm

from tests.ccr.support.fakes import make_mode


class TestKetScale(unittest.TestCase):
    def test_empty_terms_return_empty_tuple(self) -> None:
        result = ket_scale((), 2.0)
        self.assertEqual(result, ())

    def test_scales_single_term_coefficient(self) -> None:
        mode = make_mode()
        terms = (
            KetTerm(
                coeff=2.0 + 1.0j,
                monomial=Monomial(creators=(mode.cre,)),
            ),
        )

        result = ket_scale(terms, 3.0 - 2.0j)

        expected = (
            KetTerm(
                coeff=(3.0 - 2.0j) * (2.0 + 1.0j),
                monomial=Monomial(creators=(mode.cre,)),
            ),
        )
        self.assertEqual(result, expected)

    def test_scales_multiple_terms_and_preserves_monomials(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            KetTerm(
                coeff=2.0,
                monomial=Monomial(creators=(mode_a.cre,)),
            ),
            KetTerm(
                coeff=3.0 + 1.0j,
                monomial=Monomial(annihilators=(mode_b.ann,)),
            ),
        )

        result = ket_scale(terms, -2.0)

        expected = (
            KetTerm(
                coeff=-4.0,
                monomial=Monomial(creators=(mode_a.cre,)),
            ),
            KetTerm(
                coeff=-6.0 - 2.0j,
                monomial=Monomial(annihilators=(mode_b.ann,)),
            ),
        )
        self.assertEqual(result, expected)

    def test_scaling_by_zero_zeroes_coefficients_but_keeps_structure(self) -> None:
        mode = make_mode()
        terms = (
            KetTerm(
                coeff=2.0 + 3.0j,
                monomial=Monomial(creators=(mode.cre,), annihilators=(mode.ann,)),
            ),
        )

        result = ket_scale(terms, 0.0)

        expected = (
            KetTerm(
                coeff=0.0 + 0.0j,
                monomial=Monomial(creators=(mode.cre,), annihilators=(mode.ann,)),
            ),
        )
        self.assertEqual(result, expected)

    def test_scaling_by_one_returns_equal_terms(self) -> None:
        mode = make_mode()
        terms = (
            KetTerm(
                coeff=5.0,
                monomial=Monomial(creators=(mode.cre,)),
            ),
        )

        result = ket_scale(terms, 1.0)

        self.assertEqual(result, terms)


if __name__ == "__main__":
    unittest.main()
