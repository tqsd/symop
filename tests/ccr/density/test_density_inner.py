import unittest

from symop.ccr.algebra.density.inner import density_inner
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestDensityInner(unittest.TestCase):
    def test_empty_with_empty_is_zero(self) -> None:
        self.assertEqual(density_inner((), ()), 0.0 + 0.0j)

    def test_identity_density_with_itself_is_one(self) -> None:
        term = DensityTerm(
            coeff=1.0,
            left=Monomial.identity(),
            right=Monomial.identity(),
        )

        self.assertEqual(density_inner((term,), (term,)), 1.0 + 0.0j)

    def test_single_rank_one_term_uses_left_and_right_overlaps(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = (
            DensityTerm(
                coeff=2.0 + 1.0j,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )
        b = (
            DensityTerm(
                coeff=3.0 - 2.0j,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        expected = (2.0 + 1.0j).conjugate() * (3.0 - 2.0j)
        self.assertEqual(density_inner(a, b), expected)

    def test_orthogonal_left_overlap_gives_zero(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )
        b = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        self.assertEqual(density_inner(a, b), 0.0 + 0.0j)

    def test_overlapping_modes_respect_hermitian_structure(self) -> None:
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

        overlap = 0.2 + 0.7j
        set_hermitian_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            overlap,
        )

        a = (
            DensityTerm(
                coeff=2.0 - 1.0j,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )
        b = (
            DensityTerm(
                coeff=3.0 + 4.0j,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        ab = density_inner(a, b)
        ba = density_inner(b, a)

        self.assertEqual(ab, ba.conjugate())

    def test_inner_distributes_over_multiple_terms(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )
        b = (
            DensityTerm(
                coeff=5.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial.identity(),
            ),
        )

        self.assertEqual(density_inner(a, b), 10.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
