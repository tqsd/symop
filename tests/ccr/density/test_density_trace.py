import unittest

from symop.ccr.algebra.density.trace import density_trace
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestDensityTrace(unittest.TestCase):
    def test_empty_density_has_zero_trace(self) -> None:
        self.assertEqual(density_trace(()), 0.0 + 0.0j)

    def test_identity_density_has_unit_trace(self) -> None:
        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )

        self.assertEqual(density_trace(terms), 1.0 + 0.0j)

    def test_trace_uses_overlap_of_right_and_left(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        self.assertEqual(density_trace(terms), 2.0 + 0.0j)

    def test_orthogonal_term_has_zero_trace(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        self.assertEqual(density_trace(terms), 0.0 + 0.0j)

    def test_overlapping_modes_use_overlap_in_trace(self) -> None:
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
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
        )

        self.assertEqual(density_trace(terms), 3.0 * overlap)

    def test_trace_is_linear(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        self.assertEqual(density_trace(terms), 5.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
