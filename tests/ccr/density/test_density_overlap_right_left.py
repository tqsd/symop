import unittest

from symop.ccr.algebra.density.overlap_right_left import overlap_right_left
from symop.core.monomial import Monomial

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestOverlapRightLeft(unittest.TestCase):
    def test_identity_overlap_identity_is_one(self) -> None:
        self.assertEqual(
            overlap_right_left(Monomial.identity(), Monomial.identity()),
            1.0 + 0.0j,
        )

    def test_identity_overlap_non_identity_is_zero(self) -> None:
        mode = make_mode()

        self.assertEqual(
            overlap_right_left(
                Monomial.identity(),
                Monomial(creators=(mode.cre,), annihilators=()),
            ),
            0.0 + 0.0j,
        )

    def test_same_single_particle_mode_has_unit_overlap(self) -> None:
        mode = make_mode()

        r = Monomial(creators=(mode.cre,), annihilators=())
        l = Monomial(creators=(mode.cre,), annihilators=())

        self.assertEqual(overlap_right_left(r, l), 1.0 + 0.0j)

    def test_orthogonal_single_particle_modes_have_zero_overlap(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        r = Monomial(creators=(mode_a.cre,), annihilators=())
        l = Monomial(creators=(mode_b.cre,), annihilators=())

        self.assertEqual(overlap_right_left(r, l), 0.0 + 0.0j)

    def test_overlapping_single_particle_modes_use_mode_overlap(self) -> None:
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

        r = Monomial(creators=(mode_a.cre,), annihilators=())
        l = Monomial(creators=(mode_b.cre,), annihilators=())

        self.assertEqual(overlap_right_left(r, l), overlap)

    def test_two_particle_same_mode_overlap_is_two(self) -> None:
        mode = make_mode()

        r = Monomial(creators=(mode.cre, mode.cre), annihilators=())
        l = Monomial(creators=(mode.cre, mode.cre), annihilators=())

        self.assertEqual(overlap_right_left(r, l), 2.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
