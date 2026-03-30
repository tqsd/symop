import unittest

from symop.ccr.algebra.ket.from_ops import ket_from_ops
from symop.ccr.algebra.ket.inner import ket_inner

from tests.core.support.fakes import make_mode, set_hermitian_overlap


class TestKetInner(unittest.TestCase):
    def test_inner_of_empty_with_empty_is_zero(self) -> None:
        self.assertEqual(ket_inner((), ()), 0.0 + 0.0j)

    def test_inner_of_identity_with_identity_is_one(self) -> None:
        ket = ket_from_ops(coeff=1.0)

        result = ket_inner(ket, ket)

        self.assertEqual(result, 1.0 + 0.0j)

    def test_inner_is_conjugate_linear_in_left_argument(self) -> None:
        mode = make_mode()
        a = ket_from_ops(creators=(mode.cre,), coeff=2.0 + 3.0j)
        b = ket_from_ops(creators=(mode.cre,), coeff=5.0 - 1.0j)

        result = ket_inner(a, b)

        expected = (2.0 + 3.0j).conjugate() * (5.0 - 1.0j)
        self.assertEqual(result, expected)

    def test_single_particle_same_mode_has_unit_inner_product(self) -> None:
        mode = make_mode()
        a = ket_from_ops(creators=(mode.cre,), coeff=1.0)
        b = ket_from_ops(creators=(mode.cre,), coeff=1.0)

        result = ket_inner(a, b)

        self.assertEqual(result, 1.0 + 0.0j)

    def test_single_particle_orthogonal_modes_have_zero_inner_product(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = ket_from_ops(creators=(mode_a.cre,), coeff=1.0)
        b = ket_from_ops(creators=(mode_b.cre,), coeff=1.0)

        result = ket_inner(a, b)

        self.assertEqual(result, 0.0 + 0.0j)

    def test_single_particle_overlapping_modes_use_overlap(self) -> None:
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

        a = ket_from_ops(creators=(mode_a.cre,), coeff=1.0)
        b = ket_from_ops(creators=(mode_b.cre,), coeff=1.0)

        result = ket_inner(a, b)

        self.assertEqual(result, overlap)

    def test_inner_skips_small_input_terms(self) -> None:
        mode = make_mode()

        a = ket_from_ops(creators=(mode.cre,), coeff=1e-15)
        b = ket_from_ops(creators=(mode.cre,), coeff=1.0)

        result = ket_inner(a, b, eps=1e-12)

        self.assertEqual(result, 0.0 + 0.0j)

    def test_inner_of_sum_distributes_over_terms(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        a = (
            *ket_from_ops(creators=(mode_a.cre,), coeff=2.0),
            *ket_from_ops(creators=(mode_b.cre,), coeff=3.0),
        )
        b = ket_from_ops(creators=(mode_a.cre,), coeff=5.0)

        result = ket_inner(a, b)

        expected = 2.0 * 5.0
        self.assertEqual(result, expected)

    def test_inner_of_identity_with_single_particle_is_zero(self) -> None:
        mode = make_mode()

        a = ket_from_ops(coeff=1.0)
        b = ket_from_ops(creators=(mode.cre,), coeff=1.0)

        result = ket_inner(a, b)

        self.assertEqual(result, 0.0 + 0.0j)

    def test_inner_of_two_particle_same_mode_word(self) -> None:
        mode = make_mode()

        a = ket_from_ops(creators=(mode.cre, mode.cre), coeff=1.0)
        b = ket_from_ops(creators=(mode.cre, mode.cre), coeff=1.0)

        result = ket_inner(a, b)

        self.assertEqual(result, 2.0 + 0.0j)

    def test_inner_is_hermitian(self) -> None:
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

        set_hermitian_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            0.2 + 0.7j,
        )

        a = ket_from_ops(creators=(mode_a.cre,), coeff=2.0 - 1.0j)
        b = ket_from_ops(creators=(mode_b.cre,), coeff=3.0 + 4.0j)

        ab = ket_inner(a, b)
        ba = ket_inner(b, a)

        self.assertEqual(ab, ba.conjugate())


if __name__ == "__main__":
    unittest.main()
