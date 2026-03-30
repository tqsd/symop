import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import (
    make_fake_op_poly,
    make_mode,
    make_op_term,
    set_hermitian_overlap,
)


class TestDensityPolyConstructors(unittest.TestCase):
    def test_zero(self) -> None:
        poly = DensityPoly.zero()
        self.assertEqual(poly.terms, ())
        self.assertFalse(poly)

    def test_identity(self) -> None:
        poly = DensityPoly.identity()

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0 + 0.0j,
                    left=Monomial.identity(),
                    right=Monomial.identity(),
                ),
            )
        )
        self.assertEqual(poly, expected)

    def test_pure(self) -> None:
        mode = make_mode()
        ket = KetPoly.from_ops(creators=(mode.cre,), coeff=2.0)

        poly = DensityPoly.pure(ket)

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=4.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial(creators=(mode.cre,), annihilators=()),
                ),
            )
        )
        self.assertEqual(poly, expected)


class TestDensityPolyMethods(unittest.TestCase):
    def test_combine_like_terms(self) -> None:
        mode = make_mode()

        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
                DensityTerm(
                    coeff=3.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )

        result = poly.combine_like_terms()

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=5.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )
        self.assertEqual(result, expected)

    def test_normalize_aliases_combine_like_terms(self) -> None:
        mode = make_mode()

        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
                DensityTerm(
                    coeff=3.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )

        self.assertEqual(poly.normalize(), poly.combine_like_terms())

    def test_scaled(self) -> None:
        mode = make_mode()

        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )

        result = poly.scaled(3.0)

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=6.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )
        self.assertEqual(result, expected)

    def test_apply_left(self) -> None:
        mode = make_mode()

        poly = DensityPoly.identity()
        result = poly.apply_left((mode.cre,))

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0 + 0.0j,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )
        self.assertEqual(result, expected)

    def test_apply_right(self) -> None:
        mode = make_mode()

        poly = DensityPoly.identity()
        result = poly.apply_right((mode.ann,))

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0 + 0.0j,
                    left=Monomial.identity(),
                    right=Monomial(creators=(mode.cre,), annihilators=()),
                ),
            )
        )
        self.assertEqual(result, expected)

    def test_trace(self) -> None:
        self.assertEqual(DensityPoly.identity().trace(), 1.0 + 0.0j)

    def test_normalize_trace(self) -> None:
        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0,
                    left=Monomial.identity(),
                    right=Monomial.identity(),
                ),
            )
        )

        result = poly.normalize_trace()

        self.assertEqual(result.trace(), 1.0 + 0.0j)

    def test_inner(self) -> None:
        self.assertEqual(
            DensityPoly.identity().inner(DensityPoly.identity()),
            1.0 + 0.0j,
        )

    def test_purity(self) -> None:
        self.assertEqual(DensityPoly.identity().purity(), 1.0)

    def test_partial_trace(self) -> None:
        mode = make_mode()

        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial(creators=(mode.cre,), annihilators=()),
                ),
            )
        )

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0,
                    left=Monomial.identity(),
                    right=Monomial.identity(),
                ),
            )
        )

        self.assertEqual(poly.partial_trace((mode,)), expected)

    def test_hs_norm2_and_hs_norm(self) -> None:
        poly = DensityPoly.identity()

        self.assertEqual(poly.hs_norm2(), 1.0)
        self.assertEqual(poly.hs_norm(), 1.0)


class TestDensityPolyProperties(unittest.TestCase):
    def test_unique_modes_and_mode_count(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0,
                    left=Monomial(
                        creators=(mode_a.cre,),
                        annihilators=(mode_b.ann,),
                    ),
                    right=Monomial(
                        creators=(mode_b.cre,),
                        annihilators=(),
                    ),
                ),
            )
        )

        self.assertEqual(poly.unique_modes, (mode_a, mode_b))
        self.assertEqual(poly.mode_count, 2)

    def test_is_diagonal_in_monomials(self) -> None:
        mode = make_mode()

        diag = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial(creators=(mode.cre,), annihilators=()),
                ),
            )
        )
        offdiag = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0,
                    left=Monomial.identity(),
                    right=Monomial(creators=(mode.cre,), annihilators=()),
                ),
            )
        )

        self.assertTrue(diag.is_diagonal_in_monomials)
        self.assertFalse(offdiag.is_diagonal_in_monomials)

    def test_identity_side_properties(self) -> None:
        mode = make_mode()

        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0,
                    left=Monomial.identity(),
                    right=Monomial(creators=(mode.cre,), annihilators=()),
                ),
            )
        )

        self.assertTrue(poly.is_identity_left)
        self.assertFalse(poly.is_identity_right)

    def test_creator_only_properties(self) -> None:
        mode = make_mode()

        poly = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0,
                    left=Monomial(creators=(mode.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )

        self.assertTrue(poly.is_creator_only_left)
        self.assertTrue(poly.is_creator_only_right)
        self.assertTrue(poly.is_creator_only)

    def test_trace_normalization_checks(self) -> None:
        ident = DensityPoly.identity()
        self.assertTrue(ident.is_trace_normalized())
        ident.require_trace_normalized()

        nonnorm = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0,
                    left=Monomial.identity(),
                    right=Monomial.identity(),
                ),
            )
        )
        self.assertFalse(nonnorm.is_trace_normalized())
        with self.assertRaises(ValueError):
            nonnorm.require_trace_normalized()

    def test_is_pure(self) -> None:
        self.assertTrue(DensityPoly.identity().is_pure())

        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")
        mixed = DensityPoly(
            (
                DensityTerm(
                    coeff=0.5,
                    left=Monomial(creators=(mode_a.cre,), annihilators=()),
                    right=Monomial(creators=(mode_a.cre,), annihilators=()),
                ),
                DensityTerm(
                    coeff=0.5,
                    left=Monomial(creators=(mode_b.cre,), annihilators=()),
                    right=Monomial(creators=(mode_b.cre,), annihilators=()),
                ),
            )
        )
        self.assertFalse(mixed.is_pure())

    def test_is_block_diagonal_by_modes(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        block_diag = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0,
                    left=Monomial(creators=(mode_a.cre,), annihilators=()),
                    right=Monomial(creators=(mode_a.cre,), annihilators=()),
                ),
            )
        )
        not_block_diag = DensityPoly(
            (
                DensityTerm(
                    coeff=1.0,
                    left=Monomial(creators=(mode_a.cre,), annihilators=()),
                    right=Monomial(creators=(mode_b.cre,), annihilators=()),
                ),
            )
        )

        self.assertTrue(block_diag.is_block_diagonal_by_modes())
        self.assertFalse(not_block_diag.is_block_diagonal_by_modes())


class TestDensityPolyOperators(unittest.TestCase):
    def test_len_iter_bool(self) -> None:
        zero = DensityPoly.zero()
        self.assertEqual(len(zero), 0)
        self.assertEqual(list(iter(zero)), [])
        self.assertFalse(zero)

        ident = DensityPoly.identity()
        self.assertEqual(len(ident), 1)
        self.assertTrue(ident)

    def test_add_sub_neg(self) -> None:
        left = DensityPoly(
            (
                DensityTerm(
                    coeff=5.0,
                    left=Monomial.identity(),
                    right=Monomial.identity(),
                ),
            )
        )
        right = DensityPoly(
            (
                DensityTerm(
                    coeff=3.0,
                    left=Monomial.identity(),
                    right=Monomial.identity(),
                ),
            )
        )

        self.assertEqual(
            left + right,
            DensityPoly(
                (
                    DensityTerm(
                        coeff=8.0,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                )
            ),
        )
        self.assertEqual(
            left - right,
            DensityPoly(
                (
                    DensityTerm(
                        coeff=2.0,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                )
            ),
        )
        self.assertEqual(
            -right,
            DensityPoly(
                (
                    DensityTerm(
                        coeff=-3.0,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                )
            ),
        )

    def test_mul_rmul_truediv(self) -> None:
        ident = DensityPoly.identity()

        self.assertEqual(
            ident * 2.0,
            DensityPoly(
                (
                    DensityTerm(
                        coeff=2.0 + 0.0j,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                )
            ),
        )
        self.assertEqual(
            2.0 * ident,
            DensityPoly(
                (
                    DensityTerm(
                        coeff=2.0 + 0.0j,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                )
            ),
        )
        self.assertEqual(
            ident / 2.0,
            DensityPoly(
                (
                    DensityTerm(
                        coeff=0.5 + 0.0j,
                        left=Monomial.identity(),
                        right=Monomial.identity(),
                    ),
                )
            ),
        )

    def test_mul_by_density_poly(self) -> None:
        ident = DensityPoly.identity()
        self.assertEqual(ident * ident, ident)

    def test_truediv_errors(self) -> None:
        ident = DensityPoly.identity()

        with self.assertRaises(TypeError):
            _ = ident / "x"

        with self.assertRaises(ZeroDivisionError):
            _ = ident / 0.0

    def test_eq_and_repr(self) -> None:
        ident = DensityPoly.identity()
        self.assertEqual(ident, DensityPoly.identity())
        self.assertIs(ident.__eq__(object()), NotImplemented)
        self.assertIn("DensityPoly", repr(ident))

    def test_matmul_with_fake_op_poly(self) -> None:
        mode = make_mode()

        rho = DensityPoly.identity()
        op_poly = make_fake_op_poly(
            make_op_term(ops=(mode.cre,), coeff=2.0),
            make_op_term(ops=(mode.cre,), coeff=3.0),
        )

        result = rho @ op_poly

        expected = DensityPoly(
            (
                DensityTerm(
                    coeff=5.0,
                    left=Monomial.identity(),
                    right=Monomial(creators=(), annihilators=(mode.ann,)),
                ),
            )
        )
        self.assertEqual(result, expected)

    def test_matmul_returns_not_implemented_for_other_type(self) -> None:
        self.assertIs(DensityPoly.identity().__matmul__(object()), NotImplemented)

    def test_rmatmul_returns_not_implemented(self) -> None:
        self.assertIs(DensityPoly.identity().__rmatmul__(object()), NotImplemented)


class TestDensityPolyHermitianStructure(unittest.TestCase):
    def test_inner_is_hermitian_for_overlapping_modes(self) -> None:
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

        a = DensityPoly(
            (
                DensityTerm(
                    coeff=2.0 - 1.0j,
                    left=Monomial(creators=(mode_a.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )
        b = DensityPoly(
            (
                DensityTerm(
                    coeff=3.0 + 4.0j,
                    left=Monomial(creators=(mode_b.cre,), annihilators=()),
                    right=Monomial.identity(),
                ),
            )
        )

        ab = a.inner(b)
        ba = b.inner(a)

        self.assertEqual(ab, ba.conjugate())


if __name__ == "__main__":
    unittest.main()
