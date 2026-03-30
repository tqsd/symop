import unittest

from symop.ccr.algebra.op.poly import OpPoly
from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.algebra.density.poly import DensityPoly
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm
from symop.core.terms.op_term import OpTerm

from tests.ccr.support.fakes import make_mode


class TestOpPolyConstructors(unittest.TestCase):
    def test_from_words(self) -> None:
        mode = make_mode()

        result = OpPoly.from_words(
            [[mode.cre], [mode.ann]],
            coeffs=[2.0, 3.0],
        )

        expected = OpPoly(
            (
                OpTerm(ops=(mode.cre,), coeff=2.0),
                OpTerm(ops=(mode.ann,), coeff=3.0),
            )
        )
        self.assertEqual(result, expected)

    def test_identity(self) -> None:
        self.assertEqual(
            OpPoly.identity(),
            OpPoly((OpTerm.identity(),)),
        )

    def test_zero(self) -> None:
        self.assertEqual(OpPoly.zero(), OpPoly(()))

    def test_a_adag_n(self) -> None:
        mode = make_mode()

        self.assertEqual(OpPoly.a(mode), OpPoly.from_words([[mode.ann]]))
        self.assertEqual(OpPoly.adag(mode), OpPoly.from_words([[mode.cre]]))
        self.assertEqual(OpPoly.n(mode), OpPoly.from_words([[mode.cre, mode.ann]]))

    def test_q_x_p(self) -> None:
        mode = make_mode()

        q = OpPoly.q(mode)
        x = OpPoly.x(mode)
        p = OpPoly.p(mode)

        self.assertEqual(q, x)
        self.assertEqual(
            q,
            (
                OpPoly.from_words([[mode.ann]]) + OpPoly.from_words([[mode.cre]])
            ) * (1.0 / (2.0**0.5)),
        )
        self.assertEqual(
            p,
            OpPoly.from_words([[mode.cre]]) * (1j / (2.0**0.5))
            + OpPoly.from_words([[mode.ann]]) * (-1j / (2.0**0.5)),
        )

    def test_X_theta_at_zero_matches_q(self) -> None:
        mode = make_mode()

        self.assertEqual(OpPoly.X_theta(mode, 0.0), OpPoly.q(mode))

    def test_q2_p2_n2(self) -> None:
        mode = make_mode()

        self.assertEqual(OpPoly.q2(mode), (OpPoly.q(mode) * OpPoly.q(mode)).combine_like_terms())
        self.assertEqual(OpPoly.p2(mode), (OpPoly.p(mode) * OpPoly.p(mode)).combine_like_terms())
        self.assertEqual(OpPoly.n2(mode), (OpPoly.n(mode) * OpPoly.n(mode)).combine_like_terms())


class TestOpPolyMethods(unittest.TestCase):
    def test_scaled(self) -> None:
        mode = make_mode()

        poly = OpPoly((OpTerm(ops=(mode.cre,), coeff=2.0 + 1.0j),))
        result = poly.scaled(3.0 - 2.0j)

        expected = OpPoly(
            (OpTerm(ops=(mode.cre,), coeff=(2.0 + 1.0j) * (3.0 - 2.0j)),)
        )
        self.assertEqual(result, expected)

    def test_adjoint(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        poly = OpPoly((OpTerm(ops=(mode_a.cre, mode_b.ann), coeff=2.0 + 3.0j),))
        result = poly.adjoint()

        expected = OpPoly((OpTerm(ops=(mode_b.cre, mode_a.ann), coeff=2.0 - 3.0j),))
        self.assertEqual(result, expected)

    def test_combine_like_terms(self) -> None:
        mode = make_mode()

        poly = OpPoly(
            (
                OpTerm(ops=(mode.cre,), coeff=2.0),
                OpTerm(ops=(mode.cre,), coeff=3.0),
            )
        )

        result = poly.combine_like_terms()

        expected = OpPoly((OpTerm(ops=(mode.cre,), coeff=5.0),))
        self.assertEqual(result, expected)

    def test_normalize_aliases_combine_like_terms(self) -> None:
        mode = make_mode()

        poly = OpPoly(
            (
                OpTerm(ops=(mode.cre,), coeff=2.0),
                OpTerm(ops=(mode.cre,), coeff=3.0),
            )
        )

        self.assertEqual(poly.normalize(), poly.combine_like_terms())

    def test_is_zero_is_identity_unique_modes_mode_count(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        zero = OpPoly.zero()
        self.assertTrue(zero.is_zero)
        self.assertFalse(zero.is_identity)

        ident = OpPoly.identity()
        self.assertFalse(ident.is_zero)
        self.assertTrue(ident.is_identity)

        poly = OpPoly(
            (
                OpTerm(ops=(mode_a.cre, mode_b.ann, mode_a.ann), coeff=1.0),
            )
        )
        self.assertEqual(poly.unique_modes, (mode_a, mode_b))
        self.assertEqual(poly.mode_count, 2)


class TestOpPolyOperators(unittest.TestCase):
    def test_add_sub_neg(self) -> None:
        mode = make_mode()

        left = OpPoly((OpTerm(ops=(mode.cre,), coeff=5.0),))
        right = OpPoly((OpTerm(ops=(mode.cre,), coeff=3.0),))

        self.assertEqual(
            left + right,
            OpPoly(
                (
                    OpTerm(ops=(mode.cre,), coeff=5.0),
                    OpTerm(ops=(mode.cre,), coeff=3.0),
                )
            ),
        )
        self.assertEqual(
            left - right,
            OpPoly(
                (
                    OpTerm(ops=(mode.cre,), coeff=5.0),
                    OpTerm(ops=(mode.cre,), coeff=-3.0),
                )
            ),
        )
        self.assertEqual(
            -right,
            OpPoly((OpTerm(ops=(mode.cre,), coeff=-3.0),)),
        )

    def test_mul_with_scalar(self) -> None:
        mode = make_mode()

        poly = OpPoly((OpTerm(ops=(mode.cre,), coeff=2.0),))
        result = poly * 3.0

        expected = OpPoly((OpTerm(ops=(mode.cre,), coeff=6.0),))
        self.assertEqual(result, expected)

    def test_mul_with_poly(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = OpPoly((OpTerm(ops=(mode_a.cre,), coeff=2.0),))
        right = OpPoly((OpTerm(ops=(mode_b.ann,), coeff=3.0),))

        result = left * right

        expected = OpPoly((OpTerm(ops=(mode_a.cre, mode_b.ann), coeff=6.0),))
        self.assertEqual(result, expected)

    def test_rmul(self) -> None:
        mode = make_mode()

        poly = OpPoly((OpTerm(ops=(mode.cre,), coeff=2.0),))
        self.assertEqual(3.0 * poly, OpPoly((OpTerm(ops=(mode.cre,), coeff=6.0),)))

    def test_truediv(self) -> None:
        mode = make_mode()

        poly = OpPoly((OpTerm(ops=(mode.cre,), coeff=6.0),))
        self.assertEqual(poly / 3.0, OpPoly((OpTerm(ops=(mode.cre,), coeff=2.0),)))

    def test_truediv_errors(self) -> None:
        poly = OpPoly.identity()

        with self.assertRaises(TypeError):
            _ = poly / "x"

        with self.assertRaises(ZeroDivisionError):
            _ = poly / 0.0

    def test_bool_eq_repr_len_iter(self) -> None:
        zero = OpPoly.zero()
        self.assertFalse(zero)
        self.assertEqual(len(zero), 0)
        self.assertEqual(list(iter(zero)), [])

        ident = OpPoly.identity()
        self.assertTrue(ident)
        self.assertEqual(len(ident), 1)
        self.assertEqual(ident, OpPoly.identity())
        self.assertIs(ident.__eq__(object()), NotImplemented)
        self.assertIn("OpPoly", repr(ident))


class TestOpPolyMatmul(unittest.TestCase):
    def test_matmul_with_op_poly(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        left = OpPoly((OpTerm(ops=(mode_a.cre,), coeff=2.0),))
        right = OpPoly((OpTerm(ops=(mode_b.ann,), coeff=3.0),))

        result = left @ right

        expected = OpPoly((OpTerm(ops=(mode_a.cre, mode_b.ann), coeff=6.0),))
        self.assertEqual(result, expected)

    def test_matmul_with_ket_poly(self) -> None:
        mode = make_mode()

        op = OpPoly(
            (
                OpTerm(ops=(mode.cre,), coeff=2.0),
                OpTerm(ops=(mode.cre,), coeff=3.0),
            )
        )
        ket = KetPoly.from_ops(coeff=1.0)

        result = op @ ket

        expected = KetPoly.from_ops(creators=(mode.cre,), coeff=5.0)
        self.assertEqual(result, expected)

    def test_matmul_with_density_poly(self) -> None:
        mode = make_mode()

        op = OpPoly(
            (
                OpTerm(ops=(mode.cre,), coeff=2.0),
                OpTerm(ops=(mode.cre,), coeff=3.0),
            )
        )
        rho = DensityPoly.identity()

        result = op @ rho

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

    def test_rmatmul_with_density_poly(self) -> None:
        mode = make_mode()

        op = OpPoly(
            (
                OpTerm(ops=(mode.cre,), coeff=2.0),
                OpTerm(ops=(mode.cre,), coeff=3.0),
            )
        )
        rho = DensityPoly.identity()

        result = rho @ op

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

    def test_matmul_returns_not_implemented_for_unsupported_type(self) -> None:
        self.assertIs(OpPoly.identity().__matmul__(object()), NotImplemented)
        self.assertIs(OpPoly.identity().__rmatmul__(object()), NotImplemented)


if __name__ == "__main__":
    unittest.main()
